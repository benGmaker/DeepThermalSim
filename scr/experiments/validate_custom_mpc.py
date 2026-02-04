import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.optimal as opt
import time
import logging
from pathlib import Path

# Custom functions
import scr.models.one_dimensional_basic as one_dim_models
import scr.models.thermal_systems as thermal_models
import scr.controllers.custom_mpc as custom_mpc

# Logging setup
topic = 'experiment'
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(topic)


class ValidateCustomMPC:
    """
    Multi-run experiment testing:
    - Validation of custom MPC functionality against built-in MPC.
    - Performance comparison (computation time, position tracking, input usage).
    """

    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        self.run_dir = run_dir
        self.log = logging.getLogger(topic)

    def run(self):
        '''Run the validation experiment based on the system configuration.'''
        # Define a mapping between system names and their corresponding experiment functions
        # This keeps track of which functions have been implemented and validated to work on this function,
        #  if it is not mapped it has not been used within this experiment and may not be compatible
        system_function_map = {
            "mass_spring_damper": one_dim_models.mass_spring_damper,
            "two_transistor_heat": thermal_models.two_transistor_heat,
            # Add more mappings as needed
        }

        # Get the system name from the configuration
        system_name = self.cfg.system.name
        self.log.info(f"Running validation experiment for system: {system_name}") 
        self.log.info(f"System type: {self.cfg.system.type}")

        # Select the appropriate function based on the system name
        if system_name in system_function_map:
            self._basic_experiment(system_function_map[system_name])
        else:
            raise ValueError(f"Unknown system name: {system_name}, it is incorrect or not inplemented")

    def _basic_experiment(self, model_func):
        """Basic experiment comparing built-in MPC with custom CVXPY-based MPC."""
        self.log.info("Starting basic experiment: Built-in MPC vs Custom CVXPY MPC")
        # Define the plant model

        # Reference values from configuration (must be in the config file)
        xref = self.cfg.system.xref
        uref = self.cfg.system.uref
        xref = list(self.cfg.system.xref)
        uref = list(self.cfg.system.uref)
        if self.cfg.system.type == "linear":
            plant = model_func(self.cfg.system, self.cfg.simulation)
            dt_ln_plant = plant
        elif self.cfg.system.type == "nonlinear":
            plant = model_func(self.cfg.system)
            ct_ln_plant = ct.linearize(plant, xref, uref) # Plant is linearized around desired reference point, note might not be optimal condition
            dt_ln_plant = ct.c2d(ct_ln_plant, self.cfg.simulation.Ts, method='zoh')
        else:
            raise ValueError(f"Unknown system type: {self.cfg.system.type}")

        # configuration and reference values
        Ts = self.cfg.simulation.Ts
        N = self.cfg.controller.N

        # Initialize both controllers with updated configuration
        builtin_mpc = custom_mpc.default_mpc(dt_ln_plant, xref, uref, Ts, N, self.cfg)
        cvx_mpc = custom_mpc.cvxpy_solver_mpc(dt_ln_plant, xref, uref, Ts, N, self.cfg)
        opt.solve_optimal_trajectory()

        # Simulate and measure computation time for each controller
        T = np.arange(0, self.cfg.simulation.Tsim + Ts, Ts)
        xb, builtin_time = self._simulate_and_time(builtin_mpc, plant, T)
        xc, custom_time = self._simulate_and_time(cvx_mpc, plant, T)

        # Log computation time
        self.log.info(f"Builtin MPC computation time: {builtin_time:.4f} s")
        self.log.info(f"Custom MPC computation time: {custom_time:.4f} s")

        # Validation metrics
        position_diff = np.round(np.abs(xb[0] - xc[0]), 5) # Position difference
        self._log_validation_metrics(position_diff)

        # Plotting results
        self._plot_results(T, xb, xc, xref, position_diff)

    def _simulate_and_time(self, controller, plant, time_vector):
        """Simulate the system and measure computation time."""
        start_time = time.time()
        self.log.info(f"Starting simulation at t = {start_time:.2f} s")

        if self.cfg.system.type == "linear":
            control_loop = ct.feedback(plant, controller, sign=1)
            t, y = ct.input_output_response(control_loop, time_vector, U=0)
        elif self.cfg.system.type == "nonlinear":
            y = self._simulate_non_linear(controller, plant, time_vector)
        else:
            raise ValueError(f"Unknown system type: {self.cfg.system.type}")

        ct.optimal.solve_optimal_trajectory()
        final_time = time.time()
        self.log.info(f"Ending the simulation at t = {final_time:.2f} s")
        elapsed_time = final_time - start_time
        return y, elapsed_time

    def _simulate_non_linear(self, controller, plant, time_vector):
        """Simulate the nonlinear system with the given controller."""
        # Initialize the state and output storage
        num_steps = len(time_vector)
        states = np.zeros((plant.nstates, num_steps))  # State trajectory
        outputs = np.zeros((plant.noutputs, num_steps))  # Output trajectory
        inputs = np.zeros((plant.ninputs, num_steps))  # Control inputs

        # Initial state of the plant
        x = self.cfg.system.initial_state  # Initial state, e.g., np.zeros((plant.nstates,))
        states[:, 0] = x

        # Time discretization
        dt = self.cfg.simulation.Ts

        # Closed-loop control simulation loop
        for i in range(1, num_steps):
            t0 = time_vector[i - 1]
            t1 = time_vector[i]

            # (1) Compute the control input using the controller
            u = controller.output(t=t0, X0=x)  # Custom MPC logic to compute control
            inputs[:, i] = u

            # (2) Simulate the plant for the current time step
            t_plant, y_plant, x_plant = ct.input_output_response(
                plant,
                T=[t0, t1],  # Simulate over one timestep
                U=u,  # Control input for this timestep
                X0=x,  # Initial state for the plant at this timestep
                return_x=True
            )

            # Record the plant output and state (end of the timestep)
            x = x_plant[:, -1]  # Update the state for the next iteration
            states[:, i] = x
            outputs[:, i] = y_plant[-1]

        return states
    
    def _log_validation_metrics(self, position_diff):
        """Log validation comparison metrics."""
        avg_pos_diff = np.mean(position_diff)
        max_pos_diff = np.max(position_diff)        
        self.log.info(f"Average Position Difference: {avg_pos_diff:.4f}")
        self.log.info(f"Max Position Difference: {max_pos_diff:.4f}")

    def _plot_results(self, T, xb, xc, xref, position_diff):
        """Plot or save results based on configuration."""

        # Dynamically fetch axis naming variables from the configuration
        x_axis_label = self.cfg.system.get("time_axis_label", "Time [s]")
        state_labels = self.cfg.system.get("state_labels", ["State 1", "State 2"])
        diff_label = self.cfg.system.get("diff_label", "State Difference")
        state_units = self.cfg.system.get("state_units", ["[unit1]", "[unit2]"])

        plt.figure(figsize=(12, 8))

        # Plot states
        for i in range(len(state_labels)):
            plt.subplot(len(state_labels) + 1, 1, i + 1)
            plt.plot(T, xb[i], 'b', label=f'{state_labels[i]} (Built-in MPC)')
            plt.plot(T, xc[i], 'orange', linestyle='dotted', label=f'{state_labels[i]} (Custom CVX MPC)')
            plt.plot(T, xref[i] * np.ones_like(T), 'k:', label=f'Reference {state_labels[i]}')
            plt.ylabel(f"{state_labels[i]} {state_units[i]}")
            plt.legend()
            plt.grid(True)

        # Plot difference
        plt.subplot(len(state_labels) + 1, 1, len(state_labels) + 1)
        plt.plot(T, position_diff, 'r', label=f'{diff_label}')
        plt.ylabel(diff_label)
        plt.xlabel(x_axis_label)
        plt.legend()
        plt.grid(True)

        # Save or show the plot
        plt.tight_layout()
        if self.cfg.experiment.get("save_figures", False):
            save_path = Path(self.run_dir, f"results_{self.cfg.system.name}.png")
            plt.savefig(save_path)
            self.log.info(f"Figure saved at {save_path}")
        if self.cfg.experiment.get("plot_figures", False):
            plt.show()
        plt.close()