import numpy as np
import matplotlib.pyplot as plt
import control as ct
import time
import logging
from pathlib import Path

# Custom functions
import scr.models.one_dimensional_basic as model
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
        self._basic_experiment()

    def _basic_experiment(self):
        plant = model.mass_spring_damper(self.cfg.system, self.cfg.simulation)

        # configuration and reference values
        xref = np.array([1.0, 0.0])  # reference: track position x = 1.0
        uref = np.array([0.0])
        Ts = self.cfg.simulation.Ts
        N = self.cfg.controller.N

        # Initialize both controllers
        builtin_mpc = custom_mpc.default_mpc(plant, xref, uref, Ts, N)
        cvx_mpc = custom_mpc.cvxpy_solver_mpc(plant, xref, uref, Ts, N)

        # Feedback systems
        loop_builtin = ct.feedback(plant, builtin_mpc, sign=1)
        loop_custom = ct.feedback(plant, cvx_mpc, sign=1)

        T = np.arange(0, self.cfg.simulation.Tsim + Ts, Ts)

        # Simulate and measure computation time for each controller
        xb, builtin_time = self._simulate_and_time(loop_builtin, T)
        xc, custom_time = self._simulate_and_time(loop_custom, T)

        # Log computation time
        self.log.info(f"Builtin MPC computation time: {builtin_time:.4f} s")
        self.log.info(f"Custom MPC computation time: {custom_time:.4f} s")

        # Validation metrics
        position_diff = np.abs(xb[0] - xc[0])  # Position difference
        self._log_validation_metrics(position_diff)

        # Plotting results
        self._plot_results(T, xb, xc, xref, position_diff)

    def _simulate_and_time(self, system, time_vector):
        """Simulate the system and measure computation time."""
        start_time = time.time()
        self.log.info(f"Starting simulation at t = {start_time:.2f} s")
        # Simulate system output (xb or xc)
        t, y = ct.input_output_response(system, time_vector, U=0)

        final_time = time.time()
        self.log.info(f"Ending the simulation at t = {start_time:.2f} s")
        elapsed_time = final_time - start_time
        return y, elapsed_time

    def _log_validation_metrics(self, position_diff):
        """Log validation comparison metrics."""
        avg_pos_diff = np.mean(position_diff)
        max_pos_diff = np.max(position_diff)
        self.log.info(f"Average Position Difference: {avg_pos_diff:.4f}")
        self.log.info(f"Max Position Difference: {max_pos_diff:.4f}")

    def _plot_results(self, T, xb, xc, xref, position_diff):
        """Plot or save results based on configuration."""
        plt.figure(figsize=(12, 8))

        # Position and velocity trajectories
        plt.subplot(3, 1, 1)
        plt.plot(T, xb[0], label='Position (Built-in MPC)')
        plt.plot(T, xc[0], label='Position (CVX MPC)')
        plt.plot(T, xref[0] * np.ones_like(T), 'k:', label='Reference Position')
        plt.ylabel('Position [m]')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(T, xb[1], label='Velocity (Built-in MPC)')
        plt.plot(T, xc[1], label='Velocity (CVX MPC)')
        plt.ylabel('Velocity [m/s]')
        plt.xlabel('Time [s]')
        plt.grid(True)

        # Position difference plot
        plt.subplot(3, 1, 3)
        plt.plot(T, position_diff, 'r', label='Position Difference')
        plt.ylabel('Difference [m]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if self.cfg.experiment.save_figures:
            save_path = Path(self.run_dir, "results.png")
            plt.savefig(save_path)
            self.log.info(f"Figure saved at {save_path}")
        if self.cfg.experiment.plot_figures:
            plt.show()
        plt.close()