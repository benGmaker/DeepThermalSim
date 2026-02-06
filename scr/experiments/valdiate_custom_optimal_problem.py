import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
import logging
from scr.controllers.custom_optimal_problem import CustomOptimalControlProblem
from pathlib import Path

topic = 'experiment'

class ValidateCustomOptimalProblem:
    """
    Experiment to validate and compare a custom optimal control problem against a built-in implementation.
    This script demonstrates solving problems using the CustomOptimalControlProblem class and provides
    a reusable framework for defining and testing control problems.
    """

    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(topic)

    def run(self):
        """Run the experiment for the given dynamic system."""
        self.log.info("Setting up the experiment...")

        # Build the control problem
        vehicle, x0, u0, xf, uf, problem_cfg, matrices = define_vehicle_problem(self.cfg)
        Ts = problem_cfg["Ts"]
        Tf = problem_cfg["Tf"]
        N = problem_cfg["N"]

        # Create time points
        timepts = np.linspace(0, Tf, N)
        
        # STEP 1: Solve using DEFAULT python-control optimal control
        self.log.info("=" * 60)
        self.log.info("STEP 1: Solving with DEFAULT optimal control")
        self.log.info("=" * 60)
        solution_default = self.solve_default_optimal(vehicle, x0, u0, xf, uf, timepts, matrices)
        
        # STEP 2: Solve using CUSTOM optimal control problem
        self.log.info("=" * 60)
        self.log.info("STEP 2: Solving with CUSTOM optimal control")
        self.log.info("=" * 60)
        solution_custom = self.solve_custom_mpc(vehicle, timepts, x0, u0, xf, uf, matrices)

        # STEP 3: Compare results
        self.log.info("=" * 60)
        self.log.info("STEP 3: Comparing results")
        self.log.info("=" * 60)
        self.compare_results(vehicle, timepts, x0, solution_default, solution_custom)

    def solve_default_optimal(self, vehicle, x0, u0, xf, uf, timepts, matrices):
        """Solve using the default python-control optimal control solver."""
        self.log.info("Solving with built-in solve_flat_optimal...")
        
        # Unpack cost matrices
        traj_cost, term_cost, constraints, Q, R, Qf = matrices
        
        # Create a flat system representation
        vehicle_flat = self._create_flat_system(vehicle)
        
        # Use a straight line as the initial guess
        evalpts = timepts
        initial_guess = np.array(
            [x0[i] + (xf[i] - x0[i]) * evalpts / timepts[-1] for i in (0, 1)]
        )
        
        # Solve the optimal control problem using default solver
        try:
            traj = fs.solve_flat_optimal(
                vehicle_flat, 
                evalpts, 
                x0, 
                u0, 
                traj_cost,
                terminal_cost=term_cost, 
                initial_guess=initial_guess,
                basis=fs.BSplineFamily([0, timepts[-1]/2, timepts[-1]], 4)
            )
            
            # Get the response
            resp_default = traj.response(timepts)
            self.log.info("Default optimization completed successfully.")
            return resp_default
            
        except Exception as e:
            self.log.error(f"Default optimization failed: {e}")
            return None

    def _create_flat_system(self, vehicle):
        """Create a flat system from the vehicle dynamics."""
        
        # Flat system forward map
        def vehicle_flat_forward(x, u, params={}):
            b = params.get('wheelbase', 3.)
            zflag = [np.zeros(3), np.zeros(3)]
            
            # Flat output is the x, y position
            zflag[0][0] = x[0]
            zflag[1][0] = x[1]
            
            # First derivatives
            zflag[0][1] = u[0] * np.cos(x[2])
            zflag[1][1] = u[0] * np.sin(x[2])
            
            # Second derivatives
            thdot = (u[0]/b) * np.tan(u[1])
            zflag[0][2] = -u[0] * thdot * np.sin(x[2])
            zflag[1][2] = u[0] * thdot * np.cos(x[2])
            
            return zflag
        
        # Flat system reverse map
        def vehicle_flat_reverse(zflag, params={}):
            b = params.get('wheelbase', 3.)
            x = np.zeros(3)
            u = np.zeros(2)
            
            x[0] = zflag[0][0]
            x[1] = zflag[1][0]
            x[2] = np.arctan2(zflag[1][1], zflag[0][1])
            
            u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
            u[1] = np.arctan2(
                (zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])), 
                u[0]/b
            )
            
            return x, u
        
        # Get update function from vehicle system
        def vehicle_update(t, x, u, params):
            b = params.get('wheelbase', 3.)
            phimax = params.get('maxsteer', 0.5)
            phi = np.clip(u[1], -phimax, phimax)
            return np.array([
                np.cos(x[2]) * u[0],
                np.sin(x[2]) * u[0],
                (u[0] / b) * np.tan(phi)
            ])
        
        return fs.flatsys(
            vehicle_flat_forward, 
            vehicle_flat_reverse,
            updfcn=vehicle_update, 
            outfcn=None, 
            name='vehicle_flat',
            inputs=('v', 'delta'), 
            outputs=('x', 'y'), 
            states=('x', 'y', 'theta')
        )

    def solve_custom_mpc(self, vehicle, timepts, x0, u0, xf, uf, matrices):
        """Solve the custom optimal control problem using CustomOptimalControlProblem."""
        self.log.info("Solving trajectory optimization using CustomOptimalControlProblem...")

        # Unpack cost and constraints
        traj_cost, term_cost, constraints, Q, R, Qf = matrices

        # Build the custom optimal control problem
        ocp = CustomOptimalControlProblem(
            sys=vehicle,
            timepts=timepts,
            integral_cost=traj_cost,
            terminal_cost=term_cost,
            trajectory_constraints=constraints,
            Q=Q,
            R=R,
            Qf=Qf,
            x0=x0,  # Initial state for linearization
            u0=u0   # Initial input for linearization
        )

        # Solve the optimal control problem from x0
        solution = ocp.compute_trajectory(x0)

        self.log.info("Custom optimization completed.")
        return solution

    def compare_results(self, vehicle, timepts, x0, solution_default, solution_custom):
        """Compare and visualize both solutions."""
        self.log.info("Comparing default vs custom solutions...")
        
        if solution_default is None:
            self.log.warning("Default solution not available for comparison")
            self.validate_simulation(vehicle, timepts, x0, solution_custom, "custom")
            return
        
        # Simulate custom solution
        resp_custom = ct.input_output_response(
            vehicle, timepts, solution_custom.inputs, x0
        )
        
        # Plot comparison
        self.plot_comparison(solution_default, resp_custom, timepts)
        
        # Calculate cost difference
        self.log.info("Cost comparison:")
        self.log.info(f"  Custom solution cost: {solution_custom.cost}")

    def validate_simulation(self, vehicle, timepts, x0, solution, label=""):
        """Simulate and validate a single solution."""
        self.log.info(f"Validating {label} solution...")

        # Simulate the plant with the solved inputs
        resp = ct.input_output_response(vehicle, timepts, solution.inputs, x0)

        # Log and plot validation results
        self.log.info("Validation complete. Generating results...")
        self.plot_results(resp, timepts, x0, label)

    def plot_comparison(self, resp_default, resp_custom, timepts):
        """Plot comparison of default vs custom solutions."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # States comparison
        states_labels = ['x [m]', 'y [m]', 'θ [rad]']
        for i in range(3):
            axes[i, 0].plot(resp_default.time, resp_default.states[i], 'b-', label='Default', lw=2)
            axes[i, 0].plot(resp_custom.time, resp_custom.states[i], 'r--', label='Custom', lw=2)
            axes[i, 0].set_ylabel(states_labels[i])
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            if i == 0:
                axes[i, 0].set_title('State Comparison')
        
        # Inputs comparison
        inputs_labels = ['v [m/s]', 'δ [rad]']
        for i in range(2):
            axes[i, 1].plot(resp_default.time, resp_default.inputs[i], 'b-', label='Default', lw=2)
            axes[i, 1].plot(resp_custom.time, resp_custom.inputs[i], 'r--', label='Custom', lw=2)
            axes[i, 1].set_ylabel(inputs_labels[i])
            axes[i, 1].legend()
            axes[i, 1].grid(True)
            if i == 0:
                axes[i, 1].set_title('Input Comparison')
        
        # Trajectory comparison (x-y plot)
        axes[2, 1].plot(resp_default.states[0], resp_default.states[1], 'b-', label='Default', lw=2)
        axes[2, 1].plot(resp_custom.states[0], resp_custom.states[1], 'r--', label='Custom', lw=2)
        axes[2, 1].set_xlabel('x [m]')
        axes[2, 1].set_ylabel('y [m]')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        axes[2, 1].set_title('Trajectory (x-y)')
        
        axes[2, 0].set_xlabel('Time [s]')
        
        plt.tight_layout()
        
        # Save the comparison
        save_path = self.run_dir / "comparison_results.png"
        plt.savefig(save_path)
        self.log.info(f"Comparison results saved to {save_path}")
        plt.show()

    def plot_results(self, response, timepts, x0, label=""):
        """Plot and save results for a single solution."""
        t, y, u = response.time, response.outputs, response.inputs

        # Set up figure
        plt.figure(figsize=(10, 8))

        # Output trajectory
        plt.subplot(3, 1, 1)
        plt.plot(t, y[0], label="x", lw=2)
        plt.plot(t, y[1], label="y", lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()
        plt.grid(True)
        plt.title(f"{label} Solution - States")

        # Input trajectory
        plt.subplot(3, 1, 2)
        plt.plot(t, u[0], label="v (velocity)", lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()
        plt.grid(True)

        # Steering input
        plt.subplot(3, 1, 3)
        plt.plot(t, u[1], label="δ (steering)", lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Steering [rad]")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save or show the plot
        save_path = self.run_dir / f"{label}_results.png"
        plt.savefig(save_path)
        self.log.info(f"{label} results saved to {save_path}")
        plt.show()


def define_vehicle_problem(cfg):
    """Define the vehicle steering problem."""
    # Vehicle plant model
    def vehicle_update(t, x, u, params):
        l = params.get('wheelbase', 3.0)  # Wheelbase
        phimax = params.get('maxsteer', 0.5)  # Max steering angle
        phi = np.clip(u[1], -phimax, phimax)  # Saturate steering input
        return np.array([
            np.cos(x[2]) * u[0], 
            np.sin(x[2]) * u[0], 
            (u[0] / l) * np.tan(phi)
        ])

    def vehicle_output(t, x, u, params):
        """Ensure output size matches system definition."""
        return np.array([x[0], x[1], x[2]])  # Return [x, y, θ]

    vehicle = ct.NonlinearIOSystem(
        vehicle_update, vehicle_output, 
        states=3, inputs=2, outputs=3, 
        name="vehicle"
    )

    # Problem configuration
    x0 = np.array([0.0, -2.0, 0.0])  # Initial state [x, y, θ]
    u0 = np.array([10.0, 0.0])  # Initial input [v, φ]
    xf = np.array([100.0, 2.0, 0.0])  # Final state [x, y, θ]
    uf = np.array([10.0, 0.0])  # Final input [v, φ]
    Tf = 10  # Final time
    Ts = 0.5  # Sampling period (make it finer for better resolution)
    N = int(Tf / Ts) + 1  # Number of time points (include endpoint)

    # Cost functions
    Q = np.diag([0, 0, 0.1])  # Penalize orientation error
    R = np.diag([1, 1])  # Penalize input efforts
    Qf = np.diag([1000, 1000, 1000])  # Terminal state cost
    
    traj_cost = ct.optimal.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    term_cost = ct.optimal.quadratic_cost(vehicle, Qf, np.zeros((2, 2)), x0=xf)

    # Constraints
    constraints = [
        ct.optimal.input_range_constraint(vehicle, [8, -0.1], [12, 0.1])
    ]

    problem_cfg = {
        "Ts": Ts, 
        "Tf": Tf,
        "N": N
    }

    return vehicle, x0, u0, xf, uf, problem_cfg, (traj_cost, term_cost, constraints, Q, R, Qf)