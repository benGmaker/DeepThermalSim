import numpy as np
import matplotlib.pyplot as plt
import control as ct
import logging
from scr.controllers.custom_optimal_problem import CustomOptimalControlProblem
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("experiment")

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
        self.log = logging.getLogger("CustomMPCExperiment")

    def run(self):
        """Run the experiment for the given dynamic system."""
        self.log.info("Setting up the experiment...")

        # Build the control problem
        plant, xref, uref, problem_cfg, matrices = define_vehicle_problem(self.cfg) # todo implement more problems and make this more general
        Ts = problem_cfg["Ts"]
        N = problem_cfg["N"]

        # Build and solve the custom optimal control problem
        timepts = np.arange(0, N * Ts, Ts)
        solution = self.solve_custom_mpc(plant, timepts, xref, uref, matrices)

        # Validate and simulate the solved trajectory
        self.validate_simulation(plant, timepts, xref, solution)

    def solve_custom_mpc(self, plant, timepts, xref, uref, matrices):
        """Solve the custom optimal control problem using CustomOptimalControlProblem."""
        self.log.info("Solving trajectory optimization using CustomOptimalControlProblem...")

        # Unpack cost and constraints
        traj_cost, term_cost, constraints, Q, R, Qf = matrices

        # Build the custom optimal control problem
        ocp = CustomOptimalControlProblem(
            sys=plant,
            timepts=timepts,
            integral_cost=traj_cost,
            terminal_cost=term_cost,
            trajectory_constraints=constraints,
            Q=Q,
            R=R,
            Qf=Qf,  # Pass explicit cost matrices
            x0=xref,  # Pass the initial state for linearization
            u0=uref   # Pass the initial input for linearization
        )

        # Solve the optimal control problem
        solution = ocp.compute_trajectory(xref)

        self.log.info("Optimization completed.")
        return solution

    def validate_simulation(self, plant, timepts, xref, solution):
        """Simulate and validate results."""
        self.log.info("Starting validation (simulation with solved trajectory)...")

        # Simulate the plant with the solved inputs
        resp = ct.input_output_response(plant, timepts, solution.inputs, xref)

        # Log and plot validation results
        self.log.info("Validation complete. Generating results...")
        self.plot_results(resp, xref)

    def plot_results(self, response, xref):
        """Plot and save results."""
        t, y, u = response.time, response.outputs, response.inputs

        # Set up figure
        plt.figure(figsize=(10, 8))

        # Output trajectory
        plt.subplot(3, 1, 1)
        plt.plot(t, y[0], label="x", lw=2)
        plt.plot(t, y[1], label="y", lw=2)
        plt.plot(t, xref[0] * np.ones_like(t), "k--", label="xref")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()
        plt.grid(True)

        # Input trajectory
        plt.subplot(3, 1, 2)
        plt.plot(t, u[0], label="input1", lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Input 1 [m/s]")
        plt.grid(True)

        # Additional control states or inputs if necessary
        if u.shape[0] > 1:  # For multi-input systems
            plt.subplot(3, 1, 3)
            plt.plot(t, u[1], label="input2", lw=2)
            plt.xlabel("Time [s]")
            plt.ylabel("Input 2 [rad]")
            plt.grid(True)

        plt.tight_layout()

        # Save or show the plot
        save_path = self.run_dir / "experiment_results.png"
        plt.savefig(save_path)
        self.log.info(f"Experiment results saved to {save_path}")
        plt.show()


def define_vehicle_problem(cfg):
    """Define the vehicle steering problem."""
    # Vehicle plant model
    def vehicle_update(t, x, u, params):
        l = params.get('wheelbase', 3.0)  # Wheelbase
        phimax = params.get('maxsteer', 0.5)  # Max steering angle
        phi = np.clip(u[1], -phimax, phimax)  # Saturate steering input
        return np.array([np.cos(x[2]) * u[0], np.sin(x[2]) * u[0], (u[0] / l) * np.tan(phi)])

    def vehicle_output(t, x, u, params):
        """Ensure output size matches system definition."""
        return np.array([x[0], x[1], x[2]])  # Return [x, y, θ] as a 3D array

    vehicle = ct.NonlinearIOSystem(vehicle_update, vehicle_output, states=3, inputs=2, outputs=3, name="vehicle")

    # Problem configuration
    x0 = np.array([0.0, -2.0, 0.0])  # Initial state [x, y, θ]
    u0 = np.array([10.0, 0.0])  # Initial input [v, φ]
    xf = np.array([100.0, 2.0, 0.0])  # Final state [x, y, θ]
    uf = np.array([10.0, 0.0])  # Final input [v, φ]
    Tf = 10  # Final time
    Ts = 1.0  # Sampling period
    N = int(Tf / Ts)  # Horizon length

    # Cost functions
    Q = np.diag([0, 0, 0.1])  # Penalize steering angles
    R = np.diag([1, 1])  # Penalize input efforts
    P = np.diag([1000, 1000, 1000])  # Terminal state cost
    traj_cost = ct.optimal.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    term_cost = ct.optimal.quadratic_cost(vehicle, P, np.zeros((2, 2)), x0=xf)

    # Constraints
    constraints = [
        ct.optimal.input_range_constraint(vehicle, [8, -0.1], [12, 0.1])  # Input bounds: velocity and steering rates
    ]

    return vehicle, x0, u0, {"Ts": Ts, "N": N}, (traj_cost, term_cost, constraints, Q, R, P)

