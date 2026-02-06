import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
import logging
import pandas as pd
from scr.controllers.custom_optimal_problem import CustomOptimalControlProblem
from scr.models.vehicle_model import define_vehicle_problem
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
        self.compare_results(vehicle, timepts, x0, xf, uf, timepts, solution_default, solution_custom, matrices)

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
            u0=u0,  # Initial input for linearization
            xf=xf,  # Reference/goal state (NEW!)
            uf=uf   # Reference/goal input (NEW!)
        )

        # Solve the optimal control problem from x0
        solution = ocp.compute_trajectory(x0)

        self.log.info("Custom optimization completed.")
        return solution

    def compare_results(self, vehicle, timepts, x0, xf, uf, time, solution_default, solution_custom, matrices):
        """Compare and visualize both solutions with detailed metrics."""
        self.log.info("Comparing default vs custom solutions...")
        
        if solution_default is None:
            self.log.warning("Default solution not available for comparison")
            self.validate_simulation(vehicle, timepts, x0, solution_custom, "custom")
            return
        
        # Simulate custom solution
        resp_custom = ct.input_output_response(
            vehicle, timepts, solution_custom.inputs, x0
        )
        
        # ============================================================
        # DETAILED COMPARISON METRICS
        # ============================================================
        self.log.info("=" * 80)
        self.log.info("DETAILED COMPARISON METRICS")
        self.log.info("=" * 80)
        
        # Cost comparison
        self.log.info("\n--- COST COMPARISON ---")
        Q, R, Qf = matrices[3], matrices[4], matrices[5]
        default_cost = self._compute_cost(solution_default.states, solution_default.inputs, timepts, Q, R, Qf, xf, uf)
        custom_cost = solution_custom.cost
        
        self.log.info(f"Default solution cost: {default_cost:.6f}")
        self.log.info(f"Custom solution cost:  {custom_cost:.6f}")
        self.log.info(f"Difference:            {custom_cost - default_cost:.6f}")
        if abs(default_cost) > 1e-10:
            self.log.info(f"Relative difference:   {(custom_cost - default_cost) / default_cost * 100:.2f}%")
        
        # State trajectory comparison
        self.log.info("\n--- STATE TRAJECTORY COMPARISON ---")
        state_labels = ['x [m]', 'y [m]', 'θ [rad]']
        for i, label in enumerate(state_labels):
            default_states = solution_default.states[i]
            custom_states = resp_custom.states[i]
            
            abs_error = np.abs(default_states - custom_states)
            rel_error = abs_error / (np.abs(default_states) + 1e-10)
            
            self.log.info(f"\n{label}:")
            self.log.info(f"  Default final: {default_states[-1]:.6f}")
            self.log.info(f"  Custom final:  {custom_states[-1]:.6f}")
            self.log.info(f"  Max abs error: {np.max(abs_error):.6e}")
            self.log.info(f"  Mean abs error: {np.mean(abs_error):.6e}")
            self.log.info(f"  RMS error: {np.sqrt(np.mean(abs_error**2)):.6e}")
            self.log.info(f"  Max rel error: {np.max(rel_error) * 100:.4f}%")
        
        # Input trajectory comparison
        self.log.info("\n--- INPUT TRAJECTORY COMPARISON ---")
        input_labels = ['v [m/s]', 'δ [rad]']
        for i, label in enumerate(input_labels):
            default_inputs = solution_default.inputs[i]
            custom_inputs = solution_custom.inputs[i]
            
            abs_error = np.abs(default_inputs - custom_inputs)
            rel_error = abs_error / (np.abs(default_inputs) + 1e-10)
            
            self.log.info(f"\n{label}:")
            self.log.info(f"  Default range: [{np.min(default_inputs):.6f}, {np.max(default_inputs):.6f}]")
            self.log.info(f"  Custom range:  [{np.min(custom_inputs):.6f}, {np.max(custom_inputs):.6f}]")
            self.log.info(f"  Max abs error: {np.max(abs_error):.6e}")
            self.log.info(f"  Mean abs error: {np.mean(abs_error):.6e}")
            self.log.info(f"  RMS error: {np.sqrt(np.mean(abs_error**2)):.6e}")
            self.log.info(f"  Max rel error: {np.max(rel_error) * 100:.4f}%")
        
        # Constraint satisfaction check
        self.log.info("\n--- CONSTRAINT SATISFACTION ---")
        self.log.info("Input bounds: v ∈ [8, 12], δ ∈ [-0.1, 0.1]")
        
        for i, (label, lb, ub) in enumerate([('v', 8, 12), ('δ', -0.1, 0.1)]):
            default_violations = np.sum((solution_default.inputs[i] < lb - 1e-6) | (solution_default.inputs[i] > ub + 1e-6))
            custom_violations = np.sum((solution_custom.inputs[i] < lb - 1e-6) | (solution_custom.inputs[i] > ub + 1e-6))
            
            self.log.info(f"\n{label}:")
            self.log.info(f"  Default violations: {default_violations}")
            self.log.info(f"  Custom violations:  {custom_violations}")
            
            if default_violations == 0 and custom_violations == 0:
                self.log.info(f"  ✓ Both satisfy constraints")
            elif custom_violations > 0:
                self.log.warning(f"  ⚠ Custom solution violates constraints!")
                max_viol = max(np.max(lb - solution_custom.inputs[i]), np.max(solution_custom.inputs[i] - ub))
                self.log.warning(f"  Max violation magnitude: {max_viol:.6e}")
        
        self.log.info("=" * 80)
        
        # Save comparison data to CSV for further analysis
        self._save_comparison_csv(timepts, solution_default, solution_custom, resp_custom)
        
        # Plot comparison
        self.plot_comparison(solution_default, resp_custom, timepts)

    def _compute_cost(self, states, inputs, timepts, Q, R, Qf, xf, uf):
        """Manually compute the cost of a trajectory."""
        cost = 0
        dt = np.diff(timepts)
        
        # Integral cost (trapezoidal rule)
        for i in range(len(timepts) - 1):
            x_err = states[:, i] - xf
            u_err = inputs[:, i] - uf
            stage_cost = x_err @ Q @ x_err + u_err @ R @ u_err
            
            x_err_next = states[:, i+1] - xf
            u_err_next = inputs[:, i+1] - uf
            stage_cost_next = x_err_next @ Q @ x_err_next + u_err_next @ R @ u_err_next
            
            cost += 0.5 * (stage_cost + stage_cost_next) * dt[i]
        
        # Terminal cost
        x_err_final = states[:, -1] - xf
        cost += x_err_final @ Qf @ x_err_final
        
        return cost

    def _save_comparison_csv(self, timepts, solution_default, solution_custom, resp_custom):
        """Save detailed comparison data to CSV."""
        data = {
            'time': timepts,
            'default_x': solution_default.states[0],
            'default_y': solution_default.states[1],
            'default_theta': solution_default.states[2],
            'default_v': solution_default.inputs[0],
            'default_delta': solution_default.inputs[1],
            'custom_x': resp_custom.states[0],
            'custom_y': resp_custom.states[1],
            'custom_theta': resp_custom.states[2],
            'custom_v': solution_custom.inputs[0],
            'custom_delta': solution_custom.inputs[1],
        }
        
        # Compute errors
        data['error_x'] = data['custom_x'] - data['default_x']
        data['error_y'] = data['custom_y'] - data['default_y']
        data['error_theta'] = data['custom_theta'] - data['default_theta']
        data['error_v'] = data['custom_v'] - data['default_v']
        data['error_delta'] = data['custom_delta'] - data['default_delta']
        
        df = pd.DataFrame(data)
        csv_path = self.run_dir / "comparison_data.csv"
        df.to_csv(csv_path, index=False)
        self.log.info(f"Comparison data saved to: {csv_path}")

    def plot_comparison(self, resp_default, resp_custom, timepts):
        """Plot comparison of default vs custom solutions with error analysis."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # States comparison (left column)
        states_labels = ['x [m]', 'y [m]', 'θ [rad]']
        for i in range(3):
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(resp_default.time, resp_default.states[i], 'b-', label='Default', lw=2)
            ax.plot(resp_custom.time, resp_custom.states[i], 'r--', label='Custom', lw=2, alpha=0.7)
            ax.set_ylabel(states_labels[i], fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title('State Trajectories', fontsize=12, fontweight='bold')
        
        # Inputs comparison (middle column)
        inputs_labels = ['v [m/s]', 'δ [rad]']
        for i in range(2):
            ax = fig.add_subplot(gs[i, 1])
            ax.plot(resp_default.time, resp_default.inputs[i], 'b-', label='Default', lw=2)
            ax.plot(resp_custom.time, resp_custom.inputs[i], 'r--', label='Custom', lw=2, alpha=0.7)
            ax.set_ylabel(inputs_labels[i], fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add constraint bounds
            if i == 0:  # velocity
                ax.axhline(8, color='k', linestyle=':', alpha=0.5, label='Bounds')
                ax.axhline(12, color='k', linestyle=':', alpha=0.5)
                ax.set_title('Input Trajectories', fontsize=12, fontweight='bold')
            else:  # steering
                ax.axhline(-0.1, color='k', linestyle=':', alpha=0.5, label='Bounds')
                ax.axhline(0.1, color='k', linestyle=':', alpha=0.5)
        
        # State errors (right column)
        for i in range(3):
            ax = fig.add_subplot(gs[i, 2])
            error = resp_custom.states[i] - resp_default.states[i]
            ax.plot(timepts, error, 'g-', lw=2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_ylabel(f'Error {states_labels[i]}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title('Errors (Custom - Default)', fontsize=12, fontweight='bold')
            
            # Add RMS error as text
            rms = np.sqrt(np.mean(error**2))
            ax.text(0.02, 0.98, f'RMS: {rms:.2e}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Input errors
        for i in range(2):
            ax = fig.add_subplot(gs[i+1, 2])  # Skip first row
            error = resp_custom.inputs[i] - resp_default.inputs[i]
            ax.plot(timepts, error, 'g-', lw=2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_ylabel(f'Error {inputs_labels[i]}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add RMS error as text
            rms = np.sqrt(np.mean(error**2))
            ax.text(0.02, 0.98, f'RMS: {rms:.2e}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Trajectory comparison (x-y plot) - bottom row
        ax = fig.add_subplot(gs[3, :])
        ax.plot(resp_default.states[0], resp_default.states[1], 'b-', label='Default', lw=2)
        ax.plot(resp_custom.states[0], resp_custom.states[1], 'r--', label='Custom', lw=2, alpha=0.7)
        ax.plot(0, -2, 'go', markersize=10, label='Start')
        ax.plot(100, 2, 'rs', markersize=10, label='Goal')
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Trajectory Comparison (x-y plane)', fontsize=12, fontweight='bold')
        ax.axis('equal')
        
        plt.suptitle('Default vs Custom Optimal Control Comparison', fontsize=14, fontweight='bold')
        
        # Save the comparison
        save_path = self.run_dir / "comparison_results.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.log.info(f"Comparison plot saved to {save_path}")
        plt.show()

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
