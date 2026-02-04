"""
Custom Optimal Control Problem implementation using CVXPY as the backend solver.
This module defines a `CustomOptimalControlProblem` class that inherits from `OptimalControlProblem`
and overrides the optimization routine to utilize CVXPY for solving the optimal control problem.
"""
import numpy as np
import cvxpy as cp
import logging
from control.optimal import OptimalControlProblem, OptimalControlResult
import control as ct

topic = 'controller'

class CustomOptimalControlProblem(OptimalControlProblem):
    def __init__(self, *args, Q=None, R=None, Qf=None, x0=None, u0=None, **kwargs):
        """
        Initialize the CustomOptimalControlProblem with explicit cost matrices.
        Automatically handles linearizing the system if it's nonlinear.
        """
        super().__init__(*args, **kwargs)
        self.Q = Q  # State quadratic cost
        self.R = R  # Input quadratic cost
        self.Qf = Qf  # Terminal state quadratic cost
        self.logger = logging.getLogger("controller")

        # Store initial state and input
        self.x0 = np.atleast_1d(x0) if x0 is not None else None
        self.u0 = np.atleast_1d(u0) if u0 is not None else None

        # Ensure the initial state and input dimensions match the system's definition
        if not self._validate_dimensions():
            raise ValueError("System, x0, and u0 dimensions do not match.")

        # Linearize the system if required
        if hasattr(self.system, "A") and hasattr(self.system, "B"):
            self.A, self.B = self.system.A, self.system.B  # Already linear system
        else:
            self.logger.info("Linearizing the system...")
            self.A, self.B, _, _ = ct.linearize(self.system, self.x0, self.u0)

    def _validate_dimensions(self):
        """Ensure that the system's dimensions match initial conditions."""
        if self.system.nstates != self.x0.shape[0]:
            self.logger.error("Mismatch in dimensions: system states and x0.")
            return False
        if self.system.ninputs != self.u0.shape[0]:
            self.logger.error("Mismatch in dimensions: system inputs and u0.")
            return False
        return True
    
    def _cvxpy_optimizer(self, fun, x0, constraints, **kwargs):
        """
        CVXPY-based custom optimizer to minimize the cost function.
        """
        self.logger.info("Starting CVXPY optimization...")

        # Define decision variables
        nx, nu = self.A.shape[0], self.B.shape[1]
        N = len(self.timepts) - 1
        X = cp.Variable((nx, N + 1))  # State trajectory
        U = cp.Variable((nu, N))      # Input trajectory

        # Construct the objective function
        cost_terms = []
        for k in range(N):
            cost_terms.append(cp.quad_form(X[:, k], self.Q))
            cost_terms.append(cp.quad_form(U[:, k], self.R))
        cost_terms.append(cp.quad_form(X[:, N], self.Qf))
        objective = cp.Minimize(cp.sum(cost_terms))

        # Dynamics constraints
        cvxp_constraints = [X[:, 0] == self.x]  # Initial condition
        for k in range(N):
            cvxp_constraints.append(X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k])

        # Add input constraints
        for lb, ub in zip(self.constraint_lb, self.constraint_ub):
            cvxp_constraints.append(lb <= U)
            cvxp_constraints.append(U <= ub)

        # Solve the CVXPY problem
        prob = cp.Problem(objective, cvxp_constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            self.logger.info("Optimization completed successfully.")
            return {
                "x": np.hstack([U.value.flatten(), X.value.flatten()]),
                "success": True,
                "message": "Optimization was successful."
            }
        else:
            self.logger.error("Optimization failed.")
            return {
                "x": x0, 
                "success": False,
                "message": f"Optimization failed with status: {prob.status}"
            }

    def compute_trajectory(self, x, **kwargs):
        """
        Override the `compute_trajectory` method to use the custom CVXPY optimizer.

        Parameters:
        - x: Initial state of the system.
        - kwargs: Additional arguments.

        Returns:
        - Result (OptimalControlResult) of the optimization.
        """
        self.logger.info("Computing trajectory with custom optimizer...")

        # Use the custom optimizer set in the constructor
        opt_result = self._cvxpy_optimizer(
            fun=self._cost_function,  # Ignored in CVXPY
            x0=self.initial_guess,
            constraints=self.constraints,
            **kwargs
        )

        # Package the optimization results
        return OptimalControlResult(
            self, opt_result,
            transpose=kwargs.get("transpose", None),
            return_states=kwargs.get("return_states", True),
            squeeze=kwargs.get("squeeze", None),
            print_summary=kwargs.get("print_summary", True)
        )