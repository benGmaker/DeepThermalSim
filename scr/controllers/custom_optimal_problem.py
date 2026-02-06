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
import scipy.optimize as opt

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
        self.logger = logging.getLogger(topic)

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
            # ct.linearize returns a LinearIOSystem object, not a tuple
            linearized_sys = ct.linearize(self.system, self.x0, self.u0)
            self.A = linearized_sys.A
            self.B = linearized_sys.B

    def _validate_dimensions(self):
        """Ensure that the system's dimensions match initial conditions."""
        if self.system.nstates != self.x0.shape[0]:
            self.logger.error("Mismatch in dimensions: system states and x0.")
            return False
        if self.system.ninputs != self.u0.shape[0]:
            self.logger.error("Mismatch in dimensions: system inputs and u0.")
            return False
        return True
    
    def _cvxpy_optimizer(self, fun, x0_opt, constraints, **kwargs):
        """
        CVXPY-based custom optimizer to minimize the cost function.
        
        Parameters
        ----------
        fun : callable
            Cost function (not used in CVXPY, but kept for compatibility)
        x0_opt : array
            Initial guess for optimization
        constraints : list
            List of constraint objects
        """
        self.logger.info("Starting CVXPY optimization...")

        # Define decision variables
        nx, nu = self.A.shape[0], self.B.shape[1]
        N = len(self.timepts) - 1
        X = cp.Variable((nx, N + 1))  # State trajectory (N+1 points)
        U = cp.Variable((nu, N + 1))  # Input trajectory (N+1 points to match time)

        # Construct the objective function
        cost_terms = []
        for k in range(N):
            cost_terms.append(cp.quad_form(X[:, k], self.Q))
            cost_terms.append(cp.quad_form(U[:, k], self.R))
        # Terminal cost
        cost_terms.append(cp.quad_form(X[:, N], self.Qf))
        cost_terms.append(cp.quad_form(U[:, N], self.R))  # Also penalize terminal input
        objective = cp.Minimize(cp.sum(cost_terms))

        # Dynamics constraints
        cvxp_constraints = [X[:, 0] == self.x]  # Initial condition
        for k in range(N):
            cvxp_constraints.append(X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k])

        # Parse input constraints from the trajectory_constraints
        # The constraints are stored as (type, fun, lb, ub) tuples
        for ctype, fun_constraint, lb, ub in self.trajectory_constraints:
            if ctype == opt.LinearConstraint:
                # Linear constraint: A @ [x; u] in [lb, ub]
                # For input-only constraints, A should have zeros for states
                # Check the shape to determine if this is input-only
                A_matrix = fun_constraint  # For LinearConstraint, 'fun' is the A matrix
                
                # Determine the dimension
                constraint_dim = A_matrix.shape[1]
                
                # Check if this is an input constraint
                # (typically A has shape (nu, nx+nu) with zeros in first nx columns)
                if constraint_dim == (nx + nu):
                    # Split into state and input parts
                    A_x = A_matrix[:, :nx]
                    A_u = A_matrix[:, nx:]
                    
                    # Check if this is an input-only constraint
                    if np.allclose(A_x, 0):
                        # Input-only constraint: A_u @ u in [lb, ub]
                        for k in range(N + 1):  # Apply to all N+1 time points
                            cvxp_constraints.append(A_u @ U[:, k] >= lb)
                            cvxp_constraints.append(A_u @ U[:, k] <= ub)
                    else:
                        # General linear constraint on both state and input
                        for k in range(N + 1):  # Apply to all N+1 time points
                            cvxp_constraints.append(
                                A_x @ X[:, k] + A_u @ U[:, k] >= lb)
                            cvxp_constraints.append(
                                A_x @ X[:, k] + A_u @ U[:, k] <= ub)
                else:
                    self.logger.warning(
                        f"LinearConstraint with unexpected dimension {constraint_dim} "
                        f"(expected {nx + nu})")
            elif ctype == opt.NonlinearConstraint:
                self.logger.warning(
                    "NonlinearConstraint not yet implemented in CVXPY solver. "
                    "Nonlinear constraints require linearization or a different approach.")
            else:
                self.logger.warning(
                    f"Constraint type {ctype} not recognized")

        # Solve the CVXPY problem
        prob = cp.Problem(objective, cvxp_constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            self.logger.error(f"CVXPY solver failed: {e}")
            return {
                "x": x0_opt, 
                "success": False,
                "message": f"Solver exception: {e}"
            }

        if prob.status in ["optimal", "optimal_inaccurate"]:
            self.logger.info(f"Optimization completed successfully with status: {prob.status}")
            
            # Return results in the format expected by OptimalControlResult
            # For shooting method: return inputs only (flattened)
            # For collocation method: return inputs + states
            if self.shooting:
                # Flatten U to shape (nu * (N+1),)
                result_x = U.value.flatten(order='F')  # Fortran order for column-major
            else:
                # For collocation, concatenate inputs and states
                result_x = np.hstack([U.value.flatten(order='F'), X.value.flatten(order='F')])
            
            return {
                "x": result_x,
                "success": True,
                "fun": prob.value,  # The optimal cost
                "message": f"Optimization was successful ({prob.status})."
            }
        else:
            self.logger.error(f"Optimization failed with status: {prob.status}")
            return {
                "x": x0_opt, 
                "success": False,
                "fun": np.inf,
                "message": f"Optimization failed with status: {prob.status}"
            }

    def compute_trajectory(self, x, **kwargs):
        """
        Override the `compute_trajectory` method to use the custom CVXPY optimizer.

        Parameters
        ----------
        x : array_like
            Initial state of the system.
        **kwargs : dict
            Additional arguments (squeeze, transpose, return_states, etc.)

        Returns
        -------
        OptimalControlResult
            Result of the optimization.
        """
        self.logger.info("Computing trajectory with custom CVXPY optimizer...")

        # Store the initial state (used by _cvxpy_optimizer)
        self.x = np.atleast_1d(x)

        # Call the custom optimizer
        opt_result_dict = self._cvxpy_optimizer(
            fun=self._cost_function,  # Passed for compatibility but not used
            x0_opt=self.initial_guess,
            constraints=self.constraints,
        )

        # Check if optimization was successful
        if not opt_result_dict['success']:
            self.logger.error("Optimization failed!")
            self.logger.error(f"Message: {opt_result_dict['message']}")

        # Convert dictionary to OptimizeResult object
        opt_result = opt.OptimizeResult(
            x=opt_result_dict['x'],
            success=opt_result_dict['success'],
            fun=opt_result_dict.get('fun', np.nan),
            message=opt_result_dict['message']
        )

        # Package the optimization results using the parent class result handler
        return OptimalControlResult(
            self, opt_result,
            transpose=kwargs.get("transpose", None),
            return_states=kwargs.get("return_states", True),
            squeeze=kwargs.get("squeeze", None),
            print_summary=kwargs.get("print_summary", True)
        )