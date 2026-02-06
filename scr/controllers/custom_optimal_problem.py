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
    def __init__(self, *args, Q=None, R=None, Qf=None, x0=None, u0=None, xf=None, uf=None, **kwargs):
        """
        Initialize the CustomOptimalControlProblem with explicit cost matrices.
        Automatically handles linearizing the system if it's nonlinear.
        
        Parameters
        ----------
        Q : ndarray
            State cost matrix
        R : ndarray
            Input cost matrix
        Qf : ndarray
            Terminal state cost matrix
        x0 : ndarray
            Initial state for linearization
        u0 : ndarray
            Initial input for linearization
        xf : ndarray
            Reference/goal state
        uf : ndarray
            Reference/goal input
        """
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.logger = logging.getLogger(topic)

        # Store initial state and input
        self.x0 = np.atleast_1d(x0) if x0 is not None else None
        self.u0 = np.atleast_1d(u0) if u0 is not None else None
        
        # Store reference state and input for cost computation
        self.xf = np.atleast_1d(xf) if xf is not None else None
        self.uf = np.atleast_1d(uf) if uf is not None else None

        # Ensure the initial state and input dimensions match
        if not self._validate_dimensions():
            raise ValueError("System, x0, and u0 dimensions do not match.")

        # Linearize the system if required
        if hasattr(self.system, "A") and hasattr(self.system, "B"):
            self.A, self.B = self.system.A, self.system.B
        else:
            self.logger.info("Linearizing the system...")
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
        """
        self.logger.info("Starting CVXPY optimization...")

        # Define decision variables
        nx, nu = self.A.shape[0], self.B.shape[1]
        N = len(self.timepts) - 1
        X = cp.Variable((nx, N + 1))  # State trajectory (N+1 points)
        U = cp.Variable((nu, N + 1))  # Input trajectory (N+1 points to match time)

        # Reference trajectory (goal state and input)
        # Extract from the cost functions passed during initialization
        # The cost is defined as (x - xf)^T Q (x - xf) + (u - uf)^T R (u - uf)
        # We need to get xf and uf from somewhere
        # Let's add them as parameters
        
        # For now, we'll need to pass xf and uf to the optimizer
        # Let's extract them from the problem definition
        
        # Construct the objective function with PROPER reference tracking
        cost_terms = []
        
        # Get reference values from integral_cost (they're embedded in the cost function)
        # We need xf and uf - let's compute them from the initial guess or pass them explicitly
        # For now, let's assume they should be stored in the class
        
        # TEMPORARY FIX: We need xf and uf to be passed to the class
        # Let's extract them from the cost function evaluation
        if not hasattr(self, 'xf') or not hasattr(self, 'uf'):
            self.logger.error("Reference state xf and input uf not set!")
            # Try to extract from the problem
            # This is a workaround - ideally these should be passed explicitly
            raise ValueError("xf and uf must be set in the problem definition")
        
        xf = self.xf
        uf = self.uf
        
        for k in range(N):
            # Penalize deviation from reference: (x - xf)^T Q (x - xf)
            x_dev = X[:, k] - xf
            u_dev = U[:, k] - uf
            cost_terms.append(cp.quad_form(x_dev, self.Q))
            cost_terms.append(cp.quad_form(u_dev, self.R))
        
        # Terminal cost: (x_N - xf)^T Qf (x_N - xf)
        x_dev_final = X[:, N] - xf
        u_dev_final = U[:, N] - uf
        cost_terms.append(cp.quad_form(x_dev_final, self.Qf))
        cost_terms.append(cp.quad_form(u_dev_final, self.R))  # Also penalize terminal input deviation
        
        objective = cp.Minimize(cp.sum(cost_terms))

        # Dynamics constraints
        cvxp_constraints = [X[:, 0] == self.x]  # Initial condition
        for k in range(N):
            cvxp_constraints.append(X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k])

        # DEBUG: Print constraint information
        self.logger.info(f"Number of trajectory constraints: {len(self.trajectory_constraints)}")
        
        # Parse input constraints from the trajectory_constraints
        for idx, (ctype, fun_constraint, lb, ub) in enumerate(self.trajectory_constraints):
            if ctype == opt.LinearConstraint:
                A_matrix = fun_constraint
                constraint_dim = A_matrix.shape[1]
                
                if constraint_dim == (nx + nu):
                    A_x = A_matrix[:, :nx]
                    A_u = A_matrix[:, nx:]
                    
                    if np.allclose(A_x, 0):
                        # Input-only constraint: A_u @ u in [lb, ub]
                        for k in range(N + 1):
                            cvxp_constraints.append(A_u @ U[:, k] >= lb)
                            cvxp_constraints.append(A_u @ U[:, k] <= ub)
                    else:
                        # General linear constraint
                        for k in range(N + 1):
                            cvxp_constraints.append(A_x @ X[:, k] + A_u @ U[:, k] >= lb)
                            cvxp_constraints.append(A_x @ X[:, k] + A_u @ U[:, k] <= ub)
                else:
                    self.logger.warning(
                        f"LinearConstraint with unexpected dimension {constraint_dim}")
            elif ctype == opt.NonlinearConstraint:
                self.logger.warning("NonlinearConstraint not yet implemented in CVXPY solver.")
            else:
                self.logger.warning(f"Constraint type {ctype} not recognized")

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

        # DEBUG: Print solution info
        self.logger.info(f"Solver status: {prob.status}")
        self.logger.info(f"Optimal cost: {prob.value}")
        if U.value is not None:
            self.logger.info(f"U solution range: [{np.min(U.value)}, {np.max(U.value)}]")
            self.logger.info(f"U[0] (velocity) - first 3 values: {U.value[0, :3]}")
            self.logger.info(f"U[1] (steering) - first 3 values: {U.value[1, :3]}")

        if prob.status in ["optimal", "optimal_inaccurate"]:
            self.logger.info(f"Optimization completed successfully with status: {prob.status}")
            
            # Log the CVXPY solution
            self.logger.info(f"CVXPY U.value shape: {U.value.shape}")
            self.logger.info(f"CVXPY U.value:\n{U.value}")
            
            # Return results in the format expected by OptimalControlResult
            if self.shooting:
                # The parent class will reshape this as: coeffs.reshape((ninputs, -1))
                # So we need to flatten in a way that preserves [u1[:], u2[:], ...] structure
                # That means we want: [u1[0], u1[1], ..., u1[N], u2[0], u2[1], ..., u2[N]]
                # Which is ROW-major (C-order) flattening
                result_x = U.value.flatten(order='C')
                
                self.logger.info(f"Flattened result_x shape: {result_x.shape}")
                self.logger.info(f"Flattened result_x (first 10): {result_x[:10]}")
                self.logger.info(f"Flattened result_x (around middle): {result_x[len(result_x)//2-5:len(result_x)//2+5]}")
                
                # Test reshaping to verify
                test_reshape = result_x.reshape((nu, N+1), order='C')
                self.logger.info(f"Test reshape shape: {test_reshape.shape}")
                self.logger.info(f"Test reshape U[0,:3]: {test_reshape[0, :3]}")
                self.logger.info(f"Test reshape U[1,:3]: {test_reshape[1, :3]}")
                
            else:
                result_x = np.hstack([U.value.flatten(order='C'), X.value.flatten(order='C')])
            
            return {
                "x": result_x,
                "success": True,
                "fun": prob.value,
                "message": f"Optimization was successful ({prob.status})."}

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