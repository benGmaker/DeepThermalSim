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
    
    def _cvxpy_optimizer(self, fun, x0_opt, constraints, max_iterations=10, **kwargs):
        """
        CVXPY-based optimizer with successive linearization for nonlinear systems.
        """
        self.logger.info("Starting CVXPY optimization with successive linearization...")
        self.logger.info(f"self.x shape: {self.x.shape}")
        self.logger.info(f"self.timepts length: {len(self.timepts)}")

        nx, nu = self.A.shape[0], self.B.shape[1]
        N = len(self.timepts) - 1
        self.logger.info(f"N = {N}, N+1 = {N+1}")
        self.logger.info(f"Expected U shape: ({nu}, {N+1})")
        
        if not hasattr(self, 'xf') or not hasattr(self, 'uf'):
            raise ValueError("xf and uf must be set in the problem definition")
        
        xf = self.xf
        uf = self.uf
        
        # Initial guess: straight line from x0 to xf
        X_guess = np.zeros((nx, N + 1))
        U_guess = np.zeros((nu, N + 1))
        for k in range(N + 1):
            alpha = k / N
            X_guess[:, k] = (1 - alpha) * self.x + alpha * xf
            U_guess[:, k] = uf
        
        # Successive Convex Programming loop
        for iteration in range(max_iterations):
            self.logger.info(f"SCP Iteration {iteration + 1}/{max_iterations}")
            
            # Linearize around the current guess at each time point
            A_list = []
            B_list = []
            c_list = []  # Affine term: x_k+1 = A_k x_k + B_k u_k + c_k
            
            for k in range(N):
                # Linearize at (X_guess[:, k], U_guess[:, k])
                x_bar = X_guess[:, k]
                u_bar = U_guess[:, k]
                
                # Get linearization at this point
                lin_sys = ct.linearize(self.system, x_bar, u_bar)
                A_k = lin_sys.A
                B_k = lin_sys.B
                
                # Compute affine term: c_k = f(x_bar, u_bar) - A_k*x_bar - B_k*u_bar
                # For continuous time: c_k = 0 if we use exact discretization
                # For simplicity, use Euler: x_k+1 = x_k + dt * f(x_k, u_k)
                dt = self.timepts[k+1] - self.timepts[k]
                
                # Evaluate nonlinear dynamics
                f_bar = self.system._rhs(self.timepts[k], x_bar, u_bar)
                
                # Linearized dynamics: x_k+1 ≈ x_k + dt*(f_bar + A_k*(x_k - x_bar) + B_k*(u_k - u_bar))
                # Rearranging: x_k+1 = (I + dt*A_k)*x_k + dt*B_k*u_k + dt*(f_bar - A_k*x_bar - B_k*u_bar)
                A_k_disc = np.eye(nx) + dt * A_k
                B_k_disc = dt * B_k
                c_k = dt * (f_bar - A_k @ x_bar - B_k @ u_bar)
                
                A_list.append(A_k_disc)
                B_list.append(B_k_disc)
                c_list.append(c_k)
            
            # Solve the convex problem with time-varying linearization
            X = cp.Variable((nx, N + 1))
            U = cp.Variable((nu, N + 1))
            
            # Cost function
            cost_terms = []
            for k in range(N):
                x_dev = X[:, k] - xf
                u_dev = U[:, k] - uf
                cost_terms.append(cp.quad_form(x_dev, self.Q))
                cost_terms.append(cp.quad_form(u_dev, self.R))
            
            x_dev_final = X[:, N] - xf
            u_dev_final = U[:, N] - uf
            cost_terms.append(cp.quad_form(x_dev_final, self.Qf))
            cost_terms.append(cp.quad_form(u_dev_final, self.R))
            
            # Add trust region to prevent large changes
            trust_region_weight = 1.0 / (iteration + 1)  # Decrease over iterations
            for k in range(N + 1):
                cost_terms.append(trust_region_weight * cp.sum_squares(X[:, k] - X_guess[:, k]))
                cost_terms.append(trust_region_weight * cp.sum_squares(U[:, k] - U_guess[:, k]))
            
            objective = cp.Minimize(cp.sum(cost_terms))
            
            # Constraints
            cvxp_constraints = [X[:, 0] == self.x]
            
            # Time-varying linearized dynamics
            for k in range(N):
                cvxp_constraints.append(
                    X[:, k + 1] == A_list[k] @ X[:, k] + B_list[k] @ U[:, k] + c_list[k]
                )
            
            # Input constraints
            for idx, (ctype, fun_constraint, lb, ub) in enumerate(self.trajectory_constraints):
                if ctype == opt.LinearConstraint:
                    A_matrix = fun_constraint
                    if A_matrix.shape[1] == (nx + nu):
                        A_x = A_matrix[:, :nx]
                        A_u = A_matrix[:, nx:]
                        
                        if np.allclose(A_x, 0):
                            for k in range(N + 1):
                                cvxp_constraints.append(A_u @ U[:, k] >= lb)
                                cvxp_constraints.append(A_u @ U[:, k] <= ub)
            
            # Solve
            prob = cp.Problem(objective, cvxp_constraints)
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as e:
                self.logger.error(f"CVXPY solver failed: {e}")
                break
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                self.logger.error(f"Optimization failed with status: {prob.status}")
                break
            
            # Update guess
            X_new = X.value
            U_new = U.value
            
            # Check convergence
            state_change = np.linalg.norm(X_new - X_guess)
            input_change = np.linalg.norm(U_new - U_guess)
            
            self.logger.info(f"  Cost: {prob.value:.6f}")
            self.logger.info(f"  State change: {state_change:.6e}")
            self.logger.info(f"  Input change: {input_change:.6e}")
            
            X_guess = X_new
            U_guess = U_new
            
            # Convergence check
            if state_change < 1e-3 and input_change < 1e-3:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # ========================================================================
        # AFTER THE LOOP - Final solution processing
        # ========================================================================
        self.logger.info("SCP iterations completed")
        self.logger.info(f"Final U[0] (velocity) - first 3: {U_guess[0, :3]}")
        self.logger.info(f"Final U[1] (steering) - first 3: {U_guess[1, :3]}")
        self.logger.info(f"U_guess shape: {U_guess.shape}")
        
        # Compute actual cost (without trust region)
        actual_cost = 0
        dt_array = np.diff(self.timepts)
        for k in range(N):
            x_err = X_guess[:, k] - xf
            u_err = U_guess[:, k] - uf
            stage_cost = x_err @ self.Q @ x_err + u_err @ self.R @ u_err
            x_err_next = X_guess[:, k+1] - xf
            u_err_next = U_guess[:, k+1] - uf
            stage_cost_next = x_err_next @ self.Q @ x_err_next + u_err_next @ self.R @ u_err_next
            actual_cost += 0.5 * (stage_cost + stage_cost_next) * dt_array[k]
        x_err_final = X_guess[:, N] - xf
        actual_cost += x_err_final @ self.Qf @ x_err_final
        
        self.logger.info(f"Actual cost (without trust region): {actual_cost:.6f}")
        
        # Return in the correct format
        result_x = U_guess.flatten(order='C')
        
        self.logger.info(f"Result_x shape: {result_x.shape}")
        self.logger.info(f"Result_x first 10: {result_x[:10]}")
        self.logger.info(f"Expected shape: ({nu * (N+1)},) = ({nu * (N+1)},)")
        
        # Verify reshaping
        test_reshape = result_x.reshape((nu, N+1), order='C')
        self.logger.info(f"Test reshape - U[0] first 3: {test_reshape[0, :3]}")
        self.logger.info(f"Test reshape - U[1] first 3: {test_reshape[1, :3]}")
        
        return {
            "x": result_x,
            "success": True,
            "fun": actual_cost,
            "message": "SCP optimization completed successfully."
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