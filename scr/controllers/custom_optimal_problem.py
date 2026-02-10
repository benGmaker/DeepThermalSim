"""Custom Optimal Control Problem Implementation

This module demonstrates how to create a custom optimal control problem
by inheriting from OptimalControlProblem and overriding key methods.
This pattern allows you to implement custom optimization problems (e.g., DMC, DeePC)
while maintaining the same MPC interface.

This implementation shows a concrete matrix-based MPC formulation.
"""

import numpy as np
import scipy as sp
import control as ct
from control import optimal  # Assuming the original class is here
from control.optimal import OptimalControlProblem, OptimalControlResult
import logging 
topic = 'controller'

class CustomOptimalControlProblem(OptimalControlProblem):
    """Custom optimal control problem using direct matrix-based MPC formulation.
    
    This class demonstrates how to inherit from OptimalControlProblem and
    implement a custom optimization problem using explicit matrix formulations.
    
    For nonlinear systems, this class linearizes around a reference trajectory
    (x0, u0) or equilibrium point and builds prediction matrices for the
    linearized system.
    
    Parameters
    ----------
    sys : InputOutputSystem
        System to control (discrete-time, can be nonlinear)
    timepts : array
        Time points for the prediction horizon
    integral_cost : callable
        Integral cost function (from ct.optimal.quadratic_cost)
    terminal_cost : callable, optional
        Terminal cost function
    trajectory_constraints : list, optional
        Trajectory constraints
    terminal_constraints : list, optional
        Terminal constraints
    Q : array, optional
        State weight matrix (for matrix formulation)
    R : array, optional
        Input weight matrix (for matrix formulation)
    Qf : array, optional
        Terminal state weight matrix
    x0 : array, optional
        Linearization point for states (or equilibrium)
    u0 : array, optional
        Linearization point for inputs (or equilibrium)
    xf : array, optional
        Reference/goal state for tracking
    uf : array, optional
        Reference/goal input for tracking
    initial_guess : array, optional
        Initial guess for optimization
    **kwargs
        Additional arguments passed to parent class
    
    Example
    -------
    >>> # Create a custom MPC problem
    >>> ocp = CustomOptimalControlProblem(
    ...     sys=vehicle,
    ...     timepts=timepts,
    ...     integral_cost=traj_cost,
    ...     terminal_cost=term_cost,
    ...     Q=Q, R=R, Qf=Qf,
    ...     x0=x0, u0=u0, xf=xf, uf=uf
    ... )
    >>> # Solve the problem
    >>> res = ocp.compute_trajectory(x0)
    """
    
    def __init__(self, sys, timepts, integral_cost, 
                 terminal_cost=None,
                 trajectory_constraints=None, 
                 terminal_constraints=None,
                 Q=None, R=None, Qf=None,
                 x0=None, u0=None, xf=None, uf=None,
                 initial_guess=None, **kwargs):
        """Initialize the custom optimal control problem with matrix formulation."""
        
        # Initialize logger
        self.logger = logging.getLogger(topic)
        self.logger.debug("Initializing CustomOptimalControlProblem")
        
        # Store reference points
        self.x0_lin = x0 if x0 is not None else np.zeros(sys.nstates)
        self.u0_lin = u0 if u0 is not None else np.zeros(sys.ninputs)
        self.xf = xf if xf is not None else self.x0_lin
        self.uf = uf if uf is not None else self.u0_lin
        
        self.logger.debug(f"System: nstates={sys.nstates}, ninputs={sys.ninputs}, "
                      f"noutputs={sys.noutputs}, dt={sys.dt}")
        self.logger.debug(f"Linearization point: x0={self.x0_lin}, u0={self.u0_lin}")
        self.logger.debug(f"Reference point: xf={self.xf}, uf={self.uf}")
        
        # Store cost matrices (optional, for matrix-based formulation)
        self.Q = Q
        self.R = R
        self.Qf = Qf if Qf is not None else Q
        
        # Store original system
        self.original_sys = sys
        
        # Linearize if needed and extract/create state-space matrices
        if hasattr(sys, 'A') and sys.A is not None:
            # Already linear state-space
            self.logger.debug("System is already in linear state-space form")
            self.A = sys.A
            self.B = sys.B
            self.C = sys.C if sys.C is not None else np.eye(sys.nstates)
            self.D = sys.D if sys.D is not None else np.zeros((sys.noutputs, sys.ninputs))
            self.use_linearization = False
        else:
            # Nonlinear system - linearize around operating point
            self.logger.debug("Linearizing nonlinear system...")
            self._linearize_system(sys, self.x0_lin, self.u0_lin)
            self.use_linearization = True
        
        self.logger.debug(f"System matrices: A shape={self.A.shape}, B shape={self.B.shape}, "
                      f"C shape={self.C.shape}, D shape={self.D.shape}")
        
        # Compute prediction horizon length
        self.N = len(timepts)
        self.logger.debug(f"Prediction horizon: N={self.N} steps")
        
        # Build prediction matrices if we have Q and R matrices
        if self.Q is not None and self.R is not None:
            self.logger.debug("Building prediction matrices...")
            self._build_prediction_matrices()
            
            self.logger.debug("Building cost matrices...")
            self._build_cost_matrices()
            self.use_matrix_formulation = True
        else:
            self.logger.debug("No Q/R matrices provided, using parent class cost evaluation")
            self.use_matrix_formulation = False
        
        self.logger.debug("Calling parent class constructor...")
        # Call parent constructor
        super().__init__(
            sys, timepts, integral_cost,
            trajectory_constraints=trajectory_constraints,
            terminal_constraints=terminal_constraints,
            terminal_cost=terminal_cost,
            initial_guess=initial_guess,
            trajectory_method='shooting',
            **kwargs
        )
        
        self.logger.debug("CustomOptimalControlProblem initialization complete")
    
    def _linearize_system(self, sys, x0, u0):
        """Linearize a nonlinear system around an operating point.
        
        Parameters
        ----------
        sys : NonlinearIOSystem
            Nonlinear system to linearize
        x0 : array
            State operating point
        u0 : array
            Input operating point
        """
        self.logger.debug(f"Linearizing around x0={x0}, u0={u0}")
        
        # Use control toolbox linearization
        linsys = ct.linearize(sys, x0, u0)
        
        self.A = linsys.A
        self.B = linsys.B
        self.C = linsys.C if linsys.C is not None else np.eye(sys.nstates)
        self.D = linsys.D if linsys.D is not None else np.zeros((sys.noutputs, sys.ninputs))
        
        self.logger.debug("Linearization complete")
        self.logger.debug(f"A eigenvalues: {np.linalg.eigvals(self.A)}")
        
    def _build_prediction_matrices(self):
        """Build prediction matrices Phi and Gamma.
        
        These matrices predict the future states based on initial state
        and future inputs:
        
        X = Phi * x0 + Gamma * U
        
        where:
        - X = [x(1); x(2); ...; x(N)] is the stacked state vector
        - U = [u(0); u(1); ...; u(N-1)] is the stacked input vector
        - x0 is the initial state
        """
        self.logger.debug("_build_prediction_matrices called")
        
        nx = self.original_sys.nstates
        nu = self.original_sys.ninputs
        N = self.N
        
        self.logger.debug(f"Building matrices for nx={nx}, nu={nu}, N={N}")
        
        # Initialize matrices
        self.Phi = np.zeros((N * nx, nx))
        self.Gamma = np.zeros((N * nx, N * nu))
        
        self.logger.debug(f"Phi shape: {self.Phi.shape}, Gamma shape: {self.Gamma.shape}")
        
        # Build Phi matrix (state transition)
        A_power = self.A.copy()
        for i in range(N):
            self.Phi[i*nx:(i+1)*nx, :] = A_power
            A_power = A_power @ self.A
        
        self.logger.debug("Phi matrix constructed")
        
        # Build Gamma matrix (input influence)
        for i in range(N):
            A_power = np.eye(nx)
            for j in range(i + 1):
                if i - j == 0:
                    self.Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = self.B
                else:
                    self.Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_power @ self.B
                if j < i:
                    A_power = A_power @ self.A
        
        self.logger.debug("Gamma matrix constructed")
        self.logger.debug(f"Phi norm: {np.linalg.norm(self.Phi):.6f}, "
                      f"Gamma norm: {np.linalg.norm(self.Gamma):.6f}")
    
    def _build_cost_matrices(self):
        """Build quadratic cost matrices H and f for the QP problem.
        
        The cost function is formulated as:
        J = 0.5 * U^T * H * U + f(x0)^T * U + constant(x0)
        
        where U is the vector of control inputs over the horizon.
        """
        self.logger.debug("_build_cost_matrices called")
        
        nx = self.original_sys.nstates
        nu = self.original_sys.ninputs
        N = self.N
        
        Q = np.atleast_2d(self.Q)
        R = np.atleast_2d(self.R)
        Qf = np.atleast_2d(self.Qf) if self.Qf is not None else Q
        
        # Build block diagonal Q matrix for all time steps
        Q_bar = np.zeros((N * nx, N * nx))
        for i in range(N - 1):
            Q_bar[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = Q
        # Terminal cost
        Q_bar[(N-1)*nx:N*nx, (N-1)*nx:N*nx] = Qf
        
        self.logger.debug(f"Q_bar shape: {Q_bar.shape}, norm: {np.linalg.norm(Q_bar):.6f}")
        
        # Build block diagonal R matrix for all time steps
        R_bar = np.zeros((N * nu, N * nu))
        for i in range(N):
            R_bar[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = R
        
        self.logger.debug(f"R_bar shape: {R_bar.shape}, norm: {np.linalg.norm(R_bar):.6f}")
        
        # Compute Hessian matrix
        # J = (Phi*x0 + Gamma*U)^T * Q_bar * (Phi*x0 + Gamma*U) + U^T * R_bar * U
        # J = U^T * (Gamma^T * Q_bar * Gamma + R_bar) * U + ...
        self.H = self.Gamma.T @ Q_bar @ self.Gamma + R_bar
        
        # Make sure H is symmetric (numerical errors can make it slightly asymmetric)
        self.H = 0.5 * (self.H + self.H.T)
        
        self.logger.debug(f"Hessian H shape: {self.H.shape}, norm: {np.linalg.norm(self.H):.6f}")
        
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(self.H)
        min_eig = np.min(eigvals)
        self.logger.debug(f"H eigenvalues: min={min_eig:.6e}, max={np.max(eigvals):.6e}")
        if min_eig <= 0:
            self.logger.warning(f"Hessian H is not positive definite! Min eigenvalue: {min_eig:.6e}")
        
        # Linear term will be computed in cost function (depends on x0)
        # f(x0) = Gamma^T * Q_bar * Phi * x0
        self.f_matrix = self.Gamma.T @ Q_bar @ self.Phi
        
        self.logger.debug(f"f_matrix shape: {self.f_matrix.shape}, "
                      f"norm: {np.linalg.norm(self.f_matrix):.6f}")
        
        # Constant term (depends on x0, not needed for optimization)
        # c(x0) = x0^T * Phi^T * Q_bar * Phi * x0
        self.c_matrix = self.Phi.T @ Q_bar @ self.Phi
        
        self.logger.debug(f"c_matrix shape: {self.c_matrix.shape}, "
                      f"norm: {np.linalg.norm(self.c_matrix):.6f}")
        self.logger.debug("Cost matrices construction complete")
    
    def _cost_function(self, coeffs):
        """Override cost function to use matrix-based QP formulation if available.
        
        If Q and R matrices were provided, uses efficient matrix formulation.
        Otherwise, falls back to parent class implementation.
        
        Parameters
        ----------
        coeffs : array
            Optimization variables (stacked input sequence U)
        
        Returns
        -------
        float
            Cost value for the given input sequence
        """
        if not self.use_matrix_formulation:
            # Fall back to parent implementation
            return super()._cost_function(coeffs)
        
        self.logger.debug("_cost_function called (matrix formulation)")
        self.logger.debug(f"coeffs shape: {coeffs.shape}, norm: {np.linalg.norm(coeffs):.6f}")
        
        # U is the coefficient vector (stacked inputs)
        U = coeffs.reshape(-1, 1)
        
        # Linear term depends on current state
        f = self.f_matrix @ self.x.reshape(-1, 1)
        self.logger.debug(f"Linear term f norm: {np.linalg.norm(f):.6f}")
        
        # Compute quadratic cost: 0.5 * U^T * H * U + f^T * U
        # Use item() to extract scalar from 1x1 array, or flatten()[0]
        quadratic_term = 0.5 * (U.T @ self.H @ U).item()
        linear_term = (f.T @ U).item()
        constant_term = 0.5 * (self.x.reshape(1, -1) @ self.c_matrix @ self.x.reshape(-1, 1)).item()
        
        cost = quadratic_term + linear_term + constant_term
        
        self.logger.debug(f"Cost breakdown - quadratic: {quadratic_term:.6f}, "
                    f"linear: {linear_term:.6f}, constant: {constant_term:.6f}")
        self.logger.debug(f"Total cost: {cost:.6f}")
        
        # Update statistics
        self.cost_evaluations += 1

        return cost

    def _compute_states_inputs(self, coeffs):
        """Override to use matrix-based prediction if available.
        
        Parameters
        ----------
        coeffs : array
            Optimization variables (stacked input sequence)
        
        Returns
        -------
        states : array
            State trajectory (nstates x N)
        inputs : array
            Input trajectory (ninputs x N)
        """
        if not self.use_matrix_formulation:
            # Fall back to parent implementation (simulation-based)
            return super()._compute_states_inputs(coeffs)
        
        self.logger.debug("_compute_states_inputs called (matrix prediction)")
        
        nx = self.original_sys.nstates
        nu = self.original_sys.ninputs
        N = self.N
        
        # Check if we already computed this
        if np.array_equal(coeffs, self.last_coeffs) and \
           np.array_equal(self.x, self.last_x):
            self.logger.debug("Using cached states and inputs")
            return self.last_states, coeffs.reshape(nu, N)
        
        self.logger.debug(f"Computing new prediction for x0: {self.x}")
        
        # Reshape inputs
        U = coeffs.reshape(-1, 1)
        inputs = coeffs.reshape(nu, N)
        
        self.logger.debug(f"Input sequence shape: {inputs.shape}")
        self.logger.debug(f"Input range: [{np.min(inputs):.6f}, {np.max(inputs):.6f}]")
        
        # Predict states: X = Phi * x0 + Gamma * U
        X = self.Phi @ self.x.reshape(-1, 1) + self.Gamma @ U
        states = X.reshape(nx, N)
        
        self.logger.debug(f"Predicted states shape: {states.shape}")
        self.logger.debug(f"State range: [{np.min(states):.6f}, {np.max(states):.6f}]")
        
        # Cache results
        self.last_x = self.x.copy()
        self.last_states = states
        self.last_coeffs = coeffs.copy()
        
        self.logger.debug("State and input computation complete")
        
        return states, inputs
    
    def compute_trajectory(self, x, squeeze=None, transpose=None, return_states=True,
                          initial_guess=None, print_summary=True, **kwargs):
        """Compute the optimal trajectory starting at state x.
        
        Overrides parent method to add logging.
        
        Parameters
        ----------
        x : array_like or number
            Initial state for the system
        squeeze : bool, optional
            If True, squeeze output arrays
        transpose : bool, optional
            If True, transpose output arrays
        return_states : bool, optional
            If True, return state trajectory
        initial_guess : array, optional
            Initial guess for optimization
        print_summary : bool, optional
            If True, print summary statistics
        **kwargs
            Additional arguments passed to parent
        
        Returns
        -------
        OptimalControlResult
            Result object with optimal trajectory
        """
        self.logger.debug("=" * 60)
        self.logger.debug("compute_trajectory called")
        self.logger.debug(f"Initial state x: {x}")
        self.logger.debug(f"Parameters: squeeze={squeeze}, transpose={transpose}, "
                      f"return_states={return_states}, print_summary={print_summary}")
        
        result = super().compute_trajectory(
            x, squeeze=squeeze, transpose=transpose, return_states=return_states,
            initial_guess=initial_guess, print_summary=print_summary, **kwargs
        )
        
        self.logger.debug(f"Optimization success: {result.success}")
        if hasattr(result, 'cost'):
            self.logger.debug(f"Final cost: {result.cost:.6f}")
        self.logger.debug("compute_trajectory complete")
        self.logger.debug("=" * 60)
        
        return result
    
    def compute_mpc(self, x, squeeze=None):
        """Compute the optimal input at state x.
        
        Overrides parent method to add logging.
        
        Parameters
        ----------
        x : array_like or number
            Initial state for the system
        squeeze : bool, optional
            If True, squeeze output array
        
        Returns
        -------
        array
            Optimal input at current time
        """
        self.logger.debug("compute_mpc called")
        self.logger.debug(f"Current state x: {x}")
        
        result = super().compute_mpc(x, squeeze=squeeze)
        
        self.logger.debug(f"Computed optimal input: {result}")
        
        return result
    
    def get_prediction_matrices(self):
        """Return the prediction matrices for inspection or external use.
        
        Returns
        -------
        dict
            Dictionary containing Phi, Gamma, H, and f_matrix (if available)
        """
        self.logger.debug("get_prediction_matrices called")
        
        if not self.use_matrix_formulation:
            self.logger.warning("Matrix formulation not available")
            return None
        
        matrices = {
            'Phi': self.Phi,
            'Gamma': self.Gamma,
            'H': self.H,
            'f_matrix': self.f_matrix,
            'c_matrix': self.c_matrix
        }
        
        self.logger.debug("Returning prediction matrices")
        
        return matrices


def verify_custom_problem(sys, timepts, Q, R, x0,
                         trajectory_constraints=None, Qf=None):
    """Verify that the custom problem produces the same results as the original.
    
    Parameters
    ----------
    sys : InputOutputSystem
        System to control
    timepts : array
        Time points for the horizon
    Q : array
        State weight matrix
    R : array
        Input weight matrix
    x0 : array
        Initial state
    trajectory_constraints : list, optional
        Trajectory constraints
    Qf : array, optional
        Terminal weight matrix
    
    Returns
    -------
    dict
        Dictionary with 'original' and 'custom' results for comparison
    """
    log = logging.getLogger(topic)
    log.debug("verify_custom_problem called")
    log.debug(f"System: nstates={sys.nstates}, ninputs={sys.ninputs}")
    log.debug(f"Initial state: {x0}")
    log.debug(f"Horizon length: {len(timepts)}")
    
    # Define cost functions for original problem
    def integral_cost(x, u):
        return x @ Q @ x + u @ R @ u
    
    Qf_term = Qf if Qf is not None else Q
    def terminal_cost(x, u):
        return x @ Qf_term @ x
    
    # Create original problem
    log.debug("Creating original OptimalControlProblem...")
    original_problem = optimal.OptimalControlProblem(
        sys, timepts, integral_cost,
        trajectory_constraints=trajectory_constraints,
        terminal_cost=terminal_cost
    )
    
    # Create custom problem with same parameters
    log.debug("Creating CustomOptimalControlProblem...")
    custom_problem = CustomOptimalControlProblem(
        sys, timepts, integral_cost, terminal_cost=terminal_cost,
        Q=Q, R=R, Qf=Qf,
        trajectory_constraints=trajectory_constraints
    )
    
    # Solve both problems
    log.debug("Solving original problem...")
    original_res = original_problem.compute_trajectory(x0, print_summary=False)
    
    log.debug("Solving custom problem...")
    custom_res = custom_problem.compute_trajectory(x0, print_summary=False)
    
    # Compare results
    inputs_close = np.allclose(original_res.inputs, custom_res.inputs, rtol=1e-4, atol=1e-6)
    states_close = np.allclose(original_res.states, custom_res.states, rtol=1e-4, atol=1e-6)
    
    max_input_diff = np.max(np.abs(original_res.inputs - custom_res.inputs))
    max_state_diff = np.max(np.abs(original_res.states - custom_res.states))
    
    log.debug(f"Inputs match: {inputs_close}, max diff: {max_input_diff:.6e}")
    log.debug(f"States match: {states_close}, max diff: {max_state_diff:.6e}")
    
    results = {
        'original': original_res,
        'custom': custom_res,
        'inputs_match': inputs_close,
        'states_match': states_close,
        'max_input_diff': max_input_diff,
        'max_state_diff': max_state_diff,
    }
    
    log.debug("verify_custom_problem complete")
    
    return results