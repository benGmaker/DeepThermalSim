""""
This script is an implementation of a custom MPC controller and a default one.
The goal is to compare and validate a custom solution, such that we can build further on the custom solution.
The accompanying experiment is: validate_custom_mpc.py
"""
from typing import Tuple, List, Callable, Array 
import numpy as np
import control.optimal as opt
import control as ct
import cvxpy as cp # Solver
import logging

def solving_matrices(controller_cfg: dict) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Solves and returns matrices for control problem configuration.
    
    Args:
        controller_cfg (dict): Placeholder configuration dictionary for additional customization.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: 
            Q, R, Qf, u_min, u_max matrices for quadratic programming and constraints.
    """
    # Todo make these configurable via controller_cfg
    Q = np.diag([15.0, 1.0])  # penalize position error more than velocity
    R = np.diag([0.1])  # input effort
    Qf = Q  # terminal weight (used via integral + terminal cost
    u_min, u_max = np.array([-2.0]), np.array([2.0])

    return Q, R, Qf, u_min, u_max

def define_constraints(
    plant: ct.InputOutputSystem, xref: Array, uref: Array, Ts: float, N: int
) -> Tuple[Callable, Callable, List[Callable]]:
    """
    Defines the constraints for the control problem.

    Args:
        plant (ct.InputOutputSystem): The controlled plant model.
        xref (Array): Reference state trajectory.
        uref (Array): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.

    Returns:
        Tuple[Callable, Callable, List[Callable]]: Stage cost, terminal cost, and constraints.
    """
    # Quadratic weights on states/inputs (state tracking to xref)
    Q, R, Qf, u_min, u_max = solving_matrices(plant)

    # Cost: use quadratic_cost on states/inputs around nominal (xref, uref)
    stage_cost = opt.quadratic_cost(plant, Q, R, x0=xref, u0=uref)
    term_cost = opt.quadratic_cost(plant, Qf, np.zeros((1, 1)), x0=xref, u0=uref)

    # Input constraints (optional)
    constraints = [opt.input_range_constraint(plant, lb=u_min, ub=u_max)]

    return stage_cost, term_cost, constraints


def default_mpc(plant: ct.InputOutputSystem, xref: Array, uref: Array, Ts: float, N: int
    ) -> ct.InputOutputSystem:
    """
    Creates a default MPC controller using a predefined cost and constraints.

    Args:
        plant (ct.InputOutputSystem): The plant model to control.
        xref (Array): Reference state trajectory.
        uref (Array): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.

    Returns:
        ct.InputOutputSystem: The MPC I/O system.
    """
    stage_cost, term_cost, constraints = define_constraints(plant, xref, uref, Ts, N)
    timepts = np.arange(N) * Ts
    builtin_mpc = opt.create_mpc_iosystem(
        plant, timepts,
        integral_cost=stage_cost,
        terminal_cost=term_cost,
        trajectory_constraints=constraints,
        # You can pass solver knobs: minimize_method="SLSQP", minimize_kwargs={...}
    )
    return builtin_mpc


def cvxpy_solver_mpc(
    plant: ct.InputOutputSystem, xref: Array, uref: Array, Ts: float, N: int
) -> ct.InputOutputSystem:
    """
    Configures an MPC controller using CVXPY as the optimization solver.

    Args:
        plant (ct.InputOutputSystem): The plant model to control.
        xref (Array): Reference state trajectory.
        uref (Array): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.

    Returns:
        ct.InputOutputSystem: A nonlinear I/O system for online CVXPY-based MPC.
    """
    logger = logging.getLogger('controller')
    nx, nu = plant.nstates, plant.ninputs
    ny = plant.noutputs

    # --- Weights and bounds ---
    Q, R, Qf, u_min, u_max = solving_matrices(plant)

    # --- Build the QP ---
    X = cp.Variable((nx, N + 1))
    U = cp.Variable((nu, N))
    x0_param = cp.Parameter(nx)
    xref_param = cp.Parameter(nx)

    cost_terms = []
    for k in range(N):
        cost_terms.append(cp.quad_form(X[:, k] - xref_param, Q))
        cost_terms.append(cp.quad_form(U[:, k], R))
    cost_terms.append(cp.quad_form(X[:, N] - xref_param, Qf))
    objective = cp.Minimize(cp.sum(cost_terms))

    constraints_cvx = [X[:, 0] == x0_param]
    Ad, Bd = plant.A, plant.B
    for k in range(N):
        constraints_cvx += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                            u_min <= U[:, k], U[:, k] <= u_max]

    prob = cp.Problem(objective, constraints_cvx)
    solver_kwargs = dict(solver=cp.OSQP, warm_start=True, verbose=False)

    # --- Precompute the initial output ---
    def initialize_solver(x0: Array, **solver_kwargs) -> Array:
        """Solve the initial MPC problem at t=0."""
        logger.debug("[Initialize Solver] Precomputing initial control.")
        x0_param.value = x0
        xref_param.value = xref
        try:
            prob.solve(**solver_kwargs)
            if prob.status in ["optimal", "optimal_inaccurate"] and U[:, 0].value is not None:
                logger.debug("[Initialize Solver] Initial control precomputed.")
                return np.atleast_1d(U[:, 0].value)  # Use the first valid control input
            else:
                logger.warning("[Initialize Solver] Precompute failed, using default [0.].")
                return np.zeros(plant.ninputs)
        except Exception as e:
            logger.error(f"[Initialize Solver] Exception during precompute: {e}")
            return np.zeros(plant.ninputs)  # Fallback default

    # Precompute initial output
    initial_state = np.zeros(nu)
    initial_control = initialize_solver(np.zeros(nx), **solver_kwargs)

    def compute_trajectory(x_now: Array, u_hold: Array, **solver_kwargs) -> Array:
        """Solve the MPC trajectory."""
        logger.debug(f"[Compute Trajectory] Solving with state x_now={x_now}, u_hold={u_hold}")
        x0_param.value = x_now
        xref_param.value = xref
        try:
            prob.solve(**solver_kwargs)
            logger.debug(f"[Compute Trajectory] Solver status: {prob.status}")
            if prob.status in ["optimal", "optimal_inaccurate"] and U[:, 0].value is not None:
                logger.debug(f"[Compute Trajectory] Optimal control: {U[:, 0].value}")
                return np.atleast_1d(U[:, 0].value)
            else:
                logger.warning("[Compute Trajectory] MPC solve failed, using previous control input.")
                return u_hold
        except Exception as e:
            logger.error(f"[Compute Trajectory] Solver exception: {e}")
            return u_hold  # Fallback

    # --- Controller Memory ---
    def cvxmpc_update(t: float, u_hold: Array, y_meas: Array, params: dict) -> Array:
        """Controller state update."""
        x_now = np.array(y_meas).squeeze()  # Measured state
        logger.debug(f"[Update] t={t}, x_now={x_now}, u_hold={u_hold}")

        if t == 0:
            logger.debug("[Update] First timestep, solver already initialized.")
            return initial_control  # Return the precomputed control for initialization

        u_next = compute_trajectory(x_now, u_hold, **solver_kwargs)  # Solve MPC
        logger.debug(f"[Update] Computed u_next={u_next}")
        return u_next

    def cvxmpc_output(t: float, u_hold: Array, y_meas: Array, params: dict) -> Array:
        """Controller output definition."""
        # Use the precomputed initial control for the very first output at t=0
        if t == 0:
            logger.debug(f"[Output] Using precomputed control at t=0: {initial_control}")
            return initial_control  # Directly return precomputed output for first step
        logger.debug(f"[Output] t={t}, u_hold={u_hold}")
        return np.atleast_1d(u_hold)

    # Declare the controller system
    cvx_mpc = ct.NonlinearIOSystem(
        cvxmpc_update, cvxmpc_output,
        inputs=ny, outputs=nu, states=nu, dt=Ts,  # <-- Fixed number of states
        name="cvx_mpc", params={"xref": xref}
    )
    return cvx_mpc
