""""
This script is an implementation of a custom MPC controller and a default one.
The goal is to compare and validate a custom solution, such that we can build further on the custom solution.
The accompanying experiment is: validate_custom_mpc.py
"""
from typing import Tuple, List, Callable 
from numpy.typing import ArrayLike
import numpy as np
import control.optimal as opt
import control as ct
import cvxpy as cp # Solver
import logging
import control.optimal as opt

topic = 'controller'

def solving_matrices(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves and returns matrices for control problem configuration.
    
    Args:
        controller_cfg (dict): Configuration dictionary for controller parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Q, R, Qf, u_min, u_max matrices for quadratic programming and constraints.
    """
    log = logging.getLogger(topic)
    controller_cfg = cfg.controller

    # Extract the number of states (nx) and number of inputs (nu) from the system configuration
    system_cfg = cfg.system
    nx = len(system_cfg.get("xref", []))  # Number of states, based on xref dimensions
    nu = len(system_cfg.get("uref", []))  # Number of inputs, based on uref dimensions

    # Initialize matrices with controller-provided values or defaults
    Q = np.array(controller_cfg.get("Q", np.eye(nx) * 10))  # Default weights prioritize diagonals
    R = np.array(controller_cfg.get("R", np.eye(nu) * 1))
    Qf = np.array(controller_cfg.get("Qf", Q))  # Terminal state weight
    u_min = np.array(controller_cfg.get("u_min", [-2.0] * nu))  # Input limits as per system input dimension
    u_max = np.array(controller_cfg.get("u_max", [2.0] * nu))

    # Validate dimensions of provided matrices and correct if necessary
    if Q.shape != (nx, nx):
        log.warning(f"Q dimensions {Q.shape} do not match system state size {nx}. Falling back to default.")
        Q = np.eye(nx) * 10  # Default: Identity matrix scaled for nx

    if R.shape != (nu, nu):
        log.warning(f"R dimensions {R.shape} do not match system input size {nu}. Falling back to default.")
        R = np.eye(nu) * 1  # Default: Identity matrix scaled for nu

    if Qf.shape != (nx, nx):
        log.warning(f"Qf dimensions {Qf.shape} do not match system state size {nx}. Falling back to default.")
        Qf = Q  # Default: Terminal weight is same as Q

    if len(u_min) != nu:
        log.warning(f"u_min dimensions {len(u_min)} do not match system input size {nu}. Falling back to default.")
        u_min = np.array([-2.0] * nu)  # Default lower bound for inputs

    if len(u_max) != nu:
        log.warning(f"u_max dimensions {len(u_max)} do not match system input size {nu}. Falling back to default.")
        u_max = np.array([2.0] * nu)  # Default upper bound for inputs

    return Q, R, Qf, u_min, u_max
def define_constraints(
    plant: ct.InputOutputSystem,
    xref: ArrayLike,
    uref: ArrayLike,
    Ts: float,
    N: int,
    cfg: dict,
) -> Tuple[Callable, Callable, List[Callable]]:
    """
    Defines the constraints for the control problem.

    Args:
        plant (ct.InputOutputSystem): The controlled plant model.
        xref (ArrayLike): Reference state trajectory.
        uref (ArrayLike): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.
        cfg (dict): Configuration dictionary

    Returns:
        Tuple[Callable, Callable, List[Callable]]: Stage cost, terminal cost, and constraints.
    """
    # Quadratic weights on states/inputs
    Q, R, Qf, u_min, u_max = solving_matrices(cfg)

    # Cost: quadratic cost on states/inputs around nominal (xref, uref)
    stage_cost = opt.quadratic_cost(plant, Q, R, x0=xref, u0=uref)
    term_cost = opt.quadratic_cost(plant, Qf, np.zeros((1, 1)), x0=xref, u0=uref)

    # Input constraints (optional)
    constraints = [opt.input_range_constraint(plant, lb=u_min, ub=u_max)]

    return stage_cost, term_cost, constraints


def default_mpc(
    plant: ct.InputOutputSystem,
    xref: ArrayLike,
    uref: ArrayLike,
    Ts: float,
    N: int,
    cfg: dict,
) -> ct.InputOutputSystem:
    """
    Creates a default MPC controller using a predefined cost and constraints.

    Args:
        plant (ct.InputOutputSystem): Discrete-time plant model to control.
        xref (ArrayLike): Reference state trajectory.
        uref (ArrayLike): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.
        cfg (dict): Configuration dictionary
    Returns:
        ct.InputOutputSystem: The MPC I/O system.
    """
    stage_cost, term_cost, constraints = define_constraints(plant, xref, uref, Ts, N, cfg)
    timepts = np.arange(N) * Ts
    builtin_mpc = opt.create_mpc_iosystem(
        plant, timepts,
        integral_cost=stage_cost,
        terminal_cost=term_cost,
        trajectory_constraints=constraints,
    )
    return builtin_mpc

def cvxpy_solver_mpc(
    plant: ct.InputOutputSystem,
    xref: ArrayLike,
    uref: ArrayLike,
    Ts: float,
    N: int,
    cfg: dict,
) -> ct.InputOutputSystem:
    """
    Configures an MPC controller using CVXPY as the optimization solver.

    Args:
        plant (ct.InputOutputSystem): Discrete-time plant model to control.
        xref (ArrayLike): Reference state trajectory.
        uref (ArrayLike): Reference input trajectory.
        Ts (float): Sampling time.
        N (int): Horizon length.
        cfg (dict): Configuration dictionary.

    Returns:
        ct.InputOutputSystem: A nonlinear I/O system for online CVXPY-based MPC.
    """
    logger = logging.getLogger(topic)
    nx, nu = plant.nstates, plant.ninputs
    ny = plant.noutputs

    # --- Weights and bounds ---
    Q, R, Qf, u_min, u_max = solving_matrices(cfg)

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
    def initialize_solver(x0: ArrayLike, **solver_kwargs) -> ArrayLike:
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


    def compute_trajectory(x_now: ArrayLike, u_hold: ArrayLike, **solver_kwargs) -> ArrayLike:
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
    def cvxmpc_update(t: float, u_hold: ArrayLike, y_meas: ArrayLike, params: dict) -> ArrayLike:
        """Controller state update."""
        x_now = np.array(y_meas).squeeze()  # Measured state
        logger.debug(f"[Update] t={t}, x_now={x_now}, u_hold={u_hold}")

        if t == 0:
            logger.debug("[Update] First timestep, solver already initialized.")
            return initial_control  # Return the precomputed control for initialization

        u_next = compute_trajectory(x_now, u_hold, **solver_kwargs)  # Solve MPC
        logger.debug(f"[Update] Computed u_next={u_next}")
        return u_next

    def cvxmpc_output(t: float, u_hold: ArrayLike, y_meas: ArrayLike, params: dict) -> ArrayLike:
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
