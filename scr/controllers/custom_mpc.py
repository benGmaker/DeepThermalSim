""""
This script is an implementation of a custom MPC controller and a default one.
The goal is to compare and validate a custom solution, such that we can build further on the custom solution.
The accompanying experiment is: validate_custom_mpc.py
"""
import numpy as np
import control.optimal as opt
import control as ct
import cvxpy as cp # Solver
import logging

def solving_matrices(controller_cfg):
    # todo make this a configurable by cfg
    Q = np.diag([15.0, 1.0])  # penalize position error more than velocity
    R = np.diag([0.1])  # input effort
    Qf = Q  # terminal weight (used via integral + terminal cost
    u_min, u_max = np.array([-2.0]), np.array([2.0])

    return Q, R, Qf, u_min, u_max

def define_constraints(plant, xref, uref, Ts, N):
    # Quadratic weights on states/inputs (state tracking to xref)
    Q, R, Qf, u_min, u_max = solving_matrices(plant)

    # Cost: use quadratic_cost on states/inputs around nominal (xref, uref)
    stage_cost = opt.quadratic_cost(plant, Q, R, x0=xref, u0=uref)
    term_cost = opt.quadratic_cost(plant, Qf, np.zeros((1, 1)), x0=xref, u0=uref)

    # Input constraints (optional)
    constraints = [opt.input_range_constraint(plant, lb=u_min, ub=u_max)]

    return stage_cost, term_cost, constraints


def default_mpc(plant, xref, uref, Ts, N):
    # Build the MPC IO system (horizon times)
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


def cvxpy_solver_mpc(plant, xref, uref, Ts, N):
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

    def compute_trajectory(x_now, u_hold, **solver_kwargs):
        # Solve the QP problem
        x0_param.value = x_now  # Update initial state
        xref_param.value = xref  # Reference trajectory

        # Solve the problem and compute the next control input
        try:
            prob.solve(**solver_kwargs)
            if prob.status in ["optimal", "optimal_inaccurate"] and U[:, 0].value is not None:
                u_next = U[:, 0].value
            else:
                u_next = u_hold  # Fall back to previous input
                logger.warning("CVXPY solver did not find an optimal solution, using previous control input.")
        except Exception:
            u_next = u_hold  # Robust fallback
            logger.warning("CVXPY solver failed, using previous control input.")
        
        return np.atleast_1d(u_next)

    # --- Controller Memory ---
    def cvxmpc_update(t, u_hold, y_meas, params):
        x_now = np.array(y_meas).squeeze()  # Measured state
        u_next = compute_trajectory(x_now, u_hold, **solver_kwargs)  # Solve QP
        return u_next  # Update the controller state

    def cvxmpc_output(t, u_hold, y_meas, params):
        # Output the held control value (computed at the last update step)
        # No optimization or solver call is made here to avoid algebraic loops.
        return np.atleast_1d(u_hold)

    # Declare the controller system
    cvx_mpc = ct.NonlinearIOSystem(
        cvxmpc_update, cvxmpc_output,
        inputs=ny, outputs=nu, states=nu, dt=Ts,
        name="cvx_mpc", params={"xref": xref}
    )
    return cvx_mpc
