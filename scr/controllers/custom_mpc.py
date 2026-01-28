""""
This script is an implementation of a custom MPC controller and a default one.
The goal is to compare and validate a custom solution, such that we can build further on the custom solution.
The accompanying experiment is: validate_custom_mpc.py
"""
import numpy as np
import control.optimal as opt
import control as ct
import cvxpy as cp # Solver

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
    nx, nu = plant.nstates, plant.ninputs
    ny = plant.noutputs

    # --- weights and bounds (your helper) ---
    Q, R, Qf, u_min, u_max = solving_matrices(plant)

    # --- build the parameterized QP (unchanged) ---
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

    # --- sampled-data controller with memory (hold) ---
    def cvxmpc_update(t, u_hold, y_meas, params):
        # y_meas must be the measured state; with C=I we get full state.
        x_now = np.array(y_meas).squeeze()
        x0_param.value = x_now
        xref_param.value = params.get("xref", xref)

        try:
            prob.solve(**solver_kwargs)
            if prob.status in ["optimal", "optimal_inaccurate"] and U[:, 0].value is not None:
                u_next = U[:, 0].value
            else:
                u_next = u_hold  # keep previous command if solver fails
        except Exception:
            u_next = u_hold  # robust fallback

        return np.atleast_1d(u_next)  # next controller *state*

    def cvxmpc_output(t, u_hold, y_meas, params):
        # Output the *held* control value to the plant at this sample
        return np.atleast_1d(u_hold)

    # ✅ Give the controller memory: states=nu (NOT zero)
    cvx_mpc = ct.NonlinearIOSystem(
        cvxmpc_update, cvxmpc_output,
        inputs=ny, outputs=nu, states=nu, dt=Ts,
        name="cvx_mpc", params={"xref": xref}
    )
    return cvx_mpc
