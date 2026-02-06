
import numpy as np
import control as ct


def define_vehicle_problem(cfg):
    """Define the vehicle steering problem."""
    # Vehicle plant model
    def vehicle_update(t, x, u, params):
        l = params.get('wheelbase', 3.0)  # Wheelbase
        phimax = params.get('maxsteer', 0.5)  # Max steering angle
        phi = np.clip(u[1], -phimax, phimax)  # Saturate steering input
        return np.array([
            np.cos(x[2]) * u[0], 
            np.sin(x[2]) * u[0], 
            (u[0] / l) * np.tan(phi)
        ])

    def vehicle_output(t, x, u, params):
        """Ensure output size matches system definition."""
        return np.array([x[0], x[1], x[2]])  # Return [x, y, θ]

    vehicle = ct.NonlinearIOSystem(
        vehicle_update, vehicle_output, 
        states=3, inputs=2, outputs=3, 
        name="vehicle"
    )

    # Problem configuration
    x0 = np.array([0.0, -2.0, 0.0])  # Initial state [x, y, θ]
    u0 = np.array([10.0, 0.0])  # Initial input [v, φ]
    xf = np.array([100.0, 2.0, 0.0])  # Final state [x, y, θ]
    uf = np.array([10.0, 0.0])  # Final input [v, φ]
    Tf = 10  # Final time
    Ts = 0.5  # Sampling period (make it finer for better resolution)
    N = int(Tf / Ts) + 1  # Number of time points (include endpoint)

    # Cost functions
    Q = np.diag([0, 0, 0.1])  # Penalize orientation error
    R = np.diag([1, 1])  # Penalize input efforts
    Qf = np.diag([1000, 1000, 1000])  # Terminal state cost
    
    traj_cost = ct.optimal.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    term_cost = ct.optimal.quadratic_cost(vehicle, Qf, np.zeros((2, 2)), x0=xf)

    # Constraints
    constraints = [
        ct.optimal.input_range_constraint(vehicle, [8, -0.1], [12, 0.1])
    ]

    problem_cfg = {
        "Ts": Ts, 
        "Tf": Tf,
        "N": N
    }

    return vehicle, x0, u0, xf, uf, problem_cfg, (traj_cost, term_cost, constraints, Q, R, Qf)