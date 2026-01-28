import numpy as np
import control as ct

def mass_spring_damper(system_cfg, sim_cfg) -> ct.c2d:
    """"
    Creates a basic mass-spring-damper plant using configuration.

    :param system_cfg:
    :param sim_cfg:
    """
    k = system_cfg.k
    m = system_cfg.m
    d = system_cfg.d
    Ts = sim_cfg.Ts

    A = np.array([[0.0, 1.0],
                  [-k / m, -d / m]])
    B = np.array([[0.0],
                  [1.0 / m]])
    C = np.eye(2)  # expose full state as plant output
    D = np.zeros((2, 1))

    plant_c = ct.ss(A, B, C, D)  # continuous
    plant = ct.c2d(plant_c, Ts, method='zoh')  # discrete (for both controllers)

    return plant