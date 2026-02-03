import numpy as np
import control as ct
import logging

log_topic = 'model'

def transitor_heat(system_cfg, sim_cfg) -> ct.c2d:
    """
    Creates a single transistor heat system model using configuration.

    :param system_cfg:
    :param sim_cfg:
    """
    logger = logging.getLogger(log_topic)
    logger.debug("Creating single transistor heat system model.")
    C = system_cfg.C
    R1 = system_cfg.R1
    R2 = system_cfg.R2
    Ts = sim_cfg.Ts

    A = np.array([[-(1 / (R1 * C) + 1 / (R2 * C))]])
    B = np.array([[1 / C, 0.0]])
    C_mat = np.eye(1)  # expose full state as plant output
    D = np.zeros((1, 2))

    plant_c = ct.ss(A, B, C_mat, D)  # continuous
    plant = ct.c2d(plant_c, Ts, method='zoh')  # discrete (for both controllers)

    return plant    

def two_transistor_heat(system_cfg, sim_cfg) -> ct.c2d:
    """
    Creates a 2-transistor heat system model using configuration.

    :param system_cfg:
    :param sim_cfg:
    """
    logger = logging.getLogger(log_topic)
    logger.debug("Creating 2-transistor heat system model.")
    C1 = system_cfg.C1
    C2 = system_cfg.C2
    R1 = system_cfg.R1
    R2 = system_cfg.R2
    R3 = system_cfg.R3
    Ts = sim_cfg.Ts

    A = np.array([[-(1 / (R1 * C1) + 1 / (R2 * C1)), 1 / (R2 * C1)],
                  [1 / (R2 * C2), -(1 / (R2 * C2) + 1 / (R3 * C2))]])
    B = np.array([[1 / C1, 0.0],
                  [0.0, 1 / C2]])
    C = np.eye(2)  # expose full state as plant output
    D = np.zeros((2, 2))

    plant_c = ct.ss(A, B, C, D)  # continuous
    plant = ct.c2d(plant_c, Ts, method='zoh')  # discrete (for both controllers)

    return plant


