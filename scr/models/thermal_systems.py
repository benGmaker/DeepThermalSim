from typing import Any, Dict, Array 
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

def two_transistor_heat(system_cfg: Dict[str, Any]) -> ct.NonlinearIOSystem:
    """
    Creates a two-transistor heat system model using configuration.

    :param system_cfg: Dictionary containing system configuration, such as thermal and electrical constants.
    :param sim_cfg: Dictionary containing simulation configuration, such as the sampling time (Ts).
    :return: Discrete-time plant model (ct.NonlinearIOSystem) for simulation.
    """
    logger = logging.getLogger(log_topic)
    logger.debug("Creating 2-transistor heat system model.")

    if system_cfg.name != 'two_transistor_heat_system':
        logger.warning(f"System configuration name '{system_cfg.name}' does not match expected 'two_transistor_heat_system'. "
                       "Mismatching or default parameters may be used.")
    name = system_cfg.get('name', 'two_transistor_heat_system')

    # Parameters
    Ta = system_cfg.get('Ta', 25.0 + 273.15)  # Ambient temperature, K
    U = system_cfg.get('U', 10.0)           # Heat transfer coefficient, W/m^2-K
    m = system_cfg.get('m', 4.0 / 1000.0)   # Mass, kg
    Cp = system_cfg.get('Cp', 0.5 * 1000.0)  # Heat capacity, J/kg-K
    A = system_cfg.get('A', 10.0 / 100.0**2)  # Surface area, m^2
    As = system_cfg.get('As', 2.0 / 100.0**2)  # Smaller surface area, m^2
    alpha1 = system_cfg.get('alpha1', 0.0100)     # Heater 1 power coefficient, W/% heater
    alpha2 = system_cfg.get('alpha2', 0.0075)     # Heater 2 power coefficient, W/% heater
    eps = system_cfg.get('eps', 0.9)          # Emissivity of the surface
    sigma = system_cfg.get('sigma', 5.67e-8)    # Stefan-Boltzmann constant

    def update_function(t, x, u, params) -> Array:
        """Nonlinear state update function for the heat system."""
        # States
        T1 = x[0]  # Temperature of block 1
        T2 = x[1]  # Temperature of block 2
        Q1 = u[0]  # Heater power input 1 (%)
        Q2 = u[1]  # Heater power input 2 (%)

        # Heat transfer between blocks
        conv12 = U * As * (T2 - T1)
        rad12 = eps * sigma * As * (T2**4 - T1**4)

        # Nonlinear energy balances
        dT1dt = (1.0 / (m * Cp)) * (
            U * A * (Ta - T1) +
            eps * sigma * A * (Ta**4 - T1**4) +
            conv12 +
            rad12 +
            alpha1 * Q1
        )
        dT2dt = (1.0 / (m * Cp)) * (
            U * A * (Ta - T2) +
            eps * sigma * A * (Ta**4 - T2**4) -
            conv12 -
            rad12 +
            alpha2 * Q2
        )

        return np.array([dT1dt, dT2dt])
    
    def output_function(t, x, u, params) -> Array:
        """Output function for the heat system."""
        return x  # Output is the state itself (temperatures)
    
    # Create the system as a Nonlinear IO system
    heat_nonlinear_sys = ct.NonlinearIOSystem(
        update_function, output_function, name=name,
        states=['T1', 'T2'], inputs=['Q1', 'Q2'], outputs=['T1', 'T2']
    )

    return heat_nonlinear_sys


