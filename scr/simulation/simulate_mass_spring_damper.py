from control import TransferFunction, forced_response
import numpy as np
import matplotlib.pyplot as plt

def simulate_mass_spring_damper(mass, damping, stiffness, T, U):
    """
    Simulate a mass-spring-damper system.
    
    Parameters:
        mass (float): Mass (kg).
        damping (float): Damping coefficient (Ns/m).
        stiffness (float): Spring constant (N/m).
        T (numpy.ndarray): Time array.
        U (numpy.ndarray): Input signal.

    Returns:
        yout (numpy.ndarray): System response.
    """
    numerator = [1]
    denominator = [mass, damping, stiffness]  # Transfer function: ms^2 + cs + k
    system = TransferFunction(numerator, denominator)

    _, yout, _ = forced_response(system, T, U)
    return yout

if __name__ == "__main__":
    # Example standalone test
    T = np.linspace(0, 10, 1000)
    U = np.sin(2 * np.pi * T)
    yout = simulate_mass_spring_damper(1.0, 0.5, 10.0, T, U)

    plt.figure()
    plt.plot(T, U, label="Input")
    plt.plot(T, yout, label="Response")
    plt.legend()
    plt.grid()
    plt.show()