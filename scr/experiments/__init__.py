from .validate_custom_mpc import ValidateCustomMPC
from .valdiate_custom_optimal_problem import ValidateCustomOptimalProblem

AVAILABLE_EXPERIMENTS = {
    'ValidateCustomMPC': ValidateCustomMPC,
    'ValidateCustomOptimalProblem': ValidateCustomOptimalProblem,
}
