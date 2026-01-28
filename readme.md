# DeepThermalSim

Welcome to the **DeepThermalSim**, a Python-based project designed to simulate and analyze thermal systems with flexible modeling, controller integration, and experiment management.
This project provides an efficient environment for experimenting with transfer function and state-space models, implementing advanced controllers (like custom MPC), and adding real-world features such as noise simulation.

---

## **Features**
- **Modular System Dynamics**:
  - Support for interchangeable **transfer function-based** and **state-space models**.
  - Predefined configurations for common systems: e.g., mass-spring-damper.
  - Noise simulation for real-world behavior analysis.

- **Controller Integration**:
  - Includes PID controllers as a baseline.
  - Framework for implementing custom controllers, such as **MPC** or **DeePC**.
  - Intermediate controllers for stabilization (e.g., PID used with MPC or DeePC).

- **Hydra Configuration**:
  - Use hierarchical configurations to manage experiments.
  - Easily switch between models, controllers, and experiments.

- **Experiment Automation**:
  - Run parameter sweeps (e.g., varying noise, controller gains).
  - Console and data logging routines to track and analyze results.

- **Testing and Validation**:
  - Unit tests for models, controllers, and experiment flows.
  - Benchmarks for comparing controller performance.

---

## **Getting Started**

### Prerequisites
1. **Install Python (>=3.8)**.
2. **Setup Dependencies**:
   Use `pip` to install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Hydra**:
   Hydra is used for managing configurations. Ensure it's included as part of the dependencies.

---

### Project Structure
The project is organized as follows:

```plaintext
project-root/
│
├── config/                            # Hydra configs for global settings, models, and simulations
│   ├── main.yaml                      # Entry-point configuration
│   ├── system/                        # System-specific configs
│   │   ├── mass_spring_damper.yaml    # Example model config
│   ├── simulation/                    # Experimental setups
│       ├── default.yaml               # Default experiment setup
│
├── src/                               # Source code (models, controllers, experiments, etc.)
│   ├── models/                        # Models (TF, state-space, noise functionality)
│   ├── controllers/                   # PID, MPC, and other controllers
│   ├── experiments/                   # Experiment runner for automation
│   ├── simulation/
│   ├── utils/                         # Logging, plotting, etc.
│
├── tests/                             # Unit tests for models, controllers, and simulations
│
├── data/                              # Data logging for saving results
│
├── main.py                 # Entry-point for running simulations/experiments
├── environment.yml                    # Conda environment definition
└── README.md                          # This README
```

---

### **Usage**

#### 1. Run a Basic Experiment
To run a simulation, use the `main.py` script. The `config/` folder defines system, simulation, and experiment parameters.

Example:
```bash
python main.py
```

Output:
- Console logs for progress tracking.
- Data logs (e.g., system response, controller actions) saved in the `data/` folder.
- Configurable plots of system and controller behavior.

#### 2. Change System/Simulation Configurations
To modify models or simulation setups, adjust the appropriate YAML files. For example:
- Use `config/system/mass_spring_damper.yaml` to define the parameters of a mass-spring-damper system.
- Use `config/simulation/default.yaml` to specify simulation duration, input signals, and noise levels.

#### 3. Add a Custom Controller
To implement an MPC controller:
1. Create a new file under `src/controllers/` (e.g., `mpc_controller.py`).
2. Use the `BaseController` interface as a starting point.
3. Integrate the new controller into the `main_simulation.py` flow.

#### 4. Automate Experiments
Use Hydra’s multirun feature for parameter sweeps:
```bash
python main_simulation.py -m simulation.steps=500,1000 simulation.noise_level=0.0,0.1
```

---

### **Example Configurations**

#### System Example: Mass-Spring-Damper
```yaml
system:
  mass: 1.0        # Mass (kg)
  damping: 0.5     # Damping coefficient (Ns/m)
  stiffness: 10.0  # Spring constant (N/m)
```

#### Simulation Example: Noise Test
```yaml
simulation:
  duration: 10.0        # Duration (s)
  steps: 1000           # Number of time steps
  input_amplitude: 1.0  # Amplitude of input signal
  input_frequency: 1.0  # Frequency (Hz)
  noise_level: 0.1      # Noise level
```

---

### **Planned Features**
- Advanced models beyond mass-spring-damper (e.g., nonlinear systems, hysteresis models).
- Visualizations comparing controller performance.
- Extended support for complex experiment setups.

---

### **Testing**
The project uses `pytest` for unit testing.

Run all tests:
```bash
pytest tests/
```

---

### **Dependencies**
- [Python Control Systems Library](https://python-control.readthedocs.io/)
- [Hydra](https://hydra.cc/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

Additional dependencies are listed in `requirements.txt`.

---

### **Contributing**
This project is currently for personal use, but contributions and feedback may be considered if made public in the future.

---


## Hydra Configuration

Hydra controls all experiment settings via layered config files.

- Main config: `experiment/config/config.yaml`
- Defaults: define which sub-configs load and where they are mounted.
- Example: Parameters in `controller/default_drone_dpc_controller.yaml` appear under `config.controller.params`.

You can re-run past configurations by pointing Hydra to a previous run’s `.hydra` folder:
```bash
python -m experiment.main --config-dir outputs/single/2025-12-05_11-02-06_/.hydra --config-name overwrite
```
Note: Rename the run’s `config.yaml` to `overwrite.yaml` (or any name you plan to use with `--config-name`).

**Important**: Configs are applied top-to-bottom. Later-loaded configs (e.g., via `+experiment=name`) override earlier parameters.

### Loading values

When loading the configuration:
- The top-level configuration is a standard `dict`.
- Nested sections are `DictConfig` objects (from OmegaConf).

#### Recommended ways to retrieve values

1. Error if missing (preferred for required values)
   - Dot access (compact):  
     `config.key`
   - Key access (useful in scripts):  
     `config['key']`

2. Provide a default (for optional values)  
   `config.get('key', default_value)`

#### Notes

- The configuration is a Hydra/OmegaConf object, similar to a normal dictionary.
- Some functions may require a plain `dict`. Convert with:
  ```python
  from omegaconf import OmegaConf
  plain_dict = OmegaConf.to_container(cfg)
  ```

### Overrides 
Above we have already used a bit of the overides. But bellow one can see how to add overrides to a run.
It is very usefull to change a single parameter or a collection within the configuration.
On top of that within the saving of the runs all of the overrides are included in the name of the folder.
Like `yyyy-mm-dd-HH-MM-SS_parameter=value`

### Overrides

Hydra supports fine-grained overrides. Overrides are also encoded in the output directory name for reproducibility (e.g., `yyyy-mm-dd-HH-MM-SS_parameter=value`).

- Add (introduce) a parameter:
  ```
  python -m experiment.main +parameter.subparameter=value
  python -m experiment.main +parameter_collection=override_file
  ```
- Replace (override) a parameter:
  ```
  python -m experiment.main parameter.subparameter=value
  python -m experiment.main parameter_collection=override_file
  ```
  
### Multirun (Sweeps)

Launch sweeps to evaluate multiple settings:
```
python -m experiment.main --multirun +hydra=sweep_config
```
Recommendations:
- Use a dedicated sweep file: `+hydra=sweep_file` for:
  - Consistent post-processing and visualization
  - Reproducibility
- Sweep 1–2 parameters at a time for easier visualization.

---

## Outputs

### .hydra
Contains the full YAML configuration used for a run. This ensures the run is always traceable and reproducible.

### Logging
Extensive logging keeps the console focused and preserves detailed traces:
- Configured via `utils/config_logger.py` and `config.logger` in Hydra
- Topics are defined under `config.logger.topics`; create with:
  ```python
  log = logging.getLogger("desired_topic")
  ```
- A `levels` folder aggregates high-severity logs for rapid triage; a clean run should have none.
- `main.log` in the run’s parent folder contains a full log of all events.

### **License**
MIT License

---

### **References**
- Python Control Systems Library: [https://python-control.readthedocs.io/](https://python-control.readthedocs.io/)
- Hydra Documentation: [https://hydra.cc/](https://hydra.cc/)