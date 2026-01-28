# configuration management
from logging import raiseExceptions

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from pathlib import Path

# Experiments
from scr.experiments import AVAILABLE_EXPERIMENTS
from scr.utils.config_logger import configure_logging

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):
    rundir = HydraConfig.get().run.dir

    # --- Experiment logger setup ----
    logger_config = cfg.get("logger", {})
    log_root = logger_config.get("log_root", 'logs')
    root_dir = Path(rundir, log_root)
    console_level = logger_config.get("console_level", "INFO")
    topics = logger_config.get("topics", ['log'])
    configure_logging(topics=topics, log_root=root_dir, console_level=console_level)

    # --- loading experiment ---
    name = cfg.experiment.get("name", None)

    # get class from mapping
    experiment = AVAILABLE_EXPERIMENTS.get(name)
    if not experiment:
        raise RuntimeError(
            f"The experiment {name} has not been found, please set cfg.experiment.name to one of: {AVAILABLE_EXPERIMENTS.keys()}"
        )

    # Checking for configuration experiment
    if not (hasattr(experiment, "run") and callable(experiment.run)):
        raise RuntimeError(f"Selected experiment '{name}' has no runnable .run() method.")

    # Initializing experiment
    try:
        init_exp = experiment(cfg, rundir)
    except Exception as e:
        print(e)
        raise RuntimeError(f"Failed to initialize experiment '{name}'.")

    # Running experiment
    init_exp.run()

if __name__ == "__main__":
    main()