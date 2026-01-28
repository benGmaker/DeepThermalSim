#!/usr/bin/env python3
import logging
import logging.config
from pathlib import Path

def configure_logging(topics: list[str], log_root: Path = Path("../experiment_logger/logs"), level: str = "DEBUG", console_level: str = "ERROR"):
    log_root.mkdir(parents=True, exist_ok=True)
    levels_dir = log_root / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)

    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            **{
                f"h_{t}": {
                    "class": "logging.FileHandler",
                    "level": 'DEBUG',
                    "formatter": "std",
                    "filename": str(log_root / f"{t}.log"),
                    "encoding": "utf-8",
                }
                for t in topics
            },
            "h_ERROR_exact": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "formatter": "std",
                "filename": str(levels_dir / "ERROR.log"),
                "encoding": "utf-8",
            },
            "h_CRITICAL_exact": {
                "class": "logging.FileHandler",
                "level": "CRITICAL",
                "formatter": "std",
                "filename": str(levels_dir / "CRITICAL.log"),
                "encoding": "utf-8",
            },
            "h_WARNING_exact": {
                "class": "logging.FileHandler",
                "level": "WARNING",
                "formatter": "std",
                "filename": str(levels_dir / "WARNING.log"),
                "encoding": "utf-8",
            },
            "h_aggregate": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "std",
                "filename": str(log_root.parent / "main.log"),
                "encoding": "utf-8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "std",
            },
        },
        "loggers": {
            **{
                t: {
                    "level": "DEBUG",
                    "handlers": [f"h_{t}", "h_WARNING_exact", "h_ERROR_exact", "h_CRITICAL_exact", "h_aggregate", "console"],
                    "propagate": False,
                }
                for t in topics
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(LOGGING)

def main():
    topics = [
        "datalogger",
        "base_exp",
        "experiment",
        "drone_dpc",
        "dpc",
        "trajectory_sampling",
        "data_loading",
        "pre_processing",
    ]
    # You can now adjust the console log level here
    configure_logging(topics, console_level="INFO")

    datalogger = logging.getLogger("datalogger")
    base_exp = logging.getLogger("base_exp")
    experiment = logging.getLogger("experiment")
    drone_dpc = logging.getLogger("drone_dpc")
    dpc = logging.getLogger("dpc")
    trajectory_sampling = logging.getLogger("trajectory_sampling")
    data_loading = logging.getLogger("data_loading")
    pre_processing = logging.getLogger("pre_processing")

    datalogger.info("Starting data logging")
    base_exp.debug("Base experiment configuration loaded")
    experiment.warning("Experiment nearing resource limits")
    drone_dpc.error("Drone DPC encountered an error")
    dpc.critical("DPC critical failure")
    trajectory_sampling.info("Sampling trajectory with 100 steps")
    data_loading.error("Failed to parse input dataset schema")
    pre_processing.warning("Pre-processing took longer than expected")

if __name__ == "__main__":
    main()