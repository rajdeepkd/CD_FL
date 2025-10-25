import logging
import os
import json
import hashlib
from typing import Any, Dict


_LOGGER_NAME = "federated"


def init_logging(log_level: str, log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)

    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fh_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(ch)
    logger.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME).getChild(name)


def create_run_dirs(
    base_results_dir: str,
    exp_name: str,
    logs_dir: str,
    summaries_dir: str,
    artifacts_dir: str,
) -> Dict[str, str]:
    run_root = os.path.join(base_results_dir, exp_name)
    run_logs = os.path.join(run_root, "logs")
    run_summaries = os.path.join(run_root, "summaries")
    run_artifacts = os.path.join(run_root, "artifacts")

    os.makedirs(run_root, exist_ok=True)
    os.makedirs(run_logs, exist_ok=True)
    os.makedirs(run_summaries, exist_ok=True)
    os.makedirs(run_artifacts, exist_ok=True)

    return {
        "root_dir": run_root,
        "logs_dir": run_logs,
        "summaries_dir": run_summaries,
        "artifacts_dir": run_artifacts,
    }


def save_run_metadata(
    run_dirs: Dict[str, str],
    config: Dict[str, Any],
    exp_name: str,
) -> None:
    meta_path = os.path.join(run_dirs["artifacts_dir"], "run_config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    repro_path = os.path.join(run_dirs["artifacts_dir"], "README_reproducibility.txt")
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write(
            "Reproducibility Notes\n"
            "=====================\n\n"
            f"Experiment name: {exp_name}\n\n"
            "To reproduce this run:\n"
            "1. Use the same raw CSVs in data/raw_data/\n"
            "2. Use the saved config snapshot run_config.json\n"
            "3. Run `python run_experiment.py --config <copied_config.yaml/json overrides>`\n"
            "4. Ensure same seeds and same PyTorch / numpy versions.\n"
        )


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def append_client_metrics_to_txt(
    summaries_dir: str,
    client_id: str,
    strategy: str,
    metrics: Dict[str, float],
) -> None:
    """
    After training, append final test metrics to that client's summary txt.
    """
    path = os.path.join(summaries_dir, f"{client_id}_class_distribution.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n--- Final Model Performance ---\n")
        f.write(f"Strategy: {strategy}\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
