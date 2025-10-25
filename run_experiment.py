import argparse
import os
import time
from typing import Any, Dict

from src.utils.config import load_config, override_config
from src.utils.logging_utils import (
    init_logging,
    create_run_dirs,
    get_logger,
    save_run_metadata,
)
from src.experiments.runner import run_full_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments across per-client CSVs."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override: FL strategy to run (e.g., FedAvg, FedProx, PersonalizedHead).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override: number of global rounds.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        dest="local_epochs",
        help="Override: local epochs per round.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override: client batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override: learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override: random seed.",
    )
    parser.add_argument(
        "--equalize-samples",
        type=str,
        default=None,
        help="Override: 'true' or 'false' (equalize client contributions?).",
    )
    parser.add_argument(
        "--reweight-clients",
        type=str,
        default=None,
        help="Override: 'true' or 'false' (inverse-error weighting?).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--fedprox-mu",
        type=float,
        default=None,
        dest="fedprox_mu",
        help="Override: FedProx proximal mu.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. load base config
    config = load_config(args.config)

    # 2. apply overrides
    cli_overrides: Dict[str, Any] = {}
    if args.strategy is not None:
        cli_overrides["experiment.strategy"] = args.strategy
    if args.rounds is not None:
        cli_overrides["experiment.rounds"] = args.rounds
    if args.local_epochs is not None:
        cli_overrides["experiment.local_epochs"] = args.local_epochs
    if args.batch_size is not None:
        cli_overrides["experiment.batch_size"] = args.batch_size
    if args.lr is not None:
        cli_overrides["experiment.lr"] = args.lr
    if args.seed is not None:
        cli_overrides["experiment.seed"] = args.seed
    if args.equalize_samples is not None:
        cli_overrides["experiment.equalize_samples"] = (
            args.equalize_samples.lower() == "true"
        )
    if args.reweight_clients is not None:
        cli_overrides["experiment.reweight_clients"] = (
            args.reweight_clients.lower() == "true"
        )
    if args.device is not None:
        cli_overrides["experiment.device"] = args.device
    if args.fedprox_mu is not None:
        cli_overrides["experiment.fedprox_mu"] = args.fedprox_mu

    config = override_config(config, cli_overrides)

    # 3. prepare output dirs
    ts = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['experiment']['name']}_{config['experiment']['strategy']}_seed{config['experiment']['seed']}_{ts}"

    run_dirs = create_run_dirs(
        base_results_dir=config["output"]["results_dir"],
        exp_name=exp_name,
        logs_dir=config["output"]["logs_dir"],
        summaries_dir=config["output"]["summaries_dir"],
        artifacts_dir=config["output"]["artifacts_dir"],
    )

    # 4. init logger
    init_logging(
        log_level=config["logging"]["log_level"],
        log_file=os.path.join(
            run_dirs["logs_dir"],
            f"{ts}_{config['experiment']['strategy']}_{config['experiment']['seed']}.log",
        ),
    )
    logger = get_logger(__name__)
    logger.info("Starting federated experiment run.")
    logger.info("Resolved configuration:")
    logger.info(config)

    # 5. write run metadata for reproducibility
    save_run_metadata(
        run_dirs=run_dirs,
        config=config,
        exp_name=exp_name,
    )

    # 6. run experiment orchestration
    run_full_experiment(
        config=config,
        run_dirs=run_dirs,
        exp_name=exp_name,
    )

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
