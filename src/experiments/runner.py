import os
import csv
import json
import random
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.loader import prepare_all_clients
from ..utils.logging_utils import get_logger, hash_file, append_client_metrics_to_txt
from ..utils.metrics import summarize_all_clients, fairness_stats
from ..models.model_zoo import build_model
from ..fl.strategies import run_federated_training


def _fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_full_experiment(
    config: Dict[str, Any],
    run_dirs: Dict[str, str],
    exp_name: str,
) -> None:
    logger = get_logger(__name__)

    _fix_seeds(config["experiment"]["seed"])

    logger.info("Preprocessing all client data...")

    client_data = prepare_all_clients(
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
        label_column=config["data"]["label_column"],
        drop_columns=config["data"]["drop_columns"],
        numeric_features=config["data"]["numeric_features"],
        categorical_features=config["data"]["categorical_features"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        scaler_type=config["preprocess"]["scaler"],
        use_smote=config["preprocess"]["use_smote"],
        smote_minority_ratio_threshold=config["preprocess"]["smote_minority_ratio_threshold"],
        summaries_dir=run_dirs["summaries_dir"],
    )

    # client_arrays will hold TRAIN sets for FL training
    # we still keep test sets separately for final eval
    client_arrays = {
        cid: (vals["X_train"], vals["y_train"]) for cid, vals in client_data.items()
    }
    client_tests = {
        cid: (vals["X_test"], vals["y_test"]) for cid, vals in client_data.items()
    }

    # write checksums for reproducibility (train/test CSVs)
    client_checksums: Dict[str, str] = {}
    for cid in client_data.keys():
        train_path = os.path.join(config["data"]["processed_dir"], f"{cid}_train.csv")
        test_path = os.path.join(config["data"]["processed_dir"], f"{cid}_test.csv")
        # hash concatenation of train+test so it's stable
        chk = hash_file(train_path) + "_" + hash_file(test_path)
        client_checksums[cid] = chk

    # per-round CSV log
    per_round_csv_path = os.path.join(
        run_dirs["logs_dir"], f"{exp_name}_per_round.csv"
    )
    csv_fieldnames = [
        "round",
        "strategy",
        "seed",
        "client_id",
        "client_size",
        "client_update_size_bytes",
        "client_train_loss",
        "client_train_accuracy",
        "client_eval_accuracy_trainset",
        "client_eval_loss_trainset",
        "global_num_params",
        "total_comm_bytes",
    ]
    with open(per_round_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
        writer.writeheader()

    def per_round_logger(
        round_idx: int,
        strategy: str,
        client_metrics: Dict[str, Dict[str, float]],
        client_eval_metrics: Dict[str, Dict[str, float]],
        client_sizes: Dict[str, int],
        client_update_sizes: Dict[str, int],
        global_num_params: int,
        total_comm_bytes: float,
    ) -> None:
        """
        Note: client_eval_metrics here are evals on the TRAIN loaders
        (i.e. local training split). Final test eval will happen after training.
        """
        with open(per_round_csv_path, "a", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
            for cid in sorted(client_metrics.keys()):
                row = {
                    "round": round_idx,
                    "strategy": strategy,
                    "seed": config["experiment"]["seed"],
                    "client_id": cid,
                    "client_size": client_sizes[cid],
                    "client_update_size_bytes": client_update_sizes[cid],
                    "client_train_loss": client_metrics[cid]["loss"],
                    "client_train_accuracy": client_metrics[cid]["accuracy"],
                    "client_eval_accuracy_trainset": client_eval_metrics[cid]["accuracy"],
                    "client_eval_loss_trainset": client_eval_metrics[cid]["loss"],
                    "global_num_params": global_num_params,
                    "total_comm_bytes": total_comm_bytes,
                }
                writer.writerow(row)

    # run chosen FL strategy
    strategy_name = config["experiment"]["strategy"]
    results = run_federated_training(
        strategy_name=strategy_name,
        model_builder=build_model,
        client_arrays=client_arrays,
        client_tests=client_tests,  # NEW: pass test sets down
        rounds=config["experiment"]["rounds"],
        local_epochs=config["experiment"]["local_epochs"],
        lr=config["experiment"]["lr"],
        batch_size=config["experiment"]["batch_size"],
        device=config["experiment"]["device"],
        equalize_samples=config["experiment"]["equalize_samples"],
        reweight_clients=config["experiment"]["reweight_clients"],
        fedprox_mu=config["experiment"]["fedprox_mu"],
        patience=config["experiment"]["early_stopping_patience"],
        per_round_logger=per_round_logger,
        artifacts_dir=run_dirs["artifacts_dir"],
    )

    final_eval = results["final_eval"]  # test-set metrics per client
    history_per_client = results["history_per_client"]
    comm_cost_per_round = results["comm_cost_per_round"]

    # fairness summary based on test-set accuracy
    fairness_summary = summarize_all_clients(final_eval)
    acc_fairness = fairness_stats(final_eval, "accuracy")

    # append final accuracies to each client's *_class_distribution.txt
    for cid, mets in final_eval.items():
        append_client_metrics_to_txt(
            summaries_dir=run_dirs["summaries_dir"],
            client_id=cid,
            strategy=strategy_name,
            metrics=mets,
        )

    # save final global model
    final_model_path = os.path.join(
        run_dirs["artifacts_dir"], f"{strategy_name}_final_model.pt"
    )
    torch.save(results["final_global_state_dict"], final_model_path)

    # write final summary json/txt
    summary_obj = {
        "experiment_name": exp_name,
        "strategy": strategy_name,
        "seed": config["experiment"]["seed"],
        "final_eval": final_eval,
        "fairness_summary": fairness_summary,
        "accuracy_fairness": acc_fairness,
        "client_checksums": client_checksums,
        "comm_cost_per_round": comm_cost_per_round,
        "config_snapshot": config,
    }

    final_json_path = os.path.join(
        run_dirs["summaries_dir"],
        f"{exp_name}_final_summary.json",
    )
    with open(final_json_path, "w", encoding="utf-8") as f_json:
        json.dump(summary_obj, f_json, indent=2)

    final_txt_path = os.path.join(
        run_dirs["summaries_dir"],
        f"{exp_name}_final_summary.txt",
    )
    with open(final_txt_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("FINAL EVALUATION SUMMARY\n")
        f_txt.write("========================\n\n")
        f_txt.write(f"Experiment: {exp_name}\n")
        f_txt.write(f"Strategy:   {strategy_name}\n")
        f_txt.write(f"Seed:       {config['experiment']['seed']}\n\n")

        f_txt.write("Per-client final TEST metrics:\n")
        for cid, mets in final_eval.items():
            f_txt.write(f"  {cid}:\n")
            for k, v in mets.items():
                f_txt.write(f"    {k}: {v}\n")

        f_txt.write("\nFairness summary:\n")
        f_txt.write(json.dumps(fairness_summary, indent=2))
        f_txt.write("\n\nAccuracy disparity:\n")
        f_txt.write(json.dumps(acc_fairness, indent=2))
        f_txt.write("\n")

    if config["evaluation"]["save_plots"]:
        _plot_learning_curves(
            history_per_client,
            out_path=os.path.join(
                run_dirs["summaries_dir"],
                f"{exp_name}_learning_curves.png",
            ),
            title=f"Per-Client Train Accuracy vs Round ({strategy_name})",
        )

        _plot_comm_cost(
            comm_cost_per_round,
            out_path=os.path.join(
                run_dirs["summaries_dir"],
                f"{exp_name}_communication.png",
            ),
            title=f"Total Communication per Round ({strategy_name})",
        )

        _plot_final_accuracy_cdf(
            final_eval,
            out_path=os.path.join(
                run_dirs["summaries_dir"],
                f"{exp_name}_client_accuracy_cdf.png",
            ),
            title=f"Final Client TEST Accuracy CDF ({strategy_name})",
        )

        _plot_strategy_heatmap_stub(
            final_eval,
            out_path=os.path.join(
                run_dirs["summaries_dir"],
                f"{exp_name}_heatmap_strategy_delta.png",
            ),
            title="Strategy Delta Heatmap (single strategy baseline)",
        )


def _plot_learning_curves(history_per_client, out_path, title) -> None:
    plt.figure()
    for cid, hist in sorted(history_per_client.items()):
        plt.plot(range(len(hist)), hist, label=f"{cid}")
    plt.xlabel("Round")
    plt.ylabel("Train Accuracy (Global Model)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_comm_cost(comm_cost_per_round, out_path, title) -> None:
    plt.figure()
    plt.plot(range(len(comm_cost_per_round)), comm_cost_per_round)
    plt.xlabel("Round")
    plt.ylabel("Total Bytes Sent by Clients")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_final_accuracy_cdf(final_eval, out_path, title) -> None:
    accs = [mets["accuracy"] for mets in final_eval.values()]
    accs_sorted = np.sort(accs)
    cdf_y = np.arange(1, len(accs_sorted) + 1) / len(accs_sorted)

    plt.figure()
    plt.plot(accs_sorted, cdf_y, marker="o")
    plt.xlabel("Final TEST Accuracy")
    plt.ylabel("CDF")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_strategy_heatmap_stub(final_eval, out_path, title) -> None:
    cids = sorted(final_eval.keys())
    accs = [final_eval[c]["accuracy"] for c in cids]
    mat = np.array(accs).reshape(-1, 1)

    plt.figure()
    sns.heatmap(mat, annot=True, fmt=".3f", yticklabels=cids, xticklabels=["acc"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
