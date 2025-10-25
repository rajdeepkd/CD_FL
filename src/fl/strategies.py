from typing import Dict, Any, Tuple, List
import torch
import copy
import numpy as np
import logging

from .client import (
    client_local_train,
    client_evaluate,
)
from ..utils.serialization import (
    get_model_num_params,
    set_state_dict,
)
from ..utils.metrics import compute_inverse_error_weights

from torch.utils.data import TensorDataset, DataLoader


LOGGER = logging.getLogger(__name__)

# We'll keep validation slices small so per-epoch val is cheap.
_VAL_CAP = 5000


def _client_weights_equal(num_clients: int) -> List[float]:
    return [1.0 / num_clients] * num_clients


def _client_weights_size(client_sizes: Dict[str, int]) -> List[float]:
    total = sum(client_sizes.values())
    return [client_sizes[cid] / total for cid in client_sizes]


def _client_weights_inverse_error(
    client_metrics: Dict[str, Dict[str, float]]
) -> List[float]:
    """
    Produce client aggregation weights inversely proportional to each client's error
    (higher weight for clients with worse accuracy). This supports fairness.
    """
    client_order = sorted(client_metrics.keys())
    w_map = compute_inverse_error_weights(client_metrics)
    weights = [w_map[cid] for cid in client_order]
    return weights


def _make_loader_from_arrays(
    X_np: np.ndarray,
    y_np: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Helper to wrap numpy arrays into a TensorDataset + DataLoader.
    We always cast y to float32 because we're doing BCEWithLogitsLoss.
    """
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_federated_training(
    config: Dict[str, Any],
    strategy_name: str,
    model_builder,
    client_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],   # TRAIN: cid -> (X_train, y_train)
    client_tests: Dict[str, Tuple[np.ndarray, np.ndarray]],    # TEST:  cid -> (X_test, y_test)
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    equalize_samples: bool,
    reweight_clients: bool,
    fedprox_mu: float,
    patience: int,
    per_round_logger,
    artifacts_dir: str,
) -> Dict[str, Any]:
    """
    Orchestrates the FL loop across 'rounds' of communication.

    Key changes from previous version:
    - We now downsample each client's training set to at most
      config["experiment"]["max_train_samples_per_client"] rows, if provided.
    - We carve off a small validation slice (<= _VAL_CAP rows) from each client's
      (possibly downsampled) train set.
    - We pass both train_loader and val_loader, plus full client test arrays,
      into client_local_train(), which handles early stopping.
    """

    # Pull optional runtime budget cap for each client
    max_cap = config["experiment"].get("max_train_samples_per_client", None)

    # Determine input_dim from any client train matrix
    some_client = next(iter(client_arrays))
    input_dim = client_arrays[some_client][0].shape[1]
    global_model = model_builder(input_dim=input_dim)

    # init per-client models/optimizers from same global init
    client_models: Dict[str, torch.nn.Module] = {}
    client_opts: Dict[str, torch.optim.Optimizer] = {}
    for cid in client_arrays:
        m = model_builder(input_dim=input_dim)
        m.load_state_dict(global_model.state_dict())
        client_models[cid] = m

        if lr <= 0.0:
            raise ValueError("Learning rate must be > 0.")
        # We keep Adam here; could also respect config["experiment"]["optimizer"]
        client_opts[cid] = torch.optim.Adam(m.parameters(), lr=lr)

    # For PersonalizedHead strategy we track each client's head
    personalized_heads: Dict[str, torch.nn.Module] = {}

    # We'll collect history of global accuracy per client after each round
    history_per_client: Dict[str, List[float]] = {cid: [] for cid in client_arrays}

    # Track comm cost per round
    comm_cost_per_round: List[float] = []

    # Main FL loop
    for r in range(rounds):
        LOGGER.info(f"[run_federated_training] ===== Round {r} / {rounds} =====")

        # broadcast global weights to clients
        if strategy_name == "PersonalizedHead":
            # personalize only classifier head
            for cid, model in client_models.items():
                # save the client's personal head
                head_state = copy.deepcopy(model.classifier.state_dict())
                # load global backbone + global head
                model.load_state_dict(global_model.state_dict())
                # restore personal head
                model.classifier.load_state_dict(head_state)
        else:
            # full overwrite from global
            for cid in client_models:
                client_models[cid].load_state_dict(global_model.state_dict())

        round_client_metrics: Dict[str, Dict[str, float]] = {}
        round_client_sizes: Dict[str, int] = {}
        round_client_updates: Dict[str, int] = {}

        # snapshot for FedProx (global reference)
        global_snapshot = {
            n: p.detach().clone()
            for (n, p) in global_model.named_parameters()
        }
        # make sure snapshot params are on same device later inside client_local_train
        # (client_local_train will move them if needed)

        # -------- Local training loop for each client --------
        for cid in sorted(client_models.keys()):
            model = client_models[cid]
            opt = client_opts[cid]

            # grab this client's train/test arrays
            X_train_full, y_train_full = client_arrays[cid]
            X_test_full,  y_test_full  = client_tests[cid]

            orig_n = len(X_train_full)

            # (1) downsample / cap this client's training size for runtime budget
            if (max_cap is not None) and (orig_n > max_cap):
                idx = np.random.choice(orig_n, size=max_cap, replace=False)
                X_train_capped = X_train_full[idx]
                y_train_capped = y_train_full[idx]
                LOGGER.info(
                    f"[run_federated_training][client {cid}] "
                    f"downsampled train size {orig_n} -> {len(X_train_capped)}"
                )
            else:
                X_train_capped = X_train_full
                y_train_capped = y_train_full
                LOGGER.info(
                    f"[run_federated_training][client {cid}] "
                    f"using full train size {orig_n}"
                )

            # (2) carve small validation slice from capped train set
            if len(X_train_capped) > _VAL_CAP:
                X_val = X_train_capped[:_VAL_CAP]
                y_val = y_train_capped[:_VAL_CAP]
                X_tr_eff = X_train_capped[_VAL_CAP:]
                y_tr_eff = y_train_capped[_VAL_CAP:]
            else:
                # if small dataset, val = same data
                X_val = X_train_capped
                y_val = y_train_capped
                X_tr_eff = X_train_capped
                y_tr_eff = y_train_capped

            # (3) build loaders
            dl_train = _make_loader_from_arrays(
                X_tr_eff,
                y_tr_eff,
                batch_size=batch_size,
                shuffle=True,
            )
            dl_val = _make_loader_from_arrays(
                X_val,
                y_val,
                batch_size=batch_size,
                shuffle=False,
            )

            # Keep a copy of params pre-training for comm-cost calc
            before_state = {
                n: p.detach().clone()
                for (n, p) in model.named_parameters()
            }

            # fedprox mu for this client if applicable
            mu_val = fedprox_mu if strategy_name == "FedProx" else 0.0

            # (4) local train with early stopping on dl_val, final metrics from full test
            c_metrics, train_info = client_local_train(
                model=model,
                optimizer=opt,
                train_loader=dl_train,
                val_loader=dl_val,
                X_test_full=X_test_full,
                y_test_full=y_test_full,
                device=device,
                local_epochs=local_epochs,
                fedprox_mu=mu_val,
                global_params_snapshot=global_snapshot,
                patience=patience,
            )

            # Save per-client personalized head after training (for PersonalizedHead baseline)
            if strategy_name == "PersonalizedHead" and cid not in personalized_heads:
                personalized_heads[cid] = copy.deepcopy(model.classifier)

            # measure update size (communication cost)
            after_state = {
                n: p.detach().clone()
                for (n, p) in model.named_parameters()
            }
            update_bytes = 0
            for n in after_state:
                diff = after_state[n] - before_state[n]
                update_bytes += diff.nelement() * diff.element_size()

            # record round stats for this client
            round_client_metrics[cid] = c_metrics
            round_client_sizes[cid] = len(X_tr_eff)  # effective train size actually used
            round_client_updates[cid] = update_bytes

            LOGGER.info(
                f"[run_federated_training][client {cid}] "
                f"final_metrics={c_metrics} "
                f"train_used={len(X_tr_eff)} val_used={len(X_val)} "
                f"update_bytes={update_bytes} "
                f"best_val_loss={train_info.get('best_val_loss')}"
            )

        # -------- Server aggregation step --------
        if reweight_clients:
            agg_weights_list = _client_weights_inverse_error(round_client_metrics)
        else:
            if equalize_samples:
                agg_weights_list = _client_weights_equal(len(client_models))
            else:
                agg_weights_list = _client_weights_size(round_client_sizes)

        ordered_cids = sorted(client_models.keys())
        client_states = [client_models[cid].state_dict() for cid in ordered_cids]

        if strategy_name == "PersonalizedHead":
            # Only average backbone; keep heads local/personalized.
            backbones = []
            for cid in ordered_cids:
                st = {}
                for k, v in client_models[cid].state_dict().items():
                    if not k.startswith("classifier."):
                        st[k] = v.clone()
                backbones.append(st)

            avg_backbone = {}
            for k in backbones[0].keys():
                avg_backbone[k] = sum(
                    w * st[k] for w, st in zip(agg_weights_list, backbones)
                )

            new_global_state = copy.deepcopy(global_model.state_dict())
            for k, v in avg_backbone.items():
                new_global_state[k] = v.clone()
            set_state_dict(global_model, new_global_state)

        else:
            # Standard weighted FedAvg / FedProx style aggregation of all params
            weighted_state = {}
            first_state = client_states[0]
            for k in first_state.keys():
                weighted_state[k] = sum(
                    w * cs[k] for w, cs in zip(agg_weights_list, client_states)
                )
            set_state_dict(global_model, weighted_state)

        # communication accounting
        total_comm = float(sum(round_client_updates.values()))
        comm_cost_per_round.append(total_comm)

        # -------- Evaluate global model on each client's TRAIN loader for learning curves --------
        # We'll build quick loaders from *full* client train sets for curve plotting,
        # but without early stopping overhead.
        eval_metrics_train = {}
        for cid in ordered_cids:
            X_full, y_full = client_arrays[cid]
            dl_full_train = _make_loader_from_arrays(
                X_full,
                y_full,
                batch_size=batch_size,
                shuffle=False,
            )
            eval_metrics_train[cid] = client_evaluate(
                model=global_model,
                data_loader=dl_full_train,
                device=device,
            )
            history_per_client[cid].append(eval_metrics_train[cid]["accuracy"])

        # -------- Log this round to CSV/logger --------
        num_params = get_model_num_params(global_model)
        per_round_logger(
            round_idx=r,
            strategy=strategy_name,
            client_metrics=round_client_metrics,      # metrics from final local eval on full test
            client_eval_metrics=eval_metrics_train,   # eval of global model on full TRAIN
            client_sizes=round_client_sizes,          # effective sizes used this round
            client_update_sizes=round_client_updates, # bytes sent
            global_num_params=num_params,
            total_comm_bytes=total_comm,
        )

        # -------- Save a checkpoint of global model after aggregation --------
        ckpt_path = f"{artifacts_dir}/ckpt_{strategy_name}_round{r}.pt"
        torch.save(global_model.state_dict(), ckpt_path)

    # -------- After all rounds: final evaluation of global model on each client's TEST set --------
    final_eval = {}
    for cid in sorted(client_models.keys()):
        X_te, y_te = client_tests[cid]
        dl_te = _make_loader_from_arrays(
            X_te,
            y_te,
            batch_size=batch_size,
            shuffle=False,
        )
        final_eval[cid] = client_evaluate(
            model=global_model,
            data_loader=dl_te,
            device=device,
        )

    return {
        "final_eval": final_eval,
        "history_per_client": history_per_client,
        "comm_cost_per_round": comm_cost_per_round,
        "final_global_state_dict": global_model.state_dict(),
    }
