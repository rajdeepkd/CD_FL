import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LOGGER = logging.getLogger(__name__)


def _bce_logits_loss(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
    global_params_snapshot: Optional[Dict[str, torch.Tensor]] = None,
    fedprox_mu: float = 0.0,
) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss plus optional FedProx proximal term.
    """
    logits = model(xb).squeeze(-1)  # [B]
    base_loss = criterion(logits, yb)

    if global_params_snapshot is not None and fedprox_mu > 0.0:
        prox_term = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad:
                prox_term = prox_term + torch.norm(
                    param - global_params_snapshot[name]
                ) ** 2
        base_loss = base_loss + (fedprox_mu / 2.0) * prox_term

    return base_loss


def _evaluate_loader_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """
    Compute average loss on a loader without gradient.
    This is used for validation/early stopping.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    model.train()
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def _final_metrics_full_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run full test-set evaluation ONCE at the end of local training.
    Returns accuracy, precision, recall, f1.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_tensor).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def client_local_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_test_full: np.ndarray,
    y_test_full: np.ndarray,
    device: torch.device,
    local_epochs: int,
    fedprox_mu: float,
    global_params_snapshot: Optional[Dict[str, torch.Tensor]],
    patience: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Train locally for up to `local_epochs` with early stopping based on val loss.

    Arguments:
        model: client model (already copied or cloned for this client)
        optimizer: optimizer instance for this client's model
        train_loader: DataLoader for the (possibly capped) training subset
        val_loader:   DataLoader for a *small* validation subset from that same client
        X_test_full, y_test_full: numpy arrays for full client test set
        device: torch.device
        local_epochs: int
        fedprox_mu: float (0 if not FedProx)
        global_params_snapshot: dict of tensors (global weights) if FedProx is used
        patience: early stopping patience in epochs

    Returns:
        final_metrics: dict with accuracy/precision/recall/f1 on FULL test set
        train_info: dict with loss curves etc.
    """

    model = model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()

    # For FedProx, we need global params on the same device
    prox_params = None
    if global_params_snapshot is not None:
        prox_params = {
            k: v.to(device)
            for k, v in global_params_snapshot.items()
        }

    best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    bad_epochs = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for epoch in range(local_epochs):
        running_loss = 0.0
        running_count = 0

        # --- Training phase ---
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = _bce_logits_loss(
                model=model,
                xb=xb,
                yb=yb,
                criterion=criterion,
                global_params_snapshot=prox_params,
                fedprox_mu=fedprox_mu,
            )
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_count += bs

        avg_train_loss = (running_loss / running_count) if running_count > 0 else 0.0

        # --- Validation phase (cheap) ---
        avg_val_loss = _evaluate_loader_loss(model, val_loader, device, criterion)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        LOGGER.info(
            f"[client_local_train] Epoch {epoch+1}/{local_epochs} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
        )

        # --- Early stopping logic ---
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info(
                    f"[client_local_train] Early stopping at epoch {epoch+1} "
                    f"(no val improvement for {bad_epochs} epochs)."
                )
                break

    # restore best weights before evaluation
    model.load_state_dict(best_state)

    # Final evaluation on full client test set
    final_metrics = _final_metrics_full_test(
        model=model,
        X_test=X_test_full,
        y_test=y_test_full,
        device=device,
    )

    train_info = {
        "train_loss_curve": train_loss_history,
        "val_loss_curve": val_loss_history,
        "best_val_loss": float(best_val_loss),
    }

    return final_metrics, train_info
