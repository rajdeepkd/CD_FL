from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.
    Assumes y_pred is in {0,1}.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc_roc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auc_roc": float(auc_roc),
    }


def fairness_stats(
    client_metrics: Dict[str, Dict[str, float]],
    metric_key: str = "accuracy",
) -> Dict[str, float]:
    """
    Compute fairness stats across clients for a chosen metric.
    """
    vals: List[float] = []
    for _cid, m in client_metrics.items():
        if metric_key in m:
            vals.append(m[metric_key])
    if not vals:
        return {"disparity": 0.0, "std": 0.0}
    arr = np.array(vals, dtype=float)
    return {
        "disparity": float(arr.max() - arr.min()),
        "std": float(arr.std(ddof=0)),
    }


def summarize_all_clients(
    all_client_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Compute fairness stats for multiple keys.
    """
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        summary[f"{key}_fairness"] = fairness_stats(all_client_metrics, key)
    return summary


def compute_inverse_error_weights(
    client_metrics: Dict[str, Dict[str, float]],
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Produce weights inversely proportional to (1 - accuracy).
    Intuition: prioritize struggling clients (low acc).
    """
    weights = {}
    for cid, mets in client_metrics.items():
        acc = mets.get("accuracy", 0.0)
        weights[cid] = 1.0 / (max(0.0, (1.0 - acc)) + epsilon)
    s = sum(weights.values()) + epsilon
    for cid in weights:
        weights[cid] = weights[cid] / s
    return weights
