from typing import Dict, List
import copy
import torch
from ..utils.serialization import clone_state_dict, set_state_dict


def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of a list of model state_dicts.
    """
    if len(state_dicts) == 1:
        return copy.deepcopy(state_dicts[0])

    avg_state = {}
    for k in state_dicts[0].keys():
        avg_state[k] = sum(w * sd[k] for w, sd in zip(weights, state_dicts))
    return avg_state


def broadcast_global_to_clients(
    global_model: torch.nn.Module,
    client_models: Dict[str, torch.nn.Module],
) -> None:
    """
    Copy global weights into each client's model.
    """
    g_state = clone_state_dict(global_model)
    for cid, cm in client_models.items():
        set_state_dict(cm, g_state)
