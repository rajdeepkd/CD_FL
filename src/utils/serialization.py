from typing import Dict
import torch


def get_model_num_params(model: torch.nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_update_size_bytes(
    old_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
) -> int:
    """
    Approximate communication cost in bytes between two state dicts.
    We assume same keys in both.
    """
    total_bytes = 0
    for k in new_state:
        if k not in old_state:
            total_bytes += new_state[k].nelement() * new_state[k].element_size()
        else:
            diff = new_state[k] - old_state[k]
            total_bytes += diff.nelement() * diff.element_size()
    return total_bytes


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)
