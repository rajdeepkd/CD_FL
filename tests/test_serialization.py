import torch
from src.utils.serialization import (
    clone_state_dict,
    estimate_update_size_bytes,
    get_model_num_params,
)
from src.models.model_zoo import build_model


def test_serialization_utils():
    model_a = build_model(input_dim=4, hidden_dim=8)
    model_b = build_model(input_dim=4, hidden_dim=8)

    st_a = clone_state_dict(model_a)
    st_b = clone_state_dict(model_b)

    n_params = get_model_num_params(model_a)
    assert n_params > 0

    diff_size = estimate_update_size_bytes(st_a, st_b)
    assert diff_size == 0

    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(0.5)

    st_b2 = clone_state_dict(model_b)
    diff_size2 = estimate_update_size_bytes(st_a, st_b2)
    assert diff_size2 > 0
