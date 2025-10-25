import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML config into a nested dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def deep_set(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested dict item using dotted path.
    Example: deep_set(cfg, "experiment.lr", 0.01)
    """
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def override_config(
    base_cfg: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Override nested config keys with CLI-provided values.
    """
    cfg = {**base_cfg}
    for k, v in overrides.items():
        deep_set(cfg, k, v)
    return cfg
