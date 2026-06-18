"""Config loading and merging utilities.

Supports hierarchical YAML configs with override semantics:
  base config (project_defaults.yaml)
    <- model config
    <- data config
    <- train config
    <- experiment config (highest priority)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.suffix in (".yaml", ".yml"):
        raise ValueError(f"Expected a YAML file, got: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(data).__name__}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base. Override values take precedence.

    - Dicts are merged recursively.
    - Lists and scalars from override replace base entirely.
    - Keys in override that don't exist in base are added.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load and merge multiple YAML config files in order.

    Later files override earlier files. Typical usage:

        config = load_config(
            "configs/project_defaults.yaml",
            "configs/model/vanilla.yaml",
            "configs/train/debug.yaml",
        )

    Args:
        *paths: One or more paths to YAML config files.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If any config file does not exist.
        ValueError: If a file is not valid YAML or not a dict.
    """
    if not paths:
        raise ValueError("At least one config path is required")

    merged: dict[str, Any] = {}
    for path in paths:
        layer = load_yaml(path)
        merged = merge_configs(merged, layer)
    return merged


def resolve_config(
    config: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a config dict (validate, compute derived fields) and optionally save.

    Currently this is a passthrough that optionally writes the resolved config
    to disk for reproducibility. Future phases may add validation and derived
    field computation here.

    Args:
        config: Merged configuration dictionary.
        output_path: If provided, write the resolved config as YAML to this path.

    Returns:
        The resolved config dict.
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config
