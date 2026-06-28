"""Shared utilities."""

from src.utils.config import load_config, merge_configs, resolve_config
from src.utils.device import detect_device, get_precision_dtype, DeviceInfo
from src.utils.logging import setup_logging, load_metrics
from src.utils.params import count_parameters, count_parameters_by_module, format_param_table, ParamCount
from src.utils.seed import set_seed, get_rng_state, set_rng_state

__all__ = [
    # Config
    "load_config",
    "merge_configs",
    "resolve_config",
    # Device
    "detect_device",
    "get_precision_dtype",
    "DeviceInfo",
    # Logging
    "setup_logging",
    "load_metrics",
    # Parameters
    "count_parameters",
    "count_parameters_by_module",
    "format_param_table",
    "ParamCount",
    # Seed
    "set_seed",
    "get_rng_state",
    "set_rng_state",
]
