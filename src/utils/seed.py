"""Reproducibility utilities for setting random seeds."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across all RNG sources.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable PyTorch deterministic algorithms.
            This may reduce performance but guarantees bitwise reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Allow cuDNN to auto-tune for performance
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> dict:
    """Capture current RNG state for checkpoint saving.

    Returns:
        Dictionary containing RNG states for python, torch CPU, and torch CUDA.
    """
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG state from a checkpoint.

    Args:
        state: Dictionary from get_rng_state().
    """
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
