"""Parameter counting utilities for PyTorch models.
When comparing Transformer variants, we want to avoid unfair comparisons like:

Model A: 20M parameters
Model B: 80M parameters """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class ParamCount:
    """Summary of model parameter counts."""

    total: int
    trainable: int
    frozen: int

    @property
    def total_millions(self) -> float:
        """Total parameters in millions."""
        return self.total / 1_000_000

    @property
    def trainable_millions(self) -> float:
        """Trainable parameters in millions."""
        return self.trainable / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/config."""
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "total_millions": round(self.total_millions, 2),
            "trainable_millions": round(self.trainable_millions, 2),
        }

    def __str__(self) -> str:
        return (
            f"Parameters: {self.total_millions:.2f}M total, "
            f"{self.trainable_millions:.2f}M trainable, "
            f"{self.frozen:,} frozen"
        )


def count_parameters(model: nn.Module) -> ParamCount:
    """Count total, trainable, and frozen parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        ParamCount dataclass with counts.
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return ParamCount(total=total, trainable=trainable, frozen=total - trainable)


def count_parameters_by_module(model: nn.Module) -> dict[str, ParamCount]:
    """Count parameters grouped by top-level submodule.

    Useful for understanding parameter distribution (e.g., how much is in
    embeddings vs attention vs FFN).

    Args:
        model: PyTorch module.

    Returns:
        Dict mapping submodule name to its ParamCount.
    """
    counts: dict[str, ParamCount] = {}
    for name, module in model.named_children():
        total = 0
        trainable = 0
        for param in module.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
        counts[name] = ParamCount(total=total, trainable=trainable, frozen=total - trainable)
    return counts


def format_param_table(model: nn.Module) -> str:
    """Format a human-readable parameter breakdown table.

    Args:
        model: PyTorch module.

    Returns:
        Formatted string with per-module parameter counts.
    """
    by_module = count_parameters_by_module(model)
    total = count_parameters(model)

    lines = ["Parameter Breakdown:", "-" * 50]
    for name, count in by_module.items():
        pct = (count.total / total.total * 100) if total.total > 0 else 0
        lines.append(f"  {name:<30} {count.total_millions:>8.2f}M ({pct:>5.1f}%)")
    lines.append("-" * 50)
    lines.append(f"  {'TOTAL':<30} {total.total_millions:>8.2f}M")
    lines.append(f"  {'Trainable':<30} {total.trainable_millions:>8.2f}M")
    return "\n".join(lines)
