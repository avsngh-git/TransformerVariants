"""Structural protocols for training components.

Defines the DataLoader protocol — a duck-typed interface for any object
that can supply training batches via `next_batch()`.
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for any object that can supply training batches."""

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (x, y) batch of token tensors.

        Returns:
            x: Input token IDs, shape (batch_size, seq_len), dtype int64
            y: Target token IDs, shape (batch_size, seq_len), dtype int64
        """
        ...
