"""Opaque attention-cache helpers shared by decoder model wrappers."""

from __future__ import annotations

import torch


def cache_sequence_length(cache: object) -> int:
    """Return the number of populated positions in a layer cache."""
    if cache is None:
        return 0
    if not isinstance(cache, (list, tuple)) or not cache:
        raise TypeError(f"Unsupported cache type: {type(cache).__name__}")
    if len(cache) == 3:
        length = cache[2]
        if isinstance(length, torch.Tensor):
            return int(length.max().item())
        return int(length)
    first = cache[0]
    if not isinstance(first, torch.Tensor) or first.ndim < 3:
        raise TypeError("A two-tensor KV cache must store sequence length on dimension 2")
    return int(first.size(2))
