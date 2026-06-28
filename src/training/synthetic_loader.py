"""A test-only DataLoader that cycles through pre-made (x, y) tensor pairs.

Satisfies the DataLoader protocol defined in src/training/protocols.py.
Used in tests to avoid disk I/O and binary shard file dependencies.
"""

import torch


class SyntheticLoader:
    """A test-only DataLoader that cycles through pre-made (x, y) pairs."""

    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not batches:
            raise ValueError("batches must be a non-empty list")
        self.batches = batches
        self._idx = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next (x, y) batch, cycling back to start after exhaustion."""
        batch = self.batches[self._idx % len(self.batches)]
        self._idx += 1
        return batch
