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

    def state_dict(self) -> dict[str, int]:
        """Return the current batch cursor for resume tests."""
        return {"version": 1, "index": self._idx}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore a cursor produced by :meth:`state_dict`."""
        if state.get("version") != 1:
            raise ValueError("Unsupported SyntheticLoader state version")
        index = state.get("index")
        if not isinstance(index, int) or index < 0:
            raise ValueError("Invalid SyntheticLoader index")
        self._idx = index
