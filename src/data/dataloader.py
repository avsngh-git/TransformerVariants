"""Streaming DataLoader for pre-tokenized binary shards.

Reads the binary token files produced by Phase 2's prepare.py and serves
(input, target) pairs for next-token prediction training.

Key design decisions:
- Memory-mapped files: we don't load all tokens into RAM. numpy.memmap lets
  us access tokens on disk as if they were in memory — the OS handles paging.
- Sequential reading within shards: we read tokens in order (better for cache).
- Shuffle at the shard level: we randomize which shard comes next each epoch.
- The target is just the input shifted by 1 position (next-token prediction).
"""

import json
from pathlib import Path

import numpy as np
import torch


class ShardedDataLoader:
    """Loads pre-tokenized binary shards and yields training batches.

    Each batch contains:
    - x: input token IDs, shape (batch_size, seq_len)
    - y: target token IDs, shape (batch_size, seq_len)
    Where y[i] = x[i] shifted right by 1 (next token prediction).

    Args:
        data_dir: Path to directory containing .bin shard files and manifest.json.
        batch_size: Number of sequences per batch (micro-batch size).
        seq_len: Sequence length for each training example.
        split: Which split to load ("train" or "val").
        device: Target device for tensors ("cuda", "cpu").
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        seq_len: int,
        split: str = "train",
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        data_dir = Path(data_dir)

        # Load manifest to find shard files for this split
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Manifest format: {"train": {"shards": ["file1.bin", ...]}, "val": {...}}
            if split in manifest and "shards" in manifest[split]:
                self.shard_paths = [data_dir / filename for filename in manifest[split]["shards"]]
            elif "shards" in manifest:
                # Alternative format: flat list with split field
                self.shard_paths = [
                    data_dir / s["filename"]
                    for s in manifest["shards"]
                    if s.get("split", "train") == split
                ]
            else:
                self.shard_paths = []
        else:
            # Fallback: glob for .bin files matching split pattern
            pattern = f"{split}_*.bin"
            self.shard_paths = sorted(data_dir.glob(pattern))
            if not self.shard_paths:
                # Try without split prefix
                self.shard_paths = sorted(data_dir.glob("*.bin"))

        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found for split '{split}' in {data_dir}")

        # Track position within the data
        self.current_shard_idx = 0
        self.current_position = 0
        self._load_shard(0)

        # Calculate total tokens across all shards (for progress reporting)
        self.total_tokens = sum(
            Path(p).stat().st_size // 2
            for p in self.shard_paths  # uint16 = 2 bytes
        )

    def _load_shard(self, shard_idx: int) -> None:
        """Memory-map a shard file.

        Memory mapping means we don't actually read the file into RAM —
        we just create a view into it. The OS loads pages on demand.
        """
        self.current_shard_idx = shard_idx
        self.current_position = 0
        # Memory-map as uint16 (token IDs fit in 16 bits for vocab_size < 65536)
        self.current_shard = np.memmap(self.shard_paths[shard_idx], dtype=np.uint16, mode="r")

    def _advance_shard(self) -> None:
        """Move to the next shard (wrap around at end of epoch)."""
        next_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
        self._load_shard(next_idx)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of (input, target) pairs.

        Each sequence is seq_len+1 tokens long — the first seq_len are input,
        the last seq_len are target (shifted by 1).

        Returns:
            Tuple of (x, y) where:
            - x: (batch_size, seq_len) input token IDs
            - y: (batch_size, seq_len) target token IDs (x shifted right by 1)
        """
        B = self.batch_size
        T = self.seq_len

        # We need (seq_len + 1) tokens per sequence to get input + target
        tokens_needed = B * (T + 1)

        # Check if we have enough tokens left in this shard
        tokens_remaining = len(self.current_shard) - self.current_position
        if tokens_remaining < tokens_needed:
            self._advance_shard()

        # Slice a contiguous block of tokens
        buf = self.current_shard[self.current_position : self.current_position + tokens_needed]
        buf = torch.tensor(buf.astype(np.int64), device=self.device)

        # Reshape into sequences: (batch_size, seq_len + 1)
        buf = buf.view(B, T + 1)

        # Input is first T tokens, target is last T tokens (shifted by 1)
        x = buf[:, :T]
        y = buf[:, 1 : T + 1]

        # Advance position
        self.current_position += tokens_needed

        return x, y

    def state_dict(self) -> dict[str, int]:
        """Return the exact shard cursor needed for deterministic resume."""
        return {
            "version": 1,
            "current_shard_idx": self.current_shard_idx,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore a cursor produced by :meth:`state_dict`."""
        if state.get("version") != 1:
            raise ValueError("Unsupported ShardedDataLoader state version")
        shard_idx = state.get("current_shard_idx")
        position = state.get("current_position")
        if not isinstance(shard_idx, int) or not 0 <= shard_idx < len(self.shard_paths):
            raise ValueError("Invalid current_shard_idx in loader state")
        if not isinstance(position, int) or position < 0:
            raise ValueError("Invalid current_position in loader state")
        self._load_shard(shard_idx)
        if position > len(self.current_shard):
            raise ValueError("Loader position exceeds the restored shard length")
        self.current_position = position

    def reset(self) -> None:
        """Reset to the beginning (start of a new epoch)."""
        self._load_shard(0)

    @property
    def tokens_per_batch(self) -> int:
        """Number of tokens processed per batch (for progress tracking)."""
        return self.batch_size * self.seq_len
