"""Atomic checkpoint writing with SHA-256 integrity verification and ring buffer.

This module provides fault-tolerant checkpoint persistence:
- AtomicCheckpointWriter: writes checkpoints using temp file + fsync + rename
  to guarantee file integrity even during crashes.
- CheckpointRingBuffer: maintains a fixed-capacity window of verified checkpoints,
  evicting the oldest only after a new one passes integrity verification.

The atomic write protocol:
    1. Write to a temporary file in the same directory as the target
    2. fsync the file descriptor to flush OS buffers to disk
    3. Atomically rename temp → final (POSIX guarantee)
    4. Compute SHA-256 hash and write to a .sha256 sidecar file

On verification, the stored hash is compared against a freshly computed hash.
If they don't match, the checkpoint is reported as corrupted.
"""

import hashlib
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


class AtomicCheckpointWriter:
    """Writes checkpoint files atomically with integrity verification.

    Guarantees that a checkpoint file at the final path is always complete
    and valid, regardless of crashes during the write process.
    """

    def save(self, state_dict: dict, path: Path) -> str:
        """Write checkpoint atomically. Returns SHA-256 hash.

        Steps:
            1. Write state_dict to a temporary file (.pt.tmp)
            2. fsync to ensure data is on disk
            3. Atomically rename temp file to final path
            4. Compute SHA-256 and write .sha256 sidecar

        Args:
            state_dict: The state dictionary to save (model, optimizer, etc.)
            path: The final checkpoint file path.

        Returns:
            The SHA-256 hex digest of the written checkpoint file.

        Raises:
            Any exception from torch.save or filesystem operations.
            Temp file is cleaned up on failure.
        """
        path = Path(path)
        temp_path = path.with_suffix(".pt.tmp")
        try:
            # Step 1: Write to temporary file
            with open(temp_path, "wb") as f:
                torch.save(state_dict, f)
                f.flush()
                # Step 2: Force OS buffers to disk
                os.fsync(f.fileno())
            # Step 3: Atomic rename (POSIX guarantee)
            os.rename(temp_path, path)
            # Step 4: Compute and write integrity hash
            sha256 = self.compute_hash(path)
            hash_path = Path(str(path) + ".sha256")
            hash_path.write_text(sha256)
            return sha256
        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    @staticmethod
    def verify(path: Path) -> bool:
        """Verify checkpoint integrity against stored hash.

        Computes the SHA-256 hash of the checkpoint file and compares it
        against the value stored in the .sha256 sidecar file.

        If the .sha256 sidecar file is missing, the hash is recomputed
        and written (assumes the checkpoint is valid if no hash exists).

        Args:
            path: Path to the checkpoint file.

        Returns:
            True if the checkpoint matches its stored hash, False if corrupted.
        """
        path = Path(path)
        if not path.exists():
            return False

        hash_path = Path(str(path) + ".sha256")

        if not hash_path.exists():
            # Missing hash file: recompute and write it
            computed = AtomicCheckpointWriter.compute_hash(path)
            hash_path.write_text(computed)
            return True

        stored_hash = hash_path.read_text().strip()
        computed_hash = AtomicCheckpointWriter.compute_hash(path)
        return computed_hash == stored_hash

    @staticmethod
    def verify_trusted(path: Path) -> bool:
        """Verify only when an expected digest already exists.

        Unlike :meth:`verify`, this method never enrolls a legacy file by
        generating a digest from the bytes being checked.
        """
        path = Path(path)
        hash_path = Path(str(path) + ".sha256")
        if not path.exists() or not hash_path.exists():
            return False
        try:
            stored_hash = hash_path.read_text().strip()
            return AtomicCheckpointWriter.compute_hash(path) == stored_hash
        except OSError:
            return False

    @staticmethod
    def compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file.

        Reads the file in 64KB chunks for memory efficiency with large
        checkpoint files.

        Args:
            path: Path to the file to hash.

        Returns:
            Hex digest string of the SHA-256 hash.
        """
        path = Path(path)
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)  # 64KB chunks
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()


@dataclass
class RingEntry:
    """A single checkpoint entry in the ring buffer.

    Attributes:
        step: Training step when this checkpoint was saved.
        path: Relative path from checkpoint_dir to the checkpoint file.
        sha256: Hex digest of the checkpoint file's SHA-256 hash.
        timestamp: Unix timestamp (time.time()) when the entry was registered.
    """

    step: int
    path: str
    sha256: str
    timestamp: float


class CheckpointRingBuffer:
    """Manages a fixed-size window of verified checkpoints.

    Ensures that old checkpoints are only deleted after new ones are confirmed
    valid via SHA-256 integrity verification. Persists state to
    checkpoint_ring.json for crash recovery.

    Entries are ordered by step ascending.
    """

    METADATA_FILENAME = "checkpoint_ring.json"

    def __init__(self, checkpoint_dir: Path, capacity: int = 3):
        """Initialize the ring buffer.

        Args:
            checkpoint_dir: Directory where checkpoints and metadata are stored.
            capacity: Maximum number of checkpoints to retain (default 3).
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._capacity = capacity
        self._entries: list[RingEntry] = []
        self._load_metadata()

    @property
    def capacity(self) -> int:
        """Return the ring buffer's maximum capacity."""
        return self._capacity

    def register(self, step: int, path: Path, sha256: str) -> None:
        """Add a verified checkpoint to the ring. Evicts oldest if at capacity.

        The new checkpoint is verified via AtomicCheckpointWriter.verify() before
        being added. If the ring is at capacity, the oldest entry is deleted from
        disk only AFTER the new checkpoint passes verification.

        Args:
            step: Training step number for this checkpoint.
            path: Path to the checkpoint file (absolute or relative).
            sha256: Expected SHA-256 hex digest of the checkpoint.

        Raises:
            ValueError: If the new checkpoint fails integrity verification.
        """
        # Resolve the path relative to checkpoint_dir
        abs_path = Path(path)
        if not abs_path.is_absolute():
            abs_path = self._checkpoint_dir / abs_path

        # Verify the new checkpoint before adding
        if not AtomicCheckpointWriter.verify(abs_path):
            raise ValueError(
                f"Checkpoint verification failed for {abs_path}. Not adding to ring buffer."
            )

        # Evict oldest if at capacity (delete from disk after verification)
        if len(self._entries) >= self._capacity:
            oldest = self._entries[0]
            oldest_path = self._checkpoint_dir / oldest.path
            if oldest_path.exists():
                oldest_path.unlink()
            # Also remove the .sha256 sidecar
            oldest_hash_path = Path(str(oldest_path) + ".sha256")
            if oldest_hash_path.exists():
                oldest_hash_path.unlink()
            self._entries.pop(0)

        # Compute relative path for storage
        try:
            rel_path = abs_path.relative_to(self._checkpoint_dir)
        except ValueError:
            # If path is not under checkpoint_dir, store as-is
            rel_path = abs_path

        # Add the new entry
        entry = RingEntry(
            step=step,
            path=str(rel_path),
            sha256=sha256,
            timestamp=time.time(),
        )
        self._entries.append(entry)

        # Keep entries sorted by step ascending
        self._entries.sort(key=lambda e: e.step)

        # Persist metadata
        self._persist_metadata()

    def latest(self) -> Path | None:
        """Return path to the most recently registered checkpoint.

        Returns:
            Absolute path to the most recent checkpoint, or None if the ring
            is empty.
        """
        if not self._entries:
            return None
        # Most recent = highest step = last entry (sorted ascending)
        latest_entry = self._entries[-1]
        return self._checkpoint_dir / latest_entry.path

    def latest_verified(self) -> Path | None:
        """Return the newest checkpoint whose contents match the ring metadata hash."""
        for entry in reversed(self._entries):
            path = self._checkpoint_dir / entry.path
            if path.exists() and AtomicCheckpointWriter.compute_hash(path) == entry.sha256:
                return path
        return None

    def rollback_to(self, n_back: int = 1) -> Path | None:
        """Return path to the nth most recent checkpoint.

        Args:
            n_back: How many steps back from the latest (1 = previous checkpoint).

        Returns:
            Absolute path to the requested checkpoint, or None if not available.
        """
        if n_back < 0:
            return None
        # Index from end: latest is -1, one back is -2, etc.
        idx = len(self._entries) - 1 - n_back
        if idx < 0:
            return None
        entry = self._entries[idx]
        return self._checkpoint_dir / entry.path

    def list_available(self) -> list[RingEntry]:
        """List all checkpoints currently in the ring.

        Returns:
            List of RingEntry objects ordered by step ascending.
        """
        return list(self._entries)

    def _persist_metadata(self) -> None:
        """Write ring state to checkpoint_ring.json atomically."""
        metadata = {
            "capacity": self._capacity,
            "entries": [asdict(entry) for entry in self._entries],
        }
        metadata_path = self._checkpoint_dir / self.METADATA_FILENAME
        # Write atomically via temp file
        temp_path = metadata_path.with_suffix(".json.tmp")
        try:
            temp_path.write_text(json.dumps(metadata, indent=2))
            os.rename(temp_path, metadata_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_metadata(self) -> None:
        """Load ring state from checkpoint_ring.json if it exists."""
        metadata_path = self._checkpoint_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            return

        try:
            data = json.loads(metadata_path.read_text())
            self._capacity = data.get("capacity", self._capacity)
            self._entries = [
                RingEntry(
                    step=entry["step"],
                    path=entry["path"],
                    sha256=entry["sha256"],
                    timestamp=entry["timestamp"],
                )
                for entry in data.get("entries", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # If metadata is corrupted, start fresh
            self._entries = []


def _optimizer_state_to_cpu(opt_state_dict: dict) -> dict:
    """Deep copy optimizer state_dict to CPU tensors.

    Handles the nested structure of optimizer state_dicts, moving all tensors
    to CPU while preserving non-tensor values (like step counts, learning rates).

    Args:
        opt_state_dict: The optimizer's state_dict() output.

    Returns:
        A new dict with all tensors on CPU, non-tensor values copied as-is.
    """
    cpu_state: dict = {}
    for key, value in opt_state_dict.items():
        if key == "state":
            cpu_state["state"] = {}
            for param_id, param_state in value.items():
                cpu_state["state"][param_id] = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in param_state.items()
                }
        else:
            # 'param_groups' and other top-level keys are copied as-is
            cpu_state[key] = value
    return cpu_state


class AsyncCheckpointWriter:
    """Asynchronous checkpoint writer that minimizes training interruption.

    Snapshots model and optimizer state to CPU memory (fast, ~100ms), then
    delegates the slow disk I/O to a single background thread. Enforces at
    most one in-flight save at any time.

    The writer integrates with a CheckpointRingBuffer to track saved
    checkpoints and provide rollback access.
    """

    def __init__(self, ring_buffer: CheckpointRingBuffer, checkpoint_dir: Path):
        """Initialize the async checkpoint writer.

        Args:
            ring_buffer: The ring buffer to register completed saves with.
            checkpoint_dir: Directory where checkpoint files are written.
        """
        self._ring_buffer = ring_buffer
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._atomic_writer = AtomicCheckpointWriter()

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        training_state: dict,
    ) -> None:
        """Snapshot state to CPU and queue background write.

        This method:
        1. Waits for any in-flight save to complete (single-in-flight enforcement)
        2. Copies model and optimizer state_dicts to CPU tensors (fast snapshot)
        3. Submits the disk write to the background thread
        4. Returns immediately so training can resume

        Args:
            step: Current training step number.
            model: The model whose state_dict will be saved.
            optimizer: The optimizer whose state_dict will be saved.
            training_state: Additional training state (lr, epoch, etc.)
        """
        # Enforce single in-flight save
        self.wait()

        # CPU snapshot: copy state_dicts to CPU tensors
        model_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        optimizer_state_cpu = _optimizer_state_to_cpu(optimizer.state_dict())

        # Build the checkpoint path
        checkpoint_path = self._checkpoint_dir / f"checkpoint_step_{step:06d}.pt"

        # Submit background write
        self._future = self._executor.submit(
            self._background_save,
            step,
            model_state_cpu,
            optimizer_state_cpu,
            training_state,
            checkpoint_path,
        )

    def _background_save(
        self,
        step: int,
        model_state: dict,
        optimizer_state: dict,
        training_state: dict,
        path: Path,
    ) -> None:
        """Perform the actual disk write in the background thread.

        Builds the full checkpoint dict, writes atomically, and registers
        with the ring buffer.

        Args:
            step: Training step number.
            model_state: Model state_dict (already on CPU).
            optimizer_state: Optimizer state_dict (already on CPU).
            training_state: Additional training state.
            path: Target path for the checkpoint file.
        """
        checkpoint_dict = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "training_state": training_state,
        }

        # Atomic write to disk
        sha256 = self._atomic_writer.save(checkpoint_dict, path)

        # Register with ring buffer
        self._ring_buffer.register(step, path, sha256)

    def wait(self) -> None:
        """Block until the current background save completes.

        If no save is in flight, returns immediately. Propagates any
        exception raised by the background save.
        """
        if self._future is not None:
            self._future.result()  # Blocks and re-raises exceptions
            self._future = None

    def latest(self) -> Path | None:
        """Return path to the most recently saved checkpoint.

        Delegates to the ring buffer's latest() method.

        Returns:
            Absolute path to the most recent checkpoint, or None if no
            checkpoints have been saved.
        """
        return self._ring_buffer.latest()

    def rollback(self) -> Path | None:
        """Return the latest verified checkpoint path for rollback.

        Delegates to the ring buffer's latest() method, returning the most
        recent verified checkpoint that can be used for recovery.

        Returns:
            Absolute path to the latest verified checkpoint, or None if
            no checkpoints are available.
        """
        return self._ring_buffer.latest_verified()

    def shutdown(self) -> None:
        """Shut down the background thread pool.

        Waits for any in-flight save to complete, then shuts down the executor.
        Should be called when the trainer is done.
        """
        self.wait()
        self._executor.shutdown(wait=True)
