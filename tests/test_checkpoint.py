"""Tests for the AtomicCheckpointWriter.

Tests cover:
- Basic save and load round-trip
- SHA-256 hash correctness
- verify() returns True for valid checkpoints, False for corrupted
- Temp file cleanup on failure
- Handling when .sha256 sidecar file is missing
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.training.checkpoint import AtomicCheckpointWriter


@pytest.fixture
def writer():
    """Create an AtomicCheckpointWriter instance."""
    return AtomicCheckpointWriter()


@pytest.fixture
def sample_state_dict():
    """Create a sample state dict resembling a model checkpoint."""
    return {
        'model': {
            'layer1.weight': torch.randn(64, 128),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(32, 64),
        },
        'optimizer': {
            'step': 100,
            'lr': 3e-4,
        },
        'step': 42,
    }


class TestAtomicCheckpointWriterSave:
    """Tests for the save method."""

    def test_save_creates_checkpoint_file(self, writer, sample_state_dict, tmp_path):
        """Saving should create the checkpoint file at the target path."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        assert path.exists()

    def test_save_creates_sha256_sidecar(self, writer, sample_state_dict, tmp_path):
        """Saving should create a .sha256 sidecar file alongside the checkpoint."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        hash_path = Path(str(path) + '.sha256')
        assert hash_path.exists()

    def test_save_returns_sha256_hash(self, writer, sample_state_dict, tmp_path):
        """Save should return a valid 64-char hex SHA-256 hash."""
        path = tmp_path / 'checkpoint.pt'
        result = writer.save(sample_state_dict, path)
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_save_hash_matches_sidecar(self, writer, sample_state_dict, tmp_path):
        """The returned hash should match what's stored in the sidecar file."""
        path = tmp_path / 'checkpoint.pt'
        returned_hash = writer.save(sample_state_dict, path)
        hash_path = Path(str(path) + '.sha256')
        stored_hash = hash_path.read_text().strip()
        assert returned_hash == stored_hash

    def test_save_no_temp_file_remains(self, writer, sample_state_dict, tmp_path):
        """After a successful save, no .pt.tmp file should remain."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        temp_path = path.with_suffix('.pt.tmp')
        assert not temp_path.exists()

    def test_save_temp_file_in_same_directory(self, writer, sample_state_dict, tmp_path):
        """Temp file should be created in the same directory as the target."""
        path = tmp_path / 'subdir' / 'checkpoint.pt'
        path.parent.mkdir(parents=True)

        # Patch open to check that the temp file is in the right directory
        original_open = open
        temp_paths_seen = []

        def tracking_open(p, *args, **kwargs):
            if str(p).endswith('.pt.tmp'):
                temp_paths_seen.append(Path(p))
            return original_open(p, *args, **kwargs)

        with patch('builtins.open', side_effect=tracking_open):
            writer.save(sample_state_dict, path)

        assert len(temp_paths_seen) == 1
        assert temp_paths_seen[0].parent == path.parent


class TestAtomicCheckpointWriterRoundTrip:
    """Tests for save + load round-trip integrity."""

    def test_round_trip_preserves_tensors(self, writer, sample_state_dict, tmp_path):
        """Saving and loading should produce equivalent tensors."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        loaded = torch.load(path, weights_only=False)

        for key in ['layer1.weight', 'layer1.bias', 'layer2.weight']:
            assert torch.equal(
                sample_state_dict['model'][key],
                loaded['model'][key],
            )

    def test_round_trip_preserves_scalars(self, writer, sample_state_dict, tmp_path):
        """Saving and loading should preserve scalar values."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        loaded = torch.load(path, weights_only=False)

        assert loaded['step'] == 42
        assert loaded['optimizer']['step'] == 100
        assert loaded['optimizer']['lr'] == 3e-4


class TestAtomicCheckpointWriterVerify:
    """Tests for the verify method."""

    def test_verify_valid_checkpoint(self, writer, sample_state_dict, tmp_path):
        """verify() should return True for an uncorrupted checkpoint."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)
        assert AtomicCheckpointWriter.verify(path) is True

    def test_verify_corrupted_checkpoint(self, writer, sample_state_dict, tmp_path):
        """verify() should return False when checkpoint bytes are modified."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)

        # Corrupt the checkpoint by flipping some bytes
        data = path.read_bytes()
        corrupted = bytearray(data)
        corrupted[100] ^= 0xFF  # Flip bits at byte 100
        path.write_bytes(bytes(corrupted))

        assert AtomicCheckpointWriter.verify(path) is False

    def test_verify_missing_checkpoint(self, tmp_path):
        """verify() should return False when the checkpoint file doesn't exist."""
        path = tmp_path / 'nonexistent.pt'
        assert AtomicCheckpointWriter.verify(path) is False

    def test_verify_missing_hash_file_recomputes(self, writer, sample_state_dict, tmp_path):
        """verify() should recompute and write hash if .sha256 file is missing."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)

        # Delete the hash file
        hash_path = Path(str(path) + '.sha256')
        hash_path.unlink()
        assert not hash_path.exists()

        # verify should still return True (recomputes hash)
        assert AtomicCheckpointWriter.verify(path) is True

        # And the hash file should be recreated
        assert hash_path.exists()


class TestAtomicCheckpointWriterComputeHash:
    """Tests for the compute_hash static method."""

    def test_compute_hash_is_deterministic(self, writer, sample_state_dict, tmp_path):
        """Computing hash twice on the same file should give the same result."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)

        hash1 = AtomicCheckpointWriter.compute_hash(path)
        hash2 = AtomicCheckpointWriter.compute_hash(path)
        assert hash1 == hash2

    def test_compute_hash_format(self, writer, sample_state_dict, tmp_path):
        """Hash should be a 64-char lowercase hex string (SHA-256)."""
        path = tmp_path / 'checkpoint.pt'
        writer.save(sample_state_dict, path)

        h = AtomicCheckpointWriter.compute_hash(path)
        assert len(h) == 64
        assert h == h.lower()
        assert all(c in '0123456789abcdef' for c in h)

    def test_compute_hash_different_for_different_files(self, writer, tmp_path):
        """Different file contents should produce different hashes."""
        path1 = tmp_path / 'a.pt'
        path2 = tmp_path / 'b.pt'

        writer.save({'x': torch.tensor([1.0])}, path1)
        writer.save({'x': torch.tensor([2.0])}, path2)

        assert AtomicCheckpointWriter.compute_hash(path1) != AtomicCheckpointWriter.compute_hash(path2)


class TestAtomicCheckpointWriterFailure:
    """Tests for error handling and cleanup."""

    def test_temp_file_cleaned_up_on_torch_save_failure(self, writer, tmp_path):
        """If torch.save fails, the temp file should be cleaned up."""
        path = tmp_path / 'checkpoint.pt'

        # Create an object that fails to pickle
        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("Cannot pickle")

        with pytest.raises(RuntimeError, match="Cannot pickle"):
            writer.save({'bad': Unpicklable()}, path)

        # Temp file should not remain
        temp_path = path.with_suffix('.pt.tmp')
        assert not temp_path.exists()

        # Final file should not exist either
        assert not path.exists()

    def test_temp_file_cleaned_up_on_fsync_failure(self, writer, sample_state_dict, tmp_path):
        """If fsync fails, the temp file should be cleaned up."""
        path = tmp_path / 'checkpoint.pt'

        with patch('os.fsync', side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Disk full"):
                writer.save(sample_state_dict, path)

        temp_path = path.with_suffix('.pt.tmp')
        assert not temp_path.exists()
        assert not path.exists()


# ============================================================
# Tests for CheckpointRingBuffer
# ============================================================

from src.training.checkpoint import CheckpointRingBuffer, RingEntry


@pytest.fixture
def ring_dir(tmp_path):
    """Create a directory for ring buffer checkpoints."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def ring(ring_dir):
    """Create a CheckpointRingBuffer with capacity 3."""
    return CheckpointRingBuffer(ring_dir, capacity=3)


def _create_checkpoint(ring_dir, step, writer):
    """Helper to create a valid checkpoint in the ring directory."""
    path = ring_dir / f"checkpoint_step_{step:06d}.pt"
    state_dict = {"step": step, "data": torch.randn(10)}
    sha256 = writer.save(state_dict, path)
    return path, sha256


class TestCheckpointRingBufferCapacity:
    """Tests for capacity enforcement and eviction behavior."""

    def test_capacity_not_exceeded(self, ring_dir):
        """Ring buffer should never exceed its capacity."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        for step in range(1, 6):
            path, sha256 = _create_checkpoint(ring_dir, step * 1000, writer)
            ring.register(step * 1000, path, sha256)

        assert len(ring.list_available()) == 3

    def test_oldest_evicted_when_at_capacity(self, ring_dir):
        """When at capacity, the oldest entry should be evicted."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=2)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        path3, sha3 = _create_checkpoint(ring_dir, 3000, writer)
        ring.register(3000, path3, sha3)

        entries = ring.list_available()
        steps = [e.step for e in entries]
        assert steps == [2000, 3000]

    def test_evicted_file_deleted_from_disk(self, ring_dir):
        """When an entry is evicted, its file should be deleted from disk."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=2)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        # path1 still exists since capacity is 2
        assert path1.exists()

        path3, sha3 = _create_checkpoint(ring_dir, 3000, writer)
        ring.register(3000, path3, sha3)

        # path1 should be deleted now
        assert not path1.exists()

    def test_evicted_sidecar_deleted_from_disk(self, ring_dir):
        """When evicted, the .sha256 sidecar should also be deleted."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=1)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)
        sidecar1 = Path(str(path1) + '.sha256')
        assert sidecar1.exists()

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        assert not sidecar1.exists()


class TestCheckpointRingBufferLatest:
    """Tests for the latest() accessor."""

    def test_latest_returns_none_when_empty(self, ring):
        """latest() should return None for an empty ring."""
        assert ring.latest() is None

    def test_latest_returns_most_recent(self, ring_dir):
        """latest() should return the path to the most recent checkpoint."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        assert ring.latest() == path2

    def test_latest_after_eviction(self, ring_dir):
        """latest() should still work correctly after eviction."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=2)

        for step in [1000, 2000, 3000]:
            path, sha = _create_checkpoint(ring_dir, step, writer)
            ring.register(step, path, sha)

        expected = ring_dir / "checkpoint_step_003000.pt"
        assert ring.latest() == expected


class TestCheckpointRingBufferRollback:
    """Tests for the rollback_to() accessor."""

    def test_rollback_to_zero_is_latest(self, ring_dir):
        """rollback_to(0) should return the latest checkpoint."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        assert ring.rollback_to(0) == ring.latest()

    def test_rollback_to_one_returns_previous(self, ring_dir):
        """rollback_to(1) should return the second most recent checkpoint."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        path3, sha3 = _create_checkpoint(ring_dir, 3000, writer)
        ring.register(3000, path3, sha3)

        assert ring.rollback_to(1) == path2

    def test_rollback_to_out_of_range_returns_none(self, ring_dir):
        """rollback_to() beyond available entries should return None."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        assert ring.rollback_to(5) is None

    def test_rollback_to_negative_returns_none(self, ring_dir):
        """rollback_to() with negative n_back should return None."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        assert ring.rollback_to(-1) is None


class TestCheckpointRingBufferMetadata:
    """Tests for metadata persistence and reload."""

    def test_metadata_file_created(self, ring_dir):
        """Registering a checkpoint should create checkpoint_ring.json."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path, sha = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path, sha)

        metadata_path = ring_dir / "checkpoint_ring.json"
        assert metadata_path.exists()

    def test_metadata_round_trip(self, ring_dir):
        """Creating a new buffer from the same dir should load existing state."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        path2, sha2 = _create_checkpoint(ring_dir, 2000, writer)
        ring.register(2000, path2, sha2)

        # Create a new buffer from the same directory
        ring2 = CheckpointRingBuffer(ring_dir, capacity=3)

        assert len(ring2.list_available()) == 2
        assert ring2.latest() == path2

    def test_metadata_preserves_entries(self, ring_dir):
        """Metadata should preserve all entry fields."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        path, sha = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path, sha)

        # Reload and check fields
        ring2 = CheckpointRingBuffer(ring_dir, capacity=3)
        entries = ring2.list_available()
        assert len(entries) == 1
        assert entries[0].step == 1000
        assert entries[0].sha256 == sha
        assert entries[0].timestamp > 0

    def test_metadata_capacity_persisted(self, ring_dir):
        """The capacity should be saved and loaded from metadata."""
        import json

        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=5)

        path, sha = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path, sha)

        metadata_path = ring_dir / "checkpoint_ring.json"
        data = json.loads(metadata_path.read_text())
        assert data["capacity"] == 5

    def test_init_from_existing_metadata(self, ring_dir):
        """Buffer should restore full state from existing checkpoint_ring.json."""
        import json

        # Manually write metadata
        metadata = {
            "capacity": 3,
            "entries": [
                {
                    "step": 500,
                    "path": "checkpoint_step_000500.pt",
                    "sha256": "abc123",
                    "timestamp": 1720100000.0,
                },
                {
                    "step": 1000,
                    "path": "checkpoint_step_001000.pt",
                    "sha256": "def456",
                    "timestamp": 1720100100.0,
                },
            ],
        }
        metadata_path = ring_dir / "checkpoint_ring.json"
        metadata_path.write_text(json.dumps(metadata))

        ring = CheckpointRingBuffer(ring_dir)
        entries = ring.list_available()
        assert len(entries) == 2
        assert entries[0].step == 500
        assert entries[1].step == 1000


class TestCheckpointRingBufferVerification:
    """Tests for verify-before-evict behavior."""

    def test_register_rejects_corrupted_checkpoint(self, ring_dir):
        """register() should raise ValueError if the checkpoint fails verification."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=3)

        # Create a valid checkpoint then corrupt it
        path = ring_dir / "bad_checkpoint.pt"
        state = {"step": 1, "data": torch.randn(5)}
        sha = writer.save(state, path)

        # Corrupt the file
        data = path.read_bytes()
        corrupted = bytearray(data)
        corrupted[50] ^= 0xFF
        path.write_bytes(bytes(corrupted))

        with pytest.raises(ValueError, match="verification failed"):
            ring.register(1000, path, sha)

    def test_register_does_not_evict_on_failure(self, ring_dir):
        """If verification fails, existing entries should not be evicted."""
        writer = AtomicCheckpointWriter()
        ring = CheckpointRingBuffer(ring_dir, capacity=1)

        # Register a valid checkpoint
        path1, sha1 = _create_checkpoint(ring_dir, 1000, writer)
        ring.register(1000, path1, sha1)

        # Try to register a corrupted checkpoint
        path2 = ring_dir / "bad_checkpoint.pt"
        state = {"step": 2, "data": torch.randn(5)}
        sha2 = writer.save(state, path2)
        data = path2.read_bytes()
        corrupted = bytearray(data)
        corrupted[50] ^= 0xFF
        path2.write_bytes(bytes(corrupted))

        with pytest.raises(ValueError):
            ring.register(2000, path2, sha2)

        # Original entry should still be present
        assert len(ring.list_available()) == 1
        assert ring.list_available()[0].step == 1000
        assert path1.exists()


# ============================================================
# Tests for AsyncCheckpointWriter
# ============================================================

import threading
import time as time_module

import torch.nn as nn
from torch.optim import SGD

from src.training.checkpoint import AsyncCheckpointWriter, _optimizer_state_to_cpu


class SimpleModel(nn.Module):
    """A minimal model for testing checkpoint operations."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def checkpoint_setup(tmp_path):
    """Set up model, optimizer, ring buffer, and async writer for tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    ring_buffer = CheckpointRingBuffer(checkpoint_dir, capacity=3)
    async_writer = AsyncCheckpointWriter(ring_buffer, checkpoint_dir)
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    # Run a forward/backward pass to populate optimizer state
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    return {
        "writer": async_writer,
        "model": model,
        "optimizer": optimizer,
        "ring_buffer": ring_buffer,
        "checkpoint_dir": checkpoint_dir,
    }


class TestAsyncCheckpointWriterCPUSnapshot:
    """Tests that CPU snapshot creates tensors on CPU device."""

    def test_model_state_snapshot_on_cpu(self, checkpoint_setup):
        """After save, the snapshot should have all tensors on CPU."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]

        writer.save(step=100, model=model, optimizer=optimizer, training_state={"lr": 0.01})
        writer.wait()

        # Verify the checkpoint on disk has CPU tensors
        latest = writer.latest()
        assert latest is not None
        loaded = torch.load(latest, weights_only=False)
        for key, tensor in loaded["model_state_dict"].items():
            assert tensor.device == torch.device("cpu"), f"Tensor {key} not on CPU"

    def test_optimizer_state_snapshot_on_cpu(self, checkpoint_setup):
        """Optimizer state tensors in the checkpoint should be on CPU."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]

        writer.save(step=200, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        latest = writer.latest()
        loaded = torch.load(latest, weights_only=False)
        opt_state = loaded["optimizer_state_dict"]
        for param_id, param_state in opt_state.get("state", {}).items():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    assert v.device == torch.device("cpu"), (
                        f"Optimizer state tensor {param_id}/{k} not on CPU"
                    )


class TestOptimizerStateToCpu:
    """Tests for the _optimizer_state_to_cpu helper."""

    def test_tensors_moved_to_cpu(self):
        """All tensors in optimizer state should be on CPU after conversion."""
        opt_state = {
            "state": {
                0: {
                    "momentum_buffer": torch.randn(10, 5),
                    "step": torch.tensor(1),
                },
                1: {
                    "momentum_buffer": torch.randn(5),
                    "step": torch.tensor(2),
                },
            },
            "param_groups": [{"lr": 0.01, "weight_decay": 0.0}],
        }

        cpu_state = _optimizer_state_to_cpu(opt_state)

        for param_id, param_state in cpu_state["state"].items():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    assert v.device == torch.device("cpu")

    def test_non_tensor_values_preserved(self):
        """Non-tensor values like param_groups should be preserved."""
        opt_state = {
            "state": {
                0: {"momentum_buffer": torch.randn(5), "step_count": 42},
            },
            "param_groups": [{"lr": 0.01, "momentum": 0.9}],
        }

        cpu_state = _optimizer_state_to_cpu(opt_state)

        assert cpu_state["state"][0]["step_count"] == 42
        assert cpu_state["param_groups"] == [{"lr": 0.01, "momentum": 0.9}]

    def test_deep_copy_independence(self):
        """CPU snapshot should be independent of the original tensors."""
        original_tensor = torch.randn(5)
        opt_state = {
            "state": {
                0: {"momentum_buffer": original_tensor},
            },
            "param_groups": [],
        }

        cpu_state = _optimizer_state_to_cpu(opt_state)

        # Modify original tensor
        original_tensor.fill_(999.0)

        # CPU snapshot should be unaffected
        assert not torch.all(cpu_state["state"][0]["momentum_buffer"] == 999.0)


class TestAsyncCheckpointWriterWait:
    """Tests that wait() blocks until background save completes."""

    def test_wait_blocks_until_completion(self, checkpoint_setup):
        """wait() should block until the background save is done."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]

        writer.save(step=100, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        # After wait, the checkpoint must exist
        latest = writer.latest()
        assert latest is not None
        assert latest.exists()

    def test_wait_with_no_inflight_returns_immediately(self, checkpoint_setup):
        """wait() should return immediately if no save is in flight."""
        setup = checkpoint_setup
        writer = setup["writer"]
        # Should not raise or hang
        writer.wait()

    def test_wait_propagates_exceptions(self, tmp_path):
        """wait() should propagate exceptions from the background thread."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        ring_buffer = CheckpointRingBuffer(checkpoint_dir, capacity=3)
        writer = AsyncCheckpointWriter(ring_buffer, checkpoint_dir)

        # Make the checkpoint directory read-only to force a write failure
        model = SimpleModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Submit a save to a path that doesn't exist (parent removed)
        writer.save(step=100, model=model, optimizer=optimizer, training_state={})
        # Remove the checkpoint dir to trigger an error during background save
        import shutil
        shutil.rmtree(checkpoint_dir)

        with pytest.raises(Exception):
            writer.wait()


class TestAsyncCheckpointWriterSingleInflight:
    """Tests that at most one save is in flight at a time."""

    def test_second_save_waits_for_first(self, checkpoint_setup):
        """A second save should wait for the first to complete."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]

        # First save
        writer.save(step=100, model=model, optimizer=optimizer, training_state={"epoch": 1})
        # Second save (should internally wait for first)
        writer.save(step=200, model=model, optimizer=optimizer, training_state={"epoch": 2})
        writer.wait()

        # Both checkpoints should exist
        ring = setup["ring_buffer"]
        entries = ring.list_available()
        steps = [e.step for e in entries]
        assert 100 in steps
        assert 200 in steps

    def test_single_inflight_enforcement(self, checkpoint_setup):
        """Only one background save should execute at a time."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]

        # Track concurrent executions
        concurrent_count = []
        active_count = [0]
        lock = threading.Lock()

        original_save = writer._atomic_writer.save

        def tracking_save(state_dict, path):
            with lock:
                active_count[0] += 1
                concurrent_count.append(active_count[0])
            time_module.sleep(0.05)  # Simulate slow disk I/O
            result = original_save(state_dict, path)
            with lock:
                active_count[0] -= 1
            return result

        writer._atomic_writer.save = tracking_save

        # Issue multiple saves
        writer.save(step=100, model=model, optimizer=optimizer, training_state={})
        writer.save(step=200, model=model, optimizer=optimizer, training_state={})
        writer.save(step=300, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        # No more than 1 concurrent execution at any time
        assert max(concurrent_count) == 1


class TestAsyncCheckpointWriterRingBuffer:
    """Tests that ring buffer is updated after save completes."""

    def test_ring_buffer_updated_after_save(self, checkpoint_setup):
        """The ring buffer should have the checkpoint registered after wait()."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]
        ring = setup["ring_buffer"]

        assert ring.latest() is None

        writer.save(step=500, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        latest = ring.latest()
        assert latest is not None
        assert "step_000500" in str(latest)

    def test_multiple_saves_all_registered(self, checkpoint_setup):
        """Multiple saves should all be registered in the ring buffer."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]
        ring = setup["ring_buffer"]

        for step in [100, 200, 300]:
            writer.save(step=step, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        entries = ring.list_available()
        assert len(entries) == 3
        assert [e.step for e in entries] == [100, 200, 300]

    def test_latest_delegates_to_ring_buffer(self, checkpoint_setup):
        """latest() should return the same result as ring_buffer.latest()."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]
        ring = setup["ring_buffer"]

        writer.save(step=1000, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        assert writer.latest() == ring.latest()

    def test_rollback_delegates_to_ring_buffer(self, checkpoint_setup):
        """rollback() should return the latest verified checkpoint."""
        setup = checkpoint_setup
        writer = setup["writer"]
        model = setup["model"]
        optimizer = setup["optimizer"]
        ring = setup["ring_buffer"]

        writer.save(step=1000, model=model, optimizer=optimizer, training_state={})
        writer.wait()

        assert writer.rollback() == ring.latest()
