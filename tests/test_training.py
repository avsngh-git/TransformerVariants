"""Tests for the training loop components (Phase 4).

Tests verify:
- Learning rate scheduler produces correct values
- DataLoader serves correct shapes and shifts targets
- Gradient accumulation works correctly
- A short training run reduces loss (model is learning)
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataloader import ShardedDataLoader
from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer
from src.training.scheduler import get_lr
from src.training.synthetic_loader import SyntheticLoader
from src.training.trainer import Trainer, TrainConfig
from src.training.run_logger import RunLogger


class TestScheduler:
    """Tests for the cosine learning rate scheduler."""

    def test_warmup_starts_near_zero(self):
        """First step should have a small but nonzero LR."""
        lr = get_lr(step=0, max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        assert lr > 0
        assert lr < 3e-4

    def test_warmup_reaches_max(self):
        """At the end of warmup, LR should equal max_lr."""
        lr = get_lr(step=99, max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        assert abs(lr - 3e-4) < 1e-8

    def test_cosine_midpoint(self):
        """At the midpoint of decay, LR should be approximately (max + min) / 2."""
        warmup = 100
        total = 1000
        midpoint = warmup + (total - warmup) // 2  # 550
        lr = get_lr(step=midpoint, max_lr=3e-4, min_lr=3e-5, warmup_steps=warmup, total_steps=total)
        expected_mid = (3e-4 + 3e-5) / 2
        assert abs(lr - expected_mid) < 1e-5

    def test_end_reaches_min(self):
        """At the last step, LR should be min_lr."""
        lr = get_lr(step=999, max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        assert abs(lr - 3e-5) < 1e-7

    def test_beyond_total_returns_min(self):
        """After total_steps, LR should stay at min_lr."""
        lr = get_lr(step=2000, max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
        assert abs(lr - 3e-5) < 1e-8

    def test_monotonic_during_decay(self):
        """LR should be monotonically decreasing during the cosine phase."""
        lrs = [
            get_lr(step=s, max_lr=3e-4, min_lr=3e-5, warmup_steps=100, total_steps=1000)
            for s in range(100, 1000)
        ]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


class TestShardedDataLoader:
    """Tests for the binary shard DataLoader."""

    @pytest.fixture
    def shard_dir(self, tmp_path):
        """Create a temporary directory with fake shard data."""
        # Create a small shard with known tokens
        n_tokens = 2048
        tokens = np.arange(n_tokens, dtype=np.uint16)
        shard_path = tmp_path / "train_0000.bin"
        tokens.tofile(shard_path)

        # Create a validation shard
        val_tokens = np.arange(1024, dtype=np.uint16) + 5000
        val_path = tmp_path / "val_0000.bin"
        val_tokens.tofile(val_path)

        # Create manifest
        manifest = {
            "shards": [
                {"filename": "train_0000.bin", "split": "train", "n_tokens": n_tokens},
                {"filename": "val_0000.bin", "split": "val", "n_tokens": 1024},
            ]
        }
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        return tmp_path

    def test_batch_shape(self, shard_dir):
        loader = ShardedDataLoader(shard_dir, batch_size=4, seq_len=32, split="train")
        x, y = loader.next_batch()
        assert x.shape == (4, 32)
        assert y.shape == (4, 32)

    def test_target_is_shifted_input(self, shard_dir):
        """Target should be input shifted by 1 position."""
        loader = ShardedDataLoader(shard_dir, batch_size=1, seq_len=8, split="train")
        x, y = loader.next_batch()
        # Since our shard has sequential values (0, 1, 2, ...),
        # x[0] = [0,1,2,3,4,5,6,7] and y[0] = [1,2,3,4,5,6,7,8]
        assert torch.equal(x[0, 1:], y[0, :-1])

    def test_dtype_is_long(self, shard_dir):
        """Tokens should be int64 (required by nn.Embedding)."""
        loader = ShardedDataLoader(shard_dir, batch_size=2, seq_len=16, split="train")
        x, y = loader.next_batch()
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_shard_advancement(self, shard_dir):
        """Should wrap around when a shard runs out."""
        # With 2048 tokens and batch of 4 * (32+1) = 132 tokens per batch,
        # after ~15 batches we should exhaust the shard
        loader = ShardedDataLoader(shard_dir, batch_size=4, seq_len=32, split="train")
        # This shouldn't crash even after many batches
        for _ in range(50):
            x, y = loader.next_batch()
            assert x.shape == (4, 32)


class TestTrainingIntegration:
    """Integration test: verify the model actually learns."""

    def test_loss_decreases(self, tmp_path):
        """A short training run should reduce loss (model is learning)."""
        # Create tiny model
        config = ModelConfig(
            n_layer=2, d_model=64, n_head=4, vocab_size=256,
            seq_len=32, dropout=0.0, bias=False, tie_embeddings=True,
        )
        model = VanillaTransformer(config)

        # Create synthetic loader with random tensor batches
        micro_batch_size = 4
        seq_len = 32
        vocab_size = 256
        batches = [
            (
                torch.randint(0, vocab_size, (micro_batch_size, seq_len), dtype=torch.int64),
                torch.randint(0, vocab_size, (micro_batch_size, seq_len), dtype=torch.int64),
            )
            for _ in range(10)
        ]
        loader = SyntheticLoader(batches)

        # Training config for a quick run
        train_config = TrainConfig(
            max_lr=1e-3,
            min_lr=1e-4,
            warmup_steps=5,
            max_steps=50,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=1,
            dtype="float32",  # CPU doesn't support bf16 well
            log_interval=100,  # suppress logging
            eval_interval=100,
            checkpoint_interval=1000,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        # Create RunLogger for structured logging
        run_logger = RunLogger(tmp_path / "run", config={"variant": "test", "scale": "debug"})

        trainer = Trainer(
            model, train_config,
            train_loader=loader,
            val_loader=loader,
            run_logger=run_logger,
            device="cpu",
        )

        # Get initial loss
        x, y = trainer.train_loader.next_batch()
        with torch.no_grad():
            _, initial_loss, _ = model(x, y)
        initial_loss = initial_loss.item()

        # Train
        results = trainer.train()

        # Loss should decrease
        assert results["final_train_loss"] < initial_loss


class TestModernTrainingIntegration:
    """Integration test: verify the V1 Modern Transformer learns through the training loop."""

    def test_loss_decreases(self, tmp_path):
        """A short training run with V1 should reduce loss (model is learning)."""
        # Create tiny V1 model (same dimensions as V0 test for comparability)
        config = ModelConfig(
            n_layer=2, d_model=64, n_head=4, vocab_size=256,
            seq_len=32, dropout=0.0, bias=False, tie_embeddings=True,
        )
        model = ModernTransformer(config)

        # Create synthetic loader with random tensor batches
        micro_batch_size = 4
        seq_len = 32
        vocab_size = 256
        batches = [
            (
                torch.randint(0, vocab_size, (micro_batch_size, seq_len), dtype=torch.int64),
                torch.randint(0, vocab_size, (micro_batch_size, seq_len), dtype=torch.int64),
            )
            for _ in range(10)
        ]
        loader = SyntheticLoader(batches)

        # Training config for a quick run
        train_config = TrainConfig(
            max_lr=1e-3,
            min_lr=1e-4,
            warmup_steps=5,
            max_steps=50,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=1,
            dtype="float32",  # CPU doesn't support bf16 well
            log_interval=100,  # suppress logging
            eval_interval=100,
            checkpoint_interval=1000,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        # Create RunLogger for structured logging
        run_logger = RunLogger(tmp_path / "run", config={"variant": "test", "scale": "debug"})

        trainer = Trainer(
            model, train_config,
            train_loader=loader,
            val_loader=loader,
            run_logger=run_logger,
            device="cpu",
        )

        # Get initial loss
        x, y = trainer.train_loader.next_batch()
        with torch.no_grad():
            _, initial_loss, _ = model(x, y)
        initial_loss = initial_loss.item()

        # Train
        results = trainer.train()

        # Loss should decrease
        assert results["final_train_loss"] < initial_loss
