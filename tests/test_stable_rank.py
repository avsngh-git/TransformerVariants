"""Unit tests for compute_stable_rank in src/evaluation/probes.py.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.probes import StableRankResult, compute_stable_rank
from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer


class _FakeLoader:
    """Minimal loader with next_batch() for testing stable rank."""

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        return x, y


class TestStableRankResult:
    """Tests for StableRankResult dataclass."""

    def test_dataclass_fields(self):
        """StableRankResult has per_layer, mean, std fields."""
        result = StableRankResult(
            per_layer=np.array([2.0, 3.0, 4.0]),
            mean=3.0,
            std=0.816,
        )
        assert result.per_layer.shape == (3,)
        assert result.mean == 3.0
        assert result.std == 0.816


class TestComputeStableRank:
    """Tests for compute_stable_rank function."""

    def _make_model_and_loader(self, n_layer=2, d_model=64, seq_len=32, vocab_size=100):
        config = ModelConfig(
            n_layer=n_layer,
            d_model=d_model,
            n_head=4,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dropout=0.0,
        )
        model = VanillaTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=seq_len, vocab_size=vocab_size)
        return model, loader, config

    def test_returns_stable_rank_result(self):
        """compute_stable_rank returns a StableRankResult."""
        model, loader, config = self._make_model_and_loader()
        result = compute_stable_rank(model, loader, n_batches=3, device="cpu")
        assert isinstance(result, StableRankResult)

    def test_per_layer_shape(self):
        """Requirement 6.3: per_layer has shape (n_layer,)."""
        n_layer = 4
        model, loader, config = self._make_model_and_loader(n_layer=n_layer)
        result = compute_stable_rank(model, loader, n_batches=3, device="cpu")
        assert result.per_layer.shape == (n_layer,)

    def test_values_in_valid_range(self):
        """Requirement 6.4: per_layer values are in [1.0, d_model]."""
        d_model = 64
        model, loader, config = self._make_model_and_loader(d_model=d_model)
        result = compute_stable_rank(model, loader, n_batches=5, device="cpu")
        assert np.all(result.per_layer >= 1.0)
        assert np.all(result.per_layer <= d_model)

    def test_mean_and_std_consistency(self):
        """Requirement 6.3: mean and std are consistent with per_layer values."""
        model, loader, config = self._make_model_and_loader(n_layer=3)
        result = compute_stable_rank(model, loader, n_batches=5, device="cpu")
        expected_mean = result.per_layer.mean()
        expected_std = result.per_layer.std()
        assert abs(result.mean - expected_mean) < 1e-6
        assert abs(result.std - expected_std) < 1e-6

    def test_hooks_removed_after_computation(self):
        """Requirement 6.5: forward hooks are removed after computation."""
        model, loader, config = self._make_model_and_loader()

        # Count hooks before
        hooks_before = sum(
            len(block._forward_hooks) for block in model.blocks
        )

        compute_stable_rank(model, loader, n_batches=2, device="cpu")

        # Count hooks after
        hooks_after = sum(
            len(block._forward_hooks) for block in model.blocks
        )

        assert hooks_after == hooks_before

    def test_no_parameter_modification(self):
        """Requirement 6.5: model parameters are not modified."""
        model, loader, config = self._make_model_and_loader()

        # Capture parameter values before
        params_before = {
            name: p.clone() for name, p in model.named_parameters()
        }

        compute_stable_rank(model, loader, n_batches=3, device="cpu")

        # Verify parameters unchanged
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), f"Parameter {name} was modified"

    def test_averages_over_n_batches(self):
        """Requirement 6.2: averaging over specified n_batches."""
        model, loader, config = self._make_model_and_loader()

        # Running with different n_batches should give different results
        # (due to random data in loader) but both should be valid
        result_few = compute_stable_rank(model, loader, n_batches=2, device="cpu")
        result_more = compute_stable_rank(model, loader, n_batches=10, device="cpu")

        # Both should produce valid results in range
        assert np.all(result_few.per_layer >= 1.0)
        assert np.all(result_more.per_layer >= 1.0)
        assert np.all(result_few.per_layer <= config.d_model)
        assert np.all(result_more.per_layer <= config.d_model)

    def test_identity_like_matrix_has_high_stable_rank(self):
        """A model producing diverse hidden states should have stable rank > 1."""
        model, loader, config = self._make_model_and_loader(d_model=64)
        result = compute_stable_rank(model, loader, n_batches=5, device="cpu")
        # Random initialization should yield diverse representations,
        # so stable rank should be meaningfully above 1
        assert np.all(result.per_layer > 1.0)
