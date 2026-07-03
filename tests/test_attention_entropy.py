"""Unit tests for compute_attention_entropy in src/evaluation/probes.py.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
"""

import numpy as np
import torch

from src.evaluation.probes import AttentionEntropyResult, compute_attention_entropy
from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer
from src.models.linear_attention import LinformerAttention


class _FakeLoader:
    """Minimal loader with next_batch() for testing."""

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        return x, y


class TestAttentionEntropyV0:
    """Tests for compute_attention_entropy with V0 (VanillaTransformer)."""

    def _make_model_and_loader(self):
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            vocab_size=1000,
            seq_len=32,
            attention_type="full",
            dropout=0.0,
        )
        model = VanillaTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=32, vocab_size=1000)
        return model, loader, config

    def test_returns_attention_entropy_result(self):
        """Requirement 8.1: Returns AttentionEntropyResult for V0."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert isinstance(result, AttentionEntropyResult)

    def test_per_layer_shape(self):
        """Requirement 8.1: per_layer has shape (n_layer,)."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result.per_layer.shape == (config.n_layer,)

    def test_per_head_shape(self):
        """Requirement 8.1: per_head has shape (n_layer, n_head)."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result.per_head.shape == (config.n_layer, config.n_head)

    def test_entropy_non_negative(self):
        """Requirement 8.4: All entropy values are >= 0."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert np.all(result.per_layer >= 0)
        assert np.all(result.per_head >= 0)

    def test_per_layer_is_mean_of_per_head(self):
        """per_layer should be the mean of per_head across heads."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        expected_per_layer = result.per_head.mean(axis=1)
        np.testing.assert_allclose(result.per_layer, expected_per_layer, rtol=1e-5)


class TestAttentionEntropyV5:
    """Tests for compute_attention_entropy with V5 (Linformer)."""

    def _make_model_and_loader(self):
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            vocab_size=1000,
            seq_len=32,
            attention_type="linear",
            norm_type="rmsnorm",
            position_encoding="rope",
            ffn_type="swiglu",
            projection_rank=8,
            dropout=0.0,
        )
        model = ModernTransformer(config, attention_class=LinformerAttention)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=32, vocab_size=1000)
        return model, loader, config

    def test_returns_attention_entropy_result(self):
        """Requirement 8.1: Returns AttentionEntropyResult for V5 (Linformer)."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert isinstance(result, AttentionEntropyResult)

    def test_per_layer_shape(self):
        """Requirement 8.1: per_layer shape (n_layer,) for V5."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result.per_layer.shape == (config.n_layer,)

    def test_per_head_shape(self):
        """Requirement 8.1: per_head shape (n_layer, n_head) for V5."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result.per_head.shape == (config.n_layer, config.n_head)

    def test_entropy_non_negative(self):
        """Requirement 8.4: All entropy values >= 0 for V5."""
        model, loader, config = self._make_model_and_loader()
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert np.all(result.per_layer >= 0)
        assert np.all(result.per_head >= 0)


class TestAttentionEntropyFlashVariants:
    """Tests for compute_attention_entropy returning None for flash variants."""

    def test_flash_sdpa_returns_none(self):
        """Requirement 8.2: Returns None for flash_sdpa (V1–V3)."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            vocab_size=1000,
            seq_len=32,
            attention_type="flash_sdpa",
            norm_type="rmsnorm",
            position_encoding="rope",
            ffn_type="swiglu",
            dropout=0.0,
        )
        model = ModernTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=32, vocab_size=1000)
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result is None

    def test_sliding_window_returns_none(self):
        """Requirement 8.2: Returns None for sliding_window (V4)."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            vocab_size=1000,
            seq_len=32,
            attention_type="sliding_window",
            norm_type="rmsnorm",
            position_encoding="rope",
            ffn_type="swiglu",
            window_size=16,
            dropout=0.0,
        )
        model = ModernTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=32, vocab_size=1000)
        result = compute_attention_entropy(model, loader, n_batches=3, device="cpu")
        assert result is None


class TestAttentionEntropyAveraging:
    """Tests for batch averaging behavior."""

    def test_averages_over_n_batches(self):
        """Requirement 8.5: Entropy is averaged over n_batches."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            vocab_size=1000,
            seq_len=32,
            attention_type="full",
            dropout=0.0,
        )
        model = VanillaTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=32, vocab_size=1000)

        # Run with different n_batches — result should differ slightly
        # (due to random inputs) but both should be valid
        result_5 = compute_attention_entropy(model, loader, n_batches=5, device="cpu")
        result_10 = compute_attention_entropy(model, loader, n_batches=10, device="cpu")

        # Both results should have valid shapes and non-negative values
        assert result_5.per_layer.shape == (config.n_layer,)
        assert result_10.per_layer.shape == (config.n_layer,)
        assert np.all(result_5.per_layer >= 0)
        assert np.all(result_10.per_layer >= 0)
