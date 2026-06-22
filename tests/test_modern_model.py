"""Tests for the V1 Modern Transformer (RoPE, RMSNorm, SwiGLU, Flash Attention)."""

import math

import pytest
import torch

from src.models.config import ModelConfig
from src.models.rmsnorm import RMSNorm
from src.models.rope import precompute_rope_frequencies, apply_rope
from src.models.swiglu_ffn import SwiGLUFeedForward
from src.models.modern_attention import ModernAttention
from src.models.modern_transformer import ModernTransformerBlock, ModernTransformer


@pytest.fixture
def config():
    return ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=4,
        vocab_size=100,
        seq_len=32,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
    )


@pytest.fixture
def model(config):
    return ModernTransformer(config)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == (2, 16, 64)

    def test_normalized_rms(self):
        """After RMSNorm, the RMS of the output should be approximately 1."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 5  # large input
        out = norm(x)
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        # Should be close to 1 (the weight is initialized to ones)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRoPE:
    def test_frequency_shape(self):
        cos, sin = precompute_rope_frequencies(d_head=64, seq_len=128)
        assert cos.shape == (128, 32)  # d_head // 2
        assert sin.shape == (128, 32)

    def test_apply_preserves_shape(self):
        cos, sin = precompute_rope_frequencies(d_head=16, seq_len=32)
        x = torch.randn(2, 4, 10, 16)  # (B, n_head, T, d_head)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_different_positions_different_output(self):
        """Same vector at different positions should produce different rotations."""
        cos, sin = precompute_rope_frequencies(d_head=16, seq_len=32)
        x = torch.ones(1, 1, 5, 16)  # same vector at all positions
        out = apply_rope(x, cos, sin)
        # Each position should be different after rotation
        assert not torch.allclose(out[:, :, 0, :], out[:, :, 1, :])

    def test_relative_position_invariance(self):
        """The dot product between Q and K should depend on relative position."""
        cos, sin = precompute_rope_frequencies(d_head=16, seq_len=64)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)

        # Place Q at pos 5, K at pos 3 (distance = 2)
        cos5, sin5 = cos[5:6], sin[5:6]
        cos3, sin3 = cos[3:4], sin[3:4]
        q_rot_5 = apply_rope(q, cos5, sin5)
        k_rot_3 = apply_rope(k, cos3, sin3)
        dot_5_3 = (q_rot_5 * k_rot_3).sum()

        # Place Q at pos 10, K at pos 8 (same distance = 2)
        cos10, sin10 = cos[10:11], sin[10:11]
        cos8, sin8 = cos[8:9], sin[8:9]
        q_rot_10 = apply_rope(q, cos10, sin10)
        k_rot_8 = apply_rope(k, cos8, sin8)
        dot_10_8 = (q_rot_10 * k_rot_8).sum()

        # Same relative distance → same dot product
        assert torch.allclose(dot_5_3, dot_10_8, atol=1e-5)


class TestSwiGLU:
    def test_output_shape(self, config):
        ffn = SwiGLUFeedForward(config)
        x = torch.randn(2, 16, config.d_model)
        out = ffn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_hidden_dim_rounded(self, config):
        """Hidden dim should be rounded to nearest multiple of 64."""
        ffn = SwiGLUFeedForward(config)
        hidden = ffn.w_gate.out_features
        assert hidden % 64 == 0


class TestModernAttention:
    def test_output_shape(self, config):
        attn = ModernAttention(config)
        x = torch.randn(2, 16, config.d_model)
        out, _ = attn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_kv_cache_shape(self, config):
        attn = ModernAttention(config)
        x = torch.randn(2, 10, config.d_model)
        _, cache = attn(x)
        k, v = cache
        assert k.shape == (2, config.n_head, 10, config.d_head)


class TestModernTransformer:
    def test_logits_shape(self, model, config):
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, _ = model(idx)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is None

    def test_loss_computation(self, model, config):
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, _ = model(idx, targets)
        assert loss is not None
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_generate_length(self, model, config):
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert out.shape == (1, 15)

    def test_generate_greedy_deterministic(self, model, config):
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out1 = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        out2 = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_cached_matches_uncached(self, model, config):
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out_cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
        out_uncached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)
        assert torch.equal(out_cached, out_uncached)

    def test_weight_tying(self, model):
        assert model.tok_emb.weight is model.head.weight

    def test_no_position_embedding(self, model):
        """V1 should NOT have a position embedding layer."""
        assert not hasattr(model, "pos_emb")

    def test_parameter_count_comparable_to_v0(self, config):
        """V1 should have roughly similar params to V0 (within 20%)."""
        from src.models.vanilla_transformer import VanillaTransformer
        v0 = VanillaTransformer(config)
        v1 = ModernTransformer(config)
        v0_params = sum(p.numel() for p in v0.parameters())
        v1_params = sum(p.numel() for p in v1.parameters())
        ratio = v1_params / v0_params
        # SwiGLU adds a bit, no pos_emb saves a bit — should be within 20%
        assert 0.8 < ratio < 1.2, f"V1/V0 param ratio: {ratio:.2f}"
