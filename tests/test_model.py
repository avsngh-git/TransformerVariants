"""Tests for the vanilla Transformer model (Phase 3).

Tests verify:
- Output shapes are correct for all components
- Causal masking prevents information leakage from future tokens
- Generation produces the expected number of tokens
- Loss computation works correctly
- Weight tying between embeddings and output head
- Weight initialization follows GPT-2 style
"""

import math

import pytest
import torch

from src.models.config import ModelConfig
from src.models.attention import CausalSelfAttention
from src.models.ffn import FeedForward
from src.models.vanilla_transformer import TransformerBlock, VanillaTransformer
from src.models.generate import generate


# Use a small debug config for fast tests
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
    return VanillaTransformer(config)


class TestCausalSelfAttention:
    """Tests for the attention module."""

    def test_output_shape(self, config):
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 16, config.d_model)
        out, _ = attn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_causal_mask(self, config):
        """Changing a future token should NOT affect the output at earlier positions."""
        attn = CausalSelfAttention(config)
        attn.eval()

        # Create two inputs that differ only at position 5
        x1 = torch.randn(1, 10, config.d_model)
        x2 = x1.clone()
        x2[:, 5, :] = torch.randn(config.d_model)  # change position 5

        out1, _ = attn(x1)
        out2, _ = attn(x2)

        # Positions 0-4 should be identical (they can't see position 5)
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)

        # Position 5+ should differ (they CAN see the changed position)
        assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-6)


class TestFeedForward:
    """Tests for the FFN module."""

    def test_output_shape(self, config):
        ffn = FeedForward(config)
        x = torch.randn(2, 16, config.d_model)
        out = ffn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_hidden_dimension(self, config):
        """Verify the internal expansion is correct (4x)."""
        ffn = FeedForward(config)
        assert ffn.fc1.out_features == config.d_model * config.ffn_multiplier
        assert ffn.fc2.in_features == config.d_model * config.ffn_multiplier


class TestTransformerBlock:
    """Tests for a single Transformer block."""

    def test_output_shape(self, config):
        block = TransformerBlock(config)
        x = torch.randn(2, 16, config.d_model)
        out, _ = block(x)
        assert out.shape == (2, 16, config.d_model)

    def test_residual_connection(self, config):
        """With zeroed sublayers, output should equal input (residual passthrough)."""
        block = TransformerBlock(config)
        # Zero out all parameters so sublayers produce zeros
        with torch.no_grad():
            for param in block.attn.parameters():
                param.zero_()
            for param in block.ffn.parameters():
                param.zero_()

        x = torch.randn(2, 16, config.d_model)
        out, _ = block(x)
        # Residual should pass through: x + 0 + 0 = x
        assert torch.allclose(out, x, atol=1e-5)


class TestVanillaTransformer:
    """Tests for the full model."""

    def test_logits_shape(self, model, config):
        """Logits should be (batch, seq_len, vocab_size)."""
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, _ = model(idx)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is None

    def test_loss_computation(self, model, config):
        """When targets are provided, should return a scalar loss."""
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, _ = model(idx, targets)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # cross-entropy is always positive

    def test_loss_value_range(self, model, config):
        """Initial loss should be approximately -ln(1/vocab_size) = ln(vocab_size)."""
        idx = torch.randint(0, config.vocab_size, (4, 16))
        targets = torch.randint(0, config.vocab_size, (4, 16))
        _, loss, _ = model(idx, targets)
        expected = math.log(config.vocab_size)  # ~4.6 for vocab_size=100
        # Should be within 2x of random (init is noisy but not crazy)
        assert loss.item() < expected * 2

    def test_generate_length(self, model, config):
        """Generate should produce exactly max_new_tokens additional tokens."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        max_new = 10
        output = generate(model, prompt, max_new_tokens=max_new)
        assert output.shape == (1, 5 + max_new)

    def test_generate_preserves_prompt(self, model, config):
        """Generated output should start with the original prompt."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        output = generate(model, prompt, max_new_tokens=3)
        assert torch.equal(output[:, :5], prompt)

    def test_weight_tying(self, model):
        """Token embedding and output head should share the same weight tensor."""
        assert model.tok_emb.weight is model.head.weight

    def test_no_weight_tying(self, config):
        """When tie_embeddings=False, weights should be separate."""
        config.tie_embeddings = False
        model = VanillaTransformer(config)
        assert model.tok_emb.weight is not model.head.weight

    def test_weight_init_std(self, model):
        """Most linear layers should have std close to 0.02."""
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                std = param.std().item()
                # Should be in a reasonable range (0.02 or scaled version)
                assert std < 0.1, f"{name} has unexpectedly large std: {std}"

    def test_residual_projection_scaling(self, config):
        """Residual projections should have smaller init than other layers."""
        model = VanillaTransformer(config)
        regular_std = 0.02
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)

        # out_proj should be scaled down
        out_proj_std = model.blocks[0].attn.out_proj.weight.std().item()
        fc1_std = model.blocks[0].ffn.fc1.weight.std().item()

        # Residual projection should have noticeably smaller std than regular layers
        assert out_proj_std < fc1_std

    def test_sequence_length_limit(self, model, config):
        """Should raise assertion if sequence exceeds max length."""
        idx = torch.randint(0, config.vocab_size, (1, config.seq_len + 1))
        with pytest.raises(AssertionError):
            model(idx)


class TestGeneration:
    """Tests for temperature, top-k, and top-p generation."""

    @pytest.fixture
    def model(self, config):
        return VanillaTransformer(config)

    def test_greedy_is_deterministic(self, model, config):
        """Temperature=0 (greedy) should always produce the same output."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out1 = generate(model, prompt, max_new_tokens=10, temperature=0.0)
        out2 = generate(model, prompt, max_new_tokens=10, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_sampling_produces_variety(self, model, config):
        """Temperature=1.0 with sampling should sometimes produce different outputs."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        outputs = set()
        for _ in range(10):
            out = generate(model, prompt, max_new_tokens=5, temperature=1.0)
            outputs.add(tuple(out[0, 5:].tolist()))
        # With 100 vocab and random model, sampling should produce variation
        assert len(outputs) > 1

    def test_low_temperature_less_random(self, model, config):
        """Lower temperature should produce less variety than higher."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))

        low_temp_outputs = set()
        high_temp_outputs = set()
        for _ in range(20):
            out_low = generate(model, prompt, max_new_tokens=3, temperature=0.1)
            out_high = generate(model, prompt, max_new_tokens=3, temperature=2.0)
            low_temp_outputs.add(tuple(out_low[0, 5:].tolist()))
            high_temp_outputs.add(tuple(out_high[0, 5:].tolist()))

        # High temperature should produce more unique sequences
        assert len(high_temp_outputs) >= len(low_temp_outputs)

    def test_top_k_limits_choices(self, model, config):
        """Top-k should still generate valid tokens and correct length."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5)
        assert out.shape == (1, 15)

    def test_top_p_generates_correct_length(self, model, config):
        """Top-p should generate the correct number of tokens."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_p=0.9)
        assert out.shape == (1, 15)

    def test_top_k_and_top_p_together(self, model, config):
        """Both top-k and top-p can be applied simultaneously."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        out = generate(model, prompt, max_new_tokens=5, temperature=0.8, top_k=10, top_p=0.9)
        assert out.shape == (1, 10)


class TestKVCache:
    """Tests for KV-cache correctness."""

    @pytest.fixture
    def model(self, config):
        model = VanillaTransformer(config)
        model.eval()
        return model

    def test_cached_matches_uncached(self, model, config):
        """Generation with and without KV-cache should produce identical results."""
        torch.manual_seed(42)
        prompt = torch.randint(0, config.vocab_size, (1, 5))

        # Generate with cache
        torch.manual_seed(123)
        out_cached = generate(model, prompt, max_new_tokens=10, temperature=0.0, use_cache=True)

        # Generate without cache
        torch.manual_seed(123)
        out_uncached = generate(model, prompt, max_new_tokens=10, temperature=0.0, use_cache=False)

        assert torch.equal(out_cached, out_uncached)

    def test_cache_shape(self, model, config):
        """KV-cache should have correct shape after forward pass."""
        idx = torch.randint(0, config.vocab_size, (2, 10))
        _, _, kv_cache = model(idx)

        # Should have one cache entry per layer
        assert len(kv_cache) == config.n_layer

        # Each cache entry is (k, v), shape (B, n_head, seq_len, d_head)
        for k, v in kv_cache:
            assert k.shape == (2, config.n_head, 10, config.d_head)
            assert v.shape == (2, config.n_head, 10, config.d_head)

    def test_cache_grows_by_one(self, model, config):
        """Each generation step should grow the cache by 1 position."""
        # First forward: process 5 tokens
        idx = torch.randint(0, config.vocab_size, (1, 5))
        _, _, cache1 = model(idx)
        assert cache1[0][0].size(2) == 5  # 5 cached positions

        # Second forward: process 1 new token with cache
        new_token = torch.randint(0, config.vocab_size, (1, 1))
        _, _, cache2 = model(new_token, kv_cache=cache1)
        assert cache2[0][0].size(2) == 6  # grew by 1

    def test_cache_speeds_up_generation(self, model, config):
        """Cached generation should be faster (or at least not slower)."""
        import time

        prompt = torch.randint(0, config.vocab_size, (1, 5))

        # Warm up
        generate(model, prompt, max_new_tokens=5, temperature=0.0, use_cache=True)

        # Time cached
        start = time.time()
        for _ in range(5):
            generate(model, prompt, max_new_tokens=20, temperature=0.0, use_cache=True)
        cached_time = time.time() - start

        # Time uncached
        start = time.time()
        for _ in range(5):
            generate(model, prompt, max_new_tokens=20, temperature=0.0, use_cache=False)
        uncached_time = time.time() - start

        # Cached should be faster (or at worst similar for tiny models)
        # Allow some slack since model is tiny and overhead matters
        assert cached_time <= uncached_time * 1.5, (
            f"Cached ({cached_time:.3f}s) should not be much slower than uncached ({uncached_time:.3f}s)"
        )
