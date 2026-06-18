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
        out = attn(x)
        assert out.shape == (2, 16, config.d_model)

    def test_causal_mask(self, config):
        """Changing a future token should NOT affect the output at earlier positions."""
        attn = CausalSelfAttention(config)
        attn.eval()

        # Create two inputs that differ only at position 5
        x1 = torch.randn(1, 10, config.d_model)
        x2 = x1.clone()
        x2[:, 5, :] = torch.randn(config.d_model)  # change position 5

        out1 = attn(x1)
        out2 = attn(x2)

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
        out = block(x)
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
        out = block(x)
        # Residual should pass through: x + 0 + 0 = x
        assert torch.allclose(out, x, atol=1e-5)


class TestVanillaTransformer:
    """Tests for the full model."""

    def test_logits_shape(self, model, config):
        """Logits should be (batch, seq_len, vocab_size)."""
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is None

    def test_loss_computation(self, model, config):
        """When targets are provided, should return a scalar loss."""
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # cross-entropy is always positive

    def test_loss_value_range(self, model, config):
        """Initial loss should be approximately -ln(1/vocab_size) = ln(vocab_size)."""
        idx = torch.randint(0, config.vocab_size, (4, 16))
        targets = torch.randint(0, config.vocab_size, (4, 16))
        _, loss = model(idx, targets)
        expected = math.log(config.vocab_size)  # ~4.6 for vocab_size=100
        # Should be within 2x of random (init is noisy but not crazy)
        assert loss.item() < expected * 2

    def test_generate_length(self, model, config):
        """Generate should produce exactly max_new_tokens additional tokens."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        max_new = 10
        output = model.generate(prompt, max_new_tokens=max_new)
        assert output.shape == (1, 5 + max_new)

    def test_generate_preserves_prompt(self, model, config):
        """Generated output should start with the original prompt."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        output = model.generate(prompt, max_new_tokens=3)
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
