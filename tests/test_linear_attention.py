"""Unit tests for LinearAttention module."""
import pytest
import torch

from src.models.config import ModelConfig
from src.models.linear_attention import LinearAttention
from src.models import registry


@pytest.fixture
def config():
    """Debug-scale config for linear attention."""
    return ModelConfig(
        n_layer=4,
        d_model=256,
        n_head=4,
        seq_len=512,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="none",
        ffn_type="swiglu",
        attention_type="linear",
        dropout=0.0,
        bias=False,
    )


class TestLinearAttentionConstructor:
    """Tests for LinearAttention.__init__."""

    def test_creates_expected_layers(self, config):
        """Constructor creates q_proj, k_proj, v_proj, out_proj, resid_dropout."""
        attn = LinearAttention(config)
        assert hasattr(attn, 'q_proj')
        assert hasattr(attn, 'k_proj')
        assert hasattr(attn, 'v_proj')
        assert hasattr(attn, 'out_proj')
        assert hasattr(attn, 'resid_dropout')

    def test_no_rope_buffers(self, config):
        """Module state_dict should NOT contain rope_cos or rope_sin."""
        attn = LinearAttention(config)
        state_keys = set(attn.state_dict().keys())
        assert not any('rope_cos' in k for k in state_keys)
        assert not any('rope_sin' in k for k in state_keys)


class TestLinearAttentionForward:
    """Tests for LinearAttention.forward."""

    def test_kv_cache_raises_not_implemented(self, config):
        """Non-None kv_cache raises NotImplementedError with expected message."""
        attn = LinearAttention(config)
        x = torch.randn(2, 8, config.d_model)
        with pytest.raises(NotImplementedError, match="KV-cache generation is not supported"):
            attn(x, kv_cache=torch.zeros(1))

    def test_output_shape(self, config):
        """Forward returns (B, T, d_model) output and None."""
        attn = LinearAttention(config)
        B, T = 2, 16
        x = torch.randn(B, T, config.d_model)
        output, kv_out = attn(x)
        assert output.shape == (B, T, config.d_model)
        assert kv_out is None

    def test_full_model_logits_shape(self):
        """Full model at debug scale produces (B, 512, 50257) logits."""
        model, config = registry.build("linear", "debug")
        model.eval()
        B = 2
        x = torch.randint(0, config.vocab_size, (B, config.seq_len))
        with torch.no_grad():
            logits, loss, _ = model(x)
        assert logits.shape == (B, config.seq_len, config.vocab_size)
