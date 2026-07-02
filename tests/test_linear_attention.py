"""Unit tests for LinformerAttention module."""
import pytest
import torch

from src.models.config import ModelConfig
from src.models.linear_attention import LinformerAttention
from src.models import registry


@pytest.fixture
def config():
    """Debug-scale config for Linformer attention."""
    return ModelConfig(
        n_layer=4,
        d_model=256,
        n_head=4,
        seq_len=512,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        projection_rank=64,
        dropout=0.0,
        bias=False,
    )


class TestLinformerAttentionConstructor:
    """Tests for LinformerAttention.__init__."""

    def test_creates_expected_layers(self, config):
        """Constructor creates q_proj, k_proj, v_proj, out_proj, E, F, and RoPE buffers."""
        attn = LinformerAttention(config)
        # Linear projections
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "out_proj")
        # Learned projection matrices E and F
        assert hasattr(attn, "E")
        assert hasattr(attn, "F")
        assert attn.E.shape == (config.seq_len, config.projection_rank)
        assert attn.F.shape == (config.seq_len, config.projection_rank)
        # RoPE buffers (non-persistent, so not in state_dict but accessible as attributes)
        assert hasattr(attn, "rope_cos")
        assert hasattr(attn, "rope_sin")
        assert attn.rope_cos.shape[0] == config.seq_len
        assert attn.rope_sin.shape[0] == config.seq_len


class TestLinformerAttentionForward:
    """Tests for LinformerAttention.forward."""

    def test_kv_cache_raises_not_implemented(self, config):
        """Non-None kv_cache raises NotImplementedError with expected message."""
        attn = LinformerAttention(config)
        x = torch.randn(2, 8, config.d_model)
        with pytest.raises(NotImplementedError, match="E/F projection matrices are tied to fixed seq_len"):
            attn(x, kv_cache=torch.zeros(1))

    def test_output_shape_full_seq_len(self, config):
        """Forward returns (B, seq_len, d_model) output and None for full seq_len."""
        attn = LinformerAttention(config)
        B = 2
        x = torch.randn(B, config.seq_len, config.d_model)
        output, kv_out = attn(x)
        assert output.shape == (B, config.seq_len, config.d_model)
        assert kv_out is None

    def test_output_shape_variable_seq_len(self, config):
        """Forward returns correct shape for T < seq_len."""
        attn = LinformerAttention(config)
        B, T = 2, 128
        x = torch.randn(B, T, config.d_model)
        output, kv_out = attn(x)
        assert output.shape == (B, T, config.d_model)
        assert kv_out is None


class TestLinformerFullModel:
    """Integration tests via registry."""

    def test_full_model_logits_shape(self):
        """Full model at debug scale produces (B, 512, 50257) logits."""
        model, cfg = registry.build("linear", "debug")
        model.eval()
        B = 2
        x = torch.randint(0, cfg.vocab_size, (B, cfg.seq_len))
        with torch.no_grad():
            logits, loss, _ = model(x)
        assert logits.shape == (B, cfg.seq_len, cfg.vocab_size)


class TestLinformerCompile:
    """torch.compile compatibility tests."""

    def test_torch_compile_no_graph_breaks(self, config):
        """LinformerAttention forward produces no graph breaks under torch.compile."""
        attn = LinformerAttention(config)
        attn.eval()

        compiled_attn = torch.compile(attn, fullgraph=True)
        x = torch.randn(1, 64, config.d_model)
        with torch.no_grad():
            output, kv_out = compiled_attn(x)
        assert output.shape == (1, 64, config.d_model)
        assert kv_out is None


class TestLinformerBfloat16:
    """bfloat16 precision tests."""

    def test_bfloat16_forward_finite(self, config):
        """LinformerAttention in bfloat16 produces finite outputs."""
        attn = LinformerAttention(config).to(torch.bfloat16)
        attn.eval()
        x = torch.randn(2, 64, config.d_model, dtype=torch.bfloat16)
        with torch.no_grad():
            output, _ = attn(x)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
