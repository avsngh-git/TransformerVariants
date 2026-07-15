"""Unit tests for causal linear attention."""

import pytest
import torch

from src.models import registry
from src.models.config import ModelConfig
from src.models.linear_attention import CausalLinearAttention, feature_map
from src.models.rope import apply_rope


@pytest.fixture
def config():
    """Debug-scale config for causal linear attention."""
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
        dropout=0.0,
        bias=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_feature_map_large_positive_backward_is_finite(dtype):
    """ELU+1 must remain positive and finite through both autograd branches."""
    x = torch.tensor([90.0], dtype=dtype, requires_grad=True)

    feature_map(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad.float(), torch.ones(1))


def _published_roformer_linear_reference(
    attn: CausalLinearAttention, x: torch.Tensor
) -> torch.Tensor:
    """Dense reference for RoFormer linear-attention equation 19."""
    batch, length, channels = x.shape
    q = attn.q_proj(x).view(batch, length, attn.n_head, attn.d_head).transpose(1, 2)
    k = attn.k_proj(x).view(batch, length, attn.n_head, attn.d_head).transpose(1, 2)
    v = attn.v_proj(x).view(batch, length, attn.n_head, attn.d_head).transpose(1, 2)

    phi_q = feature_map(q)
    phi_k = feature_map(k)
    rotated_phi_q = apply_rope(phi_q, attn.rope_cos[:length], attn.rope_sin[:length])
    rotated_phi_k = apply_rope(phi_k, attn.rope_cos[:length], attn.rope_sin[:length])

    numerator_scores = torch.einsum("bhld,bhmd->bhlm", rotated_phi_q, rotated_phi_k)
    denominator_scores = torch.einsum("bhld,bhmd->bhlm", phi_q, phi_k)
    mask = torch.ones(length, length, dtype=torch.bool).tril()
    numerator_scores = numerator_scores.masked_fill(~mask, 0.0)
    denominator_scores = denominator_scores.masked_fill(~mask, 0.0)

    numerator = torch.matmul(numerator_scores, v)
    denominator = denominator_scores.sum(dim=-1).clamp_min(1e-6).unsqueeze(-1)
    output = (numerator / denominator).transpose(1, 2).contiguous().view(
        batch, length, channels
    )
    return attn.out_proj(output)


class TestCausalLinearAttentionConstructor:
    """Tests for CausalLinearAttention.__init__."""

    def test_creates_expected_layers(self, config):
        """Constructor creates projections, RoPE, and a chunk causal mask."""
        attn = CausalLinearAttention(config)
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "out_proj")
        assert hasattr(attn, "rope_cos")
        assert hasattr(attn, "rope_sin")
        assert attn.rope_cos.shape[0] == config.seq_len
        assert attn.rope_sin.shape[0] == config.seq_len
        assert attn.chunk_causal_mask.shape == (64, 64)
        assert torch.equal(
            attn.chunk_causal_mask,
            torch.ones_like(attn.chunk_causal_mask).tril(),
        )


class TestCausalLinearAttentionForward:
    """Tests for CausalLinearAttention.forward."""

    def test_future_tokens_do_not_change_prefix_outputs(self, config):
        """Decoder attention must not leak information from future positions."""
        torch.manual_seed(0)
        attn = CausalLinearAttention(config)
        attn.eval()

        # Cross the 64-token chunk boundary to exercise both causal paths.
        prefix_len = 70
        x1 = torch.randn(1, 96, config.d_model)
        x2 = x1.clone()
        x2[:, prefix_len:, :] = torch.randn_like(x2[:, prefix_len:, :])

        with torch.no_grad():
            out1, _ = attn(x1)
            out2, _ = attn(x2)

        assert torch.allclose(
            out1[:, :prefix_len, :],
            out2[:, :prefix_len, :],
            atol=1e-6,
            rtol=1e-5,
        )

    def test_kv_cache_raises_not_implemented(self, config):
        """Non-None kv_cache raises NotImplementedError with expected message."""
        attn = CausalLinearAttention(config)
        x = torch.randn(2, 8, config.d_model)
        with pytest.raises(NotImplementedError, match="Recurrent generation state"):
            attn(x, kv_cache=torch.zeros(1))

    def test_output_shape_full_seq_len(self, config):
        """Forward returns (B, seq_len, d_model) output and None for full seq_len."""
        attn = CausalLinearAttention(config)
        B = 2
        x = torch.randn(B, config.seq_len, config.d_model)
        output, kv_out = attn(x)
        assert output.shape == (B, config.seq_len, config.d_model)
        assert kv_out is None

    def test_output_shape_variable_seq_len(self, config):
        """Forward returns correct shape for T < seq_len."""
        attn = CausalLinearAttention(config)
        B, T = 2, 128
        x = torch.randn(B, T, config.d_model)
        output, kv_out = attn(x)
        assert output.shape == (B, T, config.d_model)
        assert kv_out is None

    def test_matches_published_roformer_linear_equation(self, config):
        """Chunked V5 matches RoFormer equation 19 with an unrotated denominator."""
        torch.manual_seed(7)
        attn = CausalLinearAttention(config).eval()
        x = torch.randn(1, 70, config.d_model)

        with torch.no_grad():
            actual, _ = attn(x)
            expected = _published_roformer_linear_reference(attn, x)

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class TestCausalLinearFullModel:
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


class TestCausalLinearCompile:
    """torch.compile compatibility tests."""

    def test_torch_compile_no_graph_breaks(self, config):
        """CausalLinearAttention forward produces no graph breaks under torch.compile."""
        attn = CausalLinearAttention(config)
        attn.eval()

        compiled_attn = torch.compile(attn, fullgraph=True)
        x = torch.randn(1, 128, config.d_model)
        with torch.no_grad():
            output, kv_out = compiled_attn(x)
        assert output.shape == (1, 128, config.d_model)
        assert kv_out is None


class TestCausalLinearBfloat16:
    """bfloat16 precision tests."""

    def test_bfloat16_forward_finite(self, config):
        """CausalLinearAttention in bfloat16 produces finite outputs."""
        attn = CausalLinearAttention(config).to(torch.bfloat16)
        attn.eval()
        x = torch.randn(2, 64, config.d_model, dtype=torch.bfloat16)
        with torch.no_grad():
            output, _ = attn(x)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()


    def test_bfloat16_autocast_backward_finite(self, config):
        """FP32 recurrence reductions keep a BF16-autocast backward pass finite."""
        attn = CausalLinearAttention(config)
        x = torch.randn(1, 70, config.d_model, requires_grad=True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output, _ = attn(x)
            loss = output.square().mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in attn.parameters()
        )
