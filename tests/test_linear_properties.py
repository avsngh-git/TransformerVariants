"""Property-based tests for LinformerAttention using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs.
"""

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from src.models.config import ModelConfig
from src.models.linear_attention import LinformerAttention


# --------------------------------------------------------------------------
# Shared debug-scale config factory
# --------------------------------------------------------------------------

def _make_config(
    d_model: int = 64,
    n_head: int = 4,
    seq_len: int = 64,
    projection_rank: int = 16,
    n_layer: int = 2,
    dropout: float = 0.0,
) -> ModelConfig:
    """Create a debug-scale Linformer config for testing."""
    return ModelConfig(
        d_model=d_model,
        n_head=n_head,
        seq_len=seq_len,
        projection_rank=projection_rank,
        n_layer=n_layer,
        dropout=dropout,
        bias=False,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
    )


# --------------------------------------------------------------------------
# Property 1: Output shape and interface contract
# Validates: Requirements 1.1, 2.3, 2.4, 2.6, 4.3
# --------------------------------------------------------------------------


class TestOutputShapeContract:
    """**Validates: Requirements 1.1, 2.3, 2.4, 2.6, 4.3**"""

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_forward_returns_correct_shape_and_none(self, batch, seq_len, seed):
        """Forward returns 2-tuple of (Tensor[B, T, d_model], None) for all valid inputs."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = min(seq_len, config.seq_len)  # Ensure T <= seq_len
        x = torch.randn(batch, T, config.d_model)

        with torch.no_grad():
            result = attn(x)

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, kv_out = result
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch, T, config.d_model)
        assert kv_out is None


# --------------------------------------------------------------------------
# Property 2: Numerical stability — finite outputs
# Validates: Requirements 7.1, 7.2
# --------------------------------------------------------------------------


class TestNumericalStability:
    """**Validates: Requirements 7.1, 7.2**"""

    @given(
        batch=st.integers(min_value=1, max_value=3),
        seq_len=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_finite_outputs_float32(self, batch, seq_len, seed):
        """Inputs in [-10, 10] produce only finite outputs in float32."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = min(seq_len, config.seq_len)
        x = torch.empty(batch, T, config.d_model).uniform_(-10, 10)

        with torch.no_grad():
            output, _ = attn(x)

        assert torch.isfinite(output).all(), "Non-finite values in float32 output"

    @given(
        batch=st.integers(min_value=1, max_value=3),
        seq_len=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_finite_outputs_bfloat16(self, batch, seq_len, seed):
        """Inputs in [-10, 10] produce only finite outputs in bfloat16."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config).to(torch.bfloat16)
        attn.eval()

        T = min(seq_len, config.seq_len)
        x = torch.empty(batch, T, config.d_model, dtype=torch.bfloat16).uniform_(-10, 10)

        with torch.no_grad():
            output, _ = attn(x)

        assert torch.isfinite(output).all(), "Non-finite values in bfloat16 output"


# --------------------------------------------------------------------------
# Property 3: Batch independence
# Validates: Requirements 8.1, 8.2
# --------------------------------------------------------------------------


class TestBatchIndependence:
    """**Validates: Requirements 8.1, 8.2**"""

    @given(
        batch=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_batch_permutation_equivariance(self, batch, seq_len, seed):
        """Permuting batch dim of input produces corresponding permutation of output."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = min(seq_len, config.seq_len)
        x = torch.randn(batch, T, config.d_model)

        # Create a random permutation of the batch dimension
        perm = torch.randperm(batch)

        with torch.no_grad():
            output_original, _ = attn(x)
            output_permuted, _ = attn(x[perm])

        # Output of permuted input should equal permuted output
        torch.testing.assert_close(
            output_permuted,
            output_original[perm],
            atol=1e-5,
            rtol=1e-5,
        )


# --------------------------------------------------------------------------
# Property 4: Attention weights sum to one
# Validates: Requirements 4.1, 4.2
# --------------------------------------------------------------------------


class TestAttentionWeightsSumToOne:
    """**Validates: Requirements 4.1, 4.2**"""

    @given(
        batch=st.integers(min_value=1, max_value=3),
        seq_len=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_attention_weights_sum_to_one(self, batch, seq_len, seed):
        """Attention weights (post-softmax) sum to 1.0 along the projected key dimension."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = min(seq_len, config.seq_len)
        x = torch.randn(batch, T, config.d_model)

        # Hook into attn_dropout to capture weights passing through it.
        # Since dropout=0.0, the output equals the softmax weights.
        captured_weights = []

        def hook_fn(module, input, output):
            # input[0] is the weights tensor passed to dropout
            captured_weights.append(input[0].detach().clone())

        handle = attn.attn_dropout.register_forward_hook(hook_fn)

        with torch.no_grad():
            attn(x)

        handle.remove()

        assert len(captured_weights) == 1, "Expected exactly one forward pass through attn_dropout"
        weights = captured_weights[0]  # (B, H, T, r)

        # Each row should sum to 1.0 (softmax output)
        row_sums = weights.sum(dim=-1)  # (B, H, T)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        )


# --------------------------------------------------------------------------
# Property 5: Position sensitivity via RoPE
# Validates: Requirements 3.2, 3.3
# --------------------------------------------------------------------------


class TestPositionSensitivity:
    """**Validates: Requirements 3.2, 3.3**"""

    @given(
        seq_len=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_position_shift_produces_different_output(self, seq_len, seed):
        """Outputs differ when input content is shifted to different positions (RoPE encodes position)."""
        config = _make_config()
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = min(seq_len, config.seq_len)

        # Create a content vector and place it at two different positions
        # input1: [content, zeros, zeros, ...]
        # input2: [zeros, content, zeros, ...]
        content = torch.randn(1, 1, config.d_model)

        input1 = torch.zeros(1, T, config.d_model)
        input1[:, 0, :] = content[:, 0, :]

        input2 = torch.zeros(1, T, config.d_model)
        input2[:, 1, :] = content[:, 0, :]

        with torch.no_grad():
            output1, _ = attn(input1)
            output2, _ = attn(input2)

        # Outputs should differ because RoPE encodes different positions
        assert not torch.allclose(output1, output2, atol=1e-6), (
            "Outputs are identical despite position shift — RoPE not effective"
        )


# --------------------------------------------------------------------------
# Property 6: Projection rank shape invariant
# Validates: Requirements 1.3, 2.1, 2.2, 3.1
# --------------------------------------------------------------------------


class TestProjectionRankShapeInvariant:
    """**Validates: Requirements 1.3, 2.1, 2.2, 3.1**"""

    @given(
        seq_len=st.integers(min_value=8, max_value=128),
        rank_fraction=st.floats(min_value=0.1, max_value=1.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_shapes_match_config(self, seq_len, rank_fraction, seed):
        """E, F, RoPE buffers, and linear layers have shapes matching the config."""
        # Derive a valid projection_rank from the fraction
        projection_rank = max(1, int(seq_len * rank_fraction))
        projection_rank = min(projection_rank, seq_len)  # Clamp to valid range

        d_model = 64
        n_head = 4
        d_head = d_model // n_head

        config = _make_config(
            d_model=d_model,
            n_head=n_head,
            seq_len=seq_len,
            projection_rank=projection_rank,
        )

        torch.manual_seed(seed)
        attn = LinformerAttention(config)

        # E and F shapes
        assert attn.E.shape == (seq_len, projection_rank), (
            f"E shape {attn.E.shape} != ({seq_len}, {projection_rank})"
        )
        assert attn.F.shape == (seq_len, projection_rank), (
            f"F shape {attn.F.shape} != ({seq_len}, {projection_rank})"
        )

        # RoPE buffer shapes: (seq_len, d_head // 2)
        assert attn.rope_cos.shape == (seq_len, d_head // 2), (
            f"rope_cos shape {attn.rope_cos.shape} != ({seq_len}, {d_head // 2})"
        )
        assert attn.rope_sin.shape == (seq_len, d_head // 2), (
            f"rope_sin shape {attn.rope_sin.shape} != ({seq_len}, {d_head // 2})"
        )

        # Linear layer shapes
        assert attn.q_proj.weight.shape == (d_model, d_model)
        assert attn.k_proj.weight.shape == (d_model, d_model)
        assert attn.v_proj.weight.shape == (d_model, d_model)
        assert attn.out_proj.weight.shape == (d_model, d_model)

        # No bias
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.v_proj.bias is None
        assert attn.out_proj.bias is None


# --------------------------------------------------------------------------
# Property 7: Invalid configuration rejection
# Validates: Requirements 5.2, 5.3, 5.4, 9.1
# --------------------------------------------------------------------------


class TestInvalidConfigRejection:
    """**Validates: Requirements 5.2, 5.3, 5.4, 9.1**"""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=50)
    def test_none_projection_rank_raises(self, seed):
        """projection_rank=None raises ValueError on construction."""
        config = _make_config(projection_rank=16)
        # Override projection_rank to None after construction (bypass ModelConfig)
        config.projection_rank = None

        with pytest.raises(ValueError, match="projection_rank must be set"):
            LinformerAttention(config)

    @given(
        rank=st.integers(min_value=-100, max_value=0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_non_positive_projection_rank_raises(self, rank, seed):
        """projection_rank <= 0 raises ValueError on construction."""
        config = _make_config(projection_rank=16)
        config.projection_rank = rank

        with pytest.raises(ValueError, match="projection_rank must be > 0"):
            LinformerAttention(config)

    @given(
        extra=st.integers(min_value=1, max_value=100),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_projection_rank_exceeds_seq_len_raises(self, extra, seed):
        """projection_rank > seq_len raises ValueError on construction."""
        seq_len = 64
        config = _make_config(seq_len=seq_len, projection_rank=16)
        config.projection_rank = seq_len + extra

        with pytest.raises(ValueError, match="projection_rank.*must be <= seq_len"):
            LinformerAttention(config)

    @given(
        extra=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50)
    def test_input_exceeding_seq_len_raises(self, extra, seed):
        """Input with T > seq_len raises AssertionError in forward."""
        config = _make_config(seq_len=64, projection_rank=16)
        torch.manual_seed(seed)
        attn = LinformerAttention(config)
        attn.eval()

        T = config.seq_len + extra
        x = torch.randn(1, T, config.d_model)

        with pytest.raises(AssertionError, match="exceeds config.seq_len"):
            with torch.no_grad():
                attn(x)


# Need pytest for raises
import pytest
