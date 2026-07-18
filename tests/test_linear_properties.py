"""Property-based tests for CausalLinearAttention module.

Uses Hypothesis to verify universal properties hold across many random inputs.
Tests run in float32 on CPU with max_examples=100.
"""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.models.config import ModelConfig
from src.models.linear_attention import CausalLinearAttention, feature_map

# --- Property 1: Feature map positivity and shape preservation ---
# Feature: linear-attention, Property 1: Feature map positivity and shape preservation


@given(
    ndims=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
@settings(max_examples=100)
def test_feature_map_positivity_and_shape(ndims, data):
    """φ(x) = ELU(x) + 1 always outputs > 0 and preserves shape.

    **Validates: Requirements 1.3, 9.2, 10.1, 10.2, 10.3**
    """
    # Generate a random shape with 1-4 dimensions, each dim size 1-16
    shape = tuple(data.draw(st.integers(min_value=1, max_value=16)) for _ in range(ndims))
    # Generate random values in [-100, 100]
    values = data.draw(
        arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    x = torch.from_numpy(values)

    result = feature_map(x)

    # All elements strictly > 0
    assert (result > 0).all(), f"Found non-positive elements: min={result.min().item()}"
    # Shape preserved
    assert result.shape == x.shape, f"Shape mismatch: input={x.shape}, output={result.shape}"


# --- Property 2: Feature map range for negative inputs ---
# Feature: linear-attention, Property 2: Feature map range for negative inputs


@given(
    x_val=st.floats(min_value=-100.0, max_value=-1e-7, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_feature_map_negative_range(x_val):
    """For x < 0, φ(x) is in the open interval (0, 1).

    **Validates: Requirements 10.4**
    """
    x = torch.tensor([x_val], dtype=torch.float32)
    result = feature_map(x)

    value = result.item()
    assert value > 0, f"φ({x_val}) = {value} is not > 0"
    assert value < 1, f"φ({x_val}) = {value} is not < 1"


# --- Property 3: Causal ordering independence ---
# Feature: linear-attention, Property 3: Causal ordering independence


@given(
    B=st.integers(min_value=1, max_value=2),
    T=st.integers(min_value=3, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_causal_ordering_independence(B, T, seed):
    """Outputs at positions 0..t-1 are identical regardless of changes at position t+.

    **Validates: Requirements 2.1, 2.2**
    """
    d_model = 64
    n_head = 4

    config = ModelConfig(
        n_layer=1,
        d_model=d_model,
        n_head=n_head,
        seq_len=T,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        dropout=0.0,
        bias=False,
    )

    torch.manual_seed(seed)
    attn = CausalLinearAttention(config)
    attn.eval()

    # Generate two inputs identical at 0..t-1, different at t+
    t = seed % (T - 1) + 1  # split position in [1, T-1]
    t = min(t, T - 1)

    torch.manual_seed(seed + 1000)
    input_a = torch.randn(B, T, d_model)
    input_b = input_a.clone()
    # Modify positions t onward in input_b
    input_b[:, t:, :] = torch.randn(B, T - t, d_model)

    with torch.no_grad():
        output_a, _ = attn(input_a)
        output_b, _ = attn(input_b)

    # Outputs at positions 0..t-1 must be identical
    assert torch.equal(output_a[:, :t, :], output_b[:, :t, :]), (
        f"Causal violation: outputs differ at positions before t={t}"
    )


# --- Property 4: Numerical stability — finite outputs ---
# Feature: linear-attention, Property 4: Numerical stability — finite outputs


@given(
    B=st.integers(min_value=1, max_value=3),
    T=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_numerical_stability_finite_outputs(B, T, seed):
    """CausalLinearAttention produces finite outputs for inputs in [-10, 10].

    **Validates: Requirements 1.6, 9.1, 9.3**
    """
    d_model = 64
    n_head = 4

    config = ModelConfig(
        n_layer=1,
        d_model=d_model,
        n_head=n_head,
        seq_len=max(T, 1),
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        dropout=0.0,
        bias=False,
    )

    torch.manual_seed(seed)
    attn = CausalLinearAttention(config)
    attn.eval()

    # Generate input with values in [-10, 10]
    torch.manual_seed(seed + 42)
    x = torch.empty(B, T, d_model).uniform_(-10.0, 10.0)

    with torch.no_grad():
        output, _ = attn(x)

    assert torch.isfinite(output).all(), (
        f"Non-finite values found in output: NaN count={torch.isnan(output).sum().item()}, "
        f"Inf count={torch.isinf(output).sum().item()}"
    )


# --- Property 5: Output shape and interface contract ---
# Feature: linear-attention, Property 5: Output shape and interface contract


@given(
    B=st.integers(min_value=1, max_value=4),
    T=st.integers(min_value=1, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_output_shape_and_interface_contract(B, T, seed):
    """Output shape is fixed and evaluation returns a reusable recurrent state.

    **Validates: Requirements 1.7, 1.9, 4.1**
    """
    d_model = 64
    n_head = 4

    config = ModelConfig(
        n_layer=1,
        d_model=d_model,
        n_head=n_head,
        seq_len=max(T, 1),
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        dropout=0.0,
        bias=False,
    )

    torch.manual_seed(seed)
    attn = CausalLinearAttention(config)
    attn.eval()

    x = torch.randn(B, T, d_model)

    with torch.no_grad():
        result = attn(x)

    # Must return a tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    output, kv_cache_out = result

    # Output shape is exactly (B, T, d_model)
    assert output.shape == (B, T, d_model), (
        f"Expected shape ({B}, {T}, {d_model}), got {output.shape}"
    )

    assert kv_cache_out is not None
    numerator_state, denominator_state, position = kv_cache_out
    d_head = d_model // n_head
    assert numerator_state.shape == (B, n_head, d_head, d_head)
    assert denominator_state.shape == (B, n_head, d_head)
    assert position == T
