"""Shared future-token leakage tests for every attention implementation."""

import pytest
import torch

from src.models.alibi_attention import ALiBiAttention
from src.models.attention import CausalSelfAttention
from src.models.config import ModelConfig
from src.models.flash_attention import FlashAttention
from src.models.gqa_attention import GQAAttention
from src.models.linear_attention import CausalLinearAttention
from src.models.modern_attention import ModernAttention


def _assert_suffix_independence(module, *, device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(123)
    module = module.to(device=device, dtype=dtype).eval()
    x1 = torch.randn(1, 12, 64, device=device, dtype=dtype)
    x2 = x1.clone()
    x2[:, 6:, :] = torch.randn_like(x2[:, 6:, :])

    with torch.no_grad():
        y1, _ = module(x1)
        y2, _ = module(x2)

    torch.testing.assert_close(y1[:, :6], y2[:, :6], atol=0.0, rtol=0.0)
    assert not torch.equal(y1[:, 6:], y2[:, 6:])


@pytest.mark.parametrize(
    ("attention_class", "config_overrides"),
    [
        pytest.param(CausalSelfAttention, {}, id="v0-vanilla"),
        pytest.param(ModernAttention, {}, id="v1-modern-and-v6-moe"),
        pytest.param(CausalLinearAttention, {}, id="v5-causal-linear"),
    ],
)
def test_cpu_attention_paths_are_causal(attention_class, config_overrides):
    config = ModelConfig(
        n_layer=1,
        d_model=64,
        n_head=4,
        vocab_size=128,
        seq_len=16,
        dropout=0.0,
        bias=False,
        **config_overrides,
    )
    _assert_suffix_independence(attention_class(config), device="cpu", dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("attention_class", "config_overrides"),
    [
        pytest.param(ALiBiAttention, {}, id="v2-alibi"),
        pytest.param(GQAAttention, {"n_kv_head": 1}, id="v3-gqa"),
        pytest.param(FlashAttention, {"window_size": 4}, id="v4-swa"),
        pytest.param(
            FlashAttention,
            {"window_size": None},
            id="v4-interleaved-full-layers",
        ),
    ],
)
def test_flash_attention_paths_are_causal(attention_class, config_overrides):
    pytest.importorskip("flash_attn")
    config = ModelConfig(
        n_layer=1,
        d_model=64,
        n_head=4,
        vocab_size=128,
        seq_len=16,
        dropout=0.0,
        bias=False,
        **config_overrides,
    )
    _assert_suffix_independence(attention_class(config), device="cuda", dtype=torch.bfloat16)
