"""Grouped-Query Attention (GQA) — thin adapter extending FlashAttentionBase.

GQA reduces the number of key-value heads relative to query heads, so that
groups of query heads share a single KV head. This reduces parameter count
and KV-cache memory without significantly impacting model quality.

The adapter overrides _project_qkv to use separate Q and KV projections
(unlike the base class's combined qkv_proj), and implements RoPE via the
_apply_position hook.

The external interface is unchanged:
    forward(x, kv_cache=None) -> (output, new_kv_cache)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.flash_attention_base import FlashAttentionBase
from src.models.rope import precompute_rope_frequencies, apply_rope


class GQAAttention(FlashAttentionBase):
    """Grouped-Query Attention using flash_attn kernels.

    Reduces KV heads relative to Q heads. Each KV head serves a group
    of query heads. The flash_attn library handles the head ratio natively.

    Args:
        config: ModelConfig with d_model, n_head, n_kv_head, dropout, bias, seq_len.
            n_kv_head must be set (not None) and must divide n_head evenly.
    """

    def __init__(self, config: ModelConfig) -> None:
        # Validate n_kv_head BEFORE calling super().__init__
        if config.n_kv_head is None:
            raise ValueError(
                "config.n_kv_head must be set for GQAAttention (not None). "
                "Set n_kv_head to a divisor of n_head (e.g., n_head // 4)."
            )

        if config.n_head % config.n_kv_head != 0:
            raise ValueError(
                f"n_head ({config.n_head}) must be divisible by n_kv_head "
                f"({config.n_kv_head}) for grouped-query attention."
            )

        # Base class creates qkv_proj, out_proj, resid_dropout, and caches flash_attn funcs
        super().__init__(config, n_kv_head=config.n_kv_head)

        # GQA uses separate projections — remove the base class's combined qkv_proj
        del self.qkv_proj

        # Q projection: d_model → n_head * d_head
        self.q_proj = nn.Linear(config.d_model, config.n_head * config.d_head, bias=config.bias)

        # KV projection (combined K and V): d_model → 2 * n_kv_head * d_head
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_head * config.d_head, bias=config.bias)

        # Precompute RoPE frequency buffers (registered as non-learnable buffers)
        cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
        self.register_buffer("rope_cos", cos)  # (seq_len, d_head // 2)
        self.register_buffer("rope_sin", sin)  # (seq_len, d_head // 2)

    def _project_qkv(
        self, x: torch.Tensor, B: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input using separate Q and KV projections.

        GQA has asymmetric head counts: Q uses n_head, K/V use n_kv_head.

        Args:
            x: Input tensor of shape (B, T, d_model).
            B: Batch size.
            T: Sequence length.

        Returns:
            Tuple of (q, k, v) where:
                q: shape (B, T, n_head, d_head)
                k: shape (B, T, n_kv_head, d_head)
                v: shape (B, T, n_kv_head, d_head)
        """
        # Q: (B, T, d_model) → (B, T, n_head * d_head) → (B, T, n_head, d_head)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head)

        # KV: (B, T, d_model) → (B, T, 2 * n_kv_head * d_head)
        kv = self.kv_proj(x)
        # Split into K and V, each (B, T, n_kv_head * d_head) → (B, T, n_kv_head, d_head)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.d_head)
        v = v.view(B, T, self.n_kv_head, self.d_head)

        return q, k, v

    def _apply_position(
        self, q: torch.Tensor, k: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE at positions [offset, offset+T).

        Base provides tensors in (B, T, n_head, d_head) layout.
        apply_rope expects (B, n_head, T, d_head), so we transpose around the call.
        Works for both n_head (Q) and n_kv_head (K) head counts.
        """
        T = q.size(1)

        # Transpose to (B, n_head, T, d_head) for RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Slice precomputed cos/sin for correct positions
        rope_cos = self.rope_cos[offset:offset + T]
        rope_sin = self.rope_sin[offset:offset + T]

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Transpose back to (B, T, n_head, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        return q, k
