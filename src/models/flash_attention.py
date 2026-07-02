"""FlashAttention — standard MHA with RoPE, backed by flash_attn.

Thin adapter extending FlashAttentionBase. Implements RoPE position encoding
via the _apply_position hook and optionally passes alibi_slopes to the kernel.

The external interface is unchanged:
    forward(x, kv_cache=None) -> (output, new_kv_cache)
"""

from __future__ import annotations

from typing import Any

import torch

from src.models.config import ModelConfig
from src.models.flash_attention_base import FlashAttentionBase
from src.models.rope import precompute_rope_frequencies, apply_rope


class FlashAttention(FlashAttentionBase):
    """Multi-head causal self-attention with RoPE using flash_attn kernels.

    Drop-in replacement for ModernAttention — same interface, same math,
    different kernel dispatch.

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
        alibi_slopes: Optional tensor of shape (n_head,) for ALiBi biases.
            When None, standard RoPE attention is used.
    """

    def __init__(self, config: ModelConfig, alibi_slopes: torch.Tensor | None = None) -> None:
        super().__init__(config)  # n_kv_head defaults to n_head (standard MHA)

        # Precompute RoPE frequency buffers (registered as non-learnable buffers)
        cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
        self.register_buffer("rope_cos", cos)  # (seq_len, d_head // 2)
        self.register_buffer("rope_sin", sin)  # (seq_len, d_head // 2)

        # Store alibi_slopes as a buffer if provided (backward compat)
        if alibi_slopes is not None:
            self.register_buffer("alibi_slopes", alibi_slopes.float())
        else:
            self.alibi_slopes = None

        self._window_size = config.window_size

    def _apply_position(
        self, q: torch.Tensor, k: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE at positions [offset, offset+T).

        Base provides tensors in (B, T, n_head, d_head) layout.
        apply_rope expects (B, n_head, T, d_head), so we transpose around the call.
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

    def _extra_training_kwargs(self) -> dict[str, Any]:
        """Return training-path kwargs: alibi_slopes + window_size when configured."""
        kwargs = self._extra_attn_kwargs()  # gets alibi_slopes if present
        if self._window_size is not None:
            kwargs["window_size"] = (self._window_size, self._window_size)
        return kwargs

    def _extra_attn_kwargs(self) -> dict[str, Any]:
        """Return alibi_slopes if configured, otherwise empty dict."""
        if self.alibi_slopes is not None:
            return {"alibi_slopes": self.alibi_slopes}
        return {}
