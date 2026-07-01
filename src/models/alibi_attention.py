"""ALiBi Attention — causal self-attention with linear position biases.

Thin adapter extending FlashAttentionBase. Implements Attention with Linear
Biases (Press et al., "Train Short, Test Long"). Instead of encoding position
via RoPE, ALiBi adds fixed per-head linear biases directly to attention logits
through the flash_attn kernel's native alibi_slopes parameter.

The external interface is unchanged:
    forward(x, kv_cache=None) -> (output, new_kv_cache)
"""

from __future__ import annotations

from typing import Any

import torch

from src.models.config import ModelConfig
from src.models.flash_attention_base import FlashAttentionBase


class ALiBiAttention(FlashAttentionBase):
    """Multi-head causal self-attention with ALiBi position biases.

    Replaces RoPE with per-head linear biases passed to the flash_attn kernel.
    Q and K are raw projections — no rotation applied.

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)  # n_kv_head defaults to n_head (standard MHA)

        # ALiBi slopes: geometric series 2^(-8*(i+1)/n_head)
        # Registered as a buffer so it moves to the correct device with the model.
        slopes = self._compute_alibi_slopes(config.n_head)
        self.register_buffer("alibi_slopes", slopes)  # (n_head,)

    @staticmethod
    def _compute_alibi_slopes(n_head: int) -> torch.Tensor:
        """Compute ALiBi slopes following geometric series.

        slopes[i] = 2^(-8 * (i+1) / n_head) for i in range(n_head)

        Returns:
            Tensor of shape (n_head,) with dtype float32.
        """
        exponents = -8.0 * torch.arange(1, n_head + 1, dtype=torch.float32) / n_head
        return torch.pow(2.0, exponents)

    def _apply_position(
        self, q: torch.Tensor, k: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q and K unchanged — ALiBi encodes position via kernel biases.

        Args:
            q: Query tensor of shape (B, T, n_head, d_head).
            k: Key tensor of shape (B, T, n_kv_head, d_head).
            offset: Position offset (unused for ALiBi).

        Returns:
            Tuple of (q, k) unchanged.
        """
        return q, k

    def _extra_attn_kwargs(self) -> dict[str, Any]:
        """Return alibi_slopes for the flash_attn kernel.

        The kernel requires slopes in float32.
        """
        return {"alibi_slopes": self.alibi_slopes.float()}
