"""Attention module protocol — the formal contract all attention variants satisfy.

Every attention module in this project (CausalSelfAttention, ModernAttention,
FlashAttention, ALiBiAttention, GQAAttention, CausalLinearAttention) implements
this interface via duck typing. This Protocol formalizes the informal contract
for discoverability and type-checking.

The KV-cache type is deliberately `Any` — different backends use different
cache shapes (tuple[Tensor, Tensor] for SDPA, tuple[Tensor, Tensor, Tensor]
for flash_attn, None for causal linear attention). Callers pass cache opaquely; they don't
inspect or construct it. See CONTEXT.md "KV-Cache unification" open question.

Usage:
    This Protocol is used as a type annotation in ModernTransformerBlock and
    TransformerBlock to document the attention_class constructor parameter.
    It is NOT enforced at runtime — it's a structural (duck) typing contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class AttentionModule(Protocol):
    """Protocol for all attention modules in the Transformer Variant Lab.

    Every attention variant satisfies this interface:
    - Accepts input tensor x of shape (B, T, d_model)
    - Optionally accepts a KV-cache (shape varies by backend)
    - Returns (output, new_kv_cache) where output has shape (B, T, d_model)

    Implementations:
        - CausalSelfAttention (V0): manual matmul, learned position encoding
        - ModernAttention (V1 SDPA): RoPE + PyTorch scaled_dot_product_attention
        - FlashAttention (V1/V4): RoPE + flash_attn kernels, optional window_size
        - ALiBiAttention (V2): no position rotation, alibi_slopes via kernel
        - GQAAttention (V3): grouped KV heads, separate Q/KV projections
        - CausalLinearAttention (V5): ELU+1 prefix-state attention, returns None cache
    """

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """Run attention on input, optionally using and updating KV-cache.

        Args:
            x: Input tensor from the residual stream, shape (B, T, d_model).
            kv_cache: Optional KV-cache from a previous forward pass.
                Shape varies by backend — callers treat this as opaque.
                None during training (teacher forcing processes all positions).

        Returns:
            Tuple of (output, new_kv_cache):
            - output: Attention output, shape (B, T, d_model).
            - new_kv_cache: Updated cache for this layer (or None for V5/training).
        """
        ...
