"""Grouped-Query Attention (GQA) module using Dao AI Lab flash_attn library.

GQA reduces the number of key-value heads relative to query heads, so that
groups of query heads share a single KV head. This reduces parameter count
and KV-cache memory without significantly impacting model quality.

The module uses separate Q and KV projections (unlike FlashAttention's combined
qkv_proj) because Q projects to n_head * d_head while K and V project to
n_kv_head * d_head.

The flash_attn library natively handles the head count mismatch — no manual
head repetition (torch.repeat_interleave) is needed.

Tensor layout note:
    flash_attn expects (B, T, n_head, d_head) — NOT (B, n_head, T, d_head).
    We transpose to (B, n_head, T, d_head) only for RoPE application (which
    expects that layout), then transpose back before calling flash_attn kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.rope import precompute_rope_frequencies, apply_rope


def _check_flash_attn_available() -> None:
    """Raise ImportError with helpful message if flash_attn is not installed."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise ImportError(
            "flash_attn is required for GQAAttention backend but is not installed. "
            "Install it with: pip install flash-attn --no-build-isolation\n"
            "See https://github.com/Dao-AILab/flash-attention for details."
        )


class GQAAttention(nn.Module):
    """Grouped-Query Attention using flash_attn library.

    Reduces KV heads relative to Q heads. Each KV head serves a group
    of query heads. The flash_attn library handles the head ratio natively.

    Args:
        config: ModelConfig with d_model, n_head, n_kv_head, dropout, bias, seq_len.
            n_kv_head must be set (not None).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Validate n_kv_head is set
        if config.n_kv_head is None:
            raise ValueError(
                "config.n_kv_head must be set for GQAAttention (not None). "
                "Set n_kv_head to a divisor of n_head (e.g., n_head // 4)."
            )

        # Validate n_head is divisible by n_kv_head
        if config.n_head % config.n_kv_head != 0:
            raise ValueError(
                f"n_head ({config.n_head}) must be divisible by n_kv_head "
                f"({config.n_kv_head}) for grouped-query attention."
            )

        # Verify flash_attn is available
        _check_flash_attn_available()

        # Import flash_attn functions (lazy — only when class is instantiated)
        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        self._flash_attn_func = flash_attn_func
        self._flash_attn_with_kvcache = flash_attn_with_kvcache

        # Store config values
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.attn_dropout = config.dropout

        # Q projection: d_model → n_head * d_head
        self.q_proj = nn.Linear(config.d_model, config.n_head * config.d_head, bias=config.bias)

        # KV projection (combined K and V): d_model → 2 * n_kv_head * d_head
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_head * config.d_head, bias=config.bias)

        # Output projection: n_head * d_head → d_model
        self.out_proj = nn.Linear(config.n_head * config.d_head, config.d_model, bias=config.bias)

        # Residual dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequency buffers (registered as non-learnable buffers)
        cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
        self.register_buffer("rope_cos", cos)  # (seq_len, d_head // 2)
        self.register_buffer("rope_sin", sin)  # (seq_len, d_head // 2)

    def allocate_kv_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate pre-sized KV-cache tensors for generation.

        Args:
            batch_size: Number of sequences in the batch.
            max_seqlen: Maximum sequence length to allocate for.
            dtype: Data type for cache tensors.
            device: Device to allocate on.

        Returns:
            Tuple of (k_cache, v_cache, cache_seqlens) where:
                k_cache: shape (B, max_seqlen, n_kv_head, d_head)
                v_cache: shape (B, max_seqlen, n_kv_head, d_head)
                cache_seqlens: shape (B,) initialized to zeros, dtype int32
        """
        k_cache = torch.zeros(
            batch_size, max_seqlen, self.n_kv_head, self.d_head,
            dtype=dtype, device=device,
        )
        v_cache = torch.zeros(
            batch_size, max_seqlen, self.n_kv_head, self.d_head,
            dtype=dtype, device=device,
        )
        cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        return k_cache, v_cache, cache_seqlens

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """Compute grouped-query attention using flash_attn kernels.

        Dispatches to training path (flash_attn_func) when kv_cache is None,
        or generation path (flash_attn_with_kvcache) when kv_cache is provided.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of (k_cache, v_cache, cache_seqlens)
                where k_cache and v_cache have shape (B, max_seqlen, n_kv_head, d_head)
                and cache_seqlens has shape (B,) indicating filled length.

        Returns:
            Tuple of (output, new_kv_cache) where output has shape (B, T, d_model)
            and new_kv_cache is None (training) or (k_cache, v_cache, cache_seqlens).
        """
        B, T, C = x.shape

        # Project Q: (B, T, d_model) → (B, T, n_head * d_head) → (B, T, n_head, d_head)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head)

        # Project KV: (B, T, d_model) → (B, T, 2 * n_kv_head * d_head)
        kv = self.kv_proj(x)
        # Split into K and V, each (B, T, n_kv_head * d_head) → (B, T, n_kv_head, d_head)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_head, self.d_head)
        v = v.view(B, T, self.n_kv_head, self.d_head)

        if kv_cache is None:
            # --- Training path ---
            return self._training_forward(q, k, v, B, T)
        else:
            # --- Generation path ---
            return self._generation_forward(q, k, v, kv_cache, B, T)

    def _training_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
    ) -> tuple[torch.Tensor, None]:
        """Training path using flash_attn_func for full-sequence attention.

        Args:
            q: Tensor of shape (B, T, n_head, d_head).
            k: Tensor of shape (B, T, n_kv_head, d_head).
            v: Tensor of shape (B, T, n_kv_head, d_head).
            B: Batch size.
            T: Sequence length.

        Returns:
            Tuple of (output, None).
        """
        # Transpose to (B, n_head/n_kv_head, T, d_head) for RoPE application
        q = q.transpose(1, 2)  # (B, n_head, T, d_head)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, d_head)

        # Apply RoPE (expects (B, n_heads, T, d_head) — works for any n_heads)
        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Transpose back to flash_attn layout: (B, T, n_head/n_kv_head, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # flash_attn_func handles head count mismatch natively:
        # Q: (B, T, n_head, d_head), K/V: (B, T, n_kv_head, d_head)
        dropout_p = self.attn_dropout if self.training else 0.0
        out = self._flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=True,
        )

        # out shape: (B, T, n_head, d_head) -> (B, T, d_model)
        out = out.reshape(B, T, self.d_model)

        # Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, None

    def _generation_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        B: int,
        T: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generation path using flash_attn_with_kvcache for cached attention.

        Args:
            q: Tensor of shape (B, T, n_head, d_head).
            k: Tensor of shape (B, T, n_kv_head, d_head).
            v: Tensor of shape (B, T, n_kv_head, d_head).
            kv_cache: Tuple of (k_cache, v_cache, cache_seqlens).
            B: Batch size.
            T: Number of new tokens.

        Returns:
            Tuple of (output, (k_cache, v_cache, cache_seqlens)).
        """
        k_cache, v_cache, cache_seqlens = kv_cache

        # Compute position offset from cache_seqlens for RoPE
        past_len = cache_seqlens[0].item()

        # Transpose to (B, n_head/n_kv_head, T, d_head) for RoPE application
        q = q.transpose(1, 2)  # (B, n_head, T, d_head)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, d_head)

        # Apply RoPE with correct position offset
        rope_cos = self.rope_cos[past_len:past_len + T]
        rope_sin = self.rope_sin[past_len:past_len + T]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Transpose back to flash_attn layout: (B, T, n_head/n_kv_head, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # flash_attn_with_kvcache handles cache append internally
        out = self._flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
        )

        # Update cache_seqlens (flash_attn_with_kvcache writes to cache but
        # doesn't update seqlens — we do it ourselves)
        cache_seqlens = cache_seqlens + T

        # out shape: (B, T, n_head, d_head) -> (B, T, d_model)
        out = out.reshape(B, T, self.d_model)

        # Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, (k_cache, v_cache, cache_seqlens)
