"""ALiBi Attention module using flash_attn library with linear position biases.

Implements Attention with Linear Biases (Press et al., "Train Short, Test Long").
Instead of encoding position in Q/K via RoPE or learned embeddings, ALiBi adds
fixed per-head linear biases directly to attention logits. Head i receives a
slope of 2^(-8*(i+1)/n_head), forming a geometric series.

Drop-in replacement for ModernAttention — same interface:
    forward(x, kv_cache=None) -> (output, new_kv_cache)

Key difference from FlashAttention module: NO RoPE. Q and K are raw projections.
Position is encoded entirely through alibi_slopes passed to the flash_attn kernel.

Tensor layout:
    flash_attn expects (B, T, n_head, d_head). Since there's no RoPE step
    requiring (B, n_head, T, d_head) transpose, tensors stay in native layout
    throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.config import ModelConfig


class ALiBiAttention(nn.Module):
    """Multi-head causal self-attention with ALiBi position biases.

    Replaces RoPE with per-head linear biases. Uses flash_attn library
    for both training and generation with native alibi_slopes support.

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Verify flash_attn is available (lazy import guard)
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise ImportError(
                "flash_attn is required for ALiBiAttention but is not installed. "
                "Install it with: pip install flash-attn --no-build-isolation\n"
                "See https://github.com/Dao-AILab/flash-attention for details."
            )

        # Import flash_attn functions lazily in constructor
        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        self._flash_attn_func = flash_attn_func
        self._flash_attn_with_kvcache = flash_attn_with_kvcache

        # Store config values
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.attn_dropout = config.dropout

        # QKV projection: d_model -> 3*d_model
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection: d_model -> d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Residual dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # ALiBi slopes: geometric series 2^(-8*(i+1)/n_head)
        # Registered as a buffer so it moves to the correct device with the model.
        # flash_attn requires slopes in fp32, so we cast to float32 at call sites.
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
                k_cache: shape (B, max_seqlen, n_head, d_head)
                v_cache: shape (B, max_seqlen, n_head, d_head)
                cache_seqlens: shape (B,) initialized to zeros, dtype int32
        """
        k_cache = torch.zeros(
            batch_size, max_seqlen, self.n_head, self.d_head,
            dtype=dtype, device=device,
        )
        v_cache = torch.zeros(
            batch_size, max_seqlen, self.n_head, self.d_head,
            dtype=dtype, device=device,
        )
        cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        return k_cache, v_cache, cache_seqlens

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """Compute attention with ALiBi biases via flash_attn.

        Dispatch logic:
        - kv_cache is None and T > 1: training path
        - kv_cache is not None: generation path
        - kv_cache is None and T == 1: allocate fresh cache, then generation path

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of (k_cache, v_cache, cache_seqlens).

        Returns:
            Tuple of (output, new_kv_cache).
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to flash_attn native layout: (B, T, n_head, d_head)
        q = q.view(B, T, self.n_head, self.d_head)
        k = k.view(B, T, self.n_head, self.d_head)
        v = v.view(B, T, self.n_head, self.d_head)

        if kv_cache is None and T > 1:
            # Training path
            return self._training_forward(q, k, v, B, T)
        elif kv_cache is not None:
            # Generation path with existing cache
            # Assert total sequence length doesn't exceed seq_len
            _, _, cache_seqlens = kv_cache
            assert (cache_seqlens + T <= self.seq_len).all(), (
                f"Total sequence length ({cache_seqlens.max().item()} + {T}) "
                f"exceeds configured seq_len ({self.seq_len})"
            )
            return self._generation_forward(q, k, v, kv_cache, B, T)
        else:
            # kv_cache is None and T == 1: allocate fresh cache
            assert T <= self.seq_len, (
                f"Sequence length {T} exceeds configured seq_len ({self.seq_len})"
            )
            kv_cache = self.allocate_kv_cache(
                batch_size=B,
                max_seqlen=self.seq_len,
                dtype=q.dtype,
                device=q.device,
            )
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

        No RoPE applied — Q and K are raw projections. ALiBi slopes are
        passed to the kernel for position-dependent attention biases.

        Args:
            q, k, v: Tensors of shape (B, T, n_head, d_head).
            B: Batch size.
            T: Sequence length.

        Returns:
            Tuple of (output, None).
        """
        # flash_attn_func expects (B, T, n_head, d_head) — already in that layout
        dropout_p = self.attn_dropout if self.training else 0.0
        out = self._flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=True,
            alibi_slopes=self.alibi_slopes.float(),
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

        No RoPE applied — K/V are stored raw in cache. ALiBi slopes are
        passed to the kernel which computes position biases from absolute
        positions derived from cache_seqlens.

        Args:
            q, k, v: Tensors of shape (B, T, n_head, d_head).
            kv_cache: Tuple of (k_cache, v_cache, cache_seqlens).
            B: Batch size.
            T: Number of new tokens.

        Returns:
            Tuple of (output, (k_cache, v_cache, cache_seqlens)).
        """
        k_cache, v_cache, cache_seqlens = kv_cache

        # flash_attn_with_kvcache handles cache append internally
        out = self._flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            alibi_slopes=self.alibi_slopes.float(),
        )

        # Update cache_seqlens (flash_attn_with_kvcache writes to cache but
        # doesn't update seqlens — we do it ourselves)
        cache_seqlens = cache_seqlens + T

        # out shape: (B, T, n_head, d_head) -> (B, T, d_model)
        out = out.reshape(B, T, self.d_model)

        # Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, (k_cache, v_cache, cache_seqlens)
