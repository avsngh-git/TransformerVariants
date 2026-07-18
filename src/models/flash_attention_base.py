"""FlashAttentionBase — abstract base class for flash_attn-backed attention modules.

Consolidates the shared skeleton duplicated across FlashAttention, ALiBiAttention,
and GQAAttention into a single base class using the Template Method pattern.

The base owns:
- QKV projection setup (configurable head counts)
- KV-cache allocation
- Forward dispatch (training vs generation path selection)
- Training kernel call (flash_attn_func)
- Generation epilogue (flash_attn_with_kvcache → increment seqlens → reshape → out_proj)

Subclasses implement two narrow hooks:
- _apply_position(q, k, offset): position encoding (RoPE, ALiBi no-op, etc.)
- _extra_attn_kwargs(): additional kernel kwargs (e.g., alibi_slopes)

And may optionally override:
- _project_qkv(x, B, T): projection strategy (GQA uses separate q_proj/kv_proj)

Tensor layout:
    All Q/K/V tensors are in flash_attn native layout: (B, T, n_head, d_head).
    The _apply_position hook receives and returns tensors in this layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.models.config import ModelConfig


class FlashAttentionBase(nn.Module, ABC):
    """Abstract base for flash_attn-backed attention modules.

    Subclasses configure head counts and implement two hooks:
    - _apply_position: position encoding (RoPE or no-op)
    - _extra_attn_kwargs: additional kernel arguments (e.g., alibi_slopes)

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
        n_kv_head: Number of KV heads. Defaults to config.n_head (standard MHA)
            when None. Must evenly divide config.n_head.
    """

    def __init__(self, config: ModelConfig, n_kv_head: int | None = None) -> None:
        super().__init__()

        # Resolve n_kv_head: default to n_head (standard MHA) when not provided
        resolved_n_kv_head = n_kv_head if n_kv_head is not None else config.n_head

        # Validate n_head is divisible by n_kv_head
        if config.n_head % resolved_n_kv_head != 0:
            raise ValueError(
                f"n_head ({config.n_head}) must be divisible by n_kv_head "
                f"({resolved_n_kv_head}) for grouped-query attention."
            )

        # Verify flash_attn is available (fail-fast with helpful message)
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise ImportError(
                "flash_attn is required but is not installed. "
                "Install it with: pip install flash-attn --no-build-isolation\n"
                "See https://github.com/Dao-AILab/flash-attention for details."
            )

        # Import flash_attn functions (lazy — only when class is instantiated)
        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        self._flash_attn_func = flash_attn_func
        self._flash_attn_with_kvcache = flash_attn_with_kvcache

        # Store config values
        self.n_head = config.n_head
        self.n_kv_head = resolved_n_kv_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.attn_dropout = config.dropout

        # Combined QKV projection: d_model → (n_head + 2*n_kv_head) * d_head
        # Subclasses (e.g., GQA) may override _project_qkv and not use this layer.
        total_proj_dim = (self.n_head + 2 * self.n_kv_head) * self.d_head
        self.qkv_proj = nn.Linear(config.d_model, total_proj_dim, bias=config.bias)

        # Output projection: n_head * d_head → d_model
        self.out_proj = nn.Linear(self.n_head * self.d_head, config.d_model, bias=config.bias)

        # Residual dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """Compute attention using flash_attn kernels.

        Three-way dispatch:
        - training and kv_cache is None: training path (flash_attn_func)
        - kv_cache is provided: generation path (flash_attn_with_kvcache)
        - evaluation and kv_cache is None: allocate a cache and prefill it

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of (k_cache, v_cache, cache_seqlens).

        Returns:
            Tuple of (output, new_kv_cache) where output has shape (B, T, d_model)
            and new_kv_cache is None (training) or (k_cache, v_cache, cache_seqlens).
        """
        B, T, C = x.shape

        # Step 1: Project input to Q, K, V (overridable by subclasses)
        q, k, v = self._project_qkv(x, B, T)
        # q: (B, T, n_head, d_head), k/v: (B, T, n_kv_head, d_head)

        # Step 2: Dispatch based on kv_cache state
        # FlashAttention's pybind KV-cache kernel is not traceable by Dynamo.
        # Compiled models are a training/evaluation optimization in this project;
        # cache-aware serving is benchmarked eagerly. Keep no-cache compiled
        # evaluation on the traceable full-sequence kernel.
        if kv_cache is None and (self.training or torch.compiler.is_compiling()):
            # Training or compiled no-cache evaluation: full-sequence attention.
            return self._training_forward(q, k, v, B, T)
        elif kv_cache is not None:
            # Generation path with existing cache
            k_cache, v_cache, cache_seqlens = kv_cache
            assert (cache_seqlens + T <= self.seq_len).all(), (
                f"Cache overflow: cache_seqlens ({cache_seqlens.max().item()}) + T ({T}) "
                f"exceeds seq_len ({self.seq_len})"
            )
            return self._generation_forward(q, k, v, kv_cache, B, T)
        else:
            # Evaluation prefill: allocate once and append the whole prompt.
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

    # ------------------------------------------------------------------
    # Template Method hooks (subclass override points)
    # ------------------------------------------------------------------

    @abstractmethod
    def _apply_position(
        self, q: torch.Tensor, k: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply position encoding to Q and K tensors.

        Subclasses implement this to inject variant-specific position encoding:
        - RoPE: rotate Q and K at positions [offset, offset+T)
        - ALiBi: return Q and K unchanged (position encoded via kernel biases)

        Args:
            q: Query tensor of shape (B, T, n_head, d_head).
            k: Key tensor of shape (B, T, n_kv_head, d_head).
            offset: Position offset (number of prior tokens in cache).

        Returns:
            Tuple of (q, k) with same shapes — position-encoded versions.
        """
        ...

    def _extra_attn_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for flash_attn kernel calls.

        Override in subclasses that need extra kernel parameters (e.g., ALiBi slopes).

        Returns:
            Dict of keyword arguments to pass to flash_attn_func and
            flash_attn_with_kvcache. Default: empty dict.
        """
        return {}

    def _extra_training_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for flash_attn_func (training path).

        Default delegates to _extra_attn_kwargs() for backward compatibility.
        Override in subclasses that need training-only kernel parameters (e.g., window_size).

        Returns:
            Dict of keyword arguments for flash_attn_func during training.
        """
        return self._extra_attn_kwargs()

    def _project_qkv(
        self, x: torch.Tensor, B: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors using combined qkv_proj.

        Default implementation uses a single combined projection split as:
        - Q: n_head * d_head dimensions
        - K: n_kv_head * d_head dimensions
        - V: n_kv_head * d_head dimensions

        Subclasses (e.g., GQA) may override to use separate projections.

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
        qkv = self.qkv_proj(x)

        # Split: Q gets n_head*d_head, K and V each get n_kv_head*d_head
        q_dim = self.n_head * self.d_head
        kv_dim = self.n_kv_head * self.d_head

        q = qkv[..., :q_dim].view(B, T, self.n_head, self.d_head)
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, T, self.n_kv_head, self.d_head)
        v = qkv[..., q_dim + kv_dim:].view(B, T, self.n_kv_head, self.d_head)

        return q, k, v

    # ------------------------------------------------------------------
    # Internal dispatch paths
    # ------------------------------------------------------------------

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
        # Apply position encoding (offset=0 for training)
        q, k = self._apply_position(q, k, offset=0)

        # Get any extra kernel kwargs from subclass (training-specific hook)
        extra_kwargs = self._extra_training_kwargs()

        # Call flash_attn_func for full-sequence attention
        dropout_p = self.attn_dropout if self.training else 0.0
        out = self._flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=True,
            **extra_kwargs,
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

        # Compute position offset from cache_seqlens
        offset = cache_seqlens[0].item()

        # Apply position encoding at correct offset
        q, k = self._apply_position(q, k, offset)

        # Get any extra kernel kwargs from subclass
        extra_kwargs = self._extra_attn_kwargs()

        # flash_attn_with_kvcache handles cache append internally
        out = self._flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            **extra_kwargs,
        )

        # Update cache_seqlens (flash_attn_with_kvcache writes to cache but
        # doesn't update seqlens — we do it ourselves)
        cache_seqlens = cache_seqlens + T

        # out shape: (B, T, n_head, d_head) -> (B, T, d_model)
        out = out.reshape(B, T, self.d_model)

        # Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, (k_cache, v_cache, cache_seqlens)
