"""Modern causal self-attention with RoPE and Flash Attention.

Differences from vanilla attention:
1. No learned position embedding — RoPE is applied directly to Q and K
2. Uses PyTorch's scaled_dot_product_attention (which dispatches to Flash Attention
   when available) for O(T) memory instead of O(T²)
3. KV-cache still supported for fast generation

Flash Attention:
- Standard attention materializes the full T×T attention matrix in GPU memory.
  For T=2048, that's 2048² × batch × heads × 4 bytes = huge.
- Flash Attention computes attention in tiles without ever storing the full matrix.
  It's mathematically identical but uses O(T) memory instead of O(T²).
- PyTorch 2.0+ provides this via torch.nn.functional.scaled_dot_product_attention,
  which automatically selects the fastest kernel (Flash, Memory-Efficient, or math).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig
from src.models.rope import apply_rope, precompute_rope_frequencies


class ModernAttention(nn.Module):
    """Multi-head causal self-attention with RoPE + Flash Attention.

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE is conditional so the learned-position counterfactual can keep
        # the same SDPA attention operator without applying a second encoding.
        if config.position_encoding == "rope":
            cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
            self.register_buffer("rope_cos", cos)  # (seq_len, d_head // 2)
            self.register_buffer("rope_sin", sin)  # (seq_len, d_head // 2)
        elif config.position_encoding != "learned":
            raise ValueError(
                "ModernAttention supports position_encoding='rope' or 'learned', "
                f"got {config.position_encoding!r}"
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute attention with RoPE and Flash Attention.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional (cached_k, cached_v) for generation.

        Returns:
            Tuple of (output, new_kv_cache).
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads: (B, T, d_model) → (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Determine position offset for RoPE
        if kv_cache is not None:
            past_len = kv_cache[0].size(2)
        else:
            past_len = 0

        if self.config.position_encoding == "rope":
            # Apply RoPE to Q and K (not V — values don't need position info).
            rope_cos = self.rope_cos[past_len:past_len + T]
            rope_sin = self.rope_sin[past_len:past_len + T]
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # KV-cache: concatenate with prior cached keys/values
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # Flash Attention via PyTorch's SDPA
        # This automatically uses the fastest available kernel:
        # - Flash Attention 2 (if available, O(T) memory)
        # - Memory-efficient attention (fallback)
        # - Math attention (fallback of fallback)
        # is_causal=True tells it to apply causal masking internally
        # (only works when there's no KV-cache, i.e., during training or first gen step)
        is_causal = (kv_cache is None) and (T > 1)
        dropout_p = self.attn_dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Concatenate heads: (B, n_head, T, d_head) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Output projection
        out = self.resid_dropout(self.out_proj(out))

        return out, new_kv_cache
