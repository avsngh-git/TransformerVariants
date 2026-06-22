"""Causal multi-head self-attention with optional KV-cache.

This module implements the standard attention mechanism used in GPT-2:
- Multi-head attention with learned Q, K, V projections
- Causal masking (tokens can only attend to past + self)
- Scaled dot-product attention (divide by sqrt(d_head))
- Optional KV-cache for fast autoregressive generation

KV-Cache Explained:
    During generation, we produce one token at a time. Without caching,
    each new token requires recomputing K and V for ALL prior positions.
    With KV-cache, we store previously computed K and V tensors and only
    compute the new token's K/V, then concatenate with the cache.

    Training: no cache (all positions computed in parallel via teacher forcing)
    Generation: use cache (only new token computed, prior K/V reused)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The data flow is:
    1. Project input into Q, K, V (three separate projections, or one combined)
    2. Split into multiple heads
    3. (If cache exists) Concatenate new K, V with cached K, V
    4. Compute scaled dot-product attention with causal mask
    5. Concatenate heads
    6. Project back to d_model

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Combined Q, K, V projection — one big linear layer that produces
        # all three at once. More efficient than three separate layers because
        # the GPU can do one large matmul instead of three small ones.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection — after concatenating heads, project back to d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout on attention weights (probability of zeroing an attention connection)
        self.attn_dropout = nn.Dropout(config.dropout)
        # Dropout on the output projection
        self.resid_dropout = nn.Dropout(config.dropout)

        # Register the causal mask as a buffer (not a parameter — it's not learned).
        # This is a lower-triangular matrix of ones: position i can attend to positions 0..i.
        # Shape: (1, 1, seq_len, seq_len) — extra dims for broadcasting over batch and heads.
        causal_mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.seq_len, config.seq_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
               During generation with cache, seq_len = 1 (just the new token).
            kv_cache: Optional tuple of (cached_k, cached_v), each of shape
                      (batch_size, n_head, past_seq_len, d_head).
                      Pass None during training or first generation step.

        Returns:
            Tuple of:
            - output: Tensor of shape (batch_size, seq_len, d_model)
            - new_kv_cache: Tuple of (k, v) including the new positions,
                           each shape (batch_size, n_head, total_seq_len, d_head)
        """
        B, T, C = x.shape  # batch, sequence length (1 during cached generation), d_model

        # Step 1: Project to Q, K, V in one shot
        # qkv shape: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V — each is (B, T, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 2: Reshape into multiple heads
        # (B, T, d_model) → (B, T, n_head, d_head) → (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Step 3: KV-cache handling
        # If we have a cache from prior generation steps, concatenate the new K, V
        # with the cached K, V from prior positions.
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # cached_k shape: (B, n_head, past_len, d_head)
            # k shape:        (B, n_head, 1, d_head)  ← just the new token
            # After cat:      (B, n_head, past_len + 1, d_head)
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Store the updated cache (always — caller decides whether to use it)
        new_kv_cache = (k, v)

        # Now: q has T positions (1 during generation), k/v have S positions (full history)
        # S = total sequence length including cache
        S = k.size(2)

        # Step 4: Scaled dot-product attention
        # Q @ K^T: (B, n_head, T, d_head) @ (B, n_head, d_head, S) = (B, n_head, T, S)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        # Apply causal mask
        # During generation with cache: T=1, S=past+1. We need the mask for the
        # new token's position attending to all prior positions + itself.
        # The mask slice [S-T:S, :S] gives us exactly the right rows.
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, S - T:S, :S] == 0, float("-inf")
        )

        # Softmax over the key dimension (last dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (B, n_head, T, S) @ (B, n_head, S, d_head)
        # = (B, n_head, T, d_head)
        out = attn_weights @ v

        # Step 5: Concatenate heads
        # (B, n_head, T, d_head) → (B, T, n_head, d_head) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Step 6: Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, new_kv_cache
