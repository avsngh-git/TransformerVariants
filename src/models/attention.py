"""Causal multi-head self-attention.

This module implements the standard attention mechanism used in GPT-2:
- Multi-head attention with learned Q, K, V projections
- Causal masking (tokens can only attend to past + self)
- Scaled dot-product attention (divide by sqrt(d_head))
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
    3. Compute scaled dot-product attention with causal mask
    4. Concatenate heads
    5. Project back to d_model

    Args:
        config: ModelConfig with d_model, n_head, dropout, bias, seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        nn.Linear()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape  # batch, sequence length, d_model

        # Step 1: Project to Q, K, V in one shot
        # qkv shape: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V — each is (B, T, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 2: Reshape into multiple heads
        # (B, T, d_model) → (B, T, n_head, d_head) → (B, n_head, T, d_head)
        # The transpose puts the head dimension before the sequence dimension,
        # so each head processes the full sequence independently.
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        # Q @ K^T gives the raw attention scores: (B, n_head, T, T)
        # Divide by sqrt(d_head) to prevent large values pre-softmax
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        # Apply causal mask: set future positions to -inf so softmax gives them 0
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        # Softmax over the key dimension (last dim) — converts scores to probabilities
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: (B, n_head, T, T) @ (B, n_head, T, d_head)
        # = (B, n_head, T, d_head)
        out = attn_weights @ v

        # Step 4: Concatenate heads
        # (B, n_head, T, d_head) → (B, T, n_head, d_head) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Step 5: Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out
