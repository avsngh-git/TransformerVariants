"""Linformer low-rank projection attention module (V5).

Implements sub-quadratic attention by projecting Key and Value sequences from
length T down to a fixed rank r using learned projection matrices E and F,
while retaining softmax attention and RoPE position encoding. This achieves
O(T·r·d) complexity instead of the O(T²·d) of full attention.

The module interface matches the standard attention contract used by
ModernTransformer: forward(x, kv_cache=None) -> (output, None).

KV-cache generation is not supported — E/F projection matrices are tied to
fixed seq_len (training-comparison variant only, per ADR 0003).
"""

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.rope import apply_rope, precompute_rope_frequencies


class LinformerAttention(nn.Module):
    """Linformer attention with learned low-rank projection, RoPE, and softmax.

    Projects Keys and Values from sequence length T to a fixed rank r using
    learned projection matrices E and F. Applies RoPE to Q and K before
    projection for position encoding parity with V1 (full attention).

    Complexity: O(T·r·d) per head, where r << T.

    Args:
        config: ModelConfig providing d_model, n_head, seq_len, projection_rank, dropout.

    Raises:
        ValueError: If projection_rank is None, <= 0, or > seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Validate projection_rank
        if config.projection_rank is None:
            raise ValueError(
                "projection_rank must be set for LinformerAttention (got None). "
                "Set config.projection_rank to a positive integer <= seq_len."
            )
        if config.projection_rank <= 0:
            raise ValueError(
                f"projection_rank must be > 0, got {config.projection_rank}"
            )
        if config.projection_rank > config.seq_len:
            raise ValueError(
                f"projection_rank ({config.projection_rank}) must be <= seq_len ({config.seq_len})"
            )

        # Q, K, V, and output projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learned projection matrices E (for K) and F (for V)
        # Shape: (seq_len, projection_rank)
        self.E = nn.Parameter(torch.empty(config.seq_len, config.projection_rank))
        self.F = nn.Parameter(torch.empty(config.seq_len, config.projection_rank))
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

        # Precompute RoPE cos/sin buffers
        cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, kv_cache=None) -> tuple[torch.Tensor, None]:
        """Compute Linformer attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            kv_cache: Must be None. If non-None, raises NotImplementedError.

        Returns:
            Tuple of (output, None) where output has shape (B, T, d_model).

        Raises:
            NotImplementedError: If kv_cache is not None.
            AssertionError: If T > config.seq_len.
        """
        if kv_cache is not None:
            raise NotImplementedError(
                "KV-cache generation is not supported for Linformer attention "
                "(E/F projection matrices are tied to fixed seq_len)"
            )

        B, T, C = x.shape
        assert T <= self.config.seq_len, (
            f"Input seq_len {T} exceeds config.seq_len {self.config.seq_len}"
        )

        # Project Q, K, V and reshape to (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (before low-rank projection)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        # Low-rank projection of K and V
        E_t = self.E[:T, :].t()  # (r, T)
        F_t = self.F[:T, :].t()  # (r, T)
        k_proj = torch.einsum('rt,bhtd->bhrd', E_t, k)  # (B, H, r, d_head)
        v_proj = torch.einsum('rt,bhtd->bhrd', F_t, v)  # (B, H, r, d_head)

        # Scaled dot-product attention with softmax
        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * scale  # (B, H, T, r)
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, r)
        weights = self.attn_dropout(weights)

        # Weighted sum of projected values
        output = torch.matmul(weights, v_proj)  # (B, H, T, d_head)

        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(output))

        return output, None
