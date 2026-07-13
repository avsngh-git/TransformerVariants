"""Causal linear attention module (V5).

Implements the autoregressive linear Transformer from Katharopoulos et al.
using the positive feature map φ(x) = ELU(x) + 1. Prefix key/value statistics
replace the quadratic attention matrix, giving O(T·d_head²) complexity while
ensuring that position i depends only on positions j <= i.

The recurrence is evaluated in fixed-size chunks. Within each chunk, a small
causal attention matrix handles local contributions; accumulated prefix state
handles all earlier chunks. This is mathematically equivalent to the token-wise
recurrence but substantially faster on GPU and under torch.compile.

RoPE is applied before the feature map for positional parity with V1. Recurrent
generation state is not yet exposed through the shared KV-cache interface.
"""

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.rope import apply_rope, precompute_rope_frequencies


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """Apply the strictly positive ELU+1 feature map."""
    return torch.where(x >= 0, x + 1.0, torch.exp(x))


class CausalLinearAttention(nn.Module):
    """ELU+1 causal linear attention with RoPE.

    For each position i, the normalized output is::

        φ(Q_i)^T Σ_{j<=i}(φ(K_j) V_j^T)
        ---------------------------------
             φ(Q_i)^T Σ_{j<=i}φ(K_j)

    The prefix sums make the computation linear in sequence length. Chunking
    preserves the same equation while avoiding a Python loop per token.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.chunk_size = min(64, config.seq_len)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        cos, sin = precompute_rope_frequencies(config.d_head, config.seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        causal_mask = torch.ones(self.chunk_size, self.chunk_size).tril()
        self.register_buffer("chunk_causal_mask", causal_mask, persistent=False)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, kv_cache=None) -> tuple[torch.Tensor, None]:
        """Compute causal linear attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            kv_cache: Must be None. If non-None, raises NotImplementedError.

        Returns:
            Tuple of (output, None) where output has shape (B, T, d_model).

        Raises:
            NotImplementedError: If kv_cache is not None.
            AssertionError: If the input exceeds the configured context length.
        """
        if kv_cache is not None:
            raise NotImplementedError(
                "Recurrent generation state is not yet supported for causal linear attention"
            )

        B, T, C = x.shape
        assert T <= self.config.seq_len, (
            f"Input seq_len {T} exceeds config.seq_len {self.config.seq_len}"
        )

        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        phi_q = feature_map(q)
        phi_k = feature_map(k)

        # Prefix state summarizing all completed chunks.
        kv_state = x.new_zeros(B, self.n_head, self.d_head, self.d_head)
        k_state = x.new_zeros(B, self.n_head, self.d_head)
        chunks: list[torch.Tensor] = []

        for start in range(0, T, self.chunk_size):
            stop = min(start + self.chunk_size, T)
            q_chunk = phi_q[:, :, start:stop, :]
            k_chunk = phi_k[:, :, start:stop, :]
            v_chunk = v[:, :, start:stop, :]
            chunk_len = stop - start

            # Contribution from every token in earlier chunks.
            history_num = torch.einsum("bhld,bhde->bhle", q_chunk, kv_state)
            history_den = torch.einsum("bhld,bhd->bhl", q_chunk, k_state)

            # Exact causal contribution from tokens in the current chunk.
            local_scores = torch.einsum("bhld,bhmd->bhlm", q_chunk, k_chunk)
            mask = self.chunk_causal_mask[:chunk_len, :chunk_len]
            local_scores = local_scores * mask
            numerator = history_num + torch.matmul(local_scores, v_chunk)
            denominator = history_den + local_scores.sum(dim=-1)
            chunks.append(numerator / denominator.clamp_min(1e-6).unsqueeze(-1))

            # State updates occur after computing this chunk so local causality is
            # governed solely by the triangular mask above.
            kv_state = kv_state + torch.einsum("bhld,bhle->bhde", k_chunk, v_chunk)
            k_state = k_state + k_chunk.sum(dim=2)

        output = torch.cat(chunks, dim=2)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(output))

        return output, None


# Temporary import compatibility for code that used the original V5 class name.
LinearAttention = CausalLinearAttention
