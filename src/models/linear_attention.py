"""Causal linear attention module (V5).

Implements the autoregressive linear Transformer from Katharopoulos et al.
using the positive feature map φ(x) = ELU(x) + 1. Prefix key/value statistics
replace the quadratic attention matrix, giving O(T·d_head²) complexity while
ensuring that position i depends only on positions j <= i.

The recurrence is evaluated in fixed-size chunks. Within each chunk, a small
causal attention matrix handles local contributions; accumulated prefix state
handles all earlier chunks. This is mathematically equivalent to the token-wise
recurrence but substantially faster on GPU and under torch.compile.

Following RoFormer equation 19, RoPE rotates the positive query and key features
used by the numerator, while the normalization denominator remains unrotated.
Critical prefix-state reductions run in float32. At evaluation time the module
returns fixed-size numerator and denominator states for recurrent generation.
"""

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.rope import apply_rope, precompute_rope_frequencies


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """Apply the strictly positive ELU+1 feature map."""
    negative_branch = torch.exp(x.clamp_max(0.0))
    return torch.where(x >= 0, x + 1.0, negative_branch)


class CausalLinearAttention(nn.Module):
    """ELU+1 causal linear attention with RoPE.

    Following RoFormer equation 19, the numerator rotates the positive
    query/key features while the denominator remains unrotated::

        (R_i φ(Q_i))^T Σ_{j<=i}((R_j φ(K_j)) V_j^T)
        ------------------------------------------------
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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, int] | None,
    ]:
        """Compute causal linear attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            kv_cache: Optional ``(numerator_state, denominator_state, position)``.

        Returns:
            Tuple of output and recurrent state. Training returns ``None`` for
            the state; evaluation returns a reusable fixed-size state.

        Raises:
            AssertionError: If the input exceeds the configured context length.
        """
        B, T, C = x.shape
        offset = int(kv_cache[2]) if kv_cache is not None else 0
        assert offset + T <= self.config.seq_len, (
            f"Total sequence length {offset + T} exceeds config.seq_len "
            f"{self.config.seq_len}"
        )

        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # The feature map and recurrent reductions are stability-sensitive. Keep
        # them in float32 even when the surrounding model runs under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            phi_q = feature_map(q.float())
            phi_k = feature_map(k.float())
            rope_slice = slice(offset, offset + T)
            rotated_phi_q = apply_rope(
                phi_q, self.rope_cos[rope_slice], self.rope_sin[rope_slice]
            )
            rotated_phi_k = apply_rope(
                phi_k, self.rope_cos[rope_slice], self.rope_sin[rope_slice]
            )
            v_float = v.float()

            # Prefix states summarize completed chunks. RoPE appears only in the
            # numerator state; the positive denominator state is unrotated.
            if kv_cache is None:
                kv_state = torch.zeros(
                    B,
                    self.n_head,
                    self.d_head,
                    self.d_head,
                    device=x.device,
                    dtype=phi_q.dtype,
                )
                k_state = torch.zeros(
                    B, self.n_head, self.d_head, device=x.device, dtype=phi_q.dtype
                )
            else:
                kv_state, k_state, _ = kv_cache
                kv_state = kv_state.to(device=x.device, dtype=phi_q.dtype)
                k_state = k_state.to(device=x.device, dtype=phi_q.dtype)
                expected_kv_shape = (B, self.n_head, self.d_head, self.d_head)
                expected_k_shape = (B, self.n_head, self.d_head)
                if kv_state.shape != expected_kv_shape or k_state.shape != expected_k_shape:
                    raise ValueError(
                        "Linear attention cache shape does not match the current batch/model"
                    )
            chunks: list[torch.Tensor] = []

            for start in range(0, T, self.chunk_size):
                stop = min(start + self.chunk_size, T)
                q_chunk = phi_q[:, :, start:stop, :]
                k_chunk = phi_k[:, :, start:stop, :]
                rotated_q_chunk = rotated_phi_q[:, :, start:stop, :]
                rotated_k_chunk = rotated_phi_k[:, :, start:stop, :]
                v_chunk = v_float[:, :, start:stop, :]
                chunk_len = stop - start

                # Contribution from every token in earlier chunks.
                history_num = torch.einsum(
                    "bhld,bhde->bhle", rotated_q_chunk, kv_state
                )
                history_den = torch.einsum("bhld,bhd->bhl", q_chunk, k_state)

                # RoFormer rotates feature maps in the numerator but leaves the
                # normalization denominator unrotated (equation 19).
                local_num_scores = torch.einsum(
                    "bhld,bhmd->bhlm", rotated_q_chunk, rotated_k_chunk
                )
                local_den_scores = torch.einsum(
                    "bhld,bhmd->bhlm", q_chunk, k_chunk
                )
                mask = self.chunk_causal_mask[:chunk_len, :chunk_len]
                local_num_scores = local_num_scores * mask
                local_den_scores = local_den_scores * mask
                numerator = history_num + torch.matmul(local_num_scores, v_chunk)
                denominator = history_den + local_den_scores.sum(dim=-1)
                chunks.append(numerator / denominator.clamp_min(1e-6).unsqueeze(-1))

                # Update after computing this chunk so the triangular local mask
                # remains the sole source of within-chunk causality.
                kv_state = kv_state + torch.einsum(
                    "bhld,bhle->bhde", rotated_k_chunk, v_chunk
                )
                k_state = k_state + k_chunk.sum(dim=2)

            output = torch.cat(chunks, dim=2)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = output.to(self.out_proj.weight.dtype)
        output = self.resid_dropout(self.out_proj(output))

        new_cache = None if self.training else (kv_state, k_state, offset + T)
        return output, new_cache


# Temporary import compatibility for code that used the original V5 class name.
LinearAttention = CausalLinearAttention
