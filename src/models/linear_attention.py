"""ELU-based causal linear attention module (V5).

Implements O(n·d²) causal attention using the feature map φ(x) = ELU(x) + 1.
Instead of materialising a T×T attention matrix, this module maintains a running
accumulator of outer products φ(K_j)^T ⊗ V_j, enabling sub-quadratic scaling.

No RoPE is applied — position information comes solely from token embeddings.
KV-cache generation is not supported (training-comparison variant only, per ADR 0002).
"""

import torch
import torch.nn as nn

from src.models.config import ModelConfig


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """Apply the ELU+1 feature map element-wise.

    φ(x) = ELU(x) + 1. Always strictly positive (> 0), which guarantees
    that the normalisation denominator is never exactly zero.

    Mathematically: φ(x) = x + 1 for x >= 0, exp(x) for x < 0.
    We use torch.where instead of F.elu(x) + 1 to avoid float32 precision
    loss where exp(x) - 1 + 1 rounds to 0 for very negative x.

    Args:
        x: Arbitrary-shaped tensor.

    Returns:
        Tensor of same shape with all elements > 0.
    """
    return torch.where(x >= 0, x + 1.0, torch.exp(x))


class LinearAttention(nn.Module):
    """Causal linear attention with ELU+1 feature map.

    Replaces the O(n²) softmax attention kernel with an O(n·d²) recurrence.
    At each position i, the output is computed as:

        numerator_i   = φ(Q_i) @ S_i     where S_i = Σ_{j≤i} φ(K_j)^T ⊗ V_j
        denominator_i = φ(Q_i) · z_i     where z_i = Σ_{j≤i} φ(K_j)
        output_i      = numerator_i / clamp(denominator_i, min=ε)

    This is strictly causal — no position attends to any future position.

    Args:
        config: ModelConfig providing d_model, n_head, dropout, etc.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache=None):
        """Compute causal linear attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            kv_cache: Must be None. If non-None, raises NotImplementedError.

        Returns:
            Tuple of (output, None) where output has shape (B, T, d_model).

        Raises:
            NotImplementedError: If kv_cache is not None.
        """
        if kv_cache is not None:
            raise NotImplementedError(
                "KV-cache generation is not supported for linear attention"
            )

        B, T, C = x.shape

        # Project to Q, K, V and reshape to (B, n_head, T, d_head)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Apply feature map: φ(x) = ELU(x) + 1
        phi_q = feature_map(q)
        phi_k = feature_map(k)

        # Causal linear attention via recurrence
        # S: running outer-product accumulator (B, n_head, d_head, d_head)
        # z: running normaliser accumulator   (B, n_head, d_head)
        S = torch.zeros(
            B, self.n_head, self.d_head, self.d_head, device=x.device, dtype=x.dtype
        )
        z = torch.zeros(B, self.n_head, self.d_head, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            k_t = phi_k[:, :, t, :]  # (B, n_head, d_head)
            v_t = v[:, :, t, :]      # (B, n_head, d_head)
            q_t = phi_q[:, :, t, :]  # (B, n_head, d_head)

            # Update accumulators: S += outer(k_t, v_t), z += k_t
            S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            z = z + k_t

            # Numerator: φ(Q_t) @ S → (B, n_head, d_head)
            num = torch.einsum('bhd,bhde->bhe', q_t, S)

            # Denominator: φ(Q_t) · z → (B, n_head) scalar per head
            denom = torch.einsum('bhd,bhd->bh', q_t, z)

            # ε-clamp to prevent division by zero
            denom = denom.clamp(min=1e-6)

            # Normalise: output_t = num / denom
            out_t = num / denom.unsqueeze(-1)  # (B, n_head, d_head)
            outputs.append(out_t)

        # Stack: (B, n_head, T, d_head) → (B, T, d_model)
        out = torch.stack(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, None
