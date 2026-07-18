"""SwiGLU Feed-Forward Network.

SwiGLU is a gated activation used in LLaMA, Mistral, PaLM.
Instead of a simple nonlinearity between two linear layers,
it uses a GATE that controls information flow.

Standard FFN:    x → Linear → ReLU → Linear → out     (2 weight matrices)
SwiGLU FFN:      x → [Linear₁ → SiLU] * Linear₂ → Linear₃ → out   (3 weight matrices)

The "gate" (Linear₁ → SiLU) decides what to let through.
The "up" (Linear₂) provides the information.
They multiply together, then project back down.

Why 3 matrices? The gating mechanism needs two parallel paths:
one to compute the gate values, one to compute the content.
The third matrix projects back to d_model.

To keep parameter count comparable to standard FFN (which uses 4x expansion):
- Standard FFN: 2 × d_model × 4*d_model = 8*d_model² params
- SwiGLU FFN:   3 × d_model × (8/3)*d_model ≈ 8*d_model² params
So we use (8/3)*d_model as the hidden dim instead of 4*d_model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig


def swiglu_hidden_dim(d_model: int) -> int:
    """Return the parameter-matched dense SwiGLU hidden width."""
    hidden_dim = int(8 * d_model / 3)
    return ((hidden_dim + 63) // 64) * 64


class SwiGLUFeedForward(nn.Module):
    """SwiGLU gated feed-forward network.

    Architecture:
        gate = SiLU(x @ W_gate)
        up   = x @ W_up
        out  = (gate * up) @ W_down

    SiLU(x) = x * sigmoid(x) — also called "Swish"

    Args:
        config: ModelConfig with d_model, bias, dropout.
    """

    def __init__(self, config: ModelConfig, hidden_dim: int | None = None) -> None:
        super().__init__()

        # Dense default: 8/3 * d_model, rounded for efficient kernels. Sparse
        # experts can provide an exact smaller width so top-k active experts
        # collectively match one dense FFN.
        hidden_dim = hidden_dim if hidden_dim is not None else swiglu_hidden_dim(config.d_model)
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim

        # Gate projection: d_model → hidden_dim (goes through SiLU)
        self.w_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)

        # Up projection: d_model → hidden_dim (linear, no activation)
        self.w_up = nn.Linear(config.d_model, hidden_dim, bias=config.bias)

        # Down projection: hidden_dim → d_model (output)
        self.w_down = nn.Linear(hidden_dim, config.d_model, bias=config.bias)

        # Dropout on output
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Gate path: apply SiLU activation
        gate = F.silu(self.w_gate(x))  # (B, T, hidden_dim)

        # Up path: linear (no activation)
        up = self.w_up(x)  # (B, T, hidden_dim)

        # Element-wise multiply: gate controls what passes through
        hidden = gate * up  # (B, T, hidden_dim)

        # Project back to d_model
        out = self.w_down(hidden)  # (B, T, d_model)
        out = self.dropout(out)

        return out
