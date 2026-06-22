"""RMSNorm (Root Mean Square Layer Normalization).

RMSNorm is a simplification of LayerNorm that skips the mean-centering step.
Used in LLaMA, Mistral, and most modern LLMs.

LayerNorm:  x_norm = (x - mean) / std * gamma + beta
RMSNorm:    x_norm = x / RMS(x) * gamma

Where RMS(x) = sqrt(mean(x²) + eps)

Why it works:
- The mean-centering in LayerNorm is theoretically useful but empirically
  unnecessary at scale. Removing it saves compute without hurting quality.
- No bias parameter (gamma only, no beta) — matches our no-bias design.
- Slightly faster than LayerNorm (one fewer reduction operation).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: The dimension to normalize over (d_model).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (like LayerNorm's gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS: sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight
