"""Position-wise Feed-Forward Network.

Each Transformer block has two sublayers:
1. Multi-head self-attention (lets tokens communicate)
2. FFN (processes each token independently)

The FFN is the same operation applied to every token position separately —
hence "position-wise." It's just a two-layer MLP with a nonlinearity in between.

Architecture: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
Where d_ff = d_model * ffn_multiplier (typically 4x).
"""

import torch
import torch.nn as nn

from src.models.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Why expand to 4x then compress back?
    The expansion gives the network a higher-dimensional space to work in.
    Think of it like: attention figures out WHAT information to gather,
    then the FFN has a big workspace to DO something with that information
    before compressing back to d_model for the next layer.

    Args:
        config: ModelConfig with d_model, d_ff, dropout, bias.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Up-projection: d_model → d_ff (e.g., 512 → 2048)
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)

        # Down-projection: d_ff → d_model (e.g., 2048 → 512)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

        # ReLU activation — the nonlinearity between the two linear layers.
        # Without this, two linear layers collapse into one (linear(linear(x)) = linear(x)).
        # ReLU is the original Transformer choice; GPT-2 uses GELU and modern
        # recipe variants in this project use SwiGLU.
        if config.activation == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif config.activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}. Use 'relu' or 'gelu'.")

        # Dropout on the output (before adding back to the residual stream)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN to each token position independently.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x: (B, T, d_model)
        x = self.fc1(x)       # (B, T, d_ff) — expand
        x = self.act(x)       # (B, T, d_ff) — nonlinearity
        x = self.fc2(x)       # (B, T, d_model) — compress back
        x = self.dropout(x)   # (B, T, d_model) — regularization
        return x
