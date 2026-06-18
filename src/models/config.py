"""Model configuration dataclass.

A single object that holds all hyperparameters needed to construct a model.
Every model component receives this config instead of individual arguments —
keeps signatures clean and makes it easy to serialize/deserialize.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a decoder-only Transformer model.

    Attributes:
        n_layer: Number of Transformer blocks (depth).
        d_model: Hidden dimension size (width of the residual stream).
        n_head: Number of attention heads.
        vocab_size: Size of the token vocabulary.
        seq_len: Maximum sequence length the model can handle.
        ffn_multiplier: FFN hidden dim = d_model * ffn_multiplier.
        dropout: Dropout probability (0.0 = no dropout).
        bias: Whether linear layers include a bias term.
        tie_embeddings: Whether to tie input/output embedding weights.
    """

    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    vocab_size: int = 50257
    seq_len: int = 512
    ffn_multiplier: int = 4
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        """Dimension of each attention head: d_model // n_head."""
        assert self.d_model % self.n_head == 0, (
            f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
        )
        return self.d_model // self.n_head

    @property
    def d_ff(self) -> int:
        """FFN hidden dimension: d_model * ffn_multiplier."""
        return self.d_model * self.ffn_multiplier
