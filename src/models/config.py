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
        ffn_hidden_dim: Optional exact hidden width for a standard FFN. This is
            used by parameter-matched counterfactuals; None uses ffn_multiplier.
        dropout: Dropout probability (0.0 = no dropout).
        bias: Whether linear layers include a bias term.
        tie_embeddings: Whether to tie input/output embedding weights.
        activation: Activation function in FFN ("relu" or "gelu").
        variant: Variant name identifying the recipe ("vanilla", "modern").
        norm_type: Normalization layer type ("layernorm" or "rmsnorm").
        position_encoding: Position encoding method ("learned", "rope", "alibi", "none").
        ffn_type: FFN architecture ("standard" or "swiglu").
        attention_type: Attention mechanism ("full", "flash_sdpa", "sliding_window", "linear").
        n_kv_head: Number of key-value heads for grouped query attention;
            None means same as n_head (full MHA).
        attention_backend: Kernel dispatch path for attention computation
            ("sdpa" for PyTorch built-in SDPA, "flash_attn" for Dao AI Lab Flash Attention 2).
            Does not change mathematical behavior — only speed and memory characteristics.
    """

    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    vocab_size: int = 50257
    seq_len: int = 512
    ffn_multiplier: int = 4
    ffn_hidden_dim: int | None = None
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    activation: str = "gelu"  # canonical V0 baseline; ReLU remains an explicit option

    # Variant identity fields
    variant: str = "vanilla"
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    ffn_type: str = "standard"
    attention_type: str = "full"
    n_kv_head: int | None = None

    # Compute optimization fields
    attention_backend: str = "sdpa"

    # Sliding Window Attention
    window_size: int | None = None

    # Legacy Linformer field retained so old run configs remain parseable.
    # The causal linear V5 implementation does not use a projection rank.
    projection_rank: int | None = None

    # Mixture of Experts
    num_experts: int | None = None  # None = dense FFN, 2+ = MoE
    moe_top_k: int = 2  # Number of experts per token
    moe_expert_hidden_dim: int | None = None  # Exact SwiGLU width of each sparse expert
    aux_loss_alpha: float = 0.01  # Load-balancing loss coefficient
    z_loss_beta: float = 0.001  # Router z-loss coefficient

    def __post_init__(self):
        if self.window_size is not None:
            if self.window_size < 1 or self.window_size > self.seq_len:
                raise ValueError(
                    f"window_size must be between 1 and seq_len ({self.seq_len}) inclusive, "
                    f"got {self.window_size}"
                )

        if self.num_experts is not None:
            if self.num_experts < 2:
                raise ValueError(f"num_experts must be >= 2 when set, got {self.num_experts}")
            if self.moe_top_k < 1 or self.moe_top_k > self.num_experts:
                raise ValueError(
                    f"moe_top_k must be between 1 and num_experts ({self.num_experts}) "
                    f"inclusive, got {self.moe_top_k}"
                )
            if self.moe_expert_hidden_dim is not None and self.moe_expert_hidden_dim < 1:
                raise ValueError(
                    "moe_expert_hidden_dim must be positive when set, got "
                    f"{self.moe_expert_hidden_dim}"
                )

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
        return (
            self.ffn_hidden_dim
            if self.ffn_hidden_dim is not None
            else self.d_model * self.ffn_multiplier
        )
