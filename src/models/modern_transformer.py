"""V1: Modern Transformer (LLaMA-style).

This model swaps 4 components from the vanilla V0:
1. RoPE (rotary position embeddings) — replaces learned position embeddings
2. RMSNorm — replaces LayerNorm
3. SwiGLU FFN — replaces standard ReLU/GELU FFN
4. Flash Attention — replaces manual attention computation

Everything else (weight tying, residual connections, generation, KV-cache)
remains the same as V0.

Architecture:
    Token embedding (no position embedding — RoPE handles it)
    → N × ModernTransformerBlock (RMSNorm → ModernAttention → residual, RMSNorm → SwiGLU → residual)
    → Final RMSNorm
    → Linear head → logits
"""

import math
from typing import TYPE_CHECKING, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cache import cache_sequence_length
from src.models.config import ModelConfig
from src.models.ffn import FeedForward
from src.models.modern_attention import ModernAttention
from src.models.rmsnorm import RMSNorm
from src.models.swiglu_ffn import SwiGLUFeedForward

if TYPE_CHECKING:
    from src.models.attention_protocol import AttentionModule


class ModernTransformerBlock(nn.Module):
    """A single Transformer block with modern components.

    Data flow (Pre-RMSNorm pattern):
        x → RMSNorm → Attention(RoPE) → + residual → RMSNorm → SwiGLU → + residual → out

    Args:
        config: ModelConfig with all hyperparameters.
        attention_class: The attention module class to use. Must satisfy the
            AttentionModule protocol: accept `config` as its sole constructor
            argument and implement `forward(x, kv_cache=None) -> (output, new_kv_cache)`.
            Defaults to ModernAttention.
    """

    def __init__(
        self,
        config: ModelConfig,
        attention_class: "Type[AttentionModule]" = ModernAttention,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.ln1 = self._build_norm(config)
        self.attn = attention_class(config)
        self.ln2 = self._build_norm(config)

        # Config-driven FFN selection
        if config.num_experts is not None:
            from src.models.moe_ffn import MoEFeedForward
            self.ffn = MoEFeedForward(config)
        elif config.ffn_type == "standard":
            self.ffn = FeedForward(config)
        elif config.ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(config)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type!r}")

    @staticmethod
    def _build_norm(config: ModelConfig) -> nn.Module:
        if config.norm_type == "rmsnorm":
            return RMSNorm(config.d_model)
        if config.norm_type == "layernorm":
            return nn.LayerNorm(config.d_model, bias=config.bias)
        raise ValueError(f"Unknown norm_type: {config.norm_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply one modern Transformer block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            kv_cache: Optional KV-cache for this layer.

        Returns:
            Tuple of (output, new_kv_cache).
        """
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class ModernTransformer(nn.Module):
    """V1: Modern decoder-only Transformer (LLaMA-style).

    Key differences from VanillaTransformer:
    - No position embedding layer (RoPE is applied inside attention)
    - RMSNorm instead of LayerNorm
    - SwiGLU instead of standard FFN
    - Flash Attention for O(T) memory

    Args:
        config: ModelConfig with all hyperparameters.
        attention_class: The attention module class to use in each block.
            Must accept `config` as its sole constructor argument and implement
            `forward(x, kv_cache=None) -> (output, new_kv_cache)`.
            Defaults to ModernAttention.
        per_layer_configs: Optional list of ModelConfig objects, one per layer.
            When provided, each block is constructed with its own config (e.g.
            to vary window_size per layer). When None, all blocks use `config`.
    """

    def __init__(
        self,
        config: ModelConfig,
        attention_class: Type[nn.Module] = ModernAttention,
        per_layer_configs: list[ModelConfig] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Learned positions are enabled only for the surgical counterfactual.
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        if config.position_encoding == "learned":
            self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        elif config.position_encoding not in {"rope", "alibi", "none"}:
            raise ValueError(
                "ModernTransformer supports learned, RoPE, ALiBi, or no positional "
                "encoding; "
                f"got {config.position_encoding!r}"
            )

        # Dropout after embedding
        self.drop = nn.Dropout(config.dropout)

        # Validate per_layer_configs length
        if per_layer_configs is not None:
            if len(per_layer_configs) != config.n_layer:
                raise ValueError(
                    f"per_layer_configs length ({len(per_layer_configs)}) must equal "
                    f"n_layer ({config.n_layer})"
                )
            self.blocks = nn.ModuleList([
                ModernTransformerBlock(per_layer_configs[i], attention_class=attention_class)
                for i in range(config.n_layer)
            ])
        else:
            self.blocks = nn.ModuleList([
                ModernTransformerBlock(config, attention_class=attention_class)
                for _ in range(config.n_layer)
            ])

        # Final norm follows the same one-factor configuration as each block.
        self.ln_f = ModernTransformerBlock._build_norm(config)

        # Output head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (same GPT-2 style as V0)."""
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * self.config.n_layer)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if (
                    name.endswith("out_proj")
                    or name.endswith("w_down")
                    or name.endswith("fc2")
                ):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_aux_loss(self) -> torch.Tensor:
        """Sum auxiliary losses across all MoE layers and clear buffers.

        Returns:
            Scalar tensor on model device. Zero for dense models.
        """
        total = torch.tensor(0.0, device=self.tok_emb.weight.device)
        for block in self.blocks:
            if hasattr(block.ffn, 'get_aux_loss'):
                layer_loss = block.ffn.get_aux_loss()
                if layer_loss is not None:
                    total = total + layer_loss
        return total

    def get_routing_data(self) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Retrieve and clear routing data from all MoE layers.

        Returns:
            Dict mapping layer index to list of (expert_indices, expert_weights).
            Empty dict if no MoE layers or no data recorded.
        """
        data = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block.ffn, 'get_routing_data'):
                layer_data = block.ffn.get_routing_data()
                if layer_data:
                    data[i] = layer_data
        return data

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass: token IDs → logits (and optionally loss).

        Args:
            idx: Token indices, shape (batch, seq_len).
            targets: Optional target tokens for loss.
            kv_cache: Optional list of KV-caches per layer.

        Returns:
            Tuple of (logits, loss, new_kv_cache).
        """
        B, T = idx.shape

        # No position offset needed here — RoPE handles it inside attention
        # But we still need to check sequence length
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = cache_sequence_length(kv_cache[0])
        else:
            past_len = 0

        total_len = past_len + T
        assert total_len <= self.config.seq_len, (
            f"Total sequence length {total_len} exceeds model max {self.config.seq_len}"
        )

        x = self.tok_emb(idx)  # (B, T, d_model)
        if self.config.position_encoding == "learned":
            positions = torch.arange(
                past_len, past_len + T, dtype=torch.long, device=idx.device
            )
            x = x + self.pos_emb(positions)
        x = self.drop(x)

        # Pass through all blocks
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = block(x, kv_cache=layer_cache)
            new_kv_cache.append(layer_new_cache)

        # Final norm + output head
        x = self.ln_f(x)
        logits = self.head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss, new_kv_cache
