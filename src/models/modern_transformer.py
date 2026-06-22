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

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig
from src.models.rmsnorm import RMSNorm
from src.models.modern_attention import ModernAttention
from src.models.swiglu_ffn import SwiGLUFeedForward


class ModernTransformerBlock(nn.Module):
    """A single Transformer block with modern components.

    Data flow (Pre-RMSNorm pattern):
        x → RMSNorm → Attention(RoPE) → + residual → RMSNorm → SwiGLU → + residual → out

    Args:
        config: ModelConfig with all hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = ModernAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config)

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
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding only (no position embedding — RoPE handles position)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Dropout after embedding
        self.drop = nn.Dropout(config.dropout)

        # Stack of modern Transformer blocks
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final RMSNorm
        self.ln_f = RMSNorm(config.d_model)

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
                if name.endswith("out_proj") or name.endswith("w_down"):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

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
            past_len = kv_cache[0][0].size(2)
        else:
            past_len = 0

        total_len = past_len + T
        assert total_len <= self.config.seq_len, (
            f"Total sequence length {total_len} exceeds model max {self.config.seq_len}"
        )

        # Token embedding only (no position embedding!)
        x = self.tok_emb(idx)  # (B, T, d_model)
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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively (same interface as V0)."""
        kv_cache = None

        for step in range(max_new_tokens):
            if use_cache:
                if kv_cache is None:
                    input_ids = idx
                else:
                    input_ids = idx[:, -1:]
            else:
                input_ids = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]

            logits, _, new_kv_cache = self(input_ids, kv_cache=kv_cache if use_cache else None)

            if use_cache:
                kv_cache = new_kv_cache

            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)
                continue

            logits = logits / temperature

            if top_k is not None:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
