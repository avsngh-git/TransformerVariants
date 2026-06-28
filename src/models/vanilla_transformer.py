"""Vanilla GPT-2 style decoder-only Transformer.

This module assembles the full model from components:
- Token + position embeddings
- Stack of TransformerBlocks (each: Pre-LN → Attention → residual, Pre-LN → FFN → residual)
- Final LayerNorm + linear output head (logits over vocabulary)
- Greedy generation method

The architecture matches GPT-2:
- Learned position embeddings
- Pre-LayerNorm (normalize before sublayer, not after)
- Weight tying between token embeddings and output head
- GPT-2 style weight initialization
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig
from src.models.attention import CausalSelfAttention
from src.models.ffn import FeedForward


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    The data flow (Pre-LN pattern):
        x → LayerNorm → Attention → + residual → LayerNorm → FFN → + residual → out
                                     ↑                              ↑
                                     x (skip connection)            x (skip connection)

    Why Pre-LN? See docs/learnings_from_project.md — it's more stable for training
    because gradients flow cleanly through the residual stream.

    Args:
        config: ModelConfig with all hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # LayerNorm before attention (Pre-LN)
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        # The attention sublayer
        self.attn = CausalSelfAttention(config)

        # LayerNorm before FFN (Pre-LN)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        # The FFN sublayer
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            kv_cache: Optional KV-cache for this layer's attention.

        Returns:
            Tuple of (output, new_kv_cache):
            - output: shape (batch_size, seq_len, d_model)
            - new_kv_cache: updated cache for this layer
        """
        # Attention with residual connection
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out

        # FFN with residual connection
        x = x + self.ffn(self.ln2(x))

        return x, new_kv_cache


class VanillaTransformer(nn.Module):
    """Complete decoder-only Transformer (GPT-2 style).

    Architecture:
        1. Token embedding + Position embedding (added together)
        2. Dropout (optional)
        3. N × TransformerBlock
        4. Final LayerNorm
        5. Linear head → logits over vocabulary

    Weight tying: the token embedding matrix is shared with the output head.
    This means the model uses the same vectors to "understand" a token (input)
    and "predict" a token (output). Saves parameters and empirically works well.

    Args:
        config: ModelConfig with all hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # --- Embeddings ---
        # Token embedding: maps token IDs (integers) to vectors
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Position embedding: maps position indices to vectors (learned, not sinusoidal)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)

        # Dropout after embedding (before entering the Transformer stack)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer blocks ---
        # ModuleList so PyTorch registers them as submodules (for .parameters(), .to(), etc.)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # --- Output ---
        # Final LayerNorm (standard in Pre-LN: one final norm after all blocks)
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)

        # Output head: projects d_model → vocab_size to get logits for next-token prediction
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- Weight tying ---
        # The output head shares weights with the token embedding.
        # tok_emb.weight is (vocab_size, d_model), head.weight is also (vocab_size, d_model).
        # By sharing them, the model uses the same representation space for input and output.
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        # --- Initialize weights ---
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2 style weight initialization.

        - All linear layers and embeddings: N(0, 0.02)
        - Residual projections (attn out_proj, ffn fc2): scaled by 1/sqrt(2*n_layers)
        - LayerNorm: weight=1, bias=0 (if bias exists)
        """
        init_std = 0.02
        residual_std = init_std / math.sqrt(2 * self.config.n_layer)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is a residual projection (output of attention or FFN)
                if name.endswith("out_proj") or name.endswith("fc2"):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                # Initialize bias if it exists
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass: token IDs → logits (and optionally loss).

        Args:
            idx: Token indices of shape (batch_size, seq_len). Integer tensor.
            targets: Target token indices for loss computation. Same shape as idx.
                     If None, only returns logits (used during generation).
            kv_cache: Optional list of KV-cache tuples, one per layer.
                      Pass None during training. During generation, pass the
                      cache returned from the previous step.

        Returns:
            Tuple of (logits, loss, new_kv_cache):
            - logits: (batch_size, seq_len, vocab_size)
            - loss: scalar cross-entropy loss, or None if targets not provided.
            - new_kv_cache: list of (k, v) tuples for each layer.
        """
        B, T = idx.shape

        # Determine position offset from cache
        # If we have cached positions, the new tokens start after them
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].size(2)  # number of cached positions
        else:
            past_len = 0

        total_len = past_len + T
        assert total_len <= self.config.seq_len, (
            f"Total sequence length {total_len} exceeds model max {self.config.seq_len}"
        )

        # Create position indices starting from past_len
        # During cached generation: pos = [past_len] (just one new position)
        # During training (no cache): pos = [0, 1, 2, ..., T-1]
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)

        # Embed tokens and positions, then add them together
        tok = self.tok_emb(idx)   # (B, T, d_model)
        pos = self.pos_emb(pos)   # (T, d_model) — broadcasts over batch
        x = self.drop(tok + pos)  # (B, T, d_model)

        # Pass through all Transformer blocks, collecting updated caches
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = block(x, kv_cache=layer_cache)
            new_kv_cache.append(layer_new_cache)

        # Final LayerNorm
        x = self.ln_f(x)  # (B, T, d_model)

        # Project to vocabulary logits
        logits = self.head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.reshape(-1),               # (B*T,)
            )

        return logits, loss, new_kv_cache


