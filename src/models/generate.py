"""Autoregressive token generation with KV-cache acceleration.

This module provides the generation logic independently of any specific
variant. Any model that exposes forward(idx, targets, kv_cache) → (logits, loss, cache)
and has a config.seq_len attribute can be used.

Decoding strategies supported:
- Greedy (temperature=0)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Any combination of the above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generate tokens autoregressively from any decoder-only Transformer variant.

    With use_cache=True (default):
    - First step: process the entire prompt, cache all K/V
    - Subsequent steps: only process the NEW token (seq_len=1),
      reusing cached K/V from prior positions. Much faster.

    Without cache (use_cache=False):
    - Every step reprocesses the full sequence. Slower but simpler.
      Useful for debugging or when you need to modify prior tokens.

    Args:
        model: Any decoder-only Transformer with interface:
               forward(idx, targets=None, kv_cache=None) → (logits, loss, new_kv_cache)
               Must also have model.config.seq_len.
        prompt: Starting token indices, shape (batch_size, prompt_len).
        max_new_tokens: How many new tokens to generate.
        temperature: Scales logits before softmax.
                    0.0 = greedy, 1.0 = natural, <1.0 = sharper, >1.0 = flatter.
        top_k: If set, only sample from the top-k tokens.
        top_p: If set, nucleus sampling (cumulative probability threshold).
        use_cache: Whether to use KV-cache for fast generation.

    Returns:
        Token indices including generated tokens,
        shape (batch_size, prompt_len + max_new_tokens).
    """
    model.eval()
    idx = prompt
    kv_cache = None
    seq_len = model.config.seq_len

    for step in range(max_new_tokens):
        if use_cache:
            if kv_cache is None:
                # First step: process the full prompt, build initial cache
                input_ids = idx
            else:
                # Subsequent steps: only the last token (new one we just picked)
                input_ids = idx[:, -1:]
        else:
            # No cache: always process the full (growing) sequence
            input_ids = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]

        # Forward pass
        logits, _, new_kv_cache = model(input_ids, kv_cache=kv_cache if use_cache else None)

        if use_cache:
            kv_cache = new_kv_cache

        # Get logits for the last position only: (B, vocab_size)
        logits = logits[:, -1, :]

        # --- Temperature = 0: greedy decoding ---
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            continue

        # --- Apply temperature scaling ---
        logits = logits / temperature

        # --- Top-k filtering ---
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # --- Top-p (nucleus) filtering ---
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # --- Sample from the (filtered) distribution ---
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx
