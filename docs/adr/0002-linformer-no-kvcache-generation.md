# ADR 0002: V5 (Linear/Linformer) Does Not Support KV-Cache Generation

## Status

Accepted

## Context

Linformer's projection matrices E and F have shape `(seq_len, r)` — they are constructed at model-init time for a specific fixed sequence length. During autoregressive generation, the KV cache grows one token at a time. There is no principled way to incrementally apply a fixed-length projection to a growing cache: you'd need to re-project the entire sequence from scratch at every generation step, which eliminates the cache benefit and adds O(n) cost per token.

## Decision

V5 (`--variant linear`) does not implement KV-cache generation. The `forward()` method returns `(output, None)` for kv_cache — no cache is allocated or returned.

Attempting to call `allocate_kv_cache()` or pass a `kv_cache` argument will raise `NotImplementedError`.

## Rationale

- The primary research question for V5 is "can linear-complexity attention match full-attention training perplexity?" This is fully answerable with training runs only.
- Generation quality comparison is not a stated goal in the Phase 7 experiment contract.
- Implementing a degraded generation path (full re-projection each step) would be misleading — it would appear to work but produce different computational characteristics than the training path.

## Consequences

- `generate.py` will fail if called with the `linear` variant. This is an explicit, documented limitation rather than a silent failure.
- When the evaluation framework (Phase 8) is designed, generation-based metrics (perplexity on held-out data via teacher forcing still works) must distinguish between variants that support generation and those that don't.
- If generation support is needed in the future, Performer (FAVOR+ with causal formulation) would be the correct replacement.
