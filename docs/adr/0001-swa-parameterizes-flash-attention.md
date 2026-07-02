# ADR 0001: SWA (V4) Parameterizes FlashAttention Rather Than Subclassing

## Status

Accepted

## Context

V4 (Sliding Window Attention) is architecturally identical to the Modern Baseline (V1) with FlashAttention — same RoPE, same projections, same cache format. The only difference is passing `window_size=(W, W)` to the flash_attn kernel during training.

We had two options:
1. Create a thin `SlidingWindowAttention(FlashAttentionBase)` subclass (matching the pattern used by ALiBi and GQA).
2. Add a `window_size` field to `ModelConfig` and have `FlashAttention._extra_attn_kwargs()` include it when non-None.

## Decision

Parameterize `FlashAttention` via `ModelConfig.window_size`. No new attention class for V4.

## Rationale

- ALiBi and GQA each introduce *structural* differences (different projections, alibi slopes buffer, reduced KV heads). SWA introduces zero structural differences — it's one kernel argument.
- A dedicated class for one kwarg is over-engineering. The forward path, position encoding, projections, and cache format are byte-for-byte identical.
- The registry `VariantSpec` for `swa` points to the same `FlashAttention` class but with a config whose `window_size = seq_len // 4`.

## Consequences

- `FlashAttention`'s behavior varies based on config (window_size None vs a value). This is a mild violation of "one class = one behavior" but acceptable given the difference is purely a kernel dispatch parameter.
- Future readers may wonder why V4 doesn't have its own file. This ADR explains why.
- If V4 later needs structural differences (e.g., different head counts per window, dilated patterns), it should graduate to its own subclass at that point.
