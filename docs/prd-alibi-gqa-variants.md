# PRD: ALiBi and GQA Variants (V2, V3)

**Label:** `ready-for-agent`

---

## Problem Statement

The project has two implemented variants (V0 Vanilla, V1 Modern) but the Phase 6 goal requires two additional attention mechanism variants for controlled comparison: ALiBi (position encoding via linear biases) and GQA (grouped-query attention). Without these, the project cannot answer its core research question: how do different attention mechanisms compare at the same token budget and model scale?

## Solution

Implement two new attention modules and register them as variants:

1. **V2 (ALiBi)**: An attention module that removes RoPE and instead passes precomputed per-head linear bias slopes to `flash_attn_func`. Position information comes entirely from the attention logit biases, not from Q/K rotations. Everything else (RMSNorm, SwiGLU, Flash Attention kernel, model shell) stays identical to V1.

2. **V3 (GQA)**: An attention module that uses fewer KV heads than query heads (`n_kv_head = n_head / 4`). K and V projections are smaller; each KV head serves a group of query heads. RoPE is applied to both Q and the reduced K. Everything else stays identical to V1.

Both variants use the `ModernTransformer` shell (parameterized in the flash-attention-backend spec) and the `flash_attn` library for kernel dispatch.

## User Stories

1. As a researcher, I want an ALiBi variant that is identical to V1 except for position encoding, so that I can isolate ALiBi's effect on training loss.
2. As a researcher, I want a GQA variant that is identical to V1 except for head structure, so that I can isolate GQA's effect on loss and parameter efficiency.
3. As a developer, I want to train V2 with `python scripts/train.py --variant alibi --scale main`, so that the CLI experience is consistent with existing variants.
4. As a developer, I want to train V3 with `python scripts/train.py --variant gqa --scale main`, so that the CLI experience is consistent with existing variants.
5. As a developer, I want V2's ALiBi slopes to follow the paper's geometric series (`2^(-8i/n)`), so that the implementation matches published results.
6. As a developer, I want V3's KV head count to be `n_head / 4` at each scale (debug=1, main=2, stretch=3), so that the GQA ratio is consistent across scales.
7. As a developer, I want V2 to have no learned position embedding and no RoPE, so that position information comes solely from ALiBi biases.
8. As a developer, I want V3 to still use RoPE (applied to Q and reduced K), so that the only change from V1 is the head structure.
9. As a developer, I want both V2 and V3 to support KV-cache for generation, so that the attention interface is uniform across all variants.
10. As a developer, I want V2's KV-cache to work by caching raw K/V (no position encoding on them) and recomputing the ALiBi bias for the full sequence length at each step.
11. As a developer, I want V3's KV-cache to be smaller than V1's (fewer KV heads), so that the memory savings of GQA are visible during generation.
12. As a developer, I want both variants registered in the registry with proper VariantSpec entries, so that `registry.build("alibi", scale)` and `registry.build("gqa", scale)` work.
13. As a developer, I want YAML config files for V2 and V3 (`configs/model/alibi.yaml`, `configs/model/gqa.yaml`), so that the config-driven workflow is consistent.
14. As a developer, I want the ALiBi slopes precomputed once at init and stored as a buffer, so that they move with the model to GPU and are included in checkpoints.
15. As a developer, I want V2 to use `flash_attn_func(alibi_slopes=...)` for training, so that the ALiBi bias is fused into the Flash Attention kernel.
16. As a developer, I want V2 to use `flash_attn_with_kvcache(alibi_slopes=...)` for generation, so that ALiBi works correctly with cached keys.
17. As a developer, I want V3's GQA to work natively with `flash_attn_func` by passing Q with `n_head` heads and K/V with `n_kv_head` heads, so that no manual head repetition is needed.
18. As a developer, I want the parameter count difference for V3 (~6% fewer than V1 at main scale) documented in results rather than compensated, so that GQA's efficiency is visible.
19. As a developer, I want V2 and V3 to work with `torch.compile`, so that the standard training workflow applies.
20. As a developer, I want both variants capped at `seq_len` (no extrapolation), so that the comparison is controlled.
21. As a researcher, I want generation (greedy decoding) to work with V2 and V3, so that I can qualitatively inspect outputs.
22. As a developer, I want tests verifying that V2's attention pattern is position-dependent (different positions produce different attention distributions), so that ALiBi is actually encoding position.
23. As a developer, I want tests verifying that V3 with `n_kv_head < n_head` produces correct output shapes and a smaller KV-cache than V1, so that GQA is actually reducing parameters.

## Implementation Decisions

- **V2 module: `src/models/alibi_attention.py`**. Contains `ALiBiAttention` class. Computes ALiBi slopes at init as `2^(-8 * i / n_head)` for head `i`, registers as buffer. QKV projection splits Q/K/V, does NOT apply RoPE to anything. Passes `alibi_slopes` to `flash_attn_func` (training) or `flash_attn_with_kvcache` (generation). KV-cache stores raw K/V.
- **V3 module: `src/models/gqa_attention.py`**. Contains `GQAAttention` class. QKV projection produces Q with `n_head` heads and K/V with `n_kv_head` heads. RoPE applied to Q (full heads) and K (reduced heads). Passes mismatched head counts to `flash_attn_func`/`flash_attn_with_kvcache` which handle GQA natively. KV-cache shape: `(B, n_kv_head, T, d_head)`.
- **n_kv_head configuration**: Stored in `ModelConfig.n_kv_head`. Set by registry to `n_head // 4` for V3. The registry computes it per-scale: debug=1, main=2, stretch=3.
- **Model shell reuse**: Both V2 and V3 use `ModernTransformer` with their respective attention class passed in. No new model classes.
- **Registry entries**: `VARIANTS["alibi"]` and `VARIANTS["gqa"]` with appropriate VariantSpec values. V2: `position_encoding="alibi"`, `attention_type="flash_alibi"`. V3: `attention_type="flash_gqa"`, `n_kv_head` set per scale.
- **YAML configs**: `configs/model/alibi.yaml` and `configs/model/gqa.yaml` with per-scale dimensions and variant identity fields.
- **No position embedding table for V2**: Same as V1 — `ModernTransformer` shell has no `wpe`. ALiBi handles position inside the attention module.
- **ALiBi bias capping**: Bias matrix precomputed to `seq_len` maximum. No extrapolation support in this implementation.

## Testing Decisions

- **Seam 1: `ALiBiAttention.forward()`** — Unit tests: output shape (B, T, d_model), KV-cache shape, position-dependent behavior (same input at different positions produces different attention patterns due to ALiBi biases), slopes match expected geometric series.
- **Seam 2: `GQAAttention.forward()`** — Unit tests: output shape, KV-cache has `n_kv_head` heads (not `n_head`), parameter count is smaller than full MHA equivalent.
- **Seam 3: `registry.build("alibi", scale)` and `registry.build("gqa", scale)`** — Integration tests: model constructs, forward pass produces logits of correct shape, loss computation works, loss decreases over 10 training steps.
- **Seam 4: Generation** — Greedy generation produces deterministic output, cached and uncached generation match for both V2 and V3.
- **Prior art**: `tests/test_modern_model.py` patterns — same structure of component-level tests + model-level integration tests. New tests in `tests/test_alibi_model.py` and `tests/test_gqa_model.py`.
- **Good tests**: Test observable behavior (shapes, loss decrease, generation consistency, cache correctness), not internal kernel dispatch or implementation details.

## Out of Scope

- ALiBi length extrapolation experiments (deferred per CONTEXT.md open questions)
- Training runs and benchmarks (separate from implementation)
- MQA (single KV head) — only GQA with `n_head / 4` ratio
- V4 (sparse attention) and V5 (linear attention)
- Evaluation framework
- Multi-GPU / distributed attention

## Further Notes

- V2 and V3 depend on the flash-attention-backend spec being complete first (ModernTransformer parameterization and flash_attn library installation).
- The `flash_attn` library natively supports both ALiBi (`alibi_slopes` param) and GQA (mismatched Q vs KV head counts). No custom kernel work needed.
- V3's ~6% parameter reduction at main scale is accepted per the updated invariants in CONTEXT.md.
- Both variants use the same experiment protocol as V0/V1: same data, token budget, optimizer, effective batch size, precision (bf16).
