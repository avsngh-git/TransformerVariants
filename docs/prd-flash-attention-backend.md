# PRD: Flash Attention Backend Infrastructure

**Label:** `ready-for-agent`

---

## Problem Statement

The project's modern variants (V1–V5) need a common high-performance attention kernel for fair comparison. V1 currently uses PyTorch's `scaled_dot_product_attention` (SDPA), which auto-selects between Flash, memory-efficient, and math kernels — the developer has no control over which kernel runs. Upcoming variants (V2 ALiBi, V3 GQA) require explicit Flash Attention features (ALiBi slopes parameter, native GQA head count mismatch) that SDPA doesn't expose. Additionally, `ModernTransformer` is currently hardcoded to `ModernAttention`, making it impossible to swap attention modules for new variants without creating separate model classes.

## Solution

1. Add an `attention_backend` field to `ModelConfig` allowing explicit kernel selection ("sdpa" or "flash_attn").
2. Install the `flash_attn` library (Dao AI Lab) and create a Flash Attention–based attention module that V1 can use.
3. Parameterize `ModernTransformer` to accept an attention class, so V2–V5 can plug in different attention modules without duplicating the model shell.
4. Use dual-path dispatch: `flash_attn_func` for training (full sequences), `flash_attn_with_kvcache` for generation (single-token steps with cache).
5. Ensure `torch.compile` compatibility with the flash_attn backend.
6. V1's existing SDPA path remains the default — the flash_attn backend is opt-in via config. Within controlled experiments, all variants use the same backend.

## User Stories

1. As a developer running controlled experiments, I want all V1–V5 variants to use the same attention kernel, so that throughput and memory comparisons are apples-to-apples.
2. As a developer, I want to switch between SDPA and flash_attn via a config field, so that I can compare kernel performance without code changes.
3. As a developer adding a new variant (V2, V3), I want to plug a custom attention module into ModernTransformer without duplicating the model shell, so that I only write the novel attention logic.
4. As a developer, I want ModernTransformer to accept any attention class that follows the standard interface, so that future variants (V4, V5) can reuse the same shell.
5. As a developer, I want `flash_attn_func` used for training and `flash_attn_with_kvcache` for generation, so that I get Flash Attention's memory benefits during training and fused KV-cache management during generation.
6. As a developer, I want the flash_attn backend to work with `torch.compile`, so that I can combine both optimizations during training runs.
7. As a developer, I want V1's existing SDPA results to remain valid, so that I don't need to re-run prior experiments.
8. As a developer, I want the attention module interface to be uniform (`forward(x, kv_cache=None) -> (output, new_kv_cache)`), so that all variants are interchangeable at the block level.
9. As a developer, I want clear error messages if `flash_attn` is not installed but the backend is set to "flash_attn", so that I know what dependency is missing.
10. As a developer, I want the registry to pass the attention class to ModernTransformer based on the VariantSpec, so that variant registration is the single point of configuration.
11. As a developer, I want generation (KV-cache path) to use Flash Attention via `flash_attn_with_kvcache`, so that both training and inference benefit from the optimized kernel.
12. As a developer, I want the ModernAttention (SDPA-based) module to remain unchanged as the default, so that existing tests pass without modification.
13. As a developer, I want a new FlashAttention module that wraps `flash_attn_func` and `flash_attn_with_kvcache` with RoPE support, so that V1 can use it as an alternative backend.
14. As a developer, I want the flash_attn backend to produce numerically equivalent results to SDPA (within floating-point tolerance), so that switching backends doesn't change model behavior.

## Implementation Decisions

- **`attention_backend` field**: Added to `ModelConfig` with values "sdpa" (default) or "flash_attn". This is a compute optimization choice, not an architectural change.
- **New module `flash_attention.py`**: Implements `FlashAttention` class using `flash_attn_func` for training and `flash_attn_with_kvcache` for generation. Applies RoPE to Q/K before calling the kernel. Same interface as `ModernAttention`.
- **`ModernTransformer` parameterization**: Constructor accepts an `attention_class` parameter (defaults to `ModernAttention` for backward compatibility). Each `ModernTransformerBlock` instantiates the provided attention class instead of hardcoded `ModernAttention`.
- **Registry integration**: `VariantSpec` gains an `attention_class` field. The registry passes it to `ModernTransformer`. V1's spec defaults to `ModernAttention` (SDPA); when `attention_backend="flash_attn"` is set, the registry uses `FlashAttention` instead.
- **Dual-path dispatch inside `FlashAttention`**: If `kv_cache is None` and `T > 1`, use `flash_attn_func` (training). Otherwise use `flash_attn_with_kvcache` (generation with cache).
- **ALiBi slopes parameter**: `FlashAttention` accepts an optional `alibi_slopes` tensor. When None, behaves as standard attention (RoPE). When provided, passes slopes to `flash_attn_func`'s `alibi_slopes` kwarg. This prepares the module for V2 without V2-specific code.
- **`flash_attn` library dependency**: Added as an optional dependency. The `FlashAttention` module raises `ImportError` with instructions if the library is missing.
- **`torch.compile` compatibility**: The FlashAttention module is tested under `torch.compile` to ensure no graph breaks or recompilation.
- **RoPE buffer handling**: `FlashAttention` precomputes RoPE frequencies as buffers (same as `ModernAttention`). V2 will skip RoPE and use ALiBi slopes instead — this is controlled by the attention class, not the shell.

## Testing Decisions

- **Seam 1: `FlashAttention.forward()`** — Unit tests verify output shape, KV-cache shape, and numerical equivalence with `ModernAttention` (SDPA) given same inputs and weights.
- **Seam 2: `registry.build()`** — Integration test that V1 with flash_attn backend builds, runs a forward pass, and produces gradients without error.
- **Seam 3: `torch.compile` + flash_attn** — Test that a compiled model using FlashAttention runs multiple steps without recompilation.
- **Prior art**: `tests/test_modern_model.py` — same patterns (output shape, KV-cache shape, cached vs uncached equivalence, generation determinism). New tests go in `tests/test_flash_attention.py`.
- **Good tests**: Test external behavior (shapes, numerical equivalence, error conditions), not implementation details (which internal function was called). Test at the highest seam possible (registry integration tests cover the full stack).

## Out of Scope

- Implementing ALiBi or GQA attention modules (that's spec 2)
- Training runs or benchmark results
- Changes to VanillaTransformer (V0)
- Evaluation framework or generation quality testing
- Multi-GPU / distributed attention

## Further Notes

- The L4-24Q GPU has compute capability 8.9, fully supported by flash_attn.
- PyTorch 2.12.1 + CUDA 13.0 is the target environment.
- The `attention_backend` concept is captured in `CONTEXT.md` under "Attention Backend" in the glossary.
- V1's existing SDPA-based benchmarks remain the baseline. The flash_attn backend will be used for the Phase 6 controlled comparison (V1 vs V2 vs V3).
