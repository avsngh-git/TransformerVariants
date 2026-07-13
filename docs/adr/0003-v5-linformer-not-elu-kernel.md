# ADR 0003: V5 is Linformer (Low-Rank Projection), Not ELU+1 Kernel Attention

## Status

Superseded by ADR 0007

## Context

During evaluation framework design (2026-07-02), we discovered a spec/code mismatch:

- `CONTEXT.md` defines V5 as **Linformer**: low-rank E/F projection matrices (shape `seq_len × r`, r=64) that project K and V from length T down to rank r. Softmax is retained. RoPE is applied to Q and K before projection. Complexity is O(n·r·d).

- The implemented code (`src/models/linear_attention.py`) is an **ELU+1 causal kernel attention**: replaces softmax with φ(x) = ELU(x) + 1 feature map, uses a running accumulator (no softmax), no RoPE, complexity O(n·d²).

These are fundamentally different architectures testing different hypotheses:
- Linformer: "Can a low-rank approximation of full softmax attention match it?"
- ELU+1 kernel: "Can a non-softmax kernel replace attention entirely?"

## Decision

V5 is Linformer, matching the domain model in `CONTEXT.md`. The current ELU+1 implementation will be replaced.

## Rationale

1. The domain model was deliberate — Linformer tests a more directly comparable hypothesis against the other variants (all use softmax).
2. Linformer with RoPE gives position encoding parity with V1/V3/V4, isolating only the low-rank approximation as the variable.
3. The ELU+1 approach is a fundamentally different paradigm (kernel methods) that muddies the controlled comparison.

## Consequences

- `src/models/linear_attention.py` must be rewritten as Linformer
- Associated tests (`test_linear_attention.py`, `test_linear_properties.py`, `test_linear_registry.py`) must be updated
- Registry entry for "linear" variant must be updated
- `configs/model/linear.yaml` must add `projection_rank: 64`
- ADR 0002 (no KV-cache for V5) still applies — E/F matrices are tied to fixed seq_len
