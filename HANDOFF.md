# Handoff — Transformer Variant Lab

**Created:** 2026-07-02  
**Workspace:** `/home/asizu7/Desktop/Projects/TransformerVariants`  
**Git state:** `main` at `7c60974` — all committed, clean working tree  
**Conda env:** `transformer_lab` (Python 3.11, PyTorch 2.12.1)

---

## Project Summary

Single-GPU benchmark framework comparing decoder-only Transformer variants on an L4-24Q (24GB VRAM). Six variants (V0–V5) plus one sub-variant are planned, each isolating a single architectural change.

| Variant | Status |
|---------|--------|
| V0 — Vanilla (GPT-2) | ✅ Implemented + trained (main + stretch) |
| V1 — Modern (LLaMA) | ✅ Implemented + trained (main + stretch) |
| V2 — ALiBi | ✅ Implemented + debug trained |
| V3 — GQA | ✅ Implemented + debug trained |
| V4 — SWA | ✅ Implemented (FlashAttention + window_size) |
| V4-interleaved — SWA alternating layers | ✅ Implemented (per-layer config) |
| V5 — Linear (ELU+1 causal) | ✅ Implemented (LinearAttention module) |

All variants are registered in `src/models/registry.py`, have YAML configs in `configs/model/`, and have test coverage.

---

## What Was Done This Session

Nothing substantive — the session produced this handoff document only. No code changes were made.

---

## Implementation Details Worth Knowing

- **V4 (SWA):** Uses `FlashAttention` (inherits `FlashAttentionBase`) with `window_size = seq_len // 4` passed to `flash_attn_func`. No custom masks. Registered as `"swa"`.
- **V4-interleaved:** Same `FlashAttention` class but `registry.build()` creates per-layer configs — even layers get `window_size=None` (full attention), odd layers get `window_size=W`. Registered as `"swa_interleaved"`.
- **V5 (Linear):** Standalone `LinearAttention` module using ELU+1 feature map with causal recurrence (O(n·d²)). No RoPE, no KV-cache. Training-comparison only per ADR 0002.
- **Attention backend:** `FlashAttentionBase` dispatches to `flash_attn` library. Supports SWA natively via kernel `window_size` param.

---

## What Has NOT Been Done

- **No main/stretch training runs** for V2 (ALiBi), V3 (GQA), V4 (SWA), V4-interleaved, or V5 (Linear). Only debug-scale checkpoint exists for V2/V3.
- **Phase 8: Evaluation framework** — no spec, no code.
- **Phase 9: Visualization dashboard** — no code.
- **Phase 10: Large-scale data pipeline** — not started.
- **Phase 11: Fault-tolerant training** — basic checkpoint resume works, no fault injection.
- **Phase 12: Main benchmarks** — blocked on Phase 8.
- **Phase 13: Packaging** — final report, not started.

---

## Natural Next Steps

1. **Run V2–V5 at main scale** — all code exists, just launch training runs via `scripts/train.py --variant <name> --scale main --compile`.
2. **Build evaluation framework** (Phase 8) — standardized comparison metrics, plotting, statistical significance.
3. **Run controlled experiments** at main + stretch scale across all variants with 3+ seeds.
4. **Visualization dashboard** (Phase 9) — Streamlit + Plotly (deps already in pyproject.toml).

---

## Key Artifacts (Reference, Don't Duplicate)

| Artifact | Path |
|----------|------|
| Domain glossary | `CONTEXT.md` |
| Project status & training results | `docs/STATUS.md` |
| Phase index | `docs/PHASE_INDEX.md` |
| ALiBi/GQA PRD | `docs/prd-alibi-gqa-variants.md` |
| Flash Attention PRD | `docs/prd-flash-attention-backend.md` |
| ADR: SWA parameterizes flash | `docs/adr/0001-swa-parameterizes-flash-attention.md` |
| ADR: Linformer no KV-cache | `docs/adr/0002-linformer-no-kvcache-generation.md` |
| All Kiro specs (15 total) | `.kiro/specs/` |
| Variant registry (all build logic) | `src/models/registry.py` |
| Flash attention base class | `src/models/flash_attention_base.py` |
| Training script | `scripts/train.py` |
| Training shell helper | `scripts/train_v2_v3.sh` |

---

## Conventions & Gotchas

- **Run tests:** `conda run -n transformer_lab python -m pytest tests/ -v` (130+ tests)
- **Lint:** `ruff check src/ tests/ scripts/`
- **Install editable:** `pip install -e ".[dev]"`
- **flash-attn** is a separate optional dep — needed for V1–V5 training.
- `torch.compile` is used during training (`--compile`). Models must avoid graph breaks.
- All controlled comparisons must share: same data, same token budget, same optimizer, same precision (bf16).
- Parameter tolerance: ±5% between compared variants.
- Config YAML files: `configs/model/<variant>.yaml` with per-scale overrides.
- V5 (Linear) uses a sequential loop over timesteps — it will be slow. This is expected.

---

## Suggested Skills

| Skill | When to Use |
|-------|-------------|
| `tdd` | When building the evaluation framework — property-based testing with Hypothesis is already established in this project. |
| `codebase-design` | When designing the evaluation framework or dashboard. Use deep-module vocabulary to keep interfaces clean. |
| `domain-modeling` | If new terms arise (evaluation metrics, Pareto concepts). Keep `CONTEXT.md` canonical. |
| `diagnosing-bugs` | If training runs diverge, produce NaN, or flash_attn gives incorrect outputs. |
| `grilling` | Before committing to the evaluation framework design — stress-test the comparison methodology and metric choices. |

---

## Open Questions (Deferred)

- Evaluation framework primary axis: fixed compute vs fixed data vs Pareto?
- ALiBi extrapolation experiment (train-short / infer-long) — after controlled comparison.
- KV-Cache unification (concat-based 2-tuple vs pre-allocated 3-tuple) — design smell, not blocking.
- V5 performance: sequential recurrence is very slow; consider chunked-parallel version if needed for practical training at main scale.
