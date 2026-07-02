# Handoff — Transformer Variant Lab

**Created:** 2026-07-02  
**Workspace:** `/home/asizu7/Desktop/Projects/TransformerVariants`  
**Git state:** `main` at `7c60974` — all committed, clean working tree  
**Conda env:** `transformer_lab` (Python 3.11, PyTorch 2.12.1)

---

## Project Summary

Single-GPU benchmark framework comparing decoder-only Transformer variants on an L4-24Q. Six variants (V0–V5) are planned, each isolated to a single architectural change against a baseline.

| Variant | Status |
|---------|--------|
| V0 — Vanilla (GPT-2) | ✅ Implemented + trained |
| V1 — Modern (LLaMA) | ✅ Implemented + trained |
| V2 — ALiBi | ✅ Implemented (not yet trained) |
| V3 — GQA | ✅ Implemented (not yet trained) |
| V4 — SWA | Spec exists, not implemented |
| V5 — Linformer | Spec exists, not implemented |

---

## What Was Done This Session

Nothing substantive — the session was used solely to produce this handoff document. No code changes were made.

---

## Recent Work (Last 3 Commits)

See `git log --oneline -3` for details. Key changes:
- ALiBi (V2) and GQA (V3) attention modules implemented with `flash_attn` backend
- `FlashAttentionBase` — shared base module for all flash-based variants
- Variant registry refactored for dependency-injected model construction
- `RunConfigBuilder` extracted from trainer to separate config assembly from training loop
- Training script (`scripts/train.py`) simplified; shell helpers added (`scripts/train_v2_v3.sh`)

---

## What the Next Session Should Focus On

No specific focus was provided. The natural next steps (from `docs/STATUS.md` and the project roadmap) are:

1. **Train V2 (ALiBi) and V3 (GQA)** at main scale — code exists, runs haven't happened yet.
2. **Implement V4 (SWA)** — spec exists at `.kiro/specs/swa-attention/`, ADR at `docs/adr/0001-swa-parameterizes-flash-attention.md`. SWA reuses `FlashAttentionBase` with `window_size` param.
3. **Implement V5 (Linformer)** — spec exists at `.kiro/specs/linear-attention/`, ADR at `docs/adr/0002-linformer-no-kvcache-generation.md`. Standalone `LinearAttention(nn.Module)`.
4. **Implement V4-interleaved sub-variant** — spec at `.kiro/specs/swa-interleaved/`.
5. **Build evaluation framework** — Phase 8; no spec yet.

---

## Key Artifacts (Reference, Don't Duplicate)

| Artifact | Path |
|----------|------|
| Domain glossary | `CONTEXT.md` |
| Project status | `docs/STATUS.md` |
| Phase index | `docs/PHASE_INDEX.md` |
| ALiBi/GQA PRD | `docs/prd-alibi-gqa-variants.md` |
| Flash Attention PRD | `docs/prd-flash-attention-backend.md` |
| ADR: SWA parameterizes flash | `docs/adr/0001-swa-parameterizes-flash-attention.md` |
| ADR: Linformer no KV-cache | `docs/adr/0002-linformer-no-kvcache-generation.md` |
| SWA spec | `.kiro/specs/swa-attention/` |
| Linear attention spec | `.kiro/specs/linear-attention/` |
| SWA-interleaved spec | `.kiro/specs/swa-interleaved/` |
| All specs index | `.kiro/specs/` (15 specs total) |
| Training script | `scripts/train.py` |
| Flash base module | `src/models/flash_attention_base.py` |
| Variant registry | `src/models/registry.py` |

---

## Conventions & Gotchas

- **Run tests:** `conda run -n transformer_lab python -m pytest tests/ -v`
- **Lint:** `ruff check src/ tests/ scripts/`
- **Install editable:** `pip install -e ".[dev]"`
- **flash-attn** is a separate optional dep (`pip install flash-attn`) — needed for V1–V5 training but not for V0.
- `torch.compile` is used during training (pass `--compile`). Models must avoid graph breaks.
- All controlled comparisons must share: same data, same token budget, same optimizer config, same precision (bf16).
- Parameter tolerance: ±5% between compared variants.
- Config YAML files live in `configs/model/` — one per variant, with per-scale overrides.
- The `FlashAttentionBase` module handles KV-cache allocation and backend dispatch. New flash-based variants inherit from it.

---

## Suggested Skills

| Skill | When to Use |
|-------|-------------|
| `tdd` | When implementing V4/V5 — build test-first, use property-based testing with Hypothesis (already in the project). |
| `codebase-design` | When designing the evaluation framework or refining the `FlashAttentionBase` → SWA inheritance. The deep-module vocabulary will help keep seams clean. |
| `domain-modeling` | If new terms arise (e.g., evaluation metrics, Pareto frontier concepts). Keep `CONTEXT.md` canonical. |
| `diagnosing-bugs` | If training runs diverge or flash_attn produces incorrect outputs — use the diagnosis loop. |
| `grilling` | Before committing to the evaluation framework design — stress-test the comparison methodology. |

---

## Open Questions (Deferred)

- Evaluation framework primary axis: fixed compute vs fixed data vs Pareto?
- ALiBi extrapolation experiment (train-short / infer-long) — after controlled comparison.
- KV-Cache unification (concat-based 2-tuple vs pre-allocated 3-tuple) — after all variants are implemented.
