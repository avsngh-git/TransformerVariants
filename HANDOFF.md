# Handoff — Transformer Variant Lab

**Created:** 2026-07-02  
**Workspace:** `/home/asizu7/Desktop/Projects/TransformerVariants`  
**Git branch:** `main` at `cc4b41a` (clean except this file)  
**Conda env:** `transformer_lab` (Python 3.11, PyTorch 2.12.1)

---

## Session Summary

This session was brief — no substantive code or document changes were made. The user requested a handoff document for a fresh agent to continue work.

---

## Current State

### Completed
- All 5 attention variants (V0–V5) implemented and passing 220 tests
- V5 (Linformer) rewritten from ELU+1 kernel to proper Linformer architecture
- Evaluation framework PRD published (GitHub Issue #17)
- Domain model and glossary up to date in `CONTEXT.md`

### Outstanding Work (from tasks.md)

The **linformer-rewrite** spec has optional property tests remaining (tasks 2.2–2.8 and 5.3). These are marked with `*` (optional) but would strengthen correctness guarantees:

- 7 property-based tests using Hypothesis (output shape, numerical stability, batch independence, attention weight normalization, RoPE position sensitivity, projection rank shape invariant, invalid config rejection)
- File: `tests/test_linear_properties.py` — needs rewriting with Linformer properties

**Note:** Hypothesis may not be installed in the conda env. Install with `pip install hypothesis` before running property tests.

### Two Parallel Tracks Available

**Track A — Training runs:** Execute V2–V5 at main scale, 3 seeds each. All code ready. Command: `scripts/train.py --variant <name> --scale main --compile`. GPU-bound (~67h total).

**Track B — Evaluation framework:** PRD at GitHub Issue #17. Create a Kiro spec and implement `src/evaluation/`. Can proceed in parallel with training since it only needs debug-scale checkpoints for testing.

---

## Key Artifacts (Do Not Duplicate)

| Artifact | Location |
|----------|----------|
| Domain glossary + eval terms | `CONTEXT.md` |
| Project status | `docs/STATUS.md` |
| Evaluation PRD | `docs/prd-evaluation-framework.md` |
| GitHub Issue #17 (eval framework) | https://github.com/avsngh-git/TransformerVariants/issues/17 |
| ADR: V5 is Linformer | `docs/adr/0003-v5-linformer-not-elu-kernel.md` |
| Linformer rewrite spec | `.kiro/specs/linformer-rewrite/` (requirements, design, tasks) |
| Variant registry | `src/models/registry.py` |
| Training script | `scripts/train.py` |

---

## Conventions

- **Tests:** `conda run -n transformer_lab python -m pytest tests/ -v`
- **Lint:** `ruff check src/ tests/ scripts/`
- **Install:** `pip install -e ".[dev]"`
- **Config YAML:** `configs/model/<variant>.yaml`
- **flash-attn** required for V1–V4 training
- **torch.compile** used for training (`--compile` flag)
- V5 Linformer has no KV-cache (training-only variant)
- `gh` CLI authenticated as `avsngh-git`

---

## Suggested Skills

| Skill | When to Invoke |
|-------|----------------|
| `tdd` | Building the evaluation framework test-first, or completing the Linformer property tests |
| `codebase-design` | Designing `src/evaluation/` module interfaces — probes, comparisons, visualizations |
| `domain-modeling` | If new evaluation or metric terms need formal definition |
| `diagnosing-bugs` | If training runs diverge, produce NaN, or give unexpected loss curves |
| `grilling` | Stress-test the evaluation framework design before committing to implementation |
| `to-issues` | Breaking the evaluation PRD into independently-grabbable GitHub issues |

---

## Open Questions

- ALiBi extrapolation experiment — deferred until controlled comparison done
- KV-cache unification across variants — deferred, non-blocking
- Hypothesis library not confirmed installed in conda env
- Property tests (tasks 2.2–2.8, 5.3) are optional but recommended for spec completeness
