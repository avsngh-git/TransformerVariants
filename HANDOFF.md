# Handoff — Transformer Variant Lab

**Created:** 2026-07-03 (updated)  
**Workspace:** `/home/asizu7/Desktop/Projects/TransformerVariants`  
**Git branch:** `main` at `4684e43` — uncommitted architecture refactors  
**Conda env:** `transformer_lab` (Python 3.11, PyTorch 2.12.1)  
**GitHub:** `gh` authenticated as `avsngh-git`

---

## Current State

All implementation specs are **complete**:
- Linformer rewrite spec: 20/20 tasks ✅
- Evaluation framework spec: 35/35 tasks ✅
- Full test suite: **430 tests pass** (~61s)

**Architecture refactors (this session, uncommitted):**
1. **Evaluation Pipeline** — `src/evaluation/pipeline.py` (new deep module), `scripts/evaluate.py` rewritten to thin CLI shell
2. **Registry VariantSpec** — `src/models/registry.py` refactored with `config_overrides` callables, no more variant-specific conditionals in `build()`
3. **ProbeTarget seam** — `src/evaluation/probe_target.py` (Protocol + ModelProbeAdapter), decouples probes from model internals
4. **AttentionModule Protocol** — `src/models/attention_protocol.py` formalizes the attention interface, annotations updated in both transformer blocks

---

## Variant Implementation Status

| Variant | Code | Trained (main) | Trained (stretch) |
|---------|------|----------------|-------------------|
| V0 — Vanilla | ✅ | ✅ | ✅ |
| V1 — Modern | ✅ | ✅ | ✅ |
| V2 — ALiBi | ✅ | ❌ | ❌ |
| V3 — GQA | ✅ | ❌ | ❌ |
| V4 — SWA | ✅ | ❌ | ❌ |
| V4-interleaved | ✅ | ❌ | ❌ |
| V5 — Linformer | ✅ | ❌ | ❌ |

---

## What the Next Session Should Focus On

**Track A: Commit the architecture refactors** — Stage and commit the 4 refactors. All tests pass, zero breaking changes.

**Track B: Training runs** — Run V2–V5 at main scale with 3 seeds each. All code is ready.  
Command: `scripts/train.py --variant <name> --scale main --compile`  
GPU-bound (~67 hours total for 7 variants × 3 seeds).

**Track C: Run evaluation on existing checkpoints** — V0 and V1 have main + stretch checkpoints. Run:
```bash
python scripts/evaluate.py \
  --checkpoints checkpoints/vanilla_main_s*/ checkpoints/modern_main_s*/ \
  --output reports/v0_v1_comparison/
```
This validates the pipeline end-to-end on real data before launching all training runs.

**Track D: Refactor probes to use ProbeTarget** — The `ProbeTarget` protocol and `ModelProbeAdapter` are defined but probes haven't been rewritten to use them yet. This is optional (existing tests pass) but would complete the decoupling.

**Track E: Dashboard/visualization improvements** — The framework produces PNGs and `summary.md`. A future Phase 9 could add an interactive HTML dashboard (Plotly/Dash).

---

## Key Artifacts (Reference, Don't Duplicate)

| Artifact | Path |
|----------|------|
| Domain glossary (with eval + infra terms) | `CONTEXT.md` |
| Project status | `docs/STATUS.md` |
| Evaluation framework PRD | `docs/prd-evaluation-framework.md` |
| GitHub Issue #17 | https://github.com/avsngh-git/TransformerVariants/issues/17 |
| ADR: V4 parameterizes FlashAttention | `docs/adr/0001-swa-parameterizes-flash-attention.md` |
| ADR: V5 no KV-cache | `docs/adr/0002-linformer-no-kvcache-generation.md` |
| ADR: V5 is Linformer | `docs/adr/0003-v5-linformer-not-elu-kernel.md` |
| Linformer spec (complete) | `.kiro/specs/linformer-rewrite/` |
| Evaluation framework spec (complete) | `.kiro/specs/evaluation-framework/` |
| Architecture review (HTML) | `architecture-review-20260703.html` |
| Evaluation Pipeline module | `src/evaluation/pipeline.py` |
| ProbeTarget protocol + adapter | `src/evaluation/probe_target.py` |
| AttentionModule protocol | `src/models/attention_protocol.py` |
| Evaluation CLI (thin shell) | `scripts/evaluate.py` |
| Evaluation source | `src/evaluation/` (pipeline, flops, metrics, probes, probe_target, comparison, visualizations) |
| Variant registry (refactored) | `src/models/registry.py` |
| Training script | `scripts/train.py` |

---

## Conventions & Gotchas

- **Run tests:** `conda run -n transformer_lab python -m pytest tests/ -v` (430 tests, ~61s)
- **Lint:** `ruff check src/ tests/ scripts/`
- **Install:** `pip install -e ".[dev]"`
- **flash-attn** needed for V1–V4 training
- `torch.compile` for training (`--compile`)
- Config YAML: `configs/model/<variant>.yaml`
- V5 Linformer has no KV-cache (training-only variant)
- Evaluation framework only needs checkpoints + `metrics.jsonl` — no retraining
- Probes (MQAR, stable rank, CKA, entropy) require `--data_dir` pointing to validation data
- Multi-seed detection: multiple checkpoint dirs with same variant name are auto-grouped
- `gh` CLI is authenticated — can create issues directly
- IDE shows import errors for `torch` and `src.*` because the conda env isn't the system Python — tests run fine inside conda

---

## Suggested Skills

| Skill | When to Use |
|-------|-------------|
| `tdd` | Adding new metrics or probes to the evaluation framework |
| `codebase-design` | If KV-cache unification is tackled (Candidate 4B) |
| `domain-modeling` | If new evaluation terms arise during analysis |
| `diagnosing-bugs` | If training runs diverge, produce NaN, or give unexpected results |
| `grilling` | Before committing to interactive dashboard design |
| `to-issues` | Breaking remaining work into independently-grabbable GitHub issues |

---

## Architecture Decisions Made This Session

| Decision | Rationale |
|----------|-----------|
| `EvaluationPipeline` is a class, not a function | Configuration capture (device, data_dir); extensible to future `run_probes_only()` |
| `ReportResult` returned alongside file writes | Tests + dashboard get structured data without re-parsing files |
| Lenient error handling with `warnings` list | Pipeline always produces partial report; callers inspect warnings |
| `VariantSpec.config_overrides` callable | Variant knowledge concentrates with its spec; `build()` is generic |
| `per_layer_config_builder` separate field | Only swa_interleaved needs it; 6/7 variants stay simple |
| `ProbeTarget` is Protocol (not ABC) | Matches existing `DataLoader` pattern; fakes satisfy without inheritance |
| Two ProbeTarget methods (forward vs forward_with_internals) | MQAR doesn't need layer outputs — avoids OOM on large models |
| `AttentionModule` Protocol only (no cache unification) | Discoverability without breaking existing KV-cache internals |

---

## Open Questions

- ALiBi extrapolation experiment — deferred until controlled comparison complete
- KV-Cache unification — AttentionModule Protocol uses `Any` for cache; real unification still deferred
- Interactive HTML dashboard — Phase 9, not yet specced
- Training data for V2–V5 needs to be kicked off (GPU-bound, ~67h total)
- Probes not yet rewritten to use ProbeTarget (existing code works, protocol exists for new probes)
