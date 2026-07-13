# PRD: Phase 8 — Evaluation Framework

> V5-specific Linformer and attention-entropy requirements are superseded by ADR 0007.

**Label:** `ready-for-agent`  
**Status:** Ready for implementation

---

## Problem Statement

All six Transformer variants (V0–V5) are implemented and registered, but there is no standardized way to compare them beyond eyeballing validation loss from training logs. An interviewer looking at this project would see training curves but no rigorous analysis of *why* the variants differ, no hardware efficiency characterization, and no statistical reporting. The project needs a paper-grade evaluation framework that produces compelling, reproducible comparisons.

## Solution

Build an evaluation module (`src/evaluation/`) with a CLI entry point (`scripts/evaluate.py`) that, given a set of training checkpoints and logs, produces:

- Structured comparison across three axes (fixed-data, fixed-compute/wall-clock, fixed-compute/FLOPs)
- Ten paper-grade metrics covering quality, retrieval capacity, representation health, efficiency, and training dynamics
- Static reports (PNG plots + markdown summary + CSV/JSON raw data) in `reports/{experiment_name}/`
- Mean ± standard deviation reporting across 3+ seeds per variant

## User Stories

1. As a researcher, I want to compare all variants at the same token budget, so that I can isolate architectural quality differences.
2. As a researcher, I want to compare all variants at the same wall-clock duration, so that I can see which variant learns fastest in real time.
3. As a researcher, I want to compare all variants at the same FLOP budget, so that I can normalize for computational cost.
4. As a researcher, I want component-level FLOP accounting per variant, so that I can see where compute is spent (projections vs attention vs FFN).
5. As a researcher, I want Model FLOPs Utilization (MFU) computed against my L4's peak (242 TFLOPS BF16), so that I can identify memory-bound vs compute-bound variants.
6. As a researcher, I want per-position validation loss plotted and the ICL decay exponent (α) fitted, so that I can see how effectively each variant utilizes context length.
7. As a researcher, I want MQAR (Multi-Query Associative Recall) scores, so that I can demonstrate that windowed/linear variants lose long-range retrieval capability.
8. As a researcher, I want stable rank per layer computed at the final checkpoint, so that I can detect representation collapse in V5 (Linformer) or V4 (SWA).
9. As a researcher, I want CKA (Centered Kernel Alignment) between layers, so that I can identify redundant computation across the model depth.
10. As a researcher, I want CKA presented as both an adjacent-layer curve and a full L×L heatmap, so that I have a one-slide summary and a detailed appendix view.
11. As a researcher, I want gradient norm per layer logged during training, so that I can compare training stability across architectures.
12. As a researcher, I want attention entropy and sparsity metrics for variants where attention weights are accessible (V0, V5), so that I can characterize attention distribution.
13. As a researcher, I want Pareto plots (loss vs FLOPs, loss vs wall-clock, loss vs peak memory), so that I can identify which variants are Pareto-dominated.
14. As a researcher, I want a roofline diagram plotting achieved TFLOPS/s vs arithmetic intensity per variant, so that I can show hardware utilization visually.
15. As a researcher, I want the fixed-compute wall-clock budget determined dynamically (minimum total time across variants), so that all variants have valid data at the comparison point.
16. As a researcher, I want multiple time slices (25%, 50%, 75%, 100% of budget) in the fixed-compute view, so that I can show learning curves under equal time.
17. As a researcher, I want MQAR implemented as a synthetic probe (planted key-value associations), so that it runs on any checkpoint without retraining.
18. As a researcher, I want stable rank averaged over ~100 validation batches, so that the measurement is robust to individual batch variance.
19. As a researcher, I want all metrics reportable as mean ± std across seeds, so that I can assess result reliability.
20. As a researcher, I want static report output (PNGs + summary.md + CSV/JSON) per experiment, so that results are reproducible and archivable.
21. As a researcher, I want a single CLI command to run the full evaluation suite on a set of checkpoints, so that evaluation is automated.
22. As a researcher, I want the evaluation framework to work on saved checkpoints without requiring re-training, so that I can analyze any historical run.
23. As a researcher, I want parameter count parity (±5%) validated as part of the comparison report, so that fairness is documented.
24. As a researcher, I want the roofline boundary drawn using my L4 specs (242 TFLOPS BF16, 300 GB/s, ridge at ~807 FLOPs/byte), so that the diagram is hardware-accurate.

## Implementation Decisions

- **Evaluation module structure**: Five submodules — metrics (val loss, perplexity, per-position loss), probes (MQAR, stable_rank, CKA, attention entropy), flops (component-level counter, MFU), comparison (fixed-data/compute slicing, Pareto analysis), visualizations (plotting functions).
- **FLOP accounting is component-level**, not the coarse 6NT approximation. Sums per-layer contributions: Q/K/V projection matmuls, attention score computation (variant-dependent: O(T²d) for full, O(T·W·d) for SWA, O(T·r·d) for Linformer), FFN matmuls. Implemented as a pure function: `compute_step_flops(config) -> int`.
- **MFU formula**: `achieved_flops / (hardware_peak_flops × wall_clock_time)`. L4 peak: 242 TFLOPS BF16, 300 GB/s bandwidth.
- **Fixed-compute comparison uses post-hoc slicing** of existing training logs (per-step elapsed time from `metrics.jsonl`). No time-based stopping logic needed. The wall-clock budget is dynamic: `min(total_time across all variants at a given scale)`.
- **MQAR probe**: Generates synthetic sequences with planted key-value token associations at controlled distances. Runs inference on the final checkpoint. Measures P(correct recall token) at recall positions. No retraining needed.
- **Stable rank**: Computed via forward hooks capturing hidden states at each layer. Uses SVD on the hidden state matrix H: `srank(H) = ||H||²_F / σ₁²`. Averaged over ~100 validation batches.
- **CKA**: Linear CKA between layer representations. Primary visualization: adjacent-layer CKA curve (one line per variant). Secondary: full L×L heatmap per variant.
- **ICL decay exponent**: Fits per-position validation loss to power-law `L(t) = A·t^(-α) + C`. Extracts α per variant.
- **Attention entropy/sparsity**: Only computed for V0 (standard attention with accessible weights) and V5 (Linformer, softmax weights accessible). Skipped for flash-based variants (V1–V4) where the attention matrix is never materialized.
- **Gradient norm per layer**: Added as a training-time metric logged to `metrics.jsonl`. Captured per step at log intervals.
- **Roofline**: Arithmetic intensity computed per variant as `FLOPs_per_step / bytes_transferred_per_step`. Bytes estimated from activation + parameter memory traffic.
- **Statistical reporting**: Mean ± std across 3+ seeds. No formal hypothesis tests (insufficient power at n=3). Additional seeds added selectively if intervals overlap.
- **Output format**: `reports/{experiment_name}/` containing individual metric PNGs (matplotlib/plotly), `summary.md` with embedded figure references and tables, raw data as CSV/JSON.
- **CLI entry point**: `scripts/evaluate.py --checkpoints <dir1> <dir2> ... --output reports/<name>/` runs the full suite.

## Testing Decisions

- **Good tests** exercise the public interface (function inputs → outputs) without mocking internal computation steps. Tests should use debug-scale models and synthetic data to run in seconds.
- **flops module**: Test `compute_step_flops` against hand-calculated FLOP counts for known configs. Verify that V4 (SWA, window_size=128) computes fewer attention FLOPs than V1 (full, seq_len=512) at debug scale.
- **probes module**: Test each probe function (MQAR, stable_rank, CKA, ICL decay) on a debug-scale model with known-seed initialization. Verify output shapes, value ranges, and that stable_rank is in [1, d_model].
- **comparison module**: Test fixed-data and fixed-compute slicing with synthetic `metrics.jsonl` data (fabricated timestamps and losses). Verify correct interpolation and budget calculation.
- **Prior art**: The project already has property-based tests using Hypothesis (`test_linear_properties.py` pattern, `test_swa_variant.py`), parametrized scale tests (`test_linear_registry.py`), and integration tests building models via the registry.

## Out of Scope

- Interactive Streamlit dashboard (Phase 9 — consumes evaluation outputs but is a separate deliverable)
- Downstream task benchmarks (HellaSwag, LAMBADA) — models too small at 51M params
- Formal hypothesis testing (requires 5+ seeds; deferred until results justify the GPU cost)
- Large-scale data pipeline (Phase 10)
- Fault-tolerant training improvements (Phase 11)
- ALiBi extrapolation experiment (train-short/infer-long — deferred per CONTEXT.md)

## Further Notes

- The evaluation framework is designed to work entirely from saved checkpoints and training logs. No retraining is needed for any metric.
- The V5 Linformer rewrite (ADR 0003) must be complete before running V5 evaluations — it was completed in this session.
- Hardware specs for roofline: NVIDIA L4-24Q, 242 TFLOPS BF16, 300 GB/s memory bandwidth, ridge point ~807 FLOPs/byte.
- Gradient norm logging requires a small addition to the training loop (`trainer.py`) to capture per-layer norms at log intervals — this is a prerequisite that should be added before running new training runs.
- Domain terms (Comparison Axis, Component-Level FLOP Accounting, MFU, Stable Rank, MQAR, ICL Loss Decay Exponent, CKA, Roofline Diagram) have been added to CONTEXT.md during this session.
