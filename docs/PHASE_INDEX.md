# Phase Index

This file summarizes all phases and their dependencies.

## Full phase list

| Phase | File | Goal | Depends on |
|---:|---|---|---|
| 00 | `phase_00_experiment_contract.md` | Define scope, metrics, and fair-comparison rules | none |
| 01 | `phase_01_repo_foundation.md` | Repository skeleton, config loading, run directories | 00 |
| 02 | `phase_02_minimal_data_pipeline.md` | Small debug dataset to token shards | 01 |
| 03 | `phase_03_baseline_transformer.md` | Vanilla decoder-only Transformer | 01, 02 |
| 04 | `phase_04_l4_training_loop.md` | L4-aware training loop and basic checkpointing | 03 |
| 05 | `phase_05_modern_baseline.md` | RoPE, RMSNorm, SwiGLU, fast attention | 04 |
| 06 | `phase_06_alibi_gqa_mqa.md` | ALiBi, GQA, MQA, inference benchmark | 05 |
| 07 | `phase_07_efficient_long_context.md` | Sparse attention and Linformer/Performer | 05, 06 |
| 08 | `phase_08_evaluation_framework.md` | Standard evaluation, plotting, statistics | 04, 05 |
| 09 | `phase_09_visualization_dashboard.md` | Interactive model-internals dashboard | 05, 06 |
| 10 | `phase_10_large_scale_data_pipeline.md` | Streamed large-scale data preparation | 02 |
| 11 | `phase_11_fault_tolerant_training.md` | Atomic checkpoints, resume, fault injection | 04 |
| 12 | `phase_12_main_benchmarks.md` | Run controlled experiments and collect results | 08, 10, 11 |
| 13 | `phase_13_packaging.md` | Final report, README, demo assets, CV bullets | 12 |

## Recommended order

Implement in this order:

```text
00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13
```

Some phases can be developed partially in parallel:

```text
Phase 08 can begin after Phase 04.
Phase 09 can begin after Phase 05.
Phase 10 can begin after Phase 02.
Phase 11 can begin after Phase 04.
```

## Minimum impressive version

A minimum CV-worthy version includes:

```text
Phases 00 through 06
Phase 08 basic evaluation
Phase 09 basic dashboard
Phase 11 basic checkpoint/resume
Phase 13 packaging
```

## Strong version

A strong final version includes all phases through Phase 13, with at least:

```text
40M to 70M parameter main runs
RoPE/SwiGLU baseline
ALiBi
GQA/MQA
Sparse local/global attention
Linformer or Performer
interactive dashboard
large-scale data preparation sample
fault-injection checkpoint tests
final comparison report
```

## Stretch work

Only after the strong version is done:

```text
100M to 125M model run
Performer if Linformer was implemented first
small Switch/MoE model
entropy-adaptive sparse attention
hosted dashboard demo
```