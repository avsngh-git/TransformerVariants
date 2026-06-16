# Transformer Variant L4 Lab - Claude Code Phase Docs

These Markdown files are implementation briefs for Claude Code. Each phase is designed to be read and executed independently, while still fitting into the full project.

## How to use these files with Claude Code

For each implementation session, give Claude Code the following context:

1. Read `CLAUDE.md` first.
2. Read `PHASE_INDEX.md` to understand where the phase fits.
3. Read the target phase file, for example `phase_03_baseline_transformer.md`.
4. Implement only the requested phase unless the file explicitly says to touch earlier shared utilities.
5. Run the phase acceptance checks.
6. Update the phase status section at the bottom of the file.

Recommended prompt pattern:

```text
Read CLAUDE.md and PHASE_INDEX.md. Then implement Phase 03 from phase_03_baseline_transformer.md. Follow the deliverables and acceptance criteria exactly. Make small commits or clearly separated changes. Do not implement later phases yet.
```

## File list

- `CLAUDE.md`: global project rules for the coding agent.
- `PHASE_INDEX.md`: short map of all phases and dependencies.
- `phase_00_experiment_contract.md`: project scope and comparison rules.
- `phase_01_repo_foundation.md`: repository skeleton, config system, run directories.
- `phase_02_minimal_data_pipeline.md`: local debug text data to token shards.
- `phase_03_baseline_transformer.md`: vanilla GPT-style model.
- `phase_04_l4_training_loop.md`: mixed precision, gradient accumulation, memory-aware training.
- `phase_05_modern_baseline.md`: RoPE, RMSNorm, SwiGLU, fast attention.
- `phase_06_alibi_gqa_mqa.md`: ALiBi, GQA, MQA, KV-cache benchmarking.
- `phase_07_efficient_long_context.md`: sparse local/global attention and Linformer/Performer.
- `phase_08_evaluation_framework.md`: metrics, plots, statistical comparison.
- `phase_09_visualization_dashboard.md`: interactive model-internals dashboard.
- `phase_10_large_scale_data_pipeline.md`: streaming, filtering, deduplication, tokenization, manifests.
- `phase_11_fault_tolerant_training.md`: robust checkpoints, resume, fault injection.
- `phase_12_main_benchmarks.md`: controlled benchmark execution plan.
- `phase_13_packaging.md`: final README, reports, screenshots, CV framing.

## Important design principle

Every phase must leave the repository runnable. Avoid large rewrites. Prefer small, testable modules with clear interfaces.