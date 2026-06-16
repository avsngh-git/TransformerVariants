# Global Instructions for Claude Code

You are implementing a portfolio-quality machine learning project called **Transformer Variant L4 Lab**.

The project goal is to build a modular PyTorch framework for comparing Transformer variants on a single NVIDIA L4-24Q GPU with 24GB VRAM. The codebase should support training small-to-medium decoder-only language models, preparing tokenized datasets, evaluating efficiency, visualizing model internals, and resuming training after crashes.

## Hardware assumptions

Target hardware:

```text
GPU: NVIDIA L4-24Q
GPU memory: 24GB
Primary mode: single GPU
Precision: prefer bf16 when available, otherwise fp16
Main model scale: 40M to 70M parameters
Stretch model scale: 100M to 125M parameters
Main sequence length: 1024 tokens
Long-context evaluation: 2048 and 4096 tokens when possible
```

Do not assume a multi-node cluster. Distributed features may be designed as future-compatible, but the core implementation must work on one GPU.

## Project principles

1. Keep the repository runnable after every phase.
2. Use config-driven experiments.
3. Prefer clear, correct implementations over clever but fragile optimizations.
4. Add tests for shape logic, masking, checkpointing, and data correctness.
5. Separate training-speed paths from visualization paths.
6. Log all metrics in machine-readable files.
7. Never silently change experiment settings in a way that breaks fair comparisons.
8. Make all outputs reproducible from configs and manifests.

## Implementation style

Use Python and PyTorch. Keep modules simple and typed where practical.

Recommended style:

```text
src/models/        model code only
src/data/          data loading and preprocessing only
src/training/      training loop, optimizer, checkpointing
src/evaluation/    evaluation and plotting
src/viz/           dashboard and interpretability tools
src/utils/         shared utilities
configs/           YAML configs
scripts/           command-line helpers
reports/           project reports and generated analysis
runs/              experiment outputs, ignored by git
```

Prefer small functions and dataclasses over large unstructured scripts.

## Required output conventions

A training run should write:

```text
runs/<run_id>/
  config_resolved.yaml
  metrics.jsonl
  summary.json
  logs/
  checkpoints/
```

A processed dataset should write:

```text
data/processed/<dataset_name>/
  train_000000.bin
  train_000001.bin
  val_000000.bin
  manifest.json
  data_report.json
```

Metrics should be JSONL, one event per line. Example:

```json
{"step": 100, "tokens": 6553600, "train_loss": 5.91, "lr": 0.00029, "tokens_per_sec": 18750, "peak_gpu_mem_gb": 14.2}
```

## Config rules

Use YAML configs. Later phases may add fields, but preserve backward compatibility.

Minimum config groups:

```text
configs/model/
configs/data/
configs/train/
configs/experiment/
```

A full experiment should be reconstructable from:

```text
model config + data config + train config + code version + dataset manifest
```

## Testing expectations

Use pytest. Add or update tests in the same phase that introduces behavior.

Minimum tests over the full project:

```text
test config loading
test token batch shapes
test labels are shifted correctly
test causal mask blocks future tokens
test model output shape
test generation returns valid token IDs
test checkpoint save/load restores state
test corrupted checkpoint fallback
test sparse mask is causal
```

## Safety and practicality constraints

Do not download very large datasets by default. Large-scale data scripts must support limits such as:

```text
--max-documents
--max-raw-bytes
--max-tokens
```

Do not require external paid services. Optional integrations such as Weights & Biases may be supported but must not be mandatory.

Do not store secrets or API tokens in the repository.

## Phase discipline

When implementing a phase:

1. Read the target phase file.
2. Implement only the deliverables for that phase.
3. Add tests required by that phase.
4. Run the acceptance checks.
5. Update the status section in the phase file.
6. Stop and summarize what changed.

## Definition of done for code changes

A phase is done only when:

```text
python -m pytest tests relevant to the phase passes
main command for the phase runs end-to-end on a tiny config
new files are documented in the phase status section
known limitations are written down clearly
```