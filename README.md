# Transformer Variant Lab on an NVIDIA L4-24Q GPU

## Overview

This project is a single-GPU research-and-systems benchmark for efficient Transformer variants.

The goal is to implement, train, compare, and visualize several decoder-only Transformer architectures under the practical constraints of an NVIDIA L4-24Q GPU with 24 GB of memory.

This project is not intended to train a frontier-scale language model. Instead, it focuses on controlled experimentation, memory-efficient architecture design, data preparation, interactive model-internal visualization, and robust training infrastructure.

## Variants Compared

| ID | Variant | Key Idea |
|----|---------|----------|
| V0 | Vanilla Transformer | Learned position embeddings, LayerNorm, standard MHA |
| V1 | Modern baseline | RoPE, RMSNorm, SwiGLU, flash attention |
| V2 | ALiBi | Attention with Linear Biases (no position embeddings) |
| V3 | GQA / MQA | Grouped-query and multi-query attention |
| V4 | Sparse local/global | Sliding window + global tokens |
| V5 | Causal Linear Attention | ELU+1 feature-map prefix-state attention |

## Hardware Target

```
GPU:            NVIDIA L4-24Q
GPU memory:     24 GB
Precision:      bf16 (preferred), fp16 fallback
Model scale:    40M–70M parameters (main), 100M–125M (stretch)
Context:        1024 tokens (main), 2048/4096 (long-context eval)
```

## Project Structure

```
TransformerVariants/
├── configs/                 # YAML experiment configs
│   └── project_defaults.yaml
├── docs/                    # Phase documents and workflow guides
├── reports/                 # Experiment contract, generated analysis
├── src/                     # Models, training, evaluation, and static reporting
│   ├── models/              # Model implementations
│   ├── data/                # Data loading and preprocessing
│   ├── training/            # Training loop, optimizer, checkpointing
│   ├── evaluation/          # Evaluation and plotting
│   ├── viz/                 # Dashboard and interpretability
│   └── utils/               # Shared utilities
├── tests/                   # pytest test suite
├── scripts/                 # CLI helpers
├── runs/                    # Experiment outputs (git-ignored)
└── data/                    # Raw and processed datasets (git-ignored)
```

## Phases

| Phase | Goal | Status |
|------:|------|--------|
| 00 | Experiment contract and project definition | ✅ Done |
| 01 | Repository skeleton, config loading, run directories | ✅ Done |
| 02 | Minimal data pipeline (small debug dataset → token shards) | ✅ Done |
| 03 | Vanilla decoder-only Transformer | ✅ Done |
| 04 | L4-aware training loop and checkpointing | ✅ Done |
| 05 | Modern baseline (RoPE, RMSNorm, SwiGLU, fast attention) | ✅ Done |
| 06 | ALiBi, GQA, MQA, inference benchmark | ✅ Done |
| 07 | Sparse and causal linear attention | ✅ Done |
| 08 | Evaluation framework (plots, statistics) | ✅ Done |
| 09 | Self-contained offline HTML dashboard | ✅ Done |
| 10 | Large-scale data pipeline | ✅ Done |
| 11 | Fault-tolerant training (resume, corruption fallback) | Partial |
| 12 | Main benchmark runs and controlled experiments | ✅ Done |
| 13 | Final report, packaging, demo assets | In progress |

## Key Metrics

Every variant is evaluated on:

- Validation loss / perplexity
- Training throughput (tokens/sec)
- Peak GPU memory (GB)
- Wall-clock time per 1M tokens
- Generation throughput (tokens/sec)
- KV-cache memory at inference
- Long-context perplexity degradation

## Fair Comparison Rules

All variants in a comparison set share:
- Same tokenized data in same order
- Same training token budget
- Same optimizer hyperparameters
- Same effective batch size
- Same precision (bf16)
- Active parameters per token within ±5% where possible
- Total stored parameters reported separately for sparse MoE variants
- Any parity exception documented rather than hidden by an unplanned retrain
- Primary training and checkpoint-quality results reported over 3+ random seeds
- Representative-checkpoint inference diagnostics labeled separately

See `reports/experiment_contract.md` for full details.

## Quick Start

```bash
# Install dependencies
pip install -e ".[data,viz]"

# Run a debug training
python scripts/train.py --variant modern --scale debug

# Run evaluation (also builds reports/<name>/index.html)
python scripts/evaluate.py --checkpoints checkpoints/vanilla_main_1B_s42 \
  checkpoints/modern_main_1B_s42 --output reports/example \
  --data_dir data/processed/wikitext-full

# Rebuild the zero-server dashboard after adding benchmark data
python scripts/build_dashboard.py --report reports/1B_comparison
# Then open reports/1B_comparison/index.html directly in a browser.
```

## License

See [LICENSE](LICENSE) for details.
