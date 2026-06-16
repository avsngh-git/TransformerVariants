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
| V5 | Linformer or Performer | Linear-complexity attention |

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
├── src/                     # Source code (future phases)
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
| 01 | Repository skeleton, config loading, run directories | Planned |
| 02 | Minimal data pipeline (small debug dataset → token shards) | Planned |
| 03 | Vanilla decoder-only Transformer | Planned |
| 04 | L4-aware training loop and checkpointing | Planned |
| 05 | Modern baseline (RoPE, RMSNorm, SwiGLU, fast attention) | Planned |
| 06 | ALiBi, GQA, MQA, inference benchmark | Planned |
| 07 | Sparse attention, Linformer/Performer | Planned |
| 08 | Evaluation framework (plots, statistics) | Planned |
| 09 | Interactive visualization dashboard | Planned |
| 10 | Large-scale data pipeline | Planned |
| 11 | Fault-tolerant training (resume, corruption fallback) | Planned |
| 12 | Main benchmark runs and controlled experiments | Planned |
| 13 | Final report, packaging, demo assets | Planned |

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
- Parameter counts within ±5%
- Results reported over 3+ random seeds

See `reports/experiment_contract.md` for full details.

## Quick Start

```bash
# (Future phases — not yet implemented)
# Install dependencies
pip install -e .

# Run a debug training
python -m scripts.train --config configs/experiment/debug.yaml

# Launch dashboard
streamlit run src/viz/dashboard.py
```

## License

See [LICENSE](LICENSE) for details.
