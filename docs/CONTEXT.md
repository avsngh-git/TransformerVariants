# Project Context — Transformer Variant Lab

This document summarizes what has been built so far. Read this to understand the current state of the project before making changes.

---

## Project Goal

Build, train, compare, and visualize 6 decoder-only Transformer variants on an NVIDIA L4-24Q GPU (24GB). The focus is controlled experimentation under practical hardware constraints, not frontier-scale training.

## Hardware

- GPU: NVIDIA L4-24Q (24GB VRAM)
- Precision: bfloat16
- Single-GPU only

---

## What's Been Built

### Phase 1: Repository Skeleton (✅ Complete)
- Project structure with `src/`, `tests/`, `configs/`, `scripts/`, `docs/`
- Utility modules: `src/utils/seed.py`, `src/utils/config.py`, `src/utils/run_dir.py`, `src/utils/logging.py`, `src/utils/params.py`
- YAML config loading with multi-file merge
- Run directory management (timestamped experiment dirs)
- 33 tests covering all utilities

### Phase 2: Data Pipeline (✅ Complete)
- `src/data/tokenizer.py` — GPT-2 tokenizer wrapper (get_tokenizer, encode, decode, get_eot_token, get_vocab_size)
- `src/data/prepare.py` — Full pipeline: HuggingFace download → tokenize → binary shards → manifest.json + data_report.json
- `src/data/dataloader.py` — Memory-mapped shard reader, serves (input, target) pairs for next-token prediction
- `scripts/prepare_data.py` — CLI entry point
- Dataset: WikiText-103-raw-v1, GPT-2 tokenizer (vocab_size=50257)
- Data format: uint16 binary shards with JSON manifest
- 9 tests

### Phase 3: Vanilla Transformer V0 (✅ Complete)
- `src/models/config.py` — ModelConfig dataclass with all hyperparameters
- `src/models/attention.py` — CausalSelfAttention with KV-cache support
- `src/models/ffn.py` — FeedForward with configurable activation (relu/gelu)
- `src/models/vanilla_transformer.py` — TransformerBlock + VanillaTransformer

**Architecture (V0):**
- Learned position embeddings
- Pre-LayerNorm
- Standard multi-head attention with causal mask
- Configurable activation: ReLU (vanilla) or GELU (GPT-2 match)
- Weight tying (embedding ↔ output head)
- GPT-2 weight initialization (N(0, 0.02), residual scaling 1/√(2*n_layers))

**Generation features:**
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- KV-cache for fast autoregressive generation
- Greedy decoding (temperature=0)

**Model sizes (all use same code, different config):**
| Scale | Layers | d_model | Heads | Seq Len | Parameters |
|-------|--------|---------|-------|---------|-----------|
| debug | 4 | 256 | 4 | 512 | 16.1M |
| main | 8 | 512 | 8 | 1024 | 51.4M |
| stretch | 12 | 768 | 12 | 1024 | 124.3M |

- 22 tests (model + generation + KV-cache)

### Phase 4: Training Loop (✅ Complete)
- `src/training/scheduler.py` — Cosine LR with linear warmup
- `src/training/trainer.py` — Full training loop with:
  - Mixed precision (bfloat16/float16/float32)
  - Gradient accumulation
  - Gradient clipping (max norm 1.0)
  - AdamW optimizer with proper weight decay groups (2D params only)
  - Periodic evaluation on validation set
  - Checkpointing (save/resume)
  - JSON logging (train_log.json on completion, train_log.jsonl live)
- `scripts/train.py` — CLI entry point with all hyperparameters
  - `--scale debug|main|stretch`
  - `--activation relu|gelu`
  - `--compile` flag for torch.compile (~15-25% speedup)
  - `--resume` for checkpoint recovery
  - Auto-names checkpoint dirs: `checkpoints/vanilla_{activation}_{scale}/`
- 11 tests (scheduler, dataloader, integration)

**Training defaults (from project_defaults.yaml):**
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1, β1=0.9, β2=0.95)
- LR schedule: cosine decay with warmup
- Grad clip: 1.0
- Precision: bfloat16
- Effective batch: micro_batch × grad_accum × seq_len tokens

---

## Training Runs Completed

| Model | Params | Data | Steps | Final Val Loss | Throughput | Time |
|-------|--------|------|-------|---------------|-----------|------|
| debug (relu) | 16.1M | 5M tokens | 2000 | 5.42 | 126K tok/s | 4.3 min |
| main (relu) | 51.4M | 100M tokens | 5000 | 4.43 | ~42K tok/s | 39 min |
| stretch (relu) | 124.3M | 120M tokens | 3000 | in progress | ~25K tok/s | ~3 hrs |

---

## Test Suite

101+ tests total, all passing:
- `tests/test_model.py` — shapes, causal mask, generation, KV-cache, weight init
- `tests/test_training.py` — scheduler, dataloader, integration (loss decreases)
- `tests/test_data_pipeline.py` — tokenizer, sharding, manifest
- `tests/test_seed.py`, `test_run_dir.py`, `test_params.py` — utilities

Run all tests: `conda run -n transformer_lab python -m pytest tests/ -v`

---

## Prepared Datasets

| Directory | Tokens | Shards | Use |
|-----------|--------|--------|-----|
| `data/processed/wikitext-103-raw-v1/` | 62K train + 55K val | 1+1 | Quick sanity checks |
| `data/processed/wikitext-full/` | 5M train + 251K val | 5+1 | Debug model training |
| `data/processed/wikitext-100M/` | 100M train + 251K val | 10+1 | Main model training |
| `data/processed/wikitext-120M/` | 120M train + 251K val | 12+1 | Stretch model training |

---

## Config Files

- `configs/project_defaults.yaml` — Global defaults (hardware, training, eval metrics)
- `configs/data/debug.yaml` — Data pipeline config (dataset, tokenizer, shard size)
- `configs/model/vanilla.yaml` — V0 vanilla (ReLU, all 3 scales)
- `configs/model/vanilla_gpt2.yaml` — V0 with GELU activation

---

## Key Design Decisions

Documented in `docs/learnings_from_project.md`:
- Learned position embeddings (V0) → will switch to RoPE in V1
- Pre-LayerNorm (not Post-LN) for training stability
- No dropout (modern pretraining best practice)
- No bias in linear layers (modern LLM standard)
- GPT-2 weight initialization with depth scaling
- KV-cache is a computation optimization (not a model change) — used in all variants
- torch.compile for training only (fixed shapes), KV-cache for generation (dynamic shapes)
- bfloat16 mixed precision (same quality, 2x speed)

---

## Conda Environment

Name: `transformer_lab` (Python 3.11)
Key packages: PyTorch 2.12.1, tiktoken, datasets, numpy

Install: `pip install -e ".[data]"`

---

## What's Next: Phase 5 — V1 Modern Baseline

The next variant to implement (V1) swaps 4 components from V0:

| Component | V0 (Vanilla) | V1 (Modern) |
|-----------|-------------|-------------|
| Position encoding | Learned embeddings | RoPE (rotary) |
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU/GELU | SwiGLU |
| Attention | Manual implementation | Flash Attention |

All other infrastructure (training loop, data pipeline, checkpointing, configs) is reusable.

---

## File Tree (Key Files)

```
src/
├── models/
│   ├── config.py              # ModelConfig dataclass
│   ├── attention.py           # CausalSelfAttention + KV-cache
│   ├── ffn.py                 # FeedForward (relu/gelu configurable)
│   └── vanilla_transformer.py # TransformerBlock + VanillaTransformer
├── data/
│   ├── tokenizer.py           # GPT-2 tokenizer wrapper
│   ├── prepare.py             # Data pipeline (HF → shards)
│   └── dataloader.py          # ShardedDataLoader
├── training/
│   ├── scheduler.py           # Cosine LR with warmup
│   └── trainer.py             # Training loop + checkpointing
└── utils/
    ├── config.py              # YAML config loading
    ├── seed.py                # Reproducibility
    ├── run_dir.py             # Run directory management
    ├── logging.py             # Logging utilities
    └── params.py              # Parameter counting

scripts/
├── prepare_data.py            # CLI: data preparation
└── train.py                   # CLI: training

tests/
├── test_model.py              # Model + generation + KV-cache tests
├── test_training.py           # Training loop tests
├── test_data_pipeline.py      # Data pipeline tests
└── ...                        # Utility tests
```
