# Project Status — Transformer Variant Lab

Operational state of the project. Read this to understand what's built, what's running, and how to work with the codebase.

---

## Hardware

- GPU: NVIDIA L4-24Q (24GB VRAM)
- Precision: bfloat16
- Single-GPU only

---

## Phases Completed

### Phase 1: Repository Skeleton ✅
- Project structure with `src/`, `tests/`, `configs/`, `scripts/`, `docs/`
- Utility modules: `src/utils/seed.py`, `src/utils/config.py`, `src/utils/run_dir.py`, `src/utils/logging.py`, `src/utils/params.py`
- YAML config loading with multi-file merge
- Run directory management (timestamped experiment dirs)
- 33 tests covering all utilities

### Phase 2: Data Pipeline ✅
- `src/data/tokenizer.py` — GPT-2 tokenizer wrapper (get_tokenizer, encode, decode, get_eot_token, get_vocab_size)
- `src/data/prepare.py` — Full pipeline: HuggingFace download → tokenize → binary shards → manifest.json + data_report.json
- `src/data/dataloader.py` — Memory-mapped shard reader, serves (input, target) pairs for next-token prediction
- `scripts/prepare_data.py` — CLI entry point
- Dataset: WikiText-103-raw-v1, GPT-2 tokenizer (vocab_size=50257)
- Data format: uint16 binary shards with JSON manifest
- 9 tests

### Phase 3: Vanilla Transformer V0 ✅
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
- Temperature scaling, top-k sampling, top-p (nucleus) sampling
- KV-cache for fast autoregressive generation
- Greedy decoding (temperature=0)

**Model sizes:**
| Scale | Layers | d_model | Heads | Seq Len | Parameters |
|-------|--------|---------|-------|---------|-----------|
| debug | 4 | 256 | 4 | 512 | 16.1M |
| main | 8 | 512 | 8 | 1024 | 51.4M |
| stretch | 12 | 768 | 12 | 1024 | 124.3M |

- 22 tests (model + generation + KV-cache)

### Phase 4: Training Loop ✅
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

### Phase 5: Modern Transformer V1 ✅
- `src/models/rope.py` — Rotary position embeddings
- `src/models/rmsnorm.py` — RMSNorm
- `src/models/swiglu_ffn.py` — SwiGLU feed-forward
- `src/models/modern_attention.py` — Attention with RoPE + Flash Attention
- `src/models/modern_transformer.py` — ModernTransformer (LLaMA-style)
- Training script supports `--variant modern`
- torch.compile stable (no recompilation per step)
- Checkpoint resume works with compiled models (fixed prefix mismatch)

### Phase 6: ALiBi (V2) & GQA (V3) ✅
- `src/models/flash_attention_base.py` — Shared base class for all flash_attn-backed variants (KV-cache allocation, training/generation dispatch)
- `src/models/flash_attention.py` — `FlashAttention` (extends FlashAttentionBase with RoPE + optional ALiBi slopes + optional window_size)
- `src/models/alibi_attention.py` — `ALiBiAttention` (extends FlashAttentionBase with ALiBi slope computation, no RoPE)
- `src/models/gqa_attention.py` — `GQAAttention` (extends FlashAttentionBase with grouped-query KV heads, n_kv_head = n_head // 4)
- `src/models/registry.py` — Variant registry with dependency-injected model construction
- `src/training/run_config_builder.py` — Extracted run configuration builder
- `configs/model/alibi.yaml`, `configs/model/gqa.yaml` — Variant configs
- `scripts/train_v2_v3.sh` — Training shell helper for V2/V3 runs
- Tests: `test_flash_attention.py`, `test_gqa_e2e_training.py`, `test_attention_backend.py`

**Architecture:**
- V2 (ALiBi): Replaces RoPE with linear position biases injected via `alibi_slopes` kwarg to flash_attn kernel. No learned/rotary position encoding.
- V3 (GQA): Groups query heads to share KV heads (4 query heads per KV head). Reduces KV memory proportionally.

### Phase 7: SWA (V4) & Linear Attention (V5) ✅
- `src/models/flash_attention.py` — SWA via `window_size` parameter passed to flash_attn kernel (no custom masks)
- `src/models/linear_attention.py` — Standalone `LinearAttention` module with ELU+1 feature map, causal recurrence O(n·d²)
- `configs/model/swa.yaml`, `configs/model/swa_interleaved.yaml`, `configs/model/linear.yaml`
- Tests: `test_swa_variant.py`, `test_swa_interleaved_unit.py`, `test_linear_attention.py`, `test_linear_properties.py`, `test_linear_registry.py`
- ADRs: `docs/adr/0001-swa-parameterizes-flash-attention.md`, `docs/adr/0002-linformer-no-kvcache-generation.md`

**Architecture:**
- V4 (SWA): Each query attends only to `window_size = seq_len // 4` preceding tokens. Uses `FlashAttention` with `window_size` kwarg — flash_attn kernel handles windowing natively.
- V4-interleaved: Even layers use full attention (window_size=None), odd layers use SWA. Per-layer configs built by registry.
- V5 (Linear): ELU+1 feature map with running accumulator. No RoPE, no KV-cache (training-comparison only). Position info from token embeddings only.

---

## Training Runs Completed

| Model | Params | Data | Steps | Final Val Loss | Throughput | Peak Memory | Time |
|-------|--------|------|-------|---------------|-----------|-------------|------|
| V0 debug (relu) | 16.1M | 5M tokens | 2000 | 5.42 | 126K tok/s | — | 4.3 min |
| V0 main (relu) | 51.4M | 328M tokens | 5000 | 3.56 | ~28K tok/s | 9.70 GB | 192.5 min |
| V0 stretch (relu) | 124.3M | 164M tokens | 5000 | 3.84 | ~27K tok/s | 15.63 GB | 99.3 min |
| V1 main | 51.4M | 328M tokens | 5000 | 3.43 | ~32K tok/s | 7.41 GB | 171.9 min |
| V1 stretch | 123.6M | 164M tokens | 5000 | 3.62 | ~58K tok/s | 10.42 GB | 47.2 min |

---

## V0 vs V1 Comparison

| Metric | Main Scale | Stretch Scale |
|--------|-----------|--------------|
| **Parameter count** | V0: 51.4M, V1: 51.4M (0.0% diff) | V0: 124.3M, V1: 123.6M (-0.6% diff) |
| **Within ±5%** | ✅ Yes | ✅ Yes |
| **Val loss** | V0: 3.56, V1: 3.43 (V1 wins) | V0: 3.84, V1: 3.62 (V1 wins) |
| **Throughput** | V0: 28K, V1: 32K tok/s (+12%) | V0: 27K, V1: 58K tok/s (+110%) |
| **Peak memory** | V0: 9.70 GB, V1: 7.41 GB (-24%) | V0: 15.63 GB, V1: 10.42 GB (-33%) |
| **Wall-clock** | V0: 192.5 min, V1: 171.9 min (-11%) | V0: 99.3 min, V1: 47.2 min (-52%) |

**Key observations:**
- V1 wins on all axes: better loss, faster throughput, lower memory usage
- Flash Attention's memory savings are dramatic at stretch scale (33% less memory)
- The throughput difference at stretch scale (2.1×) is far larger than at main scale (1.12×) — Flash Attention's O(T) memory advantage becomes more pronounced with larger models
- Parameter counts are essentially identical at both scales, confirming fair comparison

---

## Test Suite

217 tests collected:
- `tests/test_model.py` — shapes, causal mask, generation, KV-cache, weight init
- `tests/test_modern_model.py` — V1 components (RMSNorm, RoPE, SwiGLU, ModernAttention, ModernTransformer)
- `tests/test_flash_attention.py` — FlashAttentionBase, FlashAttention, ALiBi, GQA
- `tests/test_gqa_e2e_training.py` — GQA end-to-end training integration
- `tests/test_attention_backend.py` — Backend dispatch tests
- `tests/test_swa_variant.py` — SWA variant construction and forward pass
- `tests/test_swa_interleaved_unit.py` — Interleaved per-layer window config
- `tests/test_linear_attention.py` — Linear attention forward, causality
- `tests/test_linear_properties.py` — Property-based tests (Hypothesis) for linear attention
- `tests/test_linear_registry.py` — Registry build for linear variant
- `tests/test_trainer_injection.py` — Trainer dependency injection
- `tests/test_training.py` — scheduler, dataloader, V0 and V1 training integration
- `tests/test_data_pipeline.py` — tokenizer, sharding, manifest
- `tests/test_seed.py`, `test_run_dir.py`, `test_params.py`, `test_config.py`, `test_device.py`, `test_logging.py`, `test_run_logger.py` — utilities

Run all tests: `conda run -n transformer_lab python -m pytest tests/ -v`

---

## Prepared Datasets

| Directory | Tokens | Shards | Use |
|-----------|--------|--------|-----|
| `data/processed/wikitext-103-raw-v1/` | 62K train + 55K val | 1+1 | Quick sanity checks |
| `data/processed/wikitext-full/` | 5M train + 251K val | 5+1 | Debug scale training |
| `data/processed/wikitext-100M/` | 100M train + 251K val | 10+1 | Main scale training |
| `data/processed/wikitext-120M/` | 120M train + 251K val | 12+1 | Stretch scale training |

---

## Config Files

- `configs/project_defaults.yaml` — Global defaults (hardware, training, eval metrics)
- `configs/data/debug.yaml` — Data pipeline config (dataset, tokenizer, shard size)
- `configs/model/vanilla.yaml` — V0 vanilla (ReLU, all 3 scales)
- `configs/model/vanilla_gpt2.yaml` — V0 with GELU activation
- `configs/model/modern.yaml` — V1 modern (LLaMA-style)
- `configs/model/alibi.yaml` — V2 ALiBi
- `configs/model/gqa.yaml` — V3 GQA
- `configs/model/swa.yaml` — V4 Sliding Window Attention
- `configs/model/swa_interleaved.yaml` — V4-interleaved (alternating local/global layers)
- `configs/model/linear.yaml` — V5 Linear Attention (ELU-based causal)

---

## Key Design Decisions

Documented in `docs/learnings_from_project.md`:
- Learned position embeddings (V0) → RoPE in V1
- Pre-Norm (not Post-Norm) for training stability
- No dropout (modern pretraining best practice)
- No bias in linear layers (modern LLM standard)
- GPT-2 weight initialization with depth scaling
- KV-cache for generation, torch.compile for training
- bfloat16 mixed precision

---

## Conda Environment

Name: `transformer_lab` (Python 3.11)
Key packages: PyTorch 2.12.1, tiktoken, datasets, numpy

Install: `pip install -e ".[data]"`

---

## What's Next

- Train V2 (ALiBi), V3 (GQA), V4 (SWA), V4-interleaved, V5 (Linear) at main + stretch scale
- Phase 8: Evaluation framework (standardized metrics, plotting, statistical significance)
- Phase 9: Visualization dashboard (Streamlit + Plotly)
- Phase 10: Large-scale data pipeline
- Phase 11: Fault-tolerant training (fault injection tests)
- Phase 12: Main controlled benchmarks (3+ seeds per variant)
- Phase 13: Final report and packaging

---

## File Tree (Key Files)

```
src/
├── models/
│   ├── config.py              # ModelConfig dataclass
│   ├── registry.py            # Variant registry — build(variant, scale) → (model, config)
│   ├── attention.py           # V0: CausalSelfAttention + KV-cache
│   ├── ffn.py                 # V0: FeedForward (relu/gelu)
│   ├── vanilla_transformer.py # V0: TransformerBlock + VanillaTransformer
│   ├── modern_attention.py    # V1: RoPE + SDPA attention
│   ├── modern_transformer.py  # V1: ModernTransformer (LLaMA-style shell)
│   ├── rmsnorm.py             # V1: RMSNorm
│   ├── rope.py                # V1: Rotary Position Embeddings
│   ├── swiglu_ffn.py          # V1: SwiGLU feed-forward
│   ├── flash_attention_base.py# Shared base for all flash_attn-backed variants
│   ├── flash_attention.py     # V1/V4: FlashAttention (RoPE + optional window_size)
│   ├── alibi_attention.py     # V2: ALiBi slopes via flash_attn
│   ├── gqa_attention.py       # V3: Grouped-query attention
│   ├── linear_attention.py    # V5: ELU+1 causal linear attention
│   └── generate.py            # Autoregressive generation utilities
├── data/
│   ├── tokenizer.py           # GPT-2 tokenizer wrapper
│   ├── prepare.py             # Data pipeline (HF → shards)
│   └── dataloader.py          # ShardedDataLoader
├── training/
│   ├── scheduler.py           # Cosine LR with warmup
│   ├── trainer.py             # Training loop + checkpointing
│   ├── run_config_builder.py  # Config assembly (variant → training params)
│   ├── run_logger.py          # Run logging utilities
│   ├── protocols.py           # Training protocol interfaces
│   └── synthetic_loader.py    # Synthetic data for tests
└── utils/
    ├── config.py              # YAML config loading
    ├── seed.py                # Reproducibility
    ├── device.py              # Device detection
    ├── logging.py             # Logging utilities
    └── params.py              # Parameter counting

scripts/
├── prepare_data.py            # CLI: data preparation
├── train.py                   # CLI: training (all variants)
└── train_v2_v3.sh             # Shell helper for V2/V3 training runs

configs/model/
├── vanilla.yaml               # V0
├── vanilla_gpt2.yaml          # V0 (GELU sub-variant)
├── modern.yaml                # V1
├── alibi.yaml                 # V2
├── gqa.yaml                   # V3
├── swa.yaml                   # V4
├── swa_interleaved.yaml       # V4-interleaved
└── linear.yaml                # V5

tests/
├── test_model.py              # V0 model + generation + KV-cache
├── test_modern_model.py       # V1 components
├── test_flash_attention.py    # FlashAttentionBase + variants
├── test_gqa_e2e_training.py   # GQA training integration
├── test_swa_variant.py        # SWA construction and forward
├── test_swa_interleaved_unit.py # Interleaved per-layer config
├── test_linear_attention.py   # Linear attention module
├── test_linear_properties.py  # Hypothesis property-based tests
├── test_linear_registry.py    # Linear variant registry build
├── test_training.py           # Training loop integration
├── test_data_pipeline.py      # Data pipeline
└── ...                        # Utility tests
```
