# Experiment Contract

This document defines the scope, comparison rules, hardware constraints, evaluation
metrics, and success criteria for the Transformer Variant L4 Lab project.

All future phases must respect these constraints. Changes to this contract require
explicit justification and a note in the change log at the bottom of this file.

---

## 1. Hardware Constraints

| Resource | Value |
|----------|-------|
| GPU | NVIDIA L4-24Q |
| GPU memory | 24 GB |
| Number of GPUs | 1 |
| Precision preference | bf16 > fp16 > fp32 |
| Max training memory target | 22 GB (leave 2 GB headroom) |
| Max absolute memory | 24 GB |

All experiments must run on a single L4 GPU. Multi-GPU or distributed code may be
designed as future-compatible but is not required or tested.

---

## 2. Model Scale

| Tier | Parameter range | Context length |
|------|----------------|----------------|
| Debug | 10M–20M | 512 tokens |
| Main | 40M–70M | 1024 tokens |
| Stretch | 100M–125M | 1024 tokens |

Long-context evaluation lengths: 1024, 2048, 4096 tokens.

The formal long-context protocol scores the same final 256 target tokens at every
length, over eight fixed non-overlapping validation windows per checkpoint and all
three checkpoint seeds. Checkpoint means are the independent units for uncertainty.

---

## 3. Task Definition

- **Architecture family:** decoder-only Transformer
- **Objective:** next-token prediction (causal language modeling)
- **Vocabulary size:** 32,000 tokens (GPT-style BPE)

---

## 4. Variants Under Comparison

| ID | Variant | Key changes from vanilla |
|----|---------|--------------------------|
| V0 | Vanilla Transformer | Learned position embeddings, LayerNorm, standard MHA |
| V1 | Modern baseline | RoPE, RMSNorm, SwiGLU FFN, flash/memory-efficient attention |
| V2 | ALiBi | ALiBi positional bias, no position embeddings |
| V3 | GQA / MQA | Grouped-query or multi-query attention heads |
| V4 | Sparse local/global | Sliding window + global token sparse attention |
| V5 | Causal linear attention | ELU+1 prefix-state attention with RoPE |

Stretch variants (only after strong version):
- Switch/MoE small model
- Entropy-adaptive sparse attention
- 100M–125M scale runs

---

## 5. Fair Comparison Rules

To ensure meaningful comparisons between variants:

1. **Same data.** All variants train on the same tokenized dataset shards in the
   same order (controlled by random seed and manifest).
2. **Same token budget.** Main comparisons use an identical number of training
   tokens (100M–300M range, fixed per experiment set).
3. **Same optimizer settings.** AdamW with identical hyperparameters unless a
   variant's paper specifies otherwise (documented exception required).
4. **Same effective batch size.** micro_batch × grad_accum × seq_len held constant.
5. **Same evaluation protocol.** Identical validation set, identical eval code,
   identical metrics collection.
6. **Same precision.** bf16 for all variants in a comparison set.
7. **Parameter budget accounting.** Dense variants should remain within ±5% active
   parameters. Sparse MoE comparisons report both active-per-token and total stored
   parameters. A completed recipe outside tolerance is retained only as a documented
   limitation; changing its width, depth, routing, or token budget constitutes a new
   experiment.
8. **Multiple seeds.** Primary training, checkpoint quality, and paired-tail
   long-context quality are reported over at least 3 random seeds. Serving and
   KV-cache diagnostics use one explicitly identified representative checkpoint and
   are not statistical timing claims.
9. **Reproducibility.** Every run is fully specified by:
   `model_config + data_config + train_config + code_version + dataset_manifest`

---

## 6. Evaluation Metrics

### Primary metrics (reported for every variant)

| Metric | Unit | Direction |
|--------|------|-----------|
| Validation loss | nats | lower is better |
| Perplexity | exp(loss) | lower is better |
| Training throughput | tokens/sec | higher is better |
| Peak GPU memory | GB | lower is better |
| Wall-clock time per 1M tokens | seconds | lower is better |
| Generation throughput | tokens/sec | higher is better |
| KV-cache memory at inference | MB | lower is better |

### Long-context metrics

| Metric | Unit | Direction |
|--------|------|-----------|
| Paired-tail perplexity at 1024/2048/4096 | exp(tail loss) | lower is better |
| Paired-tail perplexity ratio vs. 1024 | ratio | closer to 1 is more stable |
| Paired tail-loss change vs. 1024 | nats | lower is better |
| Cross-seed uncertainty | sample standard deviation | reported |
| Prefill throughput at 2048 tokens | tokens/sec | reported |
| Prefill throughput at 4096 tokens | tokens/sec | reported |

### Efficiency summary

For the final comparison report, compute:
- Pareto frontier: validation loss vs. tokens/sec
- Pareto frontier: validation loss vs. peak GPU memory
- Memory efficiency ratio: quality / peak_memory
- Throughput efficiency ratio: quality / wall_clock_time

---

## 7. Success Criteria

### Minimum impressive version (CV-worthy)

- [x] Phases 00–06 complete and tested
- [x] Basic evaluation framework (Phase 08)
- [x] Basic static visualization dashboard (Phase 09)
- [x] Basic fault-tolerant checkpointing (Phase 11)
- [x] Final report schema and reusable publication assets (Phase 13)
- [x] At least 3 variants trained on same data with same token budget
- [x] Comparison results with error ranges over multiple seeds
- [ ] Interactive dashboard showing attention patterns
- [x] Layer/head attention-pattern JSON and PNG inputs for the external site

### Strong version

All of the above, plus:

- [x] All 6 main variants (V0–V5) trained at 40M–70M scale
- [x] Large-scale data pipeline (Phase 10)
- [x] Full fault-injection checkpoint tests (Phase 11)
- [x] Long-context evaluation at 2048 and 4096 tokens
- [ ] Final comparison report with statistical significance

### Stretch goals

- [ ] 100M–125M model run for at least 2 variants
- [x] Sparse MoE recipes
- [ ] Entropy-adaptive sparse attention
- [ ] Hosted dashboard demo

---

## 8. Infrastructure Components

| Component | Purpose |
|-----------|---------|
| Data pipeline | Raw text → filtered → tokenized → packed shards |
| Training loop | L4-optimized, bf16, gradient accumulation, eval hooks |
| Checkpointing | Atomic saves, resume, corruption fallback |
| Evaluation | Metrics collection, statistics, plotting |
| Visualization | Frontend-agnostic JSON/PNG assets for a separate GitHub Pages/Jekyll site |
| Config system | YAML-based, hierarchical, fully reproducible |

---

## 9. Data Strategy

- **Debug dataset:** small subset for rapid iteration (5M–20M tokens)
- **Main dataset:** 100M–300M tokens from publicly available text
- **No paid APIs required.** All data sources must be freely accessible.
- **Download safety:** all data scripts support `--max-documents`,
  `--max-raw-bytes`, `--max-tokens` limits.
- **Manifest tracking:** every processed dataset writes a `manifest.json` with
  checksums, token counts, and source metadata.

---

## 10. Output Conventions

### Training run outputs

```
runs/<run_id>/
  config_resolved.yaml
  metrics.jsonl
  summary.json
  logs/
  checkpoints/
```

### Processed dataset outputs

```
data/processed/<dataset_name>/
  train_000000.bin
  train_000001.bin
  val_000000.bin
  manifest.json
  data_report.json
```

### Metrics format

JSONL, one event per line:
```json
{"step": 100, "tokens": 6553600, "train_loss": 5.91, "lr": 0.00029, "tokens_per_sec": 18750, "peak_gpu_mem_gb": 14.2}
```

### Static-site asset outputs

```text
reports/<experiment>/site_assets/
  manifest.json
  model_internals.json
  attention_patterns.json
  *.png
```

---

## 11. Change Log

| Date | Change | Justification |
|------|--------|---------------|
| 2026-07-15 | Active/total parameter accounting; static HTML report | Correct the implemented SwiGLU/MoE counts, preserve documented parity exceptions, and replace the unwanted Streamlit runtime |
| 2026-07-16 | Fault-tolerant CLI and static-site asset contract | Complete recovery integration while keeping the Jekyll frontend in its own repository |
| 2025-01-01 | Initial contract | Phase 00 creation |
