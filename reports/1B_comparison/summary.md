# Evaluation Summary Report

Automated comparison of Transformer variant architectures across
controlled evaluation axes: fixed-data, fixed-wallclock, and fixed-FLOPs.

## Fixed-Data Comparison

Validation loss at the same token budget.

| Variant | Val Loss |
|---------|----------|
| alibi | 3.9025 ± 0.0023 |
| gqa | 3.9565 ± 0.0034 |
| linear | 3.8344 ± 0.0022 |
| modern | 3.6176 ± 0.0034 |
| moe | 3.4292 ± 0.0013 |
| moe_deep | 3.4846 ± 0.0010 |
| moe_interleaved | 3.4826 ± 0.0042 |
| swa | 3.9427 ± 0.0043 |
| swa_interleaved | 3.9333 ± 0.0105 |
| vanilla | 3.6984 ± 0.0035 |

## Fixed-Wallclock Comparison

Validation loss at fractions of the dynamic wall-clock budget.

Entries without an error range are incomplete historical diagnostics; copied seed histories cannot support independent variability.

| Variant | 25% | 50% | 75% | 100% |
|---------|-----|-----|-----|------|
| alibi | 4.1792 | 4.0428 | 3.9385 | 3.8408 |
| gqa | 4.2695 | 4.1140 | 4.0279 | 3.9099 |
| linear | 4.5880 ± 0.0082 | 4.1961 ± 0.0106 | 4.1049 ± 0.0319 | 3.9171 ± 0.0818 |
| modern | 4.1032 | 3.7886 | 3.7687 | 3.5834 |
| moe | 4.5403 | 4.0587 | 3.8273 | 3.7467 |
| moe_deep | 4.4250 | 3.9921 | 3.8111 | 3.6616 |
| moe_interleaved | 4.4271 | 3.9750 | 3.8029 | 3.6549 |
| swa | 4.2323 | 4.0884 | 3.9823 | 3.8777 |
| swa_interleaved | 4.2392 | 4.0875 | 4.0117 | 3.8764 |
| vanilla | 4.4324 | 4.0369 | 3.9985 | 3.9418 |

## Fixed-FLOPs Comparison

Validation loss at the same cumulative FLOP budget.

| Variant | Val Loss |
|---------|----------|
| alibi | 3.8587 |
| gqa | 3.9210 |
| linear | 3.8001 ± 0.0013 |
| modern | 3.6082 |
| moe | 3.4281 |
| moe_deep | 3.4792 |
| moe_interleaved | 3.4732 |
| swa | 3.9029 |
| swa_interleaved | 3.8829 |
| vanilla | 3.6984 ± 0.0035 |

Entries without an error range are incomplete historical diagnostics; copied seed histories cannot support independent variability.


## Parameter Parity

❌ **FAIL** — Active parameters exceed the ±5% tolerance.

Parity is assessed on active parameters per token; total parameters include all stored MoE experts.

| Variant | Active parameters | Total parameters |
|---------|------------------:|-----------------:|
| alibi | 51,430,400 | 51,430,400 |
| gqa | 48,284,672 | 48,284,672 |
| linear | 51,430,400 | 51,430,400 |
| modern | 51,430,400 | 51,430,400 |
| moe | 68,764,672 | 172,573,696 |
| moe_deep | 60,097,536 | 112,002,048 |
| moe_interleaved | 60,097,536 | 112,002,048 |
| swa | 51,430,400 | 51,430,400 |
| swa_interleaved | 51,430,400 | 51,430,400 |
| vanilla | 51,430,400 | 51,430,400 |

## Pareto Front

Pareto-optimal variants (non-dominated on FLOPs vs val_loss):

- **moe**
- **vanilla**


## Seed-Aware Long-Context Extrapolation

Each checkpoint was evaluated on eight fixed, non-overlapping validation windows.
For every window, 1K, 2K, and 4K contexts score the same final 256 target tokens.
Entries are mean ± sample standard deviation across the three checkpoint-seed means.

| Rank at 4K | Variant | Tail PPL @ 1K | Tail PPL @ 2K | Tail PPL @ 4K | 4K / 1K PPL |
|------------|---------|--------------:|--------------:|--------------:|-------------:|
| 1 | alibi | 56.07 ± 0.51 | 58.29 ± 0.88 | **58.74 ± 1.10** | 1.047 ± 0.010 |
| 2 | swa | 58.95 ± 0.45 | 58.95 ± 0.48 | **58.93 ± 0.46** | **1.000 ± 0.001** |
| 3 | swa_interleaved | 58.58 ± 0.28 | 78.48 ± 4.50 | 84.86 ± 6.12 | 1.448 ± 0.100 |
| 4 | gqa | 60.10 ± 0.41 | 370.72 ± 82.01 | 437.37 ± 47.76 | 7.279 ± 0.815 |
| 5 | moe_deep | 34.10 ± 0.65 | 458.63 ± 61.52 | 488.85 ± 23.26 | 14.329 ± 0.423 |
| 6 | moe_interleaved | 34.40 ± 0.66 | 463.84 ± 54.23 | 518.29 ± 18.30 | 15.077 ± 0.817 |
| 7 | moe | **31.91 ± 0.29** | 427.35 ± 24.78 | 526.63 ± 17.94 | 16.505 ± 0.508 |
| 8 | modern | 39.80 ± 0.31 | 453.01 ± 61.98 | 553.94 ± 18.95 | 13.917 ± 0.368 |
| 9 | linear | 50.10 ± 0.54 | 1,103.41 ± 299.49 | 1,311.78 ± 25.66 | 26.186 ± 0.789 |
| unsupported | vanilla | 44.52 ± 0.39 | unsupported | unsupported | unsupported |

ALiBi has the lowest mean 4K tail perplexity. SWA is only 0.20 perplexity
points behind, has the smallest degradation, and has the highest 4K prefill
throughput among the quality leaders. The observed ALiBi--SWA gap is smaller than
their cross-seed standard deviations and does not establish an absolute-quality
winner. SWA's stability means its fixed 256-token receptive field transfers cleanly;
it does not demonstrate retrieval from 4K-token distances. See
[the raw benchmark](raw/benchmarks.json) for checkpoint and window observations.

## Figures

### Learning Curves

![Learning Curves (Tokens)](plots/learning_curves_tokens.png)

![Learning Curves (Wall-clock)](plots/learning_curves_wallclock.png)

![Learning Curves (FLOPs)](plots/learning_curves_flops.png)

### Per-Position Loss

![Per-Position Loss](plots/per_position_loss.png)

### Probes

![MQAR by Distance](plots/mqar_by_distance.png)

![Stable Rank](plots/stable_rank.png)

![CKA Adjacent](plots/cka_adjacent.png)

### Efficiency

![FLOP Breakdown](plots/flop_breakdown.png)

![Pareto Front](plots/pareto_flops_val_loss.png)

![Roofline](plots/roofline.png)
