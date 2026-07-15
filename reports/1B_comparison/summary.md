# Evaluation Summary Report

Automated comparison of Transformer variant architectures across
controlled evaluation axes: fixed-data, fixed-wallclock, and fixed-FLOPs.

## Fixed-Data Comparison

Validation loss at the same token budget.

| Variant | Val Loss |
|---------|----------|
| alibi | 3.8408 ± 0.0000 |
| gqa | 3.9033 ± 0.0000 |
| linear | 3.7546 ± 0.0019 |
| modern | 3.5471 ± 0.0000 |
| moe | 3.3450 ± 0.0000 |
| moe_deep | 3.4067 ± 0.0000 |
| moe_interleaved | 3.4019 ± 0.0000 |
| swa | 3.8777 ± 0.0000 |
| swa_interleaved | 3.8638 ± 0.0000 |
| vanilla | 3.6301 ± 0.0000 |

## Fixed-Wallclock Comparison

Validation loss at fractions of the dynamic wall-clock budget.

| Variant | 25% | 50% | 75% | 100% |
|---------|-----|-----|-----|------|
| alibi | 4.1792 ± 0.0000 | 4.0428 ± 0.0000 | 3.9385 ± 0.0000 | 3.8408 ± 0.0000 |
| gqa | 4.2695 ± 0.0000 | 4.1140 ± 0.0000 | 4.0279 ± 0.0000 | 3.9099 ± 0.0000 |
| linear | 4.5880 ± 0.0082 | 4.1961 ± 0.0106 | 4.1049 ± 0.0319 | 3.9171 ± 0.0818 |
| modern | 4.1032 ± 0.0000 | 3.7886 ± 0.0000 | 3.7687 ± 0.0000 | 3.5834 ± 0.0000 |
| moe | 4.5403 ± 0.0000 | 4.0587 ± 0.0000 | 3.8273 ± 0.0000 | 3.7467 ± 0.0000 |
| moe_deep | 4.4250 ± 0.0000 | 3.9921 ± 0.0000 | 3.8111 ± 0.0000 | 3.6616 ± 0.0000 |
| moe_interleaved | 4.4271 ± 0.0000 | 3.9750 ± 0.0000 | 3.8029 ± 0.0000 | 3.6549 ± 0.0000 |
| swa | 4.2323 ± 0.0000 | 4.0884 ± 0.0000 | 3.9823 ± 0.0000 | 3.8777 ± 0.0000 |
| swa_interleaved | 4.2392 ± 0.0000 | 4.0875 ± 0.0000 | 4.0117 ± 0.0000 | 3.8764 ± 0.0000 |
| vanilla | 4.4324 ± 0.0000 | 4.0369 ± 0.0000 | 3.9985 ± 0.0000 | 3.9418 ± 0.0000 |

## Fixed-FLOPs Comparison

Validation loss at the same cumulative FLOP budget.

| Variant | Val Loss |
|---------|----------|
| alibi | 3.8587 ± 0.0000 |
| gqa | 3.9210 ± 0.0000 |
| linear | 3.8001 ± 0.0013 |
| modern | 3.6082 ± 0.0000 |
| moe | 3.4281 ± 0.0000 |
| moe_deep | 3.4792 ± 0.0000 |
| moe_interleaved | 3.4732 ± 0.0000 |
| swa | 3.9029 ± 0.0000 |
| swa_interleaved | 3.8829 ± 0.0000 |
| vanilla | 3.6301 ± 0.0000 |

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
