# Invalidated 500M/five-seed training attempt

The first sustained corrective-matrix attempt stopped at `gqa_s42` on 2026-07-19.
These artifacts are preserved for fault-tolerance provenance but are excluded from
all primary or supplementary model-quality comparisons.

## Cause

The health monitor used absolute z-scores and therefore classified healthy downward
gradient drift as anomalous. GQA seed 42 produced skip, skip, rollback during normal
warm-up. Rollback then failed because `map_location=cuda` moved the CPU RNG-state byte
tensor onto CUDA. The same two-sided detector had already caused silent skips in all
15 completed runs.

## Completed endpoints rejected by the zero-skip contract

| Variant | Seed | Final step | Skipped optimizer updates |
|---|---:|---:|---:|
| Vanilla | 42 | 7,629 | 25 |
| Vanilla | 137 | 7,629 | 32 |
| Vanilla | 2024 | 7,629 | 31 |
| Vanilla | 31415 | 7,629 | 35 |
| Vanilla | 271828 | 7,629 | 40 |
| Modern | 42 | 7,629 | 87 |
| Modern | 137 | 7,629 | 82 |
| Modern | 2024 | 7,629 | 81 |
| Modern | 31415 | 7,629 | 83 |
| Modern | 271828 | 7,629 | 90 |
| ALiBi | 42 | 7,629 | 97 |
| ALiBi | 137 | 7,629 | 93 |
| ALiBi | 2024 | 7,629 | 102 |
| ALiBi | 31415 | 7,629 | 97 |
| ALiBi | 271828 | 7,629 | 101 |

`gqa_s42` stopped after its logged step 10 and is also invalidated. Its deterministic
20-step replay remained finite and improved normally, proving this was infrastructure
behavior rather than model divergence.

The corresponding run directories are archived under
`runs/invalidated/main_500m_5seed_two_sided_health_monitor/`. Training remains stopped.
Machine-readable invalidation records are in `invalidated_runs_20260719.json`.
