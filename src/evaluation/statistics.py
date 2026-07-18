"""Small-sample summaries for independent seeds and paired seed comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# Two-sided 95% Student-t critical values. Five seeds use df=4 -> 2.776.
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def sample_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return raw small-sample uncertainty using a Student-t 95% interval."""
    if not values:
        raise ValueError("At least one value is required")
    numeric = [float(value) for value in values]
    count = len(numeric)
    mean = sum(numeric) / count
    if count == 1:
        return {
            "mean": mean,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci95_half_width": None,
            "n": count,
        }
    std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / (count - 1))
    critical = _T_CRITICAL_95.get(count - 1, 1.96)
    half_width = critical * std / math.sqrt(count)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
        "n": count,
    }


def paired_difference_summary(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
) -> dict:
    """Summarize candidate-minus-baseline differences for matched seeds only."""
    matched_seeds = sorted(set(candidate) & set(baseline))
    if not matched_seeds:
        return {"status": "unavailable", "reason": "no matched seeds", "n": 0}
    deltas = [float(candidate[seed]) - float(baseline[seed]) for seed in matched_seeds]
    summary = sample_summary(deltas)
    std = summary["std"]
    effect_size_dz = None
    if isinstance(std, float) and std > 0:
        effect_size_dz = float(summary["mean"]) / std
    return {
        "status": "ok",
        "sign": "candidate_minus_baseline",
        "matched_seeds": matched_seeds,
        "differences": deltas,
        "summary": summary,
        "paired_effect_size_dz": effect_size_dz,
    }
