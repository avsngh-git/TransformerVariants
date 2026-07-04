"""Pure logic for the Per-Position Loss and ICL page.

Extracted from 5_per_position_loss.py to enable unit testing without
Streamlit dependencies.
"""

from __future__ import annotations

import numpy as np


def extract_per_position_loss(seeds: list[dict]) -> list[float] | None:
    """Extract averaged per-position loss across seeds.

    Each seed should have metrics.per_position_loss as a list of floats.
    Averages across seeds for multi-seed variants.

    Args:
        seeds: List of seed entry dicts.

    Returns:
        List of mean loss values per position, or None if no data available.
    """
    valid_losses = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        metrics = seed.get("metrics")
        if not isinstance(metrics, dict):
            continue
        ppl = metrics.get("per_position_loss")
        if isinstance(ppl, list) and len(ppl) > 0:
            valid_losses.append(ppl)

    if not valid_losses:
        return None

    # Average across seeds, handling different lengths by using the minimum
    min_len = min(len(l) for l in valid_losses)
    trimmed = [l[:min_len] for l in valid_losses]
    arr = np.array(trimmed)
    mean_loss = np.mean(arr, axis=0).tolist()
    return mean_loss


def extract_icl_fit_params(seeds: list[dict]) -> dict | None:
    """Extract averaged ICL fit parameters across seeds.

    Each seed should have metrics.icl_fit_params with keys: A, alpha, C, r_squared.
    Averages across seeds for multi-seed variants.

    Args:
        seeds: List of seed entry dicts.

    Returns:
        Dict with keys A, alpha, C, r_squared (averaged), or None if unavailable.
    """
    valid_params = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        metrics = seed.get("metrics")
        if not isinstance(metrics, dict):
            continue
        params = metrics.get("icl_fit_params")
        if isinstance(params, dict) and all(
            k in params for k in ("A", "alpha", "C", "r_squared")
        ):
            valid_params.append(params)

    if not valid_params:
        return None

    # Average the parameters across seeds
    return {
        "A": float(np.mean([p["A"] for p in valid_params])),
        "alpha": float(np.mean([p["alpha"] for p in valid_params])),
        "C": float(np.mean([p["C"] for p in valid_params])),
        "r_squared": float(np.mean([p["r_squared"] for p in valid_params])),
    }


def compute_icl_curve(
    A: float, alpha: float, C: float, seq_len: int
) -> list[float]:
    """Compute the ICL power-law fit curve L(t) = A * t^(-alpha) + C.

    Args:
        A: Amplitude parameter.
        alpha: Decay exponent.
        C: Loss floor.
        seq_len: Number of positions (1 to seq_len).

    Returns:
        List of fitted loss values for positions 1 to seq_len.
    """
    positions = np.arange(1, seq_len + 1, dtype=float)
    fitted = A * np.power(positions, -alpha) + C
    return fitted.tolist()


def build_icl_table(
    variant_names: list[str],
    variants_data: dict[str, list[dict]],
) -> list[dict]:
    """Build the ICL decay comparison table data.

    Args:
        variant_names: List of variant names to include.
        variants_data: Mapping of variant name → list of seed dicts.

    Returns:
        List of row dicts with keys: variant, alpha, C, r_squared, has_data, poor_fit.
        Sorted by alpha descending, with unavailable variants at the bottom.
    """
    rows_with_data = []
    rows_without_data = []

    for name in variant_names:
        seeds = variants_data.get(name, [])
        params = extract_icl_fit_params(seeds)

        if params is not None:
            r_squared = float(params["r_squared"])
            rows_with_data.append(
                {
                    "variant": name,
                    "alpha": float(params["alpha"]),
                    "C": float(params["C"]),
                    "r_squared": r_squared,
                    "has_data": True,
                    "poor_fit": bool(r_squared < 0.8),
                }
            )
        else:
            rows_without_data.append(
                {
                    "variant": name,
                    "alpha": None,
                    "C": None,
                    "r_squared": None,
                    "has_data": False,
                    "poor_fit": False,
                }
            )

    # Sort variants with data by alpha descending
    rows_with_data.sort(key=lambda r: r["alpha"], reverse=True)

    return rows_with_data + rows_without_data
