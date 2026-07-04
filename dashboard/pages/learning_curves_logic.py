"""Pure logic for the Learning Curves page.

Extracted from 1_learning_curves.py to enable unit testing without
Streamlit dependencies.
"""

from __future__ import annotations

import numpy as np


def extract_learning_curve(
    seeds: list[dict], x_key: str
) -> dict | None:
    """Extract averaged learning curve data from seed entries.

    For single-seed variants, returns raw values directly.
    For multi-seed variants, averages val_loss and computes std across seeds
    at each step.

    Args:
        seeds: List of seed entry dicts. Each should have a "log_entries"
            key containing a list of dicts with keys: step, tokens_seen,
            wallclock, cumulative_flops, val_loss.
        x_key: Which key to use for x-axis values. One of "tokens_seen",
            "wallclock", or "cumulative_flops".

    Returns:
        Dict with keys:
            - x_values: list of x-axis values (from first seed)
            - mean_loss: list of mean val_loss values
            - std_loss: list of std val_loss values (None for single-seed)
            - num_seeds: number of seeds with valid data
        Returns None if no valid log_entries exist.
    """
    # Collect log entries from all seeds
    seed_logs = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        entries = seed.get("log_entries")
        if not isinstance(entries, list) or len(entries) == 0:
            continue
        seed_logs.append(entries)

    if not seed_logs:
        return None

    # For single-seed, just return the data directly
    if len(seed_logs) == 1:
        entries = seed_logs[0]
        return {
            "x_values": [e.get(x_key, 0) for e in entries],
            "mean_loss": [e.get("val_loss", 0) for e in entries],
            "std_loss": None,
            "num_seeds": 1,
        }

    # For multi-seed, average val_loss across seeds at each step
    # Use the steps from the first seed as the reference
    ref_entries = seed_logs[0]
    x_values = [e.get(x_key, 0) for e in ref_entries]
    num_steps = len(ref_entries)

    # Collect val_loss values per step across seeds
    all_losses = []
    for entries in seed_logs:
        losses = [e.get("val_loss", 0) for e in entries[:num_steps]]
        # Pad with last value if a seed has fewer entries
        while len(losses) < num_steps:
            losses.append(losses[-1] if losses else 0)
        all_losses.append(losses)

    losses_array = np.array(all_losses)
    mean_loss = np.mean(losses_array, axis=0).tolist()
    std_loss = np.std(losses_array, axis=0).tolist()

    return {
        "x_values": x_values,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "num_seeds": len(seed_logs),
    }
