"""Pure logic for the Probes page.

Extracted from 3_probes.py to enable unit testing without
Streamlit dependencies.
"""

from __future__ import annotations

import numpy as np


def average_lists(lists: list[list[float]]) -> list[float]:
    """Average multiple lists element-wise.

    All lists should have the same length. If only one list is provided,
    returns it unchanged.

    Args:
        lists: List of float lists to average.

    Returns:
        Element-wise mean as a list of floats. Empty list if input is empty.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]
    arr = np.array(lists)
    return np.mean(arr, axis=0).tolist()


def get_probe_field(
    variants_data: dict,
    variant_name: str,
    probe_key: str,
    field_key: str,
) -> list[list[float]]:
    """Collect probe field data from all seeds of a variant.

    Args:
        variants_data: The "variants" dict from the metrics data.
        variant_name: Name of the variant to extract data for.
        probe_key: Top-level probe key (e.g., "mqar", "stable_rank").
        field_key: Field within the probe dict (e.g., "accuracies", "per_layer").

    Returns:
        List of per-seed data arrays. Empty list if no data available.
    """
    seeds = variants_data.get(variant_name, [])
    results = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        probe = seed.get(probe_key)
        if not isinstance(probe, dict):
            continue
        field = probe.get(field_key)
        if isinstance(field, list) and len(field) > 0:
            results.append(field)
    return results


def get_cka_matrix(variants_data: dict, variant_name: str) -> list[list[float]] | None:
    """Get the averaged CKA full matrix across seeds for a variant.

    Args:
        variants_data: The "variants" dict from the metrics data.
        variant_name: Name of the variant to get the CKA matrix for.

    Returns:
        Averaged L×L CKA similarity matrix, or None if no data available.
    """
    seeds = variants_data.get(variant_name, [])
    matrices = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        cka = seed.get("cka")
        if not isinstance(cka, dict):
            continue
        matrix = cka.get("full_matrix")
        if isinstance(matrix, list) and len(matrix) > 0:
            matrices.append(matrix)

    if not matrices:
        return None

    if len(matrices) == 1:
        return matrices[0]

    # Average matrices element-wise
    arr = np.array(matrices)
    return np.mean(arr, axis=0).tolist()


def classify_variants_for_probe(
    variants_data: dict,
    selected_variants: list[str],
    probe_key: str,
    field_key: str,
) -> tuple[list[str], list[str]]:
    """Split selected variants into those with and without probe data.

    Args:
        variants_data: The "variants" dict from the metrics data.
        selected_variants: List of variant names to check.
        probe_key: Top-level probe key.
        field_key: Field within the probe dict.

    Returns:
        Tuple of (available_variants, unavailable_variants).
    """
    available = []
    unavailable = []
    for name in selected_variants:
        seed_data = get_probe_field(variants_data, name, probe_key, field_key)
        if seed_data:
            available.append(name)
        else:
            unavailable.append(name)
    return available, unavailable
