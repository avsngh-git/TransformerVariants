"""Data loading, validation, and extraction utilities for the dashboard.

This module handles reading raw/metrics.json from the report directory,
validating the data structure, and providing helper functions to extract
variant names and seed counts. All error conditions display user-friendly
messages via Streamlit's st.error() / st.warning() rather than raising
unhandled exceptions.

NO imports from src/ — this module reads JSON files only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"variants", "aggregated", "comparison"}

METRICS_SUBPATH = os.path.join("raw", "metrics.json")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@st.cache_data
def load_metrics(report_dir: str) -> dict | None:
    """Load and validate raw/metrics.json from the report directory.

    Returns the parsed dict with keys: variants, aggregated, comparison.
    Returns None and displays st.error() if loading fails.

    Error handling:
    - Directory does not exist → st.error with path + expected structure
    - raw/metrics.json not found → st.error with expected path + generation command
    - Malformed JSON → st.error identifying file as unparseable
    """
    report_path = Path(report_dir)

    # Check if report directory exists
    if not report_path.exists() or not report_path.is_dir():
        st.error(
            f"Report directory not found: `{report_dir}`\n\n"
            f"Expected structure:\n"
            f"```\n"
            f"{report_dir}/\n"
            f"├── raw/\n"
            f"│   ├── metrics.json\n"
            f"│   └── metrics.csv\n"
            f"├── plots/\n"
            f"├── metadata.json\n"
            f"└── summary.md\n"
            f"```"
        )
        return None

    # Check if raw/metrics.json exists
    metrics_path = report_path / "raw" / "metrics.json"
    if not metrics_path.exists():
        st.error(
            f"Metrics file not found: `{metrics_path}`\n\n"
            f"Run the evaluation pipeline to generate it:\n"
            f"```bash\n"
            f"python scripts/evaluate.py --output-dir {report_dir}\n"
            f"```"
        )
        return None

    # Attempt to parse the JSON
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        st.error(
            f"Unable to parse `{metrics_path}` — the file contains malformed JSON.\n\n"
            f"Try re-running the evaluation pipeline to regenerate the file."
        )
        return None

    # Validate top-level keys and issue warnings for missing ones
    warnings = validate_metrics(data)
    if warnings:
        st.warning("⚠️ " + " | ".join(warnings))

    return data


def validate_metrics(data: dict) -> list[str]:
    """Check that required top-level keys exist.

    Returns list of warning messages for missing keys.
    An empty list means all required keys are present.
    """
    missing = REQUIRED_KEYS - set(data.keys())
    return [f"Missing required key: '{k}'" for k in sorted(missing)]


def get_variant_names(data: dict) -> list[str]:
    """Extract sorted list of all variant names from the data.

    Reads from the "variants" key. Returns an empty list if the key is
    missing or not a dict.
    """
    variants = data.get("variants")
    if not isinstance(variants, dict):
        return []
    return sorted(variants.keys())


def get_seed_count(data: dict, variant: str) -> int:
    """Return number of seed entries for a given variant.

    Returns 0 if the variant is not found or the data structure is unexpected.
    """
    variants = data.get("variants")
    if not isinstance(variants, dict):
        return 0
    seeds = variants.get(variant)
    if not isinstance(seeds, list):
        return 0
    return len(seeds)
