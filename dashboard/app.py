"""
Visualization Dashboard - Entry Point (Overview Page)

Displays a summary table of all variants and their key metrics,
a parameter parity badge, and a project description.

Launch with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that `dashboard.*` imports
# resolve correctly regardless of the working directory used to launch Streamlit.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import streamlit as st

from dashboard.components.data_loader import (
    get_seed_count,
    get_variant_names,
    load_metrics,
    validate_metrics,
)
from dashboard.components.sidebar import render_sidebar
from dashboard.components.styling import format_metric_with_std

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Transformer Variants Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Title and description
# ---------------------------------------------------------------------------

st.title("📊 Transformer Variants Dashboard")
st.markdown(
    "A comparative analysis of 7 decoder-only Transformer architecture variants "
    "evaluated under controlled conditions. This dashboard visualizes pre-computed "
    "metrics from the evaluation pipeline — no GPU required."
)

# ---------------------------------------------------------------------------
# Sidebar: render_sidebar handles report_dir input + variant toggles
# ---------------------------------------------------------------------------

# render_sidebar(None) renders the report_dir input but skips variant toggles.
# We use this to get report_dir, then load data, then render toggles in a
# subsequent sidebar block if needed.
sidebar_result = render_sidebar(None)
report_dir = sidebar_result["report_dir"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = load_metrics(report_dir)

if data is None:
    st.stop()

# Display validation warnings if any
warnings = validate_metrics(data)
if warnings:
    for w in warnings:
        st.warning(f"⚠️ {w}")

# ---------------------------------------------------------------------------
# Sidebar: variant toggles (now that data is loaded)
# ---------------------------------------------------------------------------

variant_names = get_variant_names(data)

with st.sidebar:
    if variant_names:
        st.markdown("---")
        from dashboard.components.sidebar import render_variant_toggles

        selected_variants = render_variant_toggles(variant_names, key_prefix="overview_variant")
    else:
        selected_variants = []

# ---------------------------------------------------------------------------
# Overview Summary Table
# ---------------------------------------------------------------------------

st.markdown("### Overview Summary")

aggregated = data.get("aggregated", {})

if not variant_names or not aggregated:
    st.info("No aggregated variant data available. Run the evaluation pipeline to generate results.")
    st.stop()

# Build table rows sorted by val_loss ascending
rows: list[dict] = []
for variant in variant_names:
    agg = aggregated.get(variant)
    if agg is None:
        continue

    val_loss_data = agg.get("val_loss", {})
    perplexity_data = agg.get("perplexity", {})
    throughput_data = agg.get("throughput", {})
    memory_data = agg.get("peak_memory_gb", {})

    rows.append({
        "variant": variant,
        "val_loss_mean": val_loss_data.get("mean"),
        "val_loss_std": val_loss_data.get("std"),
        "perplexity_mean": perplexity_data.get("mean"),
        "perplexity_std": perplexity_data.get("std"),
        "throughput_mean": throughput_data.get("mean"),
        "throughput_std": throughput_data.get("std"),
        "memory_mean": memory_data.get("mean"),
        "memory_std": memory_data.get("std"),
        "seeds": get_seed_count(data, variant),
    })

# Sort by val_loss ascending
rows.sort(key=lambda r: r["val_loss_mean"] if r["val_loss_mean"] is not None else float("inf"))

# Determine best values for bold highlighting
# Best: lowest val_loss, lowest perplexity, highest throughput, lowest memory
val_losses = [r["val_loss_mean"] for r in rows if r["val_loss_mean"] is not None]
perplexities = [r["perplexity_mean"] for r in rows if r["perplexity_mean"] is not None]
throughputs = [r["throughput_mean"] for r in rows if r["throughput_mean"] is not None]
memories = [r["memory_mean"] for r in rows if r["memory_mean"] is not None]

best_val_loss = min(val_losses) if val_losses else None
best_perplexity = min(perplexities) if perplexities else None
best_throughput = max(throughputs) if throughputs else None
best_memory = min(memories) if memories else None


def _format_cell(mean: float | None, std: float | None, dp: int, is_best: bool) -> str:
    """Format a metric cell, bolding if it's the best value."""
    if mean is None:
        return "N/A"
    formatted = format_metric_with_std(mean, std, dp)
    if is_best:
        return f"**{formatted}**"
    return formatted


# Build display table
table_rows: list[dict[str, str]] = []
for r in rows:
    table_rows.append({
        "Variant": r["variant"],
        "Val Loss": _format_cell(
            r["val_loss_mean"], r["val_loss_std"], 4,
            r["val_loss_mean"] == best_val_loss,
        ),
        "Perplexity": _format_cell(
            r["perplexity_mean"], r["perplexity_std"], 4,
            r["perplexity_mean"] == best_perplexity,
        ),
        "Throughput (tok/s)": _format_cell(
            r["throughput_mean"], r["throughput_std"], 1,
            r["throughput_mean"] == best_throughput,
        ),
        "Peak Memory (GB)": _format_cell(
            r["memory_mean"], r["memory_std"], 1,
            r["memory_mean"] == best_memory,
        ),
        "Seeds": str(r["seeds"]),
    })

# Render as markdown table for bold support
df = pd.DataFrame(table_rows)
st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Parameter Parity Badge
# ---------------------------------------------------------------------------

st.markdown("### Parameter Parity")

comparison = data.get("comparison", {})
parameter_counts = comparison.get("parameter_counts", {})
parameter_parity_valid = comparison.get("parameter_parity_valid")

if parameter_counts:
    # Display parameter counts
    counts_str = ", ".join(
        f"{name}: {count:,}" for name, count in sorted(parameter_counts.items())
    )
    st.caption(f"Parameter counts: {counts_str}")

if parameter_parity_valid is True:
    st.success("✅ Parameter Parity Valid — all variants are within ±5% of mean parameter count.")
elif parameter_parity_valid is False:
    st.error("❌ Parameter Parity Violated — one or more variants exceed ±5% tolerance from the mean.")
else:
    st.info("Parameter parity information not available.")
