"""Efficiency page — Roofline Diagram, FLOP Breakdown, and MFU Comparison.

Visualizes computational efficiency metrics for selected Transformer variants:
1. Roofline diagram plotting achieved TFLOPS vs arithmetic intensity
2. FLOP breakdown as stacked bar chart (QKV proj, attention, FFN)
3. MFU (Model FLOPs Utilization) bar chart comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components import chart_factory
from dashboard.components.data_loader import get_variant_names, load_metrics
from dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Efficiency", layout="wide")
st.title("Efficiency")

# ---------------------------------------------------------------------------
# Data loading and sidebar
# ---------------------------------------------------------------------------

# Initial load with None to render sidebar before data is available
sidebar_selections = render_sidebar(None)
report_dir = sidebar_selections["report_dir"]

data = load_metrics(report_dir)
if data is None:
    st.stop()

# Re-render sidebar with data to get variant toggles
sidebar_selections = render_sidebar(data)
selected_variants = sidebar_selections["selected_variants"]

if not selected_variants:
    st.info("No variants selected. Use the sidebar to select at least one variant.")
    st.stop()

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

variants_data = data.get("variants", {})


def _extract_efficiency(variant_name: str) -> dict | None:
    """Extract averaged efficiency data (tflops, arithmetic_intensity, mfu) for a variant.

    Returns None if efficiency data is not available for any seed.
    """
    seeds = variants_data.get(variant_name, [])
    if not seeds:
        return None

    tflops_values = []
    ai_values = []
    mfu_values = []

    for seed in seeds:
        eff = seed.get("efficiency")
        if eff is None:
            continue
        if "tflops" in eff and "arithmetic_intensity" in eff:
            tflops_values.append(eff["tflops"])
            ai_values.append(eff["arithmetic_intensity"])
        if "mfu" in eff:
            mfu_values.append(eff["mfu"])

    if not tflops_values or not ai_values:
        return None

    result = {
        "tflops": sum(tflops_values) / len(tflops_values),
        "arithmetic_intensity": sum(ai_values) / len(ai_values),
    }

    if mfu_values:
        result["mfu"] = sum(mfu_values) / len(mfu_values)

    return result


def _extract_flop_breakdown(variant_name: str) -> dict | None:
    """Extract averaged FLOP breakdown data for a variant.

    Returns dict with keys: qkv_proj, attention_score, attention_output, ffn
    (values in TFLOPS), or None if data is unavailable.
    """
    seeds = variants_data.get(variant_name, [])
    if not seeds:
        return None

    keys = ["qkv_proj", "attention_score", "attention_output", "ffn"]
    totals = {k: [] for k in keys}

    for seed in seeds:
        breakdown = seed.get("flop_breakdown")
        if breakdown is None:
            continue
        has_all = all(k in breakdown for k in keys)
        if not has_all:
            continue
        for k in keys:
            totals[k].append(breakdown[k])

    # Need at least one seed with complete breakdown data
    if not totals["qkv_proj"]:
        return None

    # Average across seeds and convert to TFLOPS
    return {
        k: (sum(vals) / len(vals)) / 1e12
        for k, vals in totals.items()
    }


# ---------------------------------------------------------------------------
# Section 1: Roofline Diagram
# ---------------------------------------------------------------------------

st.subheader("Roofline Diagram")

roofline_variants: dict[str, dict] = {}
roofline_missing: list[str] = []

for name in selected_variants:
    eff = _extract_efficiency(name)
    if eff is not None and "tflops" in eff and "arithmetic_intensity" in eff:
        roofline_variants[name] = {
            "tflops": eff["tflops"],
            "arithmetic_intensity": eff["arithmetic_intensity"],
        }
    else:
        roofline_missing.append(name)

if roofline_variants:
    fig_roofline = chart_factory.create_roofline(
        roofline_variants,
        hw_bandwidth_gbps=300.0,
        hw_peak_tflops=242.0,
    )
    st.plotly_chart(fig_roofline, use_container_width=True)
else:
    st.info("No efficiency data available for the selected variants.")

if roofline_missing:
    st.caption(
        f"Roofline data unavailable for: {', '.join(sorted(roofline_missing))}"
    )

# ---------------------------------------------------------------------------
# Section 2: FLOP Breakdown (Stacked Bar Chart)
# ---------------------------------------------------------------------------

st.subheader("FLOP Breakdown")

flop_categories: list[str] = []
flop_stacks: dict[str, list[float]] = {
    "QKV Projection": [],
    "Attention Score": [],
    "Attention Output": [],
    "FFN": [],
}
flop_missing: list[str] = []

# Mapping from internal key names to display names
_flop_key_to_label = {
    "qkv_proj": "QKV Projection",
    "attention_score": "Attention Score",
    "attention_output": "Attention Output",
    "ffn": "FFN",
}

for name in selected_variants:
    breakdown = _extract_flop_breakdown(name)
    if breakdown is not None:
        flop_categories.append(name)
        for key, label in _flop_key_to_label.items():
            flop_stacks[label].append(breakdown[key])
    else:
        flop_missing.append(name)

if flop_categories:
    fig_flop = chart_factory.create_stacked_bar_chart(
        categories=flop_categories,
        stacks=flop_stacks,
        title="FLOP Breakdown per Variant",
        yaxis_title="TFLOPS",
    )
    st.plotly_chart(fig_flop, use_container_width=True)
else:
    st.info("No FLOP breakdown data available for the selected variants.")

if flop_missing:
    st.caption(
        f"FLOP breakdown data unavailable for: {', '.join(sorted(flop_missing))}"
    )

# ---------------------------------------------------------------------------
# Section 3: MFU Comparison (Bar Chart)
# ---------------------------------------------------------------------------

st.subheader("Model FLOPs Utilization")

mfu_categories: list[str] = []
mfu_values: list[float] = []
mfu_missing: list[str] = []

for name in selected_variants:
    eff = _extract_efficiency(name)
    if eff is not None and "mfu" in eff:
        mfu_categories.append(name)
        mfu_values.append(eff["mfu"])
    else:
        mfu_missing.append(name)

if mfu_categories:
    fig_mfu = chart_factory.create_bar_chart(
        categories=mfu_categories,
        values=mfu_values,
        title="Model FLOPs Utilization",
        yaxis_title="MFU (%)",
    )
    # Set y-axis range to 0-100%
    fig_mfu.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_mfu, use_container_width=True)
else:
    st.info("No MFU data available for the selected variants.")

if mfu_missing:
    st.caption(
        f"MFU data unavailable for: {', '.join(sorted(mfu_missing))}"
    )
