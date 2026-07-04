"""Comparison Axes page.

Displays bar charts comparing variants under controlled resource constraints:
- Fixed data (same token budget)
- Fixed wallclock (same training duration)
- Fixed FLOPs (same cumulative compute)

Each chart shows val_loss per variant with error bars from seed aggregation
and highlights Pareto-front variants with a distinct border.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components import chart_factory
from dashboard.components.data_loader import get_seed_count, get_variant_names, load_metrics
from dashboard.components.sidebar import render_sidebar

st.set_page_config(page_title="Comparison Axes", layout="wide")
st.title("Comparison Axes")

# ---------------------------------------------------------------------------
# Sidebar: report directory + variant toggles
# ---------------------------------------------------------------------------

# Load data first to populate sidebar
# We need to get the report_dir from sidebar, but sidebar also needs data.
# Follow the same two-pass pattern: render sidebar with None first for dir,
# then load, then re-render with data for toggles.

# Initial sidebar pass to get report_dir
with st.sidebar:
    st.header("Settings")
    import os

    env_report_dir = os.environ.get("REPORT_DIR")
    if env_report_dir:
        report_dir = env_report_dir
        st.text_input(
            "Report Directory",
            value=report_dir,
            disabled=True,
            help="Set via REPORT_DIR environment variable.",
        )
    else:
        report_dir = st.text_input(
            "Report Directory",
            value="reports/",
            help="Path to the evaluation report directory containing raw/metrics.json.",
        )

# Load data
data = load_metrics(report_dir)

if data is None:
    st.stop()

# Get variant names and render toggles
variant_names = get_variant_names(data)

with st.sidebar:
    st.markdown("---")
    st.subheader("Variants")
    selected_variants: list[str] = []
    for name in variant_names:
        checked = st.checkbox(
            name,
            value=True,
            key=f"comparison_{name}",
        )
        if checked:
            selected_variants.append(name)

# ---------------------------------------------------------------------------
# Extract comparison data
# ---------------------------------------------------------------------------

comparison = data.get("comparison")
if not isinstance(comparison, dict):
    st.warning("No comparison data available in the metrics file.")
    st.stop()

aggregated = data.get("aggregated", {})

fixed_data: dict = comparison.get("fixed_data", {})
fixed_wallclock: dict = comparison.get("fixed_wallclock", {})
fixed_flops: dict = comparison.get("fixed_flops", {})
pareto_front: list[str] = comparison.get("pareto_front", [])
pareto_set: set[str] = set(pareto_front)

# ---------------------------------------------------------------------------
# Handle: no variants selected
# ---------------------------------------------------------------------------

if not selected_variants:
    st.info("No variants selected. Please select at least one variant from the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Helper: build bar chart for one axis
# ---------------------------------------------------------------------------


def _build_axis_chart(
    axis_data: dict,
    axis_title: str,
    chart_title: str,
) -> None:
    """Render a bar chart for one comparison axis.

    Filters to selected variants that have data for this axis.
    Shows a message if fewer than 2 variants have data.
    """
    # Filter to selected variants with data for this axis
    categories: list[str] = []
    values: list[float] = []
    errors: list[float | None] = []

    for variant in selected_variants:
        val = axis_data.get(variant)
        if val is None:
            continue
        # For fixed_wallclock, val is a dict mapping fraction → val_loss
        if isinstance(val, dict):
            # Use the "1.0" fraction (full budget) for the bar chart
            full_budget_val = val.get("1.0")
            if full_budget_val is None:
                continue
            val = full_budget_val

        categories.append(variant)
        values.append(float(val))

        # Get std from aggregated data if available
        agg = aggregated.get(variant, {})
        val_loss_agg = agg.get("val_loss", {})
        std = val_loss_agg.get("std") if isinstance(val_loss_agg, dict) else None
        errors.append(std)

    # Check minimum variant count
    if len(categories) < 2:
        st.caption(
            f"**{chart_title}**: Insufficient data for comparison — "
            f"fewer than 2 selected variants have data for this axis."
        )
        return

    # Build hover texts with seed count
    hover_texts: list[str] = []
    for i, cat in enumerate(categories):
        val = values[i]
        std = errors[i]
        std_str = f"± {std:.4f}" if std is not None else "N/A"
        seed_count = get_seed_count(data, cat)
        hover_texts.append(
            f"<b>{cat}</b><br>"
            f"Val Loss: {val:.4f}<br>"
            f"Std: {std_str}<br>"
            f"Seeds: {seed_count}<extra></extra>"
        )

    # Create bar chart with Pareto highlighting
    fig = chart_factory.create_bar_chart(
        categories=categories,
        values=values,
        errors=errors,
        title=chart_title,
        xaxis_title="Variant",
        yaxis_title="Validation Loss",
        highlights=pareto_set,
    )

    # Override hover template with our custom one that includes seed count
    fig.data[0].hovertemplate = hover_texts

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Render the 3 comparison axis charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "Compare variants under controlled resource constraints. "
    "Pareto-front variants are highlighted with a white border."
)

_build_axis_chart(
    axis_data=fixed_data,
    axis_title="Variant",
    chart_title="Fixed Token Budget (Equal Data)",
)

_build_axis_chart(
    axis_data=fixed_wallclock,
    axis_title="Variant",
    chart_title="Fixed Wall-Clock Time",
)

_build_axis_chart(
    axis_data=fixed_flops,
    axis_title="Variant",
    chart_title="Fixed FLOP Budget",
)

# ---------------------------------------------------------------------------
# Show unavailable variants notice
# ---------------------------------------------------------------------------

# Determine which selected variants have no data across any axis
all_axis_variants = set(fixed_data.keys()) | set(fixed_wallclock.keys()) | set(fixed_flops.keys())
unavailable = [v for v in selected_variants if v not in all_axis_variants]

if unavailable:
    st.caption(f"Data unavailable for: {', '.join(sorted(unavailable))}")
