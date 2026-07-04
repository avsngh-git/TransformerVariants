"""Per-Position Loss and ICL page.

Displays per-position loss curves with ICL power-law fit overlays,
and a comparison table of ICL decay parameters across variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components import chart_factory
from dashboard.components.data_loader import get_variant_names, load_metrics
from dashboard.components.sidebar import render_sidebar
from dashboard.components.styling import get_variant_color
from dashboard.pages.per_position_logic import (
    build_icl_table,
    compute_icl_curve,
    extract_icl_fit_params,
    extract_per_position_loss,
)

st.set_page_config(page_title="Per-Position Loss", layout="wide")
st.title("Per-Position Loss & ICL")

# ---------------------------------------------------------------------------
# Data loading and sidebar
# ---------------------------------------------------------------------------

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
# Data extraction
# ---------------------------------------------------------------------------

variants_data = data.get("variants", {})
variant_names = get_variant_names(data)

# ---------------------------------------------------------------------------
# Per-position loss line chart with ICL fit overlay
# ---------------------------------------------------------------------------

st.subheader("Per-Position Loss")

traces = []
chart_missing: list[str] = []

for name in selected_variants:
    seeds = variants_data.get(name, [])
    loss_data = extract_per_position_loss(seeds)

    if loss_data is None:
        chart_missing.append(name)
        continue

    seq_len = len(loss_data)
    positions = list(range(1, seq_len + 1))
    color = get_variant_color(name, variant_names)

    # Main per-position loss trace
    traces.append(
        {
            "name": name,
            "x": positions,
            "y": loss_data,
            "color": color,
            "dash": "solid",
        }
    )

    # ICL power-law fit overlay (dashed)
    fit_params = extract_icl_fit_params(seeds)
    if fit_params is not None:
        fitted_curve = compute_icl_curve(
            A=fit_params["A"],
            alpha=fit_params["alpha"],
            C=fit_params["C"],
            seq_len=seq_len,
        )
        traces.append(
            {
                "name": f"{name} (ICL fit)",
                "x": positions,
                "y": fitted_curve,
                "color": color,
                "dash": "dash",
                "showlegend": True,
            }
        )

if traces:
    fig = chart_factory.create_line_chart(
        traces=traces,
        title="Per-Position Loss with ICL Fit",
        xaxis_title="Position Index",
        yaxis_title="Cross-Entropy Loss",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No per-position loss data available for the selected variants. "
        "Run the evaluation pipeline with per-position loss enabled."
    )

if chart_missing:
    st.caption(
        f"Per-position loss data unavailable for: {', '.join(sorted(chart_missing))}"
    )

# ---------------------------------------------------------------------------
# ICL decay comparison table
# ---------------------------------------------------------------------------

st.subheader("ICL Decay Comparison")

table_rows = build_icl_table(selected_variants, variants_data)

if table_rows:
    # Build markdown table
    header = "| Variant | α (decay exponent) | C (loss floor) | R² |"
    separator = "|---------|-------------------|----------------|-----|"
    rows_md = [header, separator]

    for row in table_rows:
        if row["has_data"]:
            alpha_str = f"{row['alpha']:.4f}"
            c_str = f"{row['C']:.4f}"
            r2_val = row["r_squared"]
            r2_str = f"{r2_val:.4f}"
            if row["poor_fit"]:
                r2_str += " ⚠️"
            rows_md.append(f"| {row['variant']} | {alpha_str} | {c_str} | {r2_str} |")
        else:
            rows_md.append(f"| {row['variant']} | N/A | N/A | N/A |")

    st.markdown("\n".join(rows_md))
else:
    st.info("No ICL decay data available for the selected variants.")
