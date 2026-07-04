"""Learning Curves page.

Displays validation loss over training progress for each selected variant.
Supports multiple x-axis views (tokens seen, wall-clock time, cumulative FLOPs)
and an optional seed envelope showing mean ± std for multi-seed variants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from dashboard.components import chart_factory, styling
from dashboard.components.data_loader import get_variant_names, load_metrics
from dashboard.components.sidebar import render_sidebar, render_variant_toggles
from dashboard.pages.learning_curves_logic import extract_learning_curve

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Learning Curves", layout="wide")
st.title("Learning Curves")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Get report_dir from env var, session state, or sidebar
import os
report_dir = os.environ.get("REPORT_DIR") or st.session_state.get("report_dir", None)
if report_dir is None:
    report_dir = st.sidebar.text_input(
        "Report Directory",
        value="reports/",
        help="Path to the evaluation report directory containing raw/metrics.json.",
    )
    st.session_state["report_dir"] = report_dir

data = load_metrics(report_dir)

if data is None:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: variant toggles
# ---------------------------------------------------------------------------

variant_names = get_variant_names(data)

# Determine which variants have log_entries data
available_variants: set[str] = set()
variants_data = data.get("variants", {})
for name in variant_names:
    seeds = variants_data.get(name, [])
    has_log_entries = any(
        isinstance(seed.get("log_entries"), list) and len(seed["log_entries"]) > 0
        for seed in seeds
        if isinstance(seed, dict)
    )
    if has_log_entries:
        available_variants.add(name)

with st.sidebar:
    st.header("Settings")
    selected_variants = render_variant_toggles(
        variant_names,
        available=available_variants,
        key_prefix="lc_variant",
    )

# ---------------------------------------------------------------------------
# No variants selected
# ---------------------------------------------------------------------------

if not selected_variants:
    st.info("Select at least one variant from the sidebar to display learning curves.")
    st.stop()

# ---------------------------------------------------------------------------
# X-axis toggle
# ---------------------------------------------------------------------------

X_AXIS_OPTIONS = {
    "Tokens Seen": "tokens_seen",
    "Wall-clock Time": "wallclock",
    "Cumulative FLOPs": "cumulative_flops",
}

x_axis_label = st.radio(
    "X-Axis",
    options=list(X_AXIS_OPTIONS.keys()),
    index=0,
    horizontal=True,
)
x_axis_key = X_AXIS_OPTIONS[x_axis_label]

# ---------------------------------------------------------------------------
# Seed envelope toggle
# ---------------------------------------------------------------------------

show_envelope = st.checkbox("Show seed envelope (mean ± std)", value=False)

# ---------------------------------------------------------------------------
# Build traces
# ---------------------------------------------------------------------------

traces = []
envelope_traces = []

for variant_name in selected_variants:
    seeds = variants_data.get(variant_name, [])
    curve_data = extract_learning_curve(seeds, x_axis_key)

    if curve_data is None:
        # Skip variants with no log_entries
        continue

    color = styling.get_variant_color(variant_name, variant_names)

    # Main line trace (mean)
    traces.append(
        {
            "name": variant_name,
            "x": curve_data["x_values"],
            "y": curve_data["mean_loss"],
            "color": color,
            "dash": "solid",
        }
    )

    # Seed envelope (mean ± std) for multi-seed variants
    if show_envelope and curve_data["std_loss"] is not None and curve_data["num_seeds"] > 1:
        mean_arr = np.array(curve_data["mean_loss"])
        std_arr = np.array(curve_data["std_loss"])
        x_vals = curve_data["x_values"]

        upper = (mean_arr + std_arr).tolist()
        lower = (mean_arr - std_arr).tolist()

        # Convert hex color to rgba for fill
        # Parse hex to get rgba with low opacity
        hex_color = color.lstrip("#")
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        fill_color = f"rgba({r},{g},{b},0.15)"

        # Lower bound trace (invisible line, anchor for fill)
        envelope_traces.append(
            {
                "name": f"{variant_name} (lower)",
                "x": x_vals,
                "y": lower,
                "color": "rgba(0,0,0,0)",
                "dash": "solid",
                "showlegend": False,
            }
        )

        # Upper bound trace with fill to lower
        envelope_traces.append(
            {
                "name": f"{variant_name} (envelope)",
                "x": x_vals,
                "y": upper,
                "color": "rgba(0,0,0,0)",
                "dash": "solid",
                "fill": "tonexty",
                "fillcolor": fill_color,
                "showlegend": False,
            }
        )

# ---------------------------------------------------------------------------
# Render chart
# ---------------------------------------------------------------------------

# Combine envelope traces first (so they appear behind), then main traces
all_traces = envelope_traces + traces

if all_traces:
    fig = chart_factory.create_line_chart(
        traces=all_traces,
        title="Validation Loss over Training",
        xaxis_title=x_axis_label,
        yaxis_title="Validation Loss",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No learning curve data available for the selected variants.")
