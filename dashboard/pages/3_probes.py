"""Probes page.

Displays diagnostic probe results for each selected variant:
- MQAR accuracy by retrieval distance
- Stable rank per layer
- CKA adjacent-layer similarity + full L×L heatmap
- Attention entropy per layer (only for variants with data)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.components import chart_factory, styling
from dashboard.components.data_loader import get_variant_names, load_metrics
from dashboard.components.sidebar import render_variant_toggles
from dashboard.pages.probes_logic import (
    average_lists,
    classify_variants_for_probe,
    get_cka_matrix,
    get_probe_field,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Probes", layout="wide")
st.title("Probes")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
variants_data = data.get("variants", {})

# Determine which variants have any probe data
available_variants: set[str] = set()
for name in variant_names:
    seeds = variants_data.get(name, [])
    has_probe = any(
        isinstance(seed, dict)
        and (
            seed.get("mqar") is not None
            or seed.get("stable_rank") is not None
            or seed.get("cka") is not None
            or seed.get("attention_entropy") is not None
        )
        for seed in seeds
    )
    if has_probe:
        available_variants.add(name)

with st.sidebar:
    st.header("Settings")
    selected_variants = render_variant_toggles(
        variant_names,
        available=available_variants,
        key_prefix="probe_variant",
    )

# ---------------------------------------------------------------------------
# No variants selected
# ---------------------------------------------------------------------------

if not selected_variants:
    st.info("Select at least one variant from the sidebar to display probe results.")
    st.stop()


# ---------------------------------------------------------------------------
# Section 1: MQAR accuracy by retrieval distance
# ---------------------------------------------------------------------------

st.subheader("MQAR Accuracy by Retrieval Distance")

mqar_available, mqar_unavailable = classify_variants_for_probe(
    variants_data, selected_variants, "mqar", "accuracies"
)

if mqar_available:
    traces = []
    for name in mqar_available:
        accuracies_per_seed = get_probe_field(variants_data, name, "mqar", "accuracies")
        distances_per_seed = get_probe_field(variants_data, name, "mqar", "distances")

        avg_accuracies = average_lists(accuracies_per_seed)
        # Use distances from the first seed
        distances = distances_per_seed[0] if distances_per_seed else list(
            range(1, len(avg_accuracies) + 1)
        )

        traces.append(
            {
                "name": name,
                "x": distances,
                "y": avg_accuracies,
                "color": styling.get_variant_color(name, variant_names),
            }
        )

    fig = chart_factory.create_line_chart(
        traces=traces,
        title="MQAR Accuracy by Retrieval Distance",
        xaxis_title="Retrieval Distance",
        yaxis_title="Recall Accuracy",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No MQAR data available for the selected variants.")

if mqar_unavailable:
    st.caption(f"MQAR data unavailable for: {', '.join(sorted(mqar_unavailable))}")

# ---------------------------------------------------------------------------
# Section 2: Stable rank per layer
# ---------------------------------------------------------------------------

st.subheader("Stable Rank per Layer")

sr_available, sr_unavailable = classify_variants_for_probe(
    variants_data, selected_variants, "stable_rank", "per_layer"
)

if sr_available:
    traces = []
    for name in sr_available:
        per_layer_per_seed = get_probe_field(variants_data, name, "stable_rank", "per_layer")
        avg_per_layer = average_lists(per_layer_per_seed)
        layer_indices = list(range(len(avg_per_layer)))

        traces.append(
            {
                "name": name,
                "x": layer_indices,
                "y": avg_per_layer,
                "color": styling.get_variant_color(name, variant_names),
            }
        )

    fig = chart_factory.create_line_chart(
        traces=traces,
        title="Stable Rank per Layer",
        xaxis_title="Layer Index",
        yaxis_title="Stable Rank",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No stable rank data available for the selected variants.")

if sr_unavailable:
    st.caption(
        f"Stable rank data unavailable for: {', '.join(sorted(sr_unavailable))}"
    )

# ---------------------------------------------------------------------------
# Section 3: CKA adjacent-layer similarity + heatmap
# ---------------------------------------------------------------------------

st.subheader("CKA Adjacent-Layer Similarity")

cka_available, cka_unavailable = classify_variants_for_probe(
    variants_data, selected_variants, "cka", "adjacent_similarities"
)

if cka_available:
    # Line chart: adjacent similarities
    traces = []
    for name in cka_available:
        adj_per_seed = get_probe_field(variants_data, name, "cka", "adjacent_similarities")
        avg_adj = average_lists(adj_per_seed)
        layer_pairs = list(range(len(avg_adj)))

        traces.append(
            {
                "name": name,
                "x": layer_pairs,
                "y": avg_adj,
                "color": styling.get_variant_color(name, variant_names),
            }
        )

    fig = chart_factory.create_line_chart(
        traces=traces,
        title="CKA Adjacent-Layer Similarity",
        xaxis_title="Layer Pair Index",
        yaxis_title="CKA Similarity",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-variant heatmap via selectbox
    with st.expander("CKA Full L×L Heatmap", expanded=False):
        heatmap_variant = st.selectbox(
            "Select variant for CKA heatmap",
            options=cka_available,
            key="cka_heatmap_variant",
        )

        if heatmap_variant:
            matrix = get_cka_matrix(variants_data, heatmap_variant)
            if matrix:
                num_layers = len(matrix)
                labels = [str(i) for i in range(num_layers)]
                heatmap_fig = chart_factory.create_heatmap(
                    matrix=matrix,
                    labels=labels,
                    title=f"CKA Similarity Matrix — {heatmap_variant}",
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.warning(
                    f"No CKA full matrix available for {heatmap_variant}."
                )
else:
    st.info("No CKA data available for the selected variants.")

if cka_unavailable:
    st.caption(f"CKA data unavailable for: {', '.join(sorted(cka_unavailable))}")

# ---------------------------------------------------------------------------
# Section 4: Attention entropy per layer
# ---------------------------------------------------------------------------

st.subheader("Attention Entropy per Layer")

entropy_available, entropy_unavailable = classify_variants_for_probe(
    variants_data, selected_variants, "attention_entropy", "per_layer"
)

if entropy_available:
    traces = []
    for name in entropy_available:
        per_layer_per_seed = get_probe_field(
            variants_data, name, "attention_entropy", "per_layer"
        )
        avg_per_layer = average_lists(per_layer_per_seed)
        layer_indices = list(range(len(avg_per_layer)))

        traces.append(
            {
                "name": name,
                "x": layer_indices,
                "y": avg_per_layer,
                "color": styling.get_variant_color(name, variant_names),
            }
        )

    fig = chart_factory.create_line_chart(
        traces=traces,
        title="Attention Entropy per Layer",
        xaxis_title="Layer Index",
        yaxis_title="Mean Shannon Entropy",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No attention entropy data available for the selected variants. "
        "Only non-flash-based variants (V0, V5) typically have this data."
    )

if entropy_unavailable:
    st.caption(
        f"Attention entropy data unavailable for: {', '.join(sorted(entropy_unavailable))}"
    )
