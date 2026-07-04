"""Sidebar controls for the dashboard.

Manages the sidebar UI: report directory selection and per-variant
show/hide toggles shared across all pages.
"""

from __future__ import annotations

import os

import streamlit as st

from dashboard.components.data_loader import get_variant_names


def render_sidebar(data: dict | None) -> dict:
    """Render sidebar controls and return current selections.

    The sidebar contains:
    1. Report directory input — sourced from the REPORT_DIR environment
       variable if set, otherwise a text input defaulting to "reports/".
    2. Variant toggles — per-variant checkboxes (all checked by default),
       rendered only when data is successfully loaded.

    Args:
        data: The parsed metrics dict, or None if loading failed.

    Returns:
        dict with keys:
            - report_dir (str): The resolved report directory path.
            - selected_variants (list[str]): Currently selected variant names.
    """
    with st.sidebar:
        st.header("Settings")

        # --- Report directory ---
        env_report_dir = os.environ.get("REPORT_DIR")

        if env_report_dir:
            # Environment variable takes precedence; show as info, not editable
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

        # --- Variant toggles ---
        selected_variants: list[str] = []
        if data is not None:
            variant_names = get_variant_names(data)
            if variant_names:
                st.markdown("---")
                selected_variants = render_variant_toggles(variant_names)

    return {
        "report_dir": report_dir,
        "selected_variants": selected_variants,
    }


def render_variant_toggles(
    variant_names: list[str],
    available: set[str] | None = None,
    key_prefix: str = "variant",
) -> list[str]:
    """Render per-variant checkboxes in the sidebar.

    All checkboxes are checked by default on initial page load. If
    `available` is provided, checkboxes for unavailable variants are
    disabled and display a "(unavailable)" suffix.

    Args:
        variant_names: Sorted list of all variant names.
        available: Optional set of variant names that have data for
            the current page. If None, all variants are treated as available.
        key_prefix: Streamlit widget key prefix to avoid key collisions
            when the same function is used on multiple pages.

    Returns:
        List of currently selected (checked) variant names.
    """
    st.subheader("Variants")

    selected: list[str] = []

    for name in variant_names:
        is_available = available is None or name in available
        label = name if is_available else f"{name} (unavailable)"

        checked = st.checkbox(
            label,
            value=is_available,  # checked by default only if available
            disabled=not is_available,
            key=f"{key_prefix}_{name}",
        )

        if checked and is_available:
            selected.append(name)

    return selected
