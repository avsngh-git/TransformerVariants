"""Tests for dashboard/components/sidebar.py.

Tests validate the sidebar module's importability, function signatures,
and behavior using Streamlit's AppTest framework.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


# ---------------------------------------------------------------------------
# Import Tests
# ---------------------------------------------------------------------------


def test_sidebar_module_imports():
    """Verify sidebar module can be imported without error."""
    from dashboard.components.sidebar import render_sidebar, render_variant_toggles

    assert callable(render_sidebar)
    assert callable(render_variant_toggles)


# ---------------------------------------------------------------------------
# AppTest-based Integration Tests
# ---------------------------------------------------------------------------


def _sidebar_with_data():
    """App script: render_sidebar with valid data."""
    import streamlit as st
    from dashboard.components.sidebar import render_sidebar

    data = {
        "variants": {
            "alibi": [{"seed_index": 0}],
            "modern": [{"seed_index": 0}],
            "vanilla": [{"seed_index": 0}],
        },
        "aggregated": {},
        "comparison": {},
    }
    result = render_sidebar(data)
    st.session_state["sidebar_result"] = result


def _sidebar_no_data():
    """App script: render_sidebar with None data."""
    import streamlit as st
    from dashboard.components.sidebar import render_sidebar

    result = render_sidebar(None)
    st.session_state["sidebar_result"] = result


def _sidebar_env_var():
    """App script: render_sidebar with REPORT_DIR env var set."""
    import os

    import streamlit as st
    from dashboard.components.sidebar import render_sidebar

    os.environ["REPORT_DIR"] = "/custom/path"
    result = render_sidebar(None)
    st.session_state["sidebar_result"] = result
    del os.environ["REPORT_DIR"]


def _variant_toggles_available():
    """App script: render_variant_toggles with available subset."""
    import streamlit as st
    from dashboard.components.sidebar import render_variant_toggles

    with st.sidebar:
        result = render_variant_toggles(
            ["alibi", "modern", "vanilla"],
            available={"alibi", "vanilla"},
            key_prefix="test",
        )
        st.session_state["toggle_result"] = result


class TestRenderSidebar:
    """Tests for render_sidebar using Streamlit's AppTest."""

    def test_sidebar_returns_dict_with_data(self):
        """render_sidebar with data returns dict with report_dir and selected_variants."""
        at = AppTest.from_function(_sidebar_with_data)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["sidebar_result"]
        assert "report_dir" in result
        assert "selected_variants" in result
        assert isinstance(result["report_dir"], str)
        assert isinstance(result["selected_variants"], list)

    def test_sidebar_default_report_dir(self):
        """Default report_dir is 'reports/' when no env var is set."""
        at = AppTest.from_function(_sidebar_no_data)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["sidebar_result"]
        assert result["report_dir"] == "reports/"

    def test_sidebar_no_data_empty_variants(self):
        """When data is None, selected_variants is empty."""
        at = AppTest.from_function(_sidebar_no_data)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["sidebar_result"]
        assert result["selected_variants"] == []

    def test_sidebar_env_var_overrides_default(self):
        """REPORT_DIR env var takes precedence over default."""
        at = AppTest.from_function(_sidebar_env_var)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["sidebar_result"]
        assert result["report_dir"] == "/custom/path"

    def test_sidebar_all_variants_selected_by_default(self):
        """All variants are checked by default when data is provided."""
        at = AppTest.from_function(_sidebar_with_data)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["sidebar_result"]
        # All 3 variants should be selected
        assert sorted(result["selected_variants"]) == ["alibi", "modern", "vanilla"]


class TestRenderVariantToggles:
    """Tests for render_variant_toggles."""

    def test_unavailable_variants_not_in_selection(self):
        """Unavailable variants are disabled and not included in the result."""
        at = AppTest.from_function(_variant_toggles_available)
        at.run(timeout=10)
        assert not at.exception

        result = at.session_state["toggle_result"]
        # "modern" is unavailable, so should not be in selected
        assert "modern" not in result
        # "alibi" and "vanilla" are available and checked by default
        assert "alibi" in result
        assert "vanilla" in result
