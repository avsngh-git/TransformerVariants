"""Tests for the Comparison Axes page logic.

Tests the data extraction and chart building logic used by
dashboard/pages/2_comparison_axes.py.
"""

import json
import pathlib

import pytest

from dashboard.components import chart_factory
from dashboard.components.data_loader import get_seed_count, get_variant_names


FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_metrics():
    with open(FIXTURES_DIR / "valid_metrics.json") as f:
        return json.load(f)


class TestComparisonDataExtraction:
    """Test extraction of comparison data from metrics JSON."""

    def test_fixed_data_keys_match_variants(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        fixed_data = comparison["fixed_data"]
        # All variants in fixed_data should be in the variants section
        variant_names = set(get_variant_names(valid_metrics))
        assert set(fixed_data.keys()).issubset(variant_names)

    def test_fixed_wallclock_uses_fraction_1_0(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        fixed_wallclock = comparison["fixed_wallclock"]
        # Each entry is a dict with fraction keys including "1.0"
        for variant, fractions in fixed_wallclock.items():
            assert isinstance(fractions, dict)
            assert "1.0" in fractions
            assert isinstance(fractions["1.0"], (int, float))

    def test_fixed_flops_keys_match_variants(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        fixed_flops = comparison["fixed_flops"]
        variant_names = set(get_variant_names(valid_metrics))
        assert set(fixed_flops.keys()).issubset(variant_names)

    def test_pareto_front_subset_of_variants(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        pareto_front = comparison["pareto_front"]
        variant_names = set(get_variant_names(valid_metrics))
        assert set(pareto_front).issubset(variant_names)


class TestBarChartCreation:
    """Test that bar charts are created correctly for comparison axes."""

    def test_bar_chart_with_valid_data(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        aggregated = valid_metrics["aggregated"]
        fixed_data = comparison["fixed_data"]
        pareto_set = set(comparison["pareto_front"])

        categories = sorted(fixed_data.keys())
        values = [fixed_data[v] for v in categories]
        errors = [
            aggregated.get(v, {}).get("val_loss", {}).get("std")
            for v in categories
        ]

        fig = chart_factory.create_bar_chart(
            categories=categories,
            values=values,
            errors=errors,
            title="Fixed Token Budget",
            xaxis_title="Variant",
            yaxis_title="Validation Loss",
            highlights=pareto_set,
        )

        # Should have exactly one bar trace
        assert len(fig.data) == 1
        bar_trace = fig.data[0]
        assert list(bar_trace.x) == categories
        assert list(bar_trace.y) == values

    def test_bar_chart_highlights_pareto_variants(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        aggregated = valid_metrics["aggregated"]
        fixed_data = comparison["fixed_data"]
        pareto_set = set(comparison["pareto_front"])

        categories = sorted(fixed_data.keys())
        values = [fixed_data[v] for v in categories]
        errors = [
            aggregated.get(v, {}).get("val_loss", {}).get("std")
            for v in categories
        ]

        fig = chart_factory.create_bar_chart(
            categories=categories,
            values=values,
            errors=errors,
            title="Fixed Token Budget",
            xaxis_title="Variant",
            yaxis_title="Validation Loss",
            highlights=pareto_set,
        )

        bar_trace = fig.data[0]
        marker_widths = bar_trace.marker.line.width
        marker_colors = bar_trace.marker.line.color

        # Pareto variants should have width 3, others 0
        for i, cat in enumerate(categories):
            if cat in pareto_set:
                assert marker_widths[i] == 3
                assert marker_colors[i] == "white"
            else:
                assert marker_widths[i] == 0

    def test_bar_chart_error_bars_present(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        aggregated = valid_metrics["aggregated"]
        fixed_data = comparison["fixed_data"]

        categories = sorted(fixed_data.keys())
        values = [fixed_data[v] for v in categories]
        errors = [
            aggregated.get(v, {}).get("val_loss", {}).get("std")
            for v in categories
        ]

        fig = chart_factory.create_bar_chart(
            categories=categories,
            values=values,
            errors=errors,
            title="Test",
            yaxis_title="Val Loss",
        )

        bar_trace = fig.data[0]
        # Error bars should be set
        assert bar_trace.error_y is not None
        assert bar_trace.error_y.visible is True
        assert bar_trace.error_y.array is not None

    def test_bar_chart_no_errors_when_none(self):
        """When errors are None, no error bars should be visible."""
        fig = chart_factory.create_bar_chart(
            categories=["a", "b", "c"],
            values=[1.0, 2.0, 3.0],
            errors=None,
            title="No errors",
        )
        bar_trace = fig.data[0]
        # Plotly returns an empty ErrorY object; check that array is not set
        error_y = bar_trace.error_y
        assert error_y.array is None or error_y.visible is not True


class TestInsufficientData:
    """Test handling of insufficient data for comparison."""

    def test_single_variant_fixed_data(self):
        """With only 1 variant, chart should not be rendered (< 2 variants)."""
        # This logic is in the page itself; we test that chart_factory
        # can handle single-element inputs (the page checks count before calling)
        fig = chart_factory.create_bar_chart(
            categories=["vanilla"],
            values=[3.5],
            errors=[0.01],
            title="Single",
        )
        # chart_factory doesn't enforce minimum; the page does
        assert len(fig.data) == 1

    def test_empty_categories(self):
        """Empty categories produces a figure with no bars."""
        fig = chart_factory.create_bar_chart(
            categories=[],
            values=[],
            errors=None,
            title="Empty",
        )
        assert len(fig.data) == 1  # Still creates a Bar trace, just empty
        assert list(fig.data[0].x) == []


class TestWallclockFractionExtraction:
    """Test extracting the 1.0 fraction from fixed_wallclock data."""

    def test_extract_full_budget_fraction(self, valid_metrics):
        comparison = valid_metrics["comparison"]
        fixed_wallclock = comparison["fixed_wallclock"]

        for variant, fractions in fixed_wallclock.items():
            val = fractions.get("1.0")
            assert val is not None
            assert isinstance(val, (int, float))

    def test_missing_fraction_key_handled(self):
        """If 1.0 fraction is missing, variant should be skipped."""
        wallclock_data = {
            "vanilla": {"0.5": 4.5},  # no "1.0" key
            "modern": {"0.5": 4.2, "1.0": 3.2},
        }

        # Simulate the page's extraction logic
        categories = []
        values = []
        for variant in sorted(wallclock_data.keys()):
            val = wallclock_data[variant]
            if isinstance(val, dict):
                full_budget_val = val.get("1.0")
                if full_budget_val is None:
                    continue
                val = full_budget_val
            categories.append(variant)
            values.append(float(val))

        # Only "modern" has the 1.0 fraction
        assert categories == ["modern"]
        assert values == [3.2]


class TestSeedCountInTooltip:
    """Test that seed count can be retrieved for tooltip."""

    def test_seed_count_from_valid_data(self, valid_metrics):
        # vanilla has 2 seeds in the fixture
        count = get_seed_count(valid_metrics, "vanilla")
        assert count == 2

    def test_seed_count_missing_variant(self, valid_metrics):
        count = get_seed_count(valid_metrics, "nonexistent")
        assert count == 0
