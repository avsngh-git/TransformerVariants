"""Unit tests for dashboard/components/chart_factory.py."""

import plotly.graph_objects as go
import pytest

from dashboard.components.chart_factory import (
    create_bar_chart,
    create_heatmap,
    create_line_chart,
    create_roofline,
    create_stacked_bar_chart,
)
from dashboard.components.styling import PALETTE, get_plotly_layout, get_variant_color


class TestCreateLineChart:
    """Tests for create_line_chart."""

    def test_returns_plotly_figure(self):
        traces = [{"name": "vanilla", "x": [1, 2], "y": [3.0, 2.5]}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert isinstance(fig, go.Figure)

    def test_correct_trace_count(self):
        traces = [
            {"name": "vanilla", "x": [1, 2], "y": [3.0, 2.5]},
            {"name": "modern", "x": [1, 2], "y": [2.8, 2.3]},
            {"name": "gqa", "x": [1, 2], "y": [2.9, 2.4]},
        ]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert len(fig.data) == 3

    def test_trace_names_preserved(self):
        traces = [
            {"name": "vanilla", "x": [1], "y": [3.0]},
            {"name": "modern", "x": [1], "y": [2.8]},
        ]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.data[0].name == "vanilla"
        assert fig.data[1].name == "modern"

    def test_dark_theme_applied(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0]}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.layout.plot_bgcolor == "rgba(17,17,17,1)"
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"

    def test_legend_shown_by_default(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0]}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.layout.showlegend is True

    def test_legend_can_be_hidden(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0]}]
        fig = create_line_chart(traces, "Title", "X", "Y", show_legend=False)
        assert fig.layout.showlegend is False

    def test_dash_style_applied(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0], "dash": "dash"}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.data[0].line.dash == "dash"

    def test_invisible_trace_becomes_legendonly(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0], "visible": False}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.data[0].visible == "legendonly"

    def test_axis_titles_set(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0]}]
        fig = create_line_chart(traces, "My Title", "Steps", "Loss")
        assert fig.layout.title.text == "My Title"
        assert fig.layout.xaxis.title.text == "Steps"
        assert fig.layout.yaxis.title.text == "Loss"

    def test_colorblind_palette_colors_used(self):
        traces = [
            {"name": "alibi", "x": [1], "y": [3.0]},
            {"name": "vanilla", "x": [1], "y": [2.8]},
        ]
        fig = create_line_chart(traces, "Title", "X", "Y")
        all_names = ["alibi", "vanilla"]
        # Colors assigned by alphabetical order
        assert fig.data[0].line.color == get_variant_color("alibi", all_names)
        assert fig.data[1].line.color == get_variant_color("vanilla", all_names)

    def test_explicit_color_overrides_palette(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0], "color": "#FF0000"}]
        fig = create_line_chart(traces, "Title", "X", "Y")
        assert fig.data[0].line.color == "#FF0000"

    def test_hover_template_present(self):
        traces = [{"name": "vanilla", "x": [1], "y": [3.0]}]
        fig = create_line_chart(traces, "Title", "Steps", "Loss")
        assert "vanilla" in fig.data[0].hovertemplate
        assert "Steps" in fig.data[0].hovertemplate
        assert "Loss" in fig.data[0].hovertemplate


class TestCreateBarChart:
    """Tests for create_bar_chart."""

    def test_returns_plotly_figure(self):
        fig = create_bar_chart(["a", "b"], [1.0, 2.0])
        assert isinstance(fig, go.Figure)

    def test_bar_values_correct(self):
        categories = ["vanilla", "modern"]
        values = [3.5, 3.2]
        fig = create_bar_chart(categories, values)
        assert list(fig.data[0].x) == categories
        assert list(fig.data[0].y) == values

    def test_error_bars_applied(self):
        categories = ["vanilla", "modern"]
        values = [3.5, 3.2]
        errors = [0.1, 0.05]
        fig = create_bar_chart(categories, values, errors=errors)
        assert fig.data[0].error_y.visible is True
        assert list(fig.data[0].error_y.array) == errors

    def test_no_error_bars_when_none(self):
        fig = create_bar_chart(["a", "b"], [1.0, 2.0], errors=None)
        assert fig.data[0].error_y.array is None

    def test_pareto_highlights_border(self):
        categories = ["vanilla", "modern", "gqa"]
        values = [3.5, 3.2, 3.1]
        fig = create_bar_chart(categories, values, highlights={"modern", "gqa"})
        widths = fig.data[0].marker.line.width
        colors = fig.data[0].marker.line.color
        # vanilla not highlighted
        assert widths[0] == 0
        # modern and gqa highlighted
        assert widths[1] == 3
        assert widths[2] == 3
        assert colors[1] == "white"
        assert colors[2] == "white"

    def test_no_highlights_no_border(self):
        categories = ["vanilla", "modern"]
        values = [3.5, 3.2]
        fig = create_bar_chart(categories, values, highlights=None)
        widths = fig.data[0].marker.line.width
        assert all(w == 0 for w in widths)

    def test_dark_theme_applied(self):
        fig = create_bar_chart(["a"], [1.0])
        assert fig.layout.plot_bgcolor == "rgba(17,17,17,1)"

    def test_colorblind_palette_colors(self):
        categories = ["alibi", "gqa", "modern"]
        values = [1.0, 2.0, 3.0]
        fig = create_bar_chart(categories, values)
        colors = fig.data[0].marker.color
        for i, cat in enumerate(categories):
            expected = get_variant_color(cat, categories)
            assert colors[i] == expected


class TestCreateStackedBarChart:
    """Tests for create_stacked_bar_chart."""

    def test_returns_plotly_figure(self):
        fig = create_stacked_bar_chart(
            ["a"], {"comp1": [1.0], "comp2": [2.0]}
        )
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_component(self):
        stacks = {"qkv_proj": [1.0, 1.1], "ffn": [2.0, 2.1], "attn": [0.5, 0.6]}
        fig = create_stacked_bar_chart(["vanilla", "modern"], stacks)
        assert len(fig.data) == 3

    def test_component_names_as_trace_names(self):
        stacks = {"qkv_proj": [1.0], "ffn": [2.0]}
        fig = create_stacked_bar_chart(["vanilla"], stacks)
        assert fig.data[0].name == "qkv_proj"
        assert fig.data[1].name == "ffn"

    def test_stacked_barmode(self):
        stacks = {"a": [1.0], "b": [2.0]}
        fig = create_stacked_bar_chart(["x"], stacks)
        assert fig.layout.barmode == "stack"

    def test_legend_shown(self):
        stacks = {"a": [1.0], "b": [2.0]}
        fig = create_stacked_bar_chart(["x"], stacks)
        assert fig.layout.showlegend is True

    def test_dark_theme_applied(self):
        stacks = {"a": [1.0]}
        fig = create_stacked_bar_chart(["x"], stacks)
        assert fig.layout.plot_bgcolor == "rgba(17,17,17,1)"


class TestCreateHeatmap:
    """Tests for create_heatmap."""

    def test_returns_plotly_figure(self):
        fig = create_heatmap([[1.0]], ["L0"])
        assert isinstance(fig, go.Figure)

    def test_z_range_0_to_1(self):
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        fig = create_heatmap(matrix, ["L0", "L1"])
        assert fig.data[0].zmin == 0
        assert fig.data[0].zmax == 1

    def test_labels_applied(self):
        matrix = [[1.0, 0.8], [0.8, 1.0]]
        labels = ["Layer 0", "Layer 1"]
        fig = create_heatmap(matrix, labels)
        assert list(fig.data[0].x) == labels
        assert list(fig.data[0].y) == labels

    def test_custom_colorscale(self):
        fig = create_heatmap([[1.0]], ["L0"], colorscale="Plasma")
        assert fig.data[0].colorscale is not None

    def test_default_viridis_colorscale(self):
        fig = create_heatmap([[1.0]], ["L0"])
        # Plotly converts "Viridis" to a tuple of color stops
        assert fig.data[0].colorscale is not None

    def test_dark_theme_applied(self):
        fig = create_heatmap([[1.0]], ["L0"])
        assert fig.layout.plot_bgcolor == "rgba(17,17,17,1)"

    def test_hover_template_present(self):
        fig = create_heatmap([[1.0, 0.5], [0.5, 1.0]], ["L0", "L1"])
        assert "Value" in fig.data[0].hovertemplate


class TestCreateRoofline:
    """Tests for create_roofline."""

    def test_returns_plotly_figure(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants)
        assert isinstance(fig, go.Figure)

    def test_log_scale_axes(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants)
        assert fig.layout.xaxis.type == "log"
        assert fig.layout.yaxis.type == "log"

    def test_ceiling_lines_present(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants)
        # At least 3 traces: mem bw, compute, ridge + variants
        assert len(fig.data) >= 4
        assert fig.data[0].name == "Memory BW Ceiling"
        assert fig.data[1].name == "Compute Ceiling"

    def test_ridge_point_calculation(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants, hw_bandwidth_gbps=300.0, hw_peak_tflops=242.0)
        expected_ridge = 242.0 / 300.0 * 1000  # ≈ 806.67
        assert abs(fig.data[2].x[0] - expected_ridge) < 0.01

    def test_variant_points_plotted(self):
        variants = {
            "modern": {"tflops": 150.0, "arithmetic_intensity": 600.0},
            "vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0},
        }
        fig = create_roofline(variants)
        # 3 ceiling traces + 2 variants = 5
        assert len(fig.data) == 5
        # Variants are sorted alphabetically: modern, vanilla
        assert fig.data[3].name == "modern"
        assert fig.data[4].name == "vanilla"

    def test_custom_hardware_params(self):
        variants = {"vanilla": {"tflops": 50.0, "arithmetic_intensity": 200.0}}
        fig = create_roofline(variants, hw_bandwidth_gbps=100.0, hw_peak_tflops=100.0)
        expected_ridge = 100.0 / 100.0 * 1000  # = 1000
        assert abs(fig.data[2].x[0] - expected_ridge) < 0.01

    def test_dark_theme_applied(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants)
        assert fig.layout.plot_bgcolor == "rgba(17,17,17,1)"

    def test_legend_shown(self):
        variants = {"vanilla": {"tflops": 100.0, "arithmetic_intensity": 500.0}}
        fig = create_roofline(variants)
        assert fig.layout.showlegend is True
