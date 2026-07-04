"""Chart factory module for creating styled Plotly figures.

Provides factory functions for line charts, bar charts, stacked bar charts,
heatmaps, and roofline diagrams. All charts use the dark theme and
colorblind-safe palette from the styling module.
"""

import numpy as np
import plotly.graph_objects as go

from dashboard.components import styling


def create_line_chart(
    traces: list[dict],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    show_legend: bool = True,
) -> go.Figure:
    """Create a styled line chart with the dark theme and colorblind palette.

    Args:
        traces: List of trace dicts with keys:
            - name (str): Trace/variant name
            - x (list): X-axis values
            - y (list): Y-axis values
            - dash (str, optional): Line style (solid, dash, dot, dashdot, longdash)
            - visible (bool, optional): Whether trace is visible (default True)
            - color (str, optional): Override color
            - fill (str, optional): Fill mode (e.g., "toself", "tonexty")
            - fillcolor (str, optional): Fill color for shaded regions
            - showlegend (bool, optional): Whether to show in legend
        title: Chart title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        show_legend: Whether to show the legend.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()

    # Collect all trace names for consistent color assignment
    all_names = [t["name"] for t in traces if "name" in t]

    for i, trace in enumerate(traces):
        name = trace.get("name", f"Trace {i}")
        x = trace.get("x", [])
        y = trace.get("y", [])
        dash = trace.get("dash", "solid")
        visible = trace.get("visible", True)
        color = trace.get("color")
        fill = trace.get("fill")
        fillcolor = trace.get("fillcolor")
        trace_showlegend = trace.get("showlegend", True)

        # Use styling color if not explicitly provided
        if color is None:
            color = styling.get_variant_color(name, all_names)

        line_kwargs = {"color": color, "dash": dash}

        scatter_kwargs = {
            "x": x,
            "y": y,
            "name": name,
            "mode": "lines",
            "line": line_kwargs,
            "visible": visible if visible else "legendonly",
            "hovertemplate": (
                f"<b>{name}</b><br>"
                f"{xaxis_title}: %{{x}}<br>"
                f"{yaxis_title}: %{{y:.4f}}<extra></extra>"
            ),
            "showlegend": trace_showlegend,
        }

        if fill is not None:
            scatter_kwargs["fill"] = fill
        if fillcolor is not None:
            scatter_kwargs["fillcolor"] = fillcolor
            # Remove hover for fill traces (envelope backgrounds)
            if fill == "tonexty":
                scatter_kwargs["hoverinfo"] = "skip"
                scatter_kwargs["hovertemplate"] = None

        fig.add_trace(go.Scatter(**scatter_kwargs))

    # Apply base layout
    layout = styling.get_plotly_layout()
    layout.update(
        {
            "title": title,
            "xaxis_title": xaxis_title,
            "yaxis_title": yaxis_title,
            "showlegend": show_legend,
        }
    )
    fig.update_layout(**layout)

    return fig


def create_bar_chart(
    categories: list[str],
    values: list[float],
    errors: list[float | None] | None = None,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    highlights: set[str] | None = None,
) -> go.Figure:
    """Create a styled bar chart with optional error bars and Pareto highlighting.

    Args:
        categories: Bar category labels (variant names).
        values: Bar heights (one per category).
        errors: Error bar values (± error) per category, or None.
        title: Chart title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        highlights: Set of category names that are Pareto-front variants
            (rendered with a distinct border/outline).

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()

    if highlights is None:
        highlights = set()

    # Assign colors from the colorblind palette
    colors = [
        styling.get_variant_color(cat, categories) for cat in categories
    ]

    # Build border styling for Pareto-front variants
    marker_line_widths = []
    marker_line_colors = []
    for cat in categories:
        if cat in highlights:
            marker_line_widths.append(3)
            marker_line_colors.append("white")
        else:
            marker_line_widths.append(0)
            marker_line_colors.append("rgba(0,0,0,0)")

    # Error bars
    error_y = None
    if errors is not None:
        # Replace None with 0 for Plotly
        error_values = [e if e is not None else 0 for e in errors]
        error_y = {
            "type": "data",
            "array": error_values,
            "visible": True,
            "color": "white",
            "thickness": 1.5,
        }

    # Build hover template
    hover_texts = []
    for i, cat in enumerate(categories):
        val = values[i]
        err_str = f"± {errors[i]:.4f}" if errors and errors[i] is not None else "N/A"
        hover_texts.append(
            f"<b>{cat}</b><br>"
            f"{yaxis_title}: {val:.4f}<br>"
            f"Error: {err_str}<extra></extra>"
        )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker={
                "color": colors,
                "line": {
                    "width": marker_line_widths,
                    "color": marker_line_colors,
                },
            },
            error_y=error_y,
            hovertemplate=hover_texts,
            showlegend=False,
        )
    )

    # Apply base layout
    layout = styling.get_plotly_layout()
    layout.update(
        {
            "title": title,
            "xaxis_title": xaxis_title,
            "yaxis_title": yaxis_title,
            "showlegend": True,
        }
    )
    fig.update_layout(**layout)

    return fig


def create_stacked_bar_chart(
    categories: list[str],
    stacks: dict[str, list[float]],
    title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """Create a stacked bar chart for FLOP breakdown.

    Args:
        categories: Category labels (variant names) for the x-axis.
        stacks: Mapping of component name → list of values (one per category).
            E.g., {"qkv_proj": [1.2, 1.3, ...], "ffn": [2.5, 2.6, ...]}.
        title: Chart title.
        yaxis_title: Y-axis label.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()

    component_names = list(stacks.keys())
    for i, component in enumerate(component_names):
        color = styling.PALETTE[i % len(styling.PALETTE)]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=stacks[component],
                name=component,
                marker={"color": color},
                hovertemplate=(
                    f"<b>{component}</b><br>"
                    f"Variant: %{{x}}<br>"
                    f"{yaxis_title}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    # Apply base layout with stacked bar mode
    layout = styling.get_plotly_layout()
    layout.update(
        {
            "title": title,
            "xaxis_title": "",
            "yaxis_title": yaxis_title,
            "showlegend": True,
            "barmode": "stack",
        }
    )
    fig.update_layout(**layout)

    return fig


def create_heatmap(
    matrix: list[list[float]],
    labels: list[str],
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create a 2D heatmap (used for CKA L×L matrix).

    Args:
        matrix: 2D list of float values (rows × cols).
        labels: Axis labels (layer indices or names).
        title: Chart title.
        colorscale: Plotly colorscale name (default "Viridis").

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            hovertemplate=(
                "Row: %{y}<br>"
                "Col: %{x}<br>"
                "Value: %{z:.4f}<extra></extra>"
            ),
        )
    )

    # Apply base layout
    layout = styling.get_plotly_layout()
    layout.update(
        {
            "title": title,
            "showlegend": True,
            "xaxis_title": "Layer",
            "yaxis_title": "Layer",
        }
    )
    fig.update_layout(**layout)

    return fig


def create_roofline(
    variants: dict[str, dict],
    hw_bandwidth_gbps: float = 300.0,
    hw_peak_tflops: float = 242.0,
) -> go.Figure:
    """Create a roofline diagram with hardware ceiling lines.

    Args:
        variants: Mapping of variant name → dict with keys:
            - tflops (float): Achieved TFLOPS
            - arithmetic_intensity (float): FLOPs/byte
        hw_bandwidth_gbps: Hardware memory bandwidth in GB/s (default 300.0 for L4).
        hw_peak_tflops: Hardware peak compute in TFLOPS (default 242.0 for L4).

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()

    # Ridge point: where memory-bound meets compute-bound
    # hw_peak_tflops / hw_bandwidth_gbps * 1000 converts to FLOPs/byte
    # (TFLOPS / (GB/s)) = (10^12 FLOP/s) / (10^9 byte/s) = 10^3 FLOP/byte
    ridge_point = hw_peak_tflops / hw_bandwidth_gbps * 1000  # FLOPs/byte

    # X-axis range for ceiling lines (log scale)
    x_min = 0.1
    x_max = max(ridge_point * 10, 10000)

    # Memory-bandwidth ceiling: achieved TFLOPS = bandwidth * intensity / 1000
    # (GB/s * FLOPs/byte) = 10^9 byte/s * FLOPs/byte = 10^9 FLOPs/s → /10^12 = TFLOPS/1000
    mem_x = np.logspace(np.log10(x_min), np.log10(ridge_point), 50)
    mem_y = hw_bandwidth_gbps * mem_x / 1000  # Convert to TFLOPS

    # Compute ceiling: horizontal line at peak TFLOPS
    compute_x = np.logspace(np.log10(ridge_point), np.log10(x_max), 50)
    compute_y = np.full_like(compute_x, hw_peak_tflops)

    # Draw memory-bandwidth line
    fig.add_trace(
        go.Scatter(
            x=mem_x.tolist(),
            y=mem_y.tolist(),
            name="Memory BW Ceiling",
            mode="lines",
            line={"color": "rgba(255,255,255,0.6)", "dash": "dash", "width": 2},
            hovertemplate=(
                "Memory BW Ceiling<br>"
                "Intensity: %{x:.1f} FLOPs/byte<br>"
                "Achievable: %{y:.1f} TFLOPS<extra></extra>"
            ),
        )
    )

    # Draw compute ceiling line
    fig.add_trace(
        go.Scatter(
            x=compute_x.tolist(),
            y=compute_y.tolist(),
            name="Compute Ceiling",
            mode="lines",
            line={"color": "rgba(255,255,255,0.6)", "dash": "dot", "width": 2},
            hovertemplate=(
                "Compute Ceiling<br>"
                "Intensity: %{x:.1f} FLOPs/byte<br>"
                "Peak: %{y:.1f} TFLOPS<extra></extra>"
            ),
        )
    )

    # Mark ridge point
    fig.add_trace(
        go.Scatter(
            x=[ridge_point],
            y=[hw_peak_tflops],
            name=f"Ridge ({ridge_point:.0f} FLOPs/byte)",
            mode="markers",
            marker={"color": "white", "size": 10, "symbol": "diamond"},
            hovertemplate=(
                f"<b>Ridge Point</b><br>"
                f"Intensity: {ridge_point:.1f} FLOPs/byte<br>"
                f"Peak: {hw_peak_tflops:.1f} TFLOPS<extra></extra>"
            ),
        )
    )

    # Plot each variant as a scatter point
    variant_names = sorted(variants.keys())
    for variant_name in variant_names:
        data = variants[variant_name]
        tflops = data["tflops"]
        intensity = data["arithmetic_intensity"]
        color = styling.get_variant_color(variant_name, variant_names)

        fig.add_trace(
            go.Scatter(
                x=[intensity],
                y=[tflops],
                name=variant_name,
                mode="markers+text",
                marker={"color": color, "size": 12, "symbol": "circle"},
                text=[variant_name],
                textposition="top center",
                textfont={"color": color, "size": 10},
                hovertemplate=(
                    f"<b>{variant_name}</b><br>"
                    f"Arithmetic Intensity: {intensity:.1f} FLOPs/byte<br>"
                    f"Achieved: {tflops:.2f} TFLOPS<extra></extra>"
                ),
            )
        )

    # Apply base layout with log scale axes
    layout = styling.get_plotly_layout()
    layout.update(
        {
            "title": "Roofline Diagram",
            "xaxis_title": "Arithmetic Intensity (FLOPs/byte)",
            "yaxis_title": "Achieved TFLOPS",
            "showlegend": True,
            "xaxis_type": "log",
            "yaxis_type": "log",
        }
    )
    fig.update_layout(**layout)

    return fig
