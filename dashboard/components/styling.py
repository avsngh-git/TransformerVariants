"""Centralizes visual constants and Plotly layout defaults for the dashboard.

Provides a colorblind-safe palette (Wong 2011 / IBM Design Library), line styles,
marker shapes, and helper functions for consistent variant styling across all pages.
"""

# Colorblind-safe palette (Wong 2011 / IBM Design Library)
# Note: #000000 (black) is NOT used for traces on dark backgrounds.
# Effectively 7 usable colors for variant traces.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # light blue
    "#D55E00",  # red-orange
    "#F0E442",  # yellow
    "#000000",  # black (unused for traces on dark bg)
]

LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]
MARKER_SHAPES = ["circle", "square", "diamond", "cross", "x"]

# Number of usable colors (excluding black for dark backgrounds)
_USABLE_COLORS = len(PALETTE) - 1  # 7


def get_variant_color(variant_name: str, all_variants: list[str]) -> str:
    """Return consistent color for a variant based on alphabetical index.

    Sorts all_variants alphabetically, finds variant_name's position,
    and indexes into PALETTE (wrapping if necessary).

    Args:
        variant_name: The variant to get the color for.
        all_variants: The full list of variant names (used for stable ordering).

    Returns:
        Hex color string from PALETTE.
    """
    sorted_variants = sorted(all_variants)
    index = sorted_variants.index(variant_name)
    return PALETTE[index % len(PALETTE)]


def get_variant_style(variant_name: str, all_variants: list[str]) -> dict:
    """Return color + line style + marker for a variant.

    For the first 7 variants (palette size - 1, skipping black), uses the
    assigned color with solid line and circle marker. For overflow variants
    (index >= 7), wraps the palette color and assigns a unique line style
    and marker shape to ensure no two variants share both color AND line style.

    Args:
        variant_name: The variant to get the style for.
        all_variants: The full list of variant names (used for stable ordering).

    Returns:
        Dict with keys "color", "line_style", "marker".
    """
    sorted_variants = sorted(all_variants)
    index = sorted_variants.index(variant_name)

    if index < _USABLE_COLORS:
        # First 7 variants: use color directly with default line/marker
        return {
            "color": PALETTE[index],
            "line_style": "solid",
            "marker": "circle",
        }
    else:
        # Overflow variants: wrap color and vary line style + marker
        color_index = index % _USABLE_COLORS
        # Use different line styles and markers for overflow variants.
        # Skip "solid" (index 0) since the first 7 variants all use solid,
        # so overflow must use non-solid styles to be distinguishable.
        overflow_index = index - _USABLE_COLORS
        line_style = LINE_STYLES[(overflow_index % (len(LINE_STYLES) - 1)) + 1]
        marker = MARKER_SHAPES[(overflow_index % (len(MARKER_SHAPES) - 1)) + 1]
        return {
            "color": PALETTE[color_index],
            "line_style": line_style,
            "marker": marker,
        }


def get_plotly_layout() -> dict:
    """Return base Plotly layout dict for dark theme.

    Returns:
        Dict suitable for passing to plotly.graph_objects.Layout or
        fig.update_layout(**get_plotly_layout()).
    """
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(17,17,17,1)",
        "font": {"color": "white"},
        "xaxis": {
            "gridcolor": "rgba(128,128,128,0.2)",
            "zerolinecolor": "rgba(128,128,128,0.3)",
        },
        "yaxis": {
            "gridcolor": "rgba(128,128,128,0.2)",
            "zerolinecolor": "rgba(128,128,128,0.3)",
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "white"},
        },
    }


def format_metric(value: float, decimal_places: int = 4) -> str:
    """Format a metric value to the specified decimal places.

    Args:
        value: The metric value to format.
        decimal_places: Number of decimal places (default 4).

    Returns:
        Formatted string representation.
    """
    return f"{value:.{decimal_places}f}"


def format_metric_with_std(
    mean: float, std: float | None, decimal_places: int = 4
) -> str:
    """Format as 'mean ± std' or just 'mean' if std is None.

    Args:
        mean: The mean value.
        std: The standard deviation, or None for single-seed results.
        decimal_places: Number of decimal places (default 4).

    Returns:
        Formatted string like "1.2345 ± 0.0012" or "1.2345".
    """
    formatted_mean = format_metric(mean, decimal_places)
    if std is not None:
        formatted_std = format_metric(std, decimal_places)
        return f"{formatted_mean} ± {formatted_std}"
    return formatted_mean
