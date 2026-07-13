"""Publication-quality matplotlib visualizations for evaluation results.

Generates PNG plots for all metrics and comparisons with consistent styling,
colorblind-safe palettes, and one function per plot type.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.comparison import ComparisonResult, VariantData, compute_pareto_front
from src.evaluation.flops import FLOPBreakdown, compute_arithmetic_intensity, compute_step_flops
from src.evaluation.metrics import MetricsResult
from src.evaluation.probes import CKAResult, MQARResult, StableRankResult

logger = logging.getLogger(__name__)

# Colorblind-safe palette (Wong 2011 / IBM Design Library)
COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# Component colors for stacked bar charts (colorblind-safe subset)
COMPONENT_COLORS = {
    "qkv_proj": "#0072B2",
    "attention_score": "#E69F00",
    "attention_output": "#009E73",
    "ffn": "#CC79A7",
}


def _setup_style() -> None:
    """Apply consistent publication-quality style settings."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ensure_plots_dir(output_dir: Path) -> Path:
    """Ensure the plots/ subdirectory exists and return its path."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_mqar_results(results: dict[str, MQARResult], output_dir: Path) -> Path:
    """Plot MQAR accuracy by retrieval distance per variant.

    Creates a line chart with one line per variant showing how recall accuracy
    varies with the distance between key and query positions.

    Args:
        results: Mapping of variant name to MQARResult.
        output_dir: Directory to save the plot (saved in output_dir/plots/).

    Returns:
        Path to the generated PNG file.

    Validates: Requirements 12.4, 12.5
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (variant_name, result) in enumerate(sorted(results.items())):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        distances = sorted(result.accuracy_by_distance.keys())
        accuracies = [result.accuracy_by_distance[d] for d in distances]

        ax.plot(
            distances,
            accuracies,
            marker="o",
            color=color,
            label=f"{variant_name} (avg={result.accuracy:.3f})",
        )

    ax.set_xlabel("Retrieval Distance (query pos \u2212 key pos)")
    ax.set_ylabel("Accuracy")
    ax.set_title("MQAR Accuracy by Retrieval Distance")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = plots_dir / "mqar_by_distance.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved MQAR results plot to %s", output_path)
    return output_path


def plot_stable_rank(results: dict[str, StableRankResult], output_dir: Path) -> Path:
    """Plot stable rank per layer for all variants.

    Creates a line plot with one line per variant showing stable rank at each
    transformer layer. Includes mean \u00b1 std annotation in the legend.

    Args:
        results: Mapping of variant name to StableRankResult.
        output_dir: Directory to save the plot (saved in output_dir/plots/).

    Returns:
        Path to the generated PNG file.

    Validates: Requirements 12.4, 12.5
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (variant_name, result) in enumerate(sorted(results.items())):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        layers = np.arange(len(result.per_layer))

        ax.plot(
            layers,
            result.per_layer,
            marker="s",
            color=color,
            label=f"{variant_name} (\u03bc={result.mean:.1f} \u00b1 {result.std:.1f})",
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Stable Rank")
    ax.set_title("Stable Rank per Layer")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Ensure x-axis uses integer ticks for layers
    if results:
        first_result = next(iter(results.values()))
        n_layers = len(first_result.per_layer)
        if n_layers <= 20:
            ax.set_xticks(range(n_layers))

    fig.tight_layout()
    output_path = plots_dir / "stable_rank.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved stable rank plot to %s", output_path)
    return output_path


def plot_cka_adjacent(results: dict[str, CKAResult], output_dir: Path) -> Path:
    """Plot adjacent-layer CKA curves (one line per variant on shared axes).

    Shows how CKA similarity between consecutive layers varies across depth.
    Higher values indicate more redundant/similar representations between
    adjacent layers.

    Args:
        results: Mapping of variant name to CKAResult.
        output_dir: Directory to save the plot (saved in output_dir/plots/).

    Returns:
        Path to the generated PNG file.

    Validates: Requirements 12.4, 12.5
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (variant_name, result) in enumerate(sorted(results.items())):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        # adjacent_curve[i] = CKA(layer_i, layer_{i+1})
        layer_pairs = np.arange(len(result.adjacent_curve))

        ax.plot(
            layer_pairs,
            result.adjacent_curve,
            marker="^",
            color=color,
            label=variant_name,
        )

    ax.set_xlabel("Layer Pair Index (i \u2192 i+1)")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Adjacent-Layer CKA Similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Use integer x-ticks for layer pairs
    if results:
        first_result = next(iter(results.values()))
        n_pairs = len(first_result.adjacent_curve)
        if n_pairs <= 20:
            ax.set_xticks(range(n_pairs))

    fig.tight_layout()
    output_path = plots_dir / "cka_adjacent.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved CKA adjacent plot to %s", output_path)
    return output_path


def plot_cka_heatmap(
    result: CKAResult,
    variant_name: str,
    output_dir: Path,
) -> Path:
    """Plot full L\u00d7L CKA heatmap for a single variant.

    Creates a heatmap visualization of the full CKA similarity matrix between
    all layer pairs. Diagonal is 1.0 (self-similarity), off-diagonal values
    show cross-layer representation similarity.

    Args:
        result: CKAResult containing the full_matrix.
        variant_name: Name of the variant (used in title and filename).
        output_dir: Directory to save the plot (saved in output_dir/plots/).

    Returns:
        Path to the generated PNG file.

    Validates: Requirements 12.4, 12.5
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 6))

    n_layers = result.full_matrix.shape[0]

    # Use imshow for the heatmap with vmin=0, vmax=1
    im = ax.imshow(
        result.full_matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="equal",
        origin="lower",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA Similarity")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"CKA Similarity Matrix \u2014 {variant_name}")

    # Use integer ticks for layers
    if n_layers <= 20:
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))

    fig.tight_layout()
    # Sanitize variant name for filename
    safe_name = variant_name.replace(" ", "_").replace("/", "_").lower()
    output_path = plots_dir / f"cka_heatmap_{safe_name}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved CKA heatmap for %s to %s", variant_name, output_path)
    return output_path


def plot_learning_curves(
    variants: list[VariantData],
    output_dir: Path,
    x_axis: str = "tokens",
) -> Path:
    """Plot validation loss curves for all variants on shared axes.

    Supports multiple x-axis choices: tokens seen, wall-clock time, or
    cumulative FLOPs. Each variant is plotted as a separate line using
    the colorblind-safe palette.

    Args:
        variants: List of VariantData with populated log_entries.
            Each log entry should have tokens_seen, elapsed_time, val_loss,
            and step fields.
        output_dir: Base output directory. Plots are saved under plots/.
        x_axis: X-axis metric — one of "tokens", "wallclock", "flops".

    Returns:
        Path to the saved PNG file.

    Validates: Requirements 12.1, 12.2
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    x_axis_config = {
        "tokens": ("tokens_seen", "Tokens Seen"),
        "wallclock": ("elapsed_time", "Wall-Clock Time (s)"),
        "flops": ("cumulative_flops", "Cumulative FLOPs"),
    }

    if x_axis not in x_axis_config:
        raise ValueError(
            f"Unsupported x_axis: {x_axis!r}. Supported: 'tokens', 'wallclock', 'flops'"
        )

    x_key, x_label = x_axis_config[x_axis]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, variant in enumerate(variants):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]

        # Filter to entries with non-None val_loss
        entries = [
            e for e in variant.log_entries if e.get("val_loss") is not None
        ]
        if not entries:
            continue

        # Build x values based on the chosen axis
        if x_axis == "flops":
            # Compute cumulative FLOPs from step × per-step FLOPs
            if variant.flop_breakdown is not None:
                step_flops = variant.flop_breakdown.total
            else:
                step_flops = compute_step_flops(variant.config).total

            x_values = [e.get("step", 0) * step_flops for e in entries]
        else:
            x_values = [e.get(x_key, 0) for e in entries]

        y_values = [e["val_loss"] for e in entries]

        ax.plot(x_values, y_values, color=color, label=variant.name, linewidth=1.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"Learning Curves ({x_label})")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    filename = f"learning_curves_{x_axis}.png"
    save_path = plots_dir / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved learning curves plot to %s", save_path)
    return save_path


def plot_per_position_loss(
    variants: list[VariantData],
    output_dir: Path,
) -> Path:
    """Plot per-position loss with ICL power-law fit overlaid.

    For each variant that has computed per_position_loss and icl_fit_params
    in its MetricsResult, plots the raw loss curve and overlays the fitted
    power-law: L(t) = A * t^(-alpha) + C.

    Args:
        variants: List of VariantData with populated metrics field.
            metrics.per_position_loss should be a numpy array of shape (seq_len,).
            metrics.icl_fit_params should be a dict with keys "A", "alpha", "C".
        output_dir: Base output directory. Plots are saved under plots/.

    Returns:
        Path to the saved PNG file.

    Validates: Requirements 12.1, 12.2, 12.3
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, variant in enumerate(variants):
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]

        if variant.metrics is None:
            continue
        if variant.metrics.per_position_loss is None:
            continue

        per_pos = variant.metrics.per_position_loss
        seq_len = len(per_pos)
        positions = np.arange(1, seq_len + 1)

        # Plot raw per-position loss
        ax.plot(
            positions,
            per_pos,
            color=color,
            label=variant.name,
            linewidth=1.2,
            alpha=0.7,
        )

        # Overlay ICL power-law fit if available
        if variant.metrics.icl_fit_params is not None:
            params = variant.metrics.icl_fit_params
            A = params.get("A")
            alpha = params.get("alpha")
            C = params.get("C")

            if A is not None and alpha is not None and C is not None:
                # Skip overlay if fit failed (NaN values)
                if not (np.isnan(A) or np.isnan(alpha) or np.isnan(C)):
                    fit_curve = A * positions.astype(np.float64) ** (-alpha) + C
                    ax.plot(
                        positions,
                        fit_curve,
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"{variant.name} fit (\u03b1={alpha:.3f})",
                    )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Position Loss with ICL Power-Law Fit")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    save_path = plots_dir / "per_position_loss.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved per-position loss plot to %s", save_path)
    return save_path


def plot_flop_breakdown(
    breakdowns: dict[str, FLOPBreakdown],
    output_dir: Path,
) -> Path:
    """Stacked bar chart of FLOP components per variant.

    Creates a stacked horizontal bar chart showing QKV projections, attention
    score, attention output, and FFN components for each variant.

    Args:
        breakdowns: Dict mapping variant name to FLOPBreakdown.
        output_dir: Report output directory (PNG saved to output_dir/plots/).

    Returns:
        Path to the saved PNG file.

    Validates: Requirements 12.6
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    variant_names = list(breakdowns.keys())
    n_variants = len(variant_names)

    # Extract component values (in TFLOPs for readability)
    qkv = np.array([breakdowns[v].qkv_proj for v in variant_names], dtype=np.float64)
    attn_score = np.array(
        [breakdowns[v].attention_score for v in variant_names], dtype=np.float64
    )
    attn_output = np.array(
        [breakdowns[v].attention_output for v in variant_names], dtype=np.float64
    )
    ffn = np.array([breakdowns[v].ffn for v in variant_names], dtype=np.float64)

    # Convert to TFLOPs for axis readability
    scale = 1e12
    qkv_t = qkv / scale
    attn_score_t = attn_score / scale
    attn_output_t = attn_output / scale
    ffn_t = ffn / scale

    fig, ax = plt.subplots(figsize=(9, max(4, n_variants * 0.8)))

    y_pos = np.arange(n_variants)

    # Stacked horizontal bars
    bars_qkv = ax.barh(
        y_pos, qkv_t, color=COMPONENT_COLORS["qkv_proj"], label="QKV Projections"
    )
    bars_attn = ax.barh(
        y_pos,
        attn_score_t,
        left=qkv_t,
        color=COMPONENT_COLORS["attention_score"],
        label="Attention Score",
    )
    bars_out = ax.barh(
        y_pos,
        attn_output_t,
        left=qkv_t + attn_score_t,
        color=COMPONENT_COLORS["attention_output"],
        label="Attention Output",
    )
    bars_ffn = ax.barh(
        y_pos,
        ffn_t,
        left=qkv_t + attn_score_t + attn_output_t,
        color=COMPONENT_COLORS["ffn"],
        label="FFN",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variant_names)
    ax.set_xlabel("TFLOPs per training step")
    ax.set_title("FLOP Breakdown by Component")
    ax.legend(loc="lower right")

    plt.tight_layout()

    output_path = plots_dir / "flop_breakdown.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved FLOP breakdown plot to %s", output_path)
    return output_path


def plot_pareto(
    variants: list[VariantData],
    output_dir: Path,
    axes: list[tuple[str, str]] | None = None,
) -> list[Path]:
    """Plot Pareto frontier diagrams for multiple objective pairs.

    Generates one scatter plot per objective pair, with Pareto-optimal variants
    visually distinguished (star markers, different color) from dominated points.

    Args:
        variants: List of VariantData with populated log_entries and/or metrics.
        output_dir: Report output directory (PNGs saved to output_dir/plots/).
        axes: List of (x_metric, y_metric) pairs to plot. Defaults to
            [("flops", "val_loss"), ("wallclock", "val_loss"), ("peak_memory", "val_loss")].

    Returns:
        List of paths to saved PNG files.

    Validates: Requirements 12.6
    """
    if axes is None:
        axes = [
            ("flops", "val_loss"),
            ("wallclock", "val_loss"),
            ("peak_memory", "val_loss"),
        ]

    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    output_paths: list[Path] = []

    # Axis labels mapping
    axis_labels = {
        "flops": "FLOPs per step",
        "wallclock": "Wall-clock time (s)",
        "peak_memory": "Peak memory (MB)",
        "val_loss": "Validation loss",
    }

    for x_metric, y_metric in axes:
        # Compute Pareto front for this objective pair
        pareto_names = compute_pareto_front(variants, x_metric, y_metric)
        pareto_set = set(pareto_names)

        # Collect data points
        x_vals: list[float] = []
        y_vals: list[float] = []
        names: list[str] = []
        is_pareto: list[bool] = []

        for v in variants:
            x_val = _get_metric_value_for_pareto(v, x_metric)
            y_val = _get_metric_value_for_pareto(v, y_metric)
            if x_val is None or y_val is None:
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)
            names.append(v.name)
            is_pareto.append(v.name in pareto_set)

        if not x_vals:
            logger.warning(
                "No valid data points for Pareto plot (%s vs %s), skipping.",
                x_metric,
                y_metric,
            )
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot dominated points
        dom_x = [x for x, p in zip(x_vals, is_pareto) if not p]
        dom_y = [y for y, p in zip(y_vals, is_pareto) if not p]
        dom_names = [n for n, p in zip(names, is_pareto) if not p]
        if dom_x:
            ax.scatter(
                dom_x,
                dom_y,
                c="#999999",
                marker="o",
                s=80,
                alpha=0.7,
                label="Dominated",
                zorder=2,
            )

        # Plot Pareto-optimal points
        par_x = [x for x, p in zip(x_vals, is_pareto) if p]
        par_y = [y for y, p in zip(y_vals, is_pareto) if p]
        par_names = [n for n, p in zip(names, is_pareto) if p]
        if par_x:
            ax.scatter(
                par_x,
                par_y,
                c="#D55E00",
                marker="*",
                s=200,
                edgecolors="black",
                linewidths=0.5,
                label="Pareto-optimal",
                zorder=3,
            )

        # Annotate all points with variant names
        for x, y, name in zip(x_vals, y_vals, names):
            ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.8,
            )

        ax.set_xlabel(axis_labels.get(x_metric, x_metric))
        ax.set_ylabel(axis_labels.get(y_metric, y_metric))
        ax.set_title(f"Pareto Front: {x_metric} vs {y_metric}")
        ax.legend(loc="upper right")

        plt.tight_layout()

        filename = f"pareto_{x_metric}_{y_metric}.png"
        output_path = plots_dir / filename
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        output_paths.append(output_path)
        logger.info("Saved Pareto plot to %s", output_path)

    return output_paths


def plot_roofline(
    variants: list[VariantData],
    output_dir: Path,
    peak_tflops: float = 242.0,
    bandwidth_gbs: float = 300.0,
) -> Path:
    """Plot roofline diagram with L4 hardware boundary.

    Draws the compute bound and memory bound lines, marks the ridge point,
    and positions each variant by arithmetic intensity vs achieved TFLOPS.

    Args:
        variants: List of VariantData with populated log_entries and flop_breakdown.
        output_dir: Report output directory (PNG saved to output_dir/plots/).
        peak_tflops: Hardware peak TFLOPS (default 242.0 for L4 BF16).
        bandwidth_gbs: Memory bandwidth in GB/s (default 300.0 for L4).

    Returns:
        Path to the saved PNG file.

    Validates: Requirements 12.7
    """
    _setup_style()
    plots_dir = _ensure_plots_dir(output_dir)

    # Ridge point: peak_tflops × 1e12 / (bandwidth_gbs × 1e9) = FLOPs/byte
    ridge_point = (peak_tflops * 1e12) / (bandwidth_gbs * 1e9)  # ~807 FLOPs/byte

    fig, ax = plt.subplots(figsize=(9, 6))

    # Define x-axis range (arithmetic intensity in FLOPs/byte)
    x_min = 1.0
    x_max = max(ridge_point * 10, 10000.0)
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # Memory-bound line: achieved TFLOPS = bandwidth_gbs × arithmetic_intensity × 1e-3
    # bandwidth_gbs GB/s × AI FLOPs/byte = AI × bandwidth_gbs × 1e9 FLOPs/s
    # = AI × bandwidth_gbs × 1e9 / 1e12 TFLOPS = AI × bandwidth_gbs × 1e-3 TFLOPS
    memory_bound = x_range * bandwidth_gbs * 1e-3  # TFLOPS

    # Compute-bound line: achieved TFLOPS = peak_tflops (horizontal line)
    compute_bound = np.full_like(x_range, peak_tflops)

    # The roofline is the minimum of the two
    roofline = np.minimum(memory_bound, compute_bound)

    # Plot the roofline boundary
    ax.plot(x_range, roofline, "k-", linewidth=2.5, label="Hardware roofline", zorder=1)

    # Shade memory-bound and compute-bound regions
    ax.fill_between(
        x_range,
        roofline,
        0.1,
        alpha=0.05,
        color="gray",
    )

    # Mark the ridge point
    ax.axvline(
        ridge_point,
        color="#999999",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.annotate(
        f"Ridge: {ridge_point:.0f} FLOPs/byte",
        (ridge_point, peak_tflops * 0.6),
        fontsize=8,
        ha="left",
        alpha=0.8,
        xytext=(ridge_point * 1.1, peak_tflops * 0.6),
    )

    # Position each variant
    colors = COLORBLIND_PALETTE
    for i, v in enumerate(variants):
        # X-axis: arithmetic intensity from config
        ai = compute_arithmetic_intensity(v.config)

        # Y-axis: achieved TFLOPS = flop_breakdown.total / step_time
        achieved = _compute_achieved_tflops(v)
        if achieved is None:
            logger.warning(
                "Variant '%s' lacks timing data for roofline plot; skipping.",
                v.name,
            )
            continue

        color = colors[i % len(colors)]
        ax.scatter(
            ai,
            achieved,
            c=color,
            marker="o",
            s=120,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )
        ax.annotate(
            v.name,
            (ai, achieved),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(
        f"Roofline Diagram (L4: {peak_tflops} TFLOPS, {bandwidth_gbs} GB/s)"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.1, peak_tflops * 2)
    ax.legend(loc="lower right")

    plt.tight_layout()

    output_path = plots_dir / "roofline.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved roofline plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_metric_value_for_pareto(
    variant: VariantData, metric: str
) -> float | None:
    """Extract a metric value from a variant for Pareto plotting.

    Args:
        variant: VariantData instance.
        metric: One of "flops", "wallclock", "peak_memory", "val_loss".

    Returns:
        Metric value or None if unavailable.
    """
    if metric == "flops":
        if variant.flop_breakdown is not None:
            return float(variant.flop_breakdown.total)

        return float(compute_step_flops(variant.config).total)

    elif metric == "wallclock":
        if not variant.log_entries:
            return None
        times = [
            e.get("elapsed_time", 0.0)
            for e in variant.log_entries
            if e.get("elapsed_time") is not None
        ]
        return max(times) if times else None

    elif metric == "peak_memory":
        if not variant.log_entries:
            return None
        memories = [
            e.get("peak_memory_mb")
            for e in variant.log_entries
            if e.get("peak_memory_mb") is not None
        ]
        return max(memories) if memories else None

    elif metric == "val_loss":
        if variant.metrics is not None:
            return variant.metrics.val_loss
        if not variant.log_entries:
            return None
        for entry in reversed(variant.log_entries):
            val_loss = entry.get("val_loss")
            if val_loss is not None:
                return float(val_loss)
        return None

    else:
        return None


def _compute_achieved_tflops(variant: VariantData) -> float | None:
    """Compute achieved TFLOPS for a variant from FLOP breakdown and step time.

    Uses flop_breakdown.total / average_step_time to determine achieved throughput.

    Args:
        variant: VariantData with flop_breakdown and log_entries containing timing.

    Returns:
        Achieved TFLOPS, or None if data is insufficient.
    """
    # Get total FLOPs per step
    if variant.flop_breakdown is not None:
        total_flops = variant.flop_breakdown.total
    else:
        total_flops = compute_step_flops(variant.config).total

    # Estimate step time from log entries
    # Use elapsed_time and step counts to compute average step time
    if not variant.log_entries:
        return None

    # Find entries with both step and elapsed_time
    timed_entries = [
        e
        for e in variant.log_entries
        if e.get("elapsed_time") is not None and e.get("step") is not None
    ]

    if len(timed_entries) < 2:
        return None

    # Sort by step
    timed_entries.sort(key=lambda e: e["step"])

    # Compute average step time from total elapsed / total steps
    first = timed_entries[0]
    last = timed_entries[-1]
    total_time = last["elapsed_time"] - first["elapsed_time"]
    total_steps = last["step"] - first["step"]

    if total_steps <= 0 or total_time <= 0:
        return None

    avg_step_time = total_time / total_steps
    achieved_tflops = total_flops / (avg_step_time * 1e12)

    return achieved_tflops


# ---------------------------------------------------------------------------
# Summary markdown generation
# ---------------------------------------------------------------------------


def _format_metric(value: float | tuple[float, float]) -> str:
    """Format a metric value for display in markdown tables.

    If value is a tuple of (mean, std), formats as "mean ± std".
    If std is NaN, only the mean is displayed.
    Otherwise formats as a plain float with 4 decimal places.

    Args:
        value: Either a float, or a (mean, std) tuple.

    Returns:
        Formatted string representation.
    """
    if isinstance(value, tuple):
        mean, std = value
        if math.isnan(std):
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"
    return f"{value:.4f}"


def generate_summary_md(comparison: ComparisonResult, output_dir: Path) -> Path:
    """Generate summary.md with embedded figure references and formatted tables.

    Produces a human-readable markdown report including:
    - Title and overview section
    - Fixed-data comparison table (variant → val_loss at token budget)
    - Fixed-wallclock comparison table (variant → val_loss at 25/50/75/100%)
    - Fixed-FLOPs comparison table (variant → val_loss at FLOP budget)
    - Parameter parity validation results
    - Pareto front section listing Pareto-optimal variants
    - Figures section with relative image links to plots/*.png

    Statistical results are formatted as "mean ± std" when std is available.

    Args:
        comparison: ComparisonResult with populated comparison fields.
        output_dir: Directory where summary.md will be written. The directory
            is created if it doesn't exist.

    Returns:
        Path to the generated summary.md file.

    Validates: Requirements 12.8, 14.3, 15.1, 15.5
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Title and overview
    lines.append("# Evaluation Summary Report")
    lines.append("")
    lines.append("Automated comparison of Transformer variant architectures across")
    lines.append("controlled evaluation axes: fixed-data, fixed-wallclock, and fixed-FLOPs.")
    lines.append("")

    # --- Fixed-Data Comparison Table ---
    lines.append("## Fixed-Data Comparison")
    lines.append("")
    if comparison.fixed_data:
        lines.append("Validation loss at the same token budget.")
        lines.append("")
        lines.append("| Variant | Val Loss |")
        lines.append("|---------|----------|")
        for variant, val_loss in sorted(comparison.fixed_data.items()):
            lines.append(f"| {variant} | {_format_metric(val_loss)} |")
    else:
        lines.append("No fixed-data comparison results available.")
    lines.append("")

    # --- Fixed-Wallclock Comparison Table ---
    lines.append("## Fixed-Wallclock Comparison")
    lines.append("")
    if comparison.fixed_wallclock:
        lines.append("Validation loss at fractions of the dynamic wall-clock budget.")
        lines.append("")

        # Collect all time fractions across all variants
        all_fractions: set[float] = set()
        for frac_dict in comparison.fixed_wallclock.values():
            if isinstance(frac_dict, dict):
                all_fractions.update(frac_dict.keys())
        fractions_sorted = sorted(all_fractions)

        if fractions_sorted:
            # Build header
            frac_headers = [f"{int(f * 100)}%" for f in fractions_sorted]
            header = "| Variant | " + " | ".join(frac_headers) + " |"
            separator = "|---------|" + "|".join(
                ["-" * (len(h) + 2) for h in frac_headers]
            ) + "|"
            lines.append(header)
            lines.append(separator)

            for variant in sorted(comparison.fixed_wallclock.keys()):
                frac_dict = comparison.fixed_wallclock[variant]
                row_values: list[str] = []
                for f in fractions_sorted:
                    if isinstance(frac_dict, dict) and f in frac_dict:
                        row_values.append(_format_metric(frac_dict[f]))
                    else:
                        row_values.append("—")
                lines.append(f"| {variant} | " + " | ".join(row_values) + " |")
        else:
            lines.append("No wallclock fraction data available.")
    else:
        lines.append("No fixed-wallclock comparison results available.")
    lines.append("")

    # --- Fixed-FLOPs Comparison Table ---
    lines.append("## Fixed-FLOPs Comparison")
    lines.append("")
    if comparison.fixed_flops:
        lines.append("Validation loss at the same cumulative FLOP budget.")
        lines.append("")
        lines.append("| Variant | Val Loss |")
        lines.append("|---------|----------|")
        for variant, val_loss in sorted(comparison.fixed_flops.items()):
            lines.append(f"| {variant} | {_format_metric(val_loss)} |")
    else:
        lines.append("No fixed-FLOPs comparison results available.")
    lines.append("")

    # --- Parameter Parity ---
    lines.append("## Parameter Parity")
    lines.append("")
    if comparison.parameter_parity_valid:
        lines.append("✅ **PASS** — All variants are within ±5% of mean parameter count.")
    else:
        lines.append("❌ **FAIL** — Parameter counts exceed ±5% tolerance.")
    lines.append("")

    if comparison.parameter_counts:
        lines.append("| Variant | Parameters |")
        lines.append("|---------|------------|")
        for variant, count in sorted(comparison.parameter_counts.items()):
            lines.append(f"| {variant} | {count:,} |")
    lines.append("")

    # --- Pareto Front ---
    lines.append("## Pareto Front")
    lines.append("")
    if comparison.pareto_front:
        lines.append("Pareto-optimal variants (non-dominated on FLOPs vs val_loss):")
        lines.append("")
        for variant in sorted(comparison.pareto_front):
            lines.append(f"- **{variant}**")
    else:
        lines.append("No Pareto front data available.")
    lines.append("")

    # --- Figures ---
    lines.append("## Figures")
    lines.append("")
    lines.append("### Learning Curves")
    lines.append("")
    lines.append("![Learning Curves (Tokens)](plots/learning_curves_tokens.png)")
    lines.append("")
    lines.append("![Learning Curves (Wall-clock)](plots/learning_curves_wallclock.png)")
    lines.append("")
    lines.append("![Learning Curves (FLOPs)](plots/learning_curves_flops.png)")
    lines.append("")
    lines.append("### Per-Position Loss")
    lines.append("")
    lines.append("![Per-Position Loss](plots/per_position_loss.png)")
    lines.append("")
    lines.append("### Probes")
    lines.append("")
    lines.append("![MQAR by Distance](plots/mqar_by_distance.png)")
    lines.append("")
    lines.append("![Stable Rank](plots/stable_rank.png)")
    lines.append("")
    lines.append("![CKA Adjacent](plots/cka_adjacent.png)")
    lines.append("")
    lines.append("### Efficiency")
    lines.append("")
    lines.append("![FLOP Breakdown](plots/flop_breakdown.png)")
    lines.append("")
    lines.append("![Pareto Front](plots/pareto_flops_val_loss.png)")
    lines.append("")
    lines.append("![Roofline](plots/roofline.png)")
    lines.append("")

    # Write the summary file
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return summary_path
