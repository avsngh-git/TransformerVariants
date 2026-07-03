"""Evaluation CLI entry point.

Orchestrates the full evaluation pipeline: loads checkpoints, computes metrics,
runs probes, performs comparisons, generates visualizations, and writes reports.

Usage:
    python scripts/evaluate.py \
        --checkpoints checkpoints/vanilla_main_s42/ checkpoints/modern_main_s42/ \
        --output reports/main_comparison/ \
        --device cuda \
        --data_dir data/processed/wikitext-full

The script detects multiple seeds for the same variant (by naming convention or
config) and aggregates results as mean ± std. Output is a self-contained report
directory with plots/, raw/, summary.md, and metadata.json.
"""

import argparse
import csv
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.evaluation.comparison import (
    ComparisonResult,
    VariantData,
    aggregate_across_seeds,
    compute_pareto_front,
    load_variant_data,
    slice_fixed_data,
    slice_fixed_flops,
    slice_fixed_wallclock,
    validate_parameter_parity,
)
from src.evaluation.flops import compute_mfu, compute_step_flops
from src.evaluation.metrics import (
    compute_per_position_loss,
    compute_perplexity,
    compute_val_loss,
    fit_icl_decay,
    load_metrics_log,
)
from src.evaluation.probes import (
    compute_attention_entropy,
    compute_cka,
    compute_stable_rank,
    run_mqar_probe,
)
from src.evaluation.visualizations import (
    generate_summary_md,
    plot_cka_adjacent,
    plot_cka_heatmap,
    plot_flop_breakdown,
    plot_learning_curves,
    plot_mqar_results,
    plot_pareto,
    plot_per_position_loss,
    plot_roofline,
    plot_stable_rank,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full evaluation suite on Transformer variant checkpoints."
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        required=True,
        help="Paths to checkpoint directories to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the report (plots/, raw/, summary.md, metadata.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for GPU computations (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to validation data directory for probe computations. "
        "If not provided, probe computations requiring a val_loader are skipped.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_device(device_arg: str | None) -> str:
    """Resolve device: use provided arg, or auto-detect cuda availability."""
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _detect_seed_groups(
    variants: list[VariantData],
) -> dict[str, list[VariantData]]:
    """Group variants by name to detect multiple seeds for the same architecture.

    If multiple VariantData entries share the same `name`, they are treated as
    different seeds of the same variant and will be aggregated statistically.

    Args:
        variants: List of loaded VariantData objects.

    Returns:
        Dict mapping variant name to a list of VariantData (one per seed).
    """
    groups: dict[str, list[VariantData]] = {}
    for v in variants:
        groups.setdefault(v.name, []).append(v)
    return groups


def _load_model_from_checkpoint(variant_data: VariantData, device: str):
    """Load a model from checkpoint weights using the registry.

    Attempts to:
    1. Build a model from the variant config via registry
    2. Load state_dict from checkpoint file (checkpoint_latest.pt or model.pt)

    Returns the model on the specified device, or None if loading fails.
    """
    from src.models.registry import build as registry_build, SCALES

    config = variant_data.config
    if config is None:
        logger.warning(
            "No config available for %s, cannot load model.", variant_data.name
        )
        return None

    # Build model from config via registry
    try:
        variant_name = config.variant

        # Determine scale from config dimensions
        scale = "debug"
        for scale_name, dims in SCALES.items():
            if (
                dims["n_layer"] == config.n_layer
                and dims["d_model"] == config.d_model
                and dims["n_head"] == config.n_head
                and dims["seq_len"] == config.seq_len
            ):
                scale = scale_name
                break

        model, _ = registry_build(
            variant_name,
            scale,
            activation=config.activation,
            dtype="float32",  # Load in fp32 for evaluation precision
        )
    except (ValueError, KeyError) as e:
        logger.warning("Failed to build model for %s: %s", variant_data.name, e)
        return None

    # Look for checkpoint weights
    checkpoint_dir = Path(variant_data.checkpoint_dir)
    weight_paths = [
        checkpoint_dir / "checkpoint_latest.pt",
        checkpoint_dir / "model.pt",
        checkpoint_dir / "checkpoint_best.pt",
    ]

    state_dict_loaded = False
    for weight_path in weight_paths:
        if weight_path.exists():
            try:
                checkpoint = torch.load(
                    weight_path, map_location="cpu", weights_only=True
                )
                # Handle both bare state_dicts and wrapped checkpoints
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict, strict=False)
                state_dict_loaded = True
                logger.info(
                    "Loaded weights for %s from %s",
                    variant_data.name,
                    weight_path.name,
                )
                break
            except Exception as e:
                logger.warning(
                    "Failed to load weights from %s: %s", weight_path, e
                )
                continue

    if not state_dict_loaded:
        logger.warning(
            "No checkpoint weights found for %s in %s. "
            "Using randomly initialized model.",
            variant_data.name,
            checkpoint_dir,
        )

    model = model.to(device)
    model.eval()
    return model


def _create_val_loader(data_dir: str, seq_len: int, device: str):
    """Create a validation data loader from the data directory.

    Returns None if the loader cannot be created (e.g., missing data files).
    """
    from src.data.dataloader import ShardedDataLoader

    try:
        val_loader = ShardedDataLoader(
            data_dir=data_dir,
            batch_size=8,
            seq_len=seq_len,
            split="val",
            device=device,
        )
        return val_loader
    except FileNotFoundError as e:
        logger.warning("Could not create validation loader: %s", e)
        return None


# ---------------------------------------------------------------------------
# Metadata and raw data output
# ---------------------------------------------------------------------------


def _write_metadata(
    output_dir: Path,
    variants: list[VariantData],
    device: str,
) -> Path:
    """Write metadata.json with run information.

    Includes timestamp, software versions, hardware identifier, and the list
    of evaluated checkpoint paths.

    Args:
        output_dir: Report output directory.
        variants: All loaded variant data (including all seeds).
        device: Device string used for evaluation (e.g., "cuda", "cpu").

    Returns:
        Path to the written metadata.json file.

    Validates: Requirements 15.3, 15.4
    """
    # Determine hardware identifier
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        hardware = f"{device} ({gpu_name})"
    else:
        hardware = "cpu"

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "hardware": hardware,
        "evaluated_checkpoints": [
            str(v.checkpoint_dir) for v in variants
        ],
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote metadata.json to %s", metadata_path)
    return metadata_path


def _write_raw_data(
    output_dir: Path,
    variants: list[VariantData],
    comparison_result: ComparisonResult,
    seed_aggregated: dict[str, dict[str, tuple[float, float]]],
) -> tuple[Path, Path]:
    """Write raw/metrics.csv and raw/metrics.json with all computed data.

    Args:
        output_dir: Report output directory.
        variants: All loaded variant data.
        comparison_result: The computed ComparisonResult.
        seed_aggregated: Dict mapping variant name to aggregated metrics
            (metric_name → (mean, std)).

    Returns:
        Tuple of (csv_path, json_path).

    Validates: Requirements 15.3, 15.4
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "metrics.csv"
    _write_metrics_csv(csv_path, variants)

    json_path = raw_dir / "metrics.json"
    _write_metrics_json(json_path, variants, comparison_result, seed_aggregated)

    logger.info("Wrote raw data to %s", raw_dir)
    return csv_path, json_path


def _write_metrics_csv(csv_path: Path, variants: list[VariantData]) -> None:
    """Write a flat CSV with per-variant, per-seed metric values."""
    seed_groups = _detect_seed_groups(variants)

    rows: list[dict[str, str | float]] = []
    for variant_name, seeds in seed_groups.items():
        for seed_idx, v in enumerate(seeds):
            row: dict[str, str | float] = {
                "variant": variant_name,
                "seed_index": seed_idx,
                "checkpoint_dir": str(v.checkpoint_dir),
            }

            # Extract key metrics from log entries
            if v.log_entries:
                val_loss = None
                for entry in reversed(v.log_entries):
                    if entry.get("val_loss") is not None:
                        val_loss = entry["val_loss"]
                        break
                if val_loss is not None:
                    row["val_loss"] = val_loss
                    row["perplexity"] = float(np.exp(val_loss))

                tokens_values = [
                    e["tokens_seen"] for e in v.log_entries
                    if e.get("tokens_seen") is not None
                ]
                if tokens_values:
                    row["total_tokens_seen"] = max(tokens_values)

                time_values = [
                    e["elapsed_time"] for e in v.log_entries
                    if e.get("elapsed_time") is not None
                ]
                if time_values:
                    row["total_elapsed_time"] = max(time_values)

            # FLOPs
            if v.flop_breakdown is not None:
                row["step_flops"] = v.flop_breakdown.total

            # MetricsResult (if computed via probes)
            if v.metrics is not None:
                row["val_loss"] = v.metrics.val_loss
                row["perplexity"] = v.metrics.perplexity
                if v.metrics.icl_exponent is not None:
                    row["icl_alpha"] = v.metrics.icl_exponent

            rows.append(row)

    if not rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "seed_index", "checkpoint_dir"])
        return

    fixed_cols = ["variant", "seed_index", "checkpoint_dir"]
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    extra_cols = sorted(all_keys - set(fixed_cols))
    fieldnames = fixed_cols + extra_cols

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_json(
    json_path: Path,
    variants: list[VariantData],
    comparison_result: ComparisonResult,
    seed_aggregated: dict[str, dict[str, tuple[float, float]]],
) -> None:
    """Write full structured metrics data as JSON."""
    seed_groups = _detect_seed_groups(variants)

    # Build per-variant data
    variants_data: dict[str, list[dict]] = {}
    for variant_name, seeds in seed_groups.items():
        seed_list = []
        for seed_idx, v in enumerate(seeds):
            seed_entry: dict = {
                "seed_index": seed_idx,
                "checkpoint_dir": str(v.checkpoint_dir),
            }

            if v.log_entries:
                val_loss = None
                for entry in reversed(v.log_entries):
                    if entry.get("val_loss") is not None:
                        val_loss = entry["val_loss"]
                        break
                if val_loss is not None:
                    seed_entry["val_loss"] = val_loss
                    seed_entry["perplexity"] = float(np.exp(val_loss))

                tokens_values = [
                    e["tokens_seen"] for e in v.log_entries
                    if e.get("tokens_seen") is not None
                ]
                if tokens_values:
                    seed_entry["total_tokens_seen"] = max(tokens_values)

                time_values = [
                    e["elapsed_time"] for e in v.log_entries
                    if e.get("elapsed_time") is not None
                ]
                if time_values:
                    seed_entry["total_elapsed_time"] = max(time_values)

            if v.flop_breakdown is not None:
                seed_entry["flop_breakdown"] = {
                    "qkv_proj": v.flop_breakdown.qkv_proj,
                    "attention_score": v.flop_breakdown.attention_score,
                    "attention_output": v.flop_breakdown.attention_output,
                    "ffn": v.flop_breakdown.ffn,
                    "total": v.flop_breakdown.total,
                }

            if v.metrics is not None:
                seed_entry["metrics"] = {
                    "val_loss": v.metrics.val_loss,
                    "perplexity": v.metrics.perplexity,
                    "icl_exponent": v.metrics.icl_exponent,
                    "icl_fit_params": v.metrics.icl_fit_params,
                }

            seed_list.append(seed_entry)
        variants_data[variant_name] = seed_list

    # Build aggregated section
    aggregated_section: dict[str, dict] = {}
    for variant_name, metrics in seed_aggregated.items():
        aggregated_section[variant_name] = {
            metric_name: {
                "mean": mean,
                "std": std if not (isinstance(std, float) and np.isnan(std)) else None,
            }
            for metric_name, (mean, std) in metrics.items()
        }

    # Build comparison section
    comparison_section = {
        "fixed_data": comparison_result.fixed_data,
        "fixed_wallclock": comparison_result.fixed_wallclock,
        "fixed_flops": comparison_result.fixed_flops,
        "pareto_front": comparison_result.pareto_front,
        "parameter_counts": comparison_result.parameter_counts,
        "parameter_parity_valid": comparison_result.parameter_parity_valid,
    }

    output = {
        "variants": variants_data,
        "aggregated": aggregated_section,
        "comparison": comparison_section,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Seed aggregation helper
# ---------------------------------------------------------------------------


def _aggregate_seed_metrics(
    seed_groups: dict[str, list[VariantData]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Aggregate metrics across seeds for each variant group.

    For each variant with multiple seeds, collects key metrics from each seed
    and calls aggregate_across_seeds to compute mean ± std.

    Args:
        seed_groups: Dict mapping variant name to list of VariantData (seeds).

    Returns:
        Dict mapping variant name to aggregated metrics dict
        (metric_name → (mean, std)).

    Validates: Requirements 13.6, 14.1, 14.2, 14.3, 14.4
    """
    aggregated: dict[str, dict[str, tuple[float, float]]] = {}

    for variant_name, seeds in seed_groups.items():
        seed_results: list[dict] = []

        for v in seeds:
            result: dict[str, float] = {}

            if v.log_entries:
                val_loss = None
                for entry in reversed(v.log_entries):
                    if entry.get("val_loss") is not None:
                        val_loss = entry["val_loss"]
                        break
                if val_loss is not None:
                    result["val_loss"] = val_loss
                    result["perplexity"] = float(np.exp(val_loss))

                tokens_values = [
                    e["tokens_seen"] for e in v.log_entries
                    if e.get("tokens_seen") is not None
                ]
                if tokens_values:
                    result["total_tokens_seen"] = float(max(tokens_values))

                time_values = [
                    e["elapsed_time"] for e in v.log_entries
                    if e.get("elapsed_time") is not None
                ]
                if time_values:
                    result["total_elapsed_time"] = max(time_values)

            if v.flop_breakdown is not None:
                result["step_flops"] = float(v.flop_breakdown.total)

                # Compute MFU if timing info available
                if v.log_entries:
                    steps_with_time = [
                        e for e in v.log_entries
                        if e.get("elapsed_time") is not None
                        and e.get("step") is not None
                    ]
                    if len(steps_with_time) >= 2:
                        sorted_entries = sorted(
                            steps_with_time, key=lambda e: e["step"]
                        )
                        total_time = (
                            sorted_entries[-1]["elapsed_time"]
                            - sorted_entries[0]["elapsed_time"]
                        )
                        total_steps = (
                            sorted_entries[-1]["step"]
                            - sorted_entries[0]["step"]
                        )
                        if total_steps > 0 and total_time > 0:
                            avg_step_time = total_time / total_steps
                            mfu_result = compute_mfu(
                                v.flop_breakdown.total, avg_step_time
                            )
                            result["mfu"] = mfu_result.mfu

            if v.metrics is not None:
                result["val_loss"] = v.metrics.val_loss
                result["perplexity"] = v.metrics.perplexity
                if v.metrics.icl_exponent is not None:
                    result["icl_alpha"] = v.metrics.icl_exponent

            if result:
                seed_results.append(result)

        if seed_results:
            aggregated[variant_name] = aggregate_across_seeds(seed_results)

    return aggregated


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full evaluation pipeline.

    Steps:
    1. Parse CLI arguments
    2. Load variant data from checkpoint directories (gracefully skip bad ones)
    3. Compute FLOPs from configs
    4. Load models and run probes (if --data_dir provided)
    5. Run comparison slicing (fixed-data, fixed-wallclock, fixed-FLOPs)
    6. Compute Pareto front and validate parameter parity
    7. Aggregate metrics across seeds (mean ± std)
    8. Generate all visualizations
    9. Generate summary.md
    10. Write metadata.json and raw/ data files

    Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.7,
               15.1, 15.2, 15.3, 15.4, 15.6
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    device = _resolve_device(args.device)
    output_dir = Path(args.output)

    logger.info("Evaluation pipeline starting")
    logger.info("  Checkpoints: %s", args.checkpoints)
    logger.info("  Output: %s", output_dir)
    logger.info("  Device: %s", device)
    logger.info(
        "  Data dir: %s",
        args.data_dir if args.data_dir else "(none — probes will be skipped)",
    )

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Load variant data from checkpoints ───────────────────────
    logger.info("Step 1: Loading variant data from checkpoint directories...")
    checkpoint_dirs = [Path(d) for d in args.checkpoints]
    variants = load_variant_data(checkpoint_dirs)

    if not variants:
        logger.error("No valid variants loaded from provided checkpoints. Exiting.")
        sys.exit(1)

    logger.info(
        "  Loaded %d variant(s): %s",
        len(variants),
        [v.name for v in variants],
    )

    # ─── Step 2: Compute FLOPs from configs ───────────────────────────────
    logger.info("Step 2: Computing FLOP breakdowns...")
    for v in variants:
        if v.config is not None and v.flop_breakdown is None:
            try:
                v.flop_breakdown = compute_step_flops(v.config)
                logger.info(
                    "  %s: %s FLOPs/step", v.name, f"{v.flop_breakdown.total:,.0f}"
                )
            except Exception as e:
                logger.warning("  Failed to compute FLOPs for %s: %s", v.name, e)

    # ─── Step 3: Run model-based metrics and probes ───────────────────────
    mqar_results: dict[str, object] = {}
    stable_rank_results: dict[str, object] = {}
    cka_results: dict[str, object] = {}
    attention_entropy_results: dict[str, object] = {}

    if args.data_dir:
        logger.info("Step 3: Running model-based metrics and probes...")

        for v in variants:
            if v.config is None:
                logger.warning(
                    "  Skipping probes for %s: no config available.", v.name
                )
                continue

            # Create val_loader for this variant's seq_len
            val_loader = _create_val_loader(args.data_dir, v.config.seq_len, device)
            if val_loader is None:
                logger.warning(
                    "  Skipping probes for %s: could not create val_loader.", v.name
                )
                continue

            # Load model from checkpoint
            model = _load_model_from_checkpoint(v, device)
            if model is None:
                logger.warning(
                    "  Skipping probes for %s: could not load model.", v.name
                )
                continue

            try:
                # Per-position loss
                logger.info("  Computing per-position loss for %s...", v.name)
                per_pos_loss = compute_per_position_loss(
                    model, val_loader, v.config.seq_len, device
                )

                # ICL decay fit
                icl_params = fit_icl_decay(per_pos_loss)
                logger.info(
                    "    ICL alpha=%.3f, R²=%.3f",
                    icl_params.get("alpha", float("nan")),
                    icl_params.get("r_squared", 0.0),
                )

                # Store metrics on variant data
                from src.evaluation.metrics import MetricsResult

                val_loss_value = compute_val_loss(v.log_entries) if v.log_entries else 0.0
                v.metrics = MetricsResult(
                    val_loss=val_loss_value,
                    perplexity=compute_perplexity(val_loss_value),
                    per_position_loss=per_pos_loss,
                    icl_exponent=icl_params.get("alpha"),
                    icl_fit_params=icl_params,
                )

                # MQAR probe
                logger.info("  Running MQAR probe for %s...", v.name)
                mqar_result = run_mqar_probe(model, v.config, device=device)
                mqar_results[v.name] = mqar_result
                logger.info("    MQAR accuracy: %.3f", mqar_result.accuracy)

                # Stable rank
                logger.info("  Computing stable rank for %s...", v.name)
                srank_result = compute_stable_rank(
                    model, val_loader, n_batches=50, device=device
                )
                stable_rank_results[v.name] = srank_result
                logger.info(
                    "    Stable rank mean: %.2f ± %.2f",
                    srank_result.mean,
                    srank_result.std,
                )

                # CKA
                logger.info("  Computing CKA for %s...", v.name)
                cka_result = compute_cka(
                    model, val_loader, n_batches=25, device=device
                )
                cka_results[v.name] = cka_result

                # Attention entropy (only for non-flash variants)
                logger.info("  Computing attention entropy for %s...", v.name)
                entropy_result = compute_attention_entropy(
                    model, val_loader, n_batches=25, device=device
                )
                if entropy_result is not None:
                    attention_entropy_results[v.name] = entropy_result
                    logger.info(
                        "    Attention entropy mean: %.3f",
                        entropy_result.per_layer.mean(),
                    )
                else:
                    logger.info(
                        "    Attention entropy: N/A (flash-based variant)"
                    )

            except Exception as e:
                logger.warning(
                    "  Error computing probes for %s: %s", v.name, e, exc_info=True
                )
                continue
            finally:
                # Free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        logger.info(
            "Step 3: Skipping model-based probes (no --data_dir provided)."
        )

    # ─── Step 4: Detect seed groups ───────────────────────────────────────
    seed_groups = _detect_seed_groups(variants)
    for name, seeds in seed_groups.items():
        if len(seeds) > 1:
            logger.info(
                "Detected %d seeds for variant '%s' — will aggregate.",
                len(seeds),
                name,
            )

    # Use one representative per variant for comparison (first seed)
    representative_variants = [seeds[0] for seeds in seed_groups.values()]

    # ─── Step 5: Run comparison analysis ──────────────────────────────────
    logger.info("Step 4: Running comparison analysis...")

    fixed_data: dict[str, float] = {}
    fixed_wallclock: dict[str, dict[float, float]] = {}
    fixed_flops: dict[str, float] = {}
    pareto_front: list[str] = []
    parameter_parity_valid = False
    parameter_counts: dict[str, int] = {}

    try:
        fixed_data = slice_fixed_data(representative_variants)
        logger.info("  Fixed-data comparison: %s", fixed_data)
    except Exception as e:
        logger.warning("  slice_fixed_data failed: %s", e)

    try:
        fixed_wallclock = slice_fixed_wallclock(representative_variants)
        logger.info(
            "  Fixed-wallclock slices computed for %d variants",
            len(fixed_wallclock),
        )
    except Exception as e:
        logger.warning("  slice_fixed_wallclock failed: %s", e)

    try:
        fixed_flops = slice_fixed_flops(representative_variants)
        logger.info("  Fixed-FLOPs comparison: %s", fixed_flops)
    except Exception as e:
        logger.warning("  slice_fixed_flops failed: %s", e)

    try:
        pareto_front = compute_pareto_front(representative_variants)
        logger.info("  Pareto-optimal variants: %s", pareto_front)
    except Exception as e:
        logger.warning("  compute_pareto_front failed: %s", e)

    try:
        parameter_parity_valid, parameter_counts = validate_parameter_parity(
            representative_variants
        )
        logger.info(
            "  Parameter parity valid: %s, counts: %s",
            parameter_parity_valid,
            parameter_counts,
        )
    except Exception as e:
        logger.warning("  validate_parameter_parity failed: %s", e)

    comparison_result = ComparisonResult(
        fixed_data=fixed_data,
        fixed_wallclock=fixed_wallclock,
        fixed_flops=fixed_flops,
        pareto_front=pareto_front,
        parameter_counts=parameter_counts,
        parameter_parity_valid=parameter_parity_valid,
    )

    # ─── Step 6: Aggregate metrics across seeds ───────────────────────────
    logger.info("Step 5: Aggregating metrics across seeds...")
    seed_aggregated = _aggregate_seed_metrics(seed_groups)

    # ─── Step 7: Generate visualizations ──────────────────────────────────
    logger.info("Step 6: Generating visualizations...")

    # Learning curves
    try:
        plot_learning_curves(representative_variants, output_dir, x_axis="tokens")
        logger.info("  Generated learning_curves_tokens.png")
    except Exception as e:
        logger.warning("  plot_learning_curves (tokens) failed: %s", e)

    try:
        plot_learning_curves(representative_variants, output_dir, x_axis="wallclock")
        logger.info("  Generated learning_curves_wallclock.png")
    except Exception as e:
        logger.warning("  plot_learning_curves (wallclock) failed: %s", e)

    try:
        plot_learning_curves(representative_variants, output_dir, x_axis="flops")
        logger.info("  Generated learning_curves_flops.png")
    except Exception as e:
        logger.warning("  plot_learning_curves (flops) failed: %s", e)

    # Per-position loss (only if probe data exists)
    try:
        plot_per_position_loss(representative_variants, output_dir)
        logger.info("  Generated per_position_loss.png")
    except Exception as e:
        logger.warning("  plot_per_position_loss failed: %s", e)

    # MQAR results
    if mqar_results:
        try:
            plot_mqar_results(mqar_results, output_dir)
            logger.info("  Generated mqar_by_distance.png")
        except Exception as e:
            logger.warning("  plot_mqar_results failed: %s", e)

    # Stable rank
    if stable_rank_results:
        try:
            plot_stable_rank(stable_rank_results, output_dir)
            logger.info("  Generated stable_rank.png")
        except Exception as e:
            logger.warning("  plot_stable_rank failed: %s", e)

    # CKA
    if cka_results:
        try:
            plot_cka_adjacent(cka_results, output_dir)
            logger.info("  Generated cka_adjacent.png")
        except Exception as e:
            logger.warning("  plot_cka_adjacent failed: %s", e)

        for variant_name, cka_result in cka_results.items():
            try:
                plot_cka_heatmap(cka_result, variant_name, output_dir)
                logger.info("  Generated cka_heatmap_%s.png", variant_name)
            except Exception as e:
                logger.warning(
                    "  plot_cka_heatmap (%s) failed: %s", variant_name, e
                )

    # FLOP breakdown
    flop_breakdowns = {
        v.name: v.flop_breakdown
        for v in representative_variants
        if v.flop_breakdown is not None
    }
    if flop_breakdowns:
        try:
            plot_flop_breakdown(flop_breakdowns, output_dir)
            logger.info("  Generated flop_breakdown.png")
        except Exception as e:
            logger.warning("  plot_flop_breakdown failed: %s", e)

    # Pareto plots
    try:
        plot_pareto(representative_variants, output_dir)
        logger.info("  Generated pareto plots")
    except Exception as e:
        logger.warning("  plot_pareto failed: %s", e)

    # Roofline
    try:
        plot_roofline(representative_variants, output_dir)
        logger.info("  Generated roofline.png")
    except Exception as e:
        logger.warning("  plot_roofline failed: %s", e)

    # ─── Step 8: Generate summary report ──────────────────────────────────
    logger.info("Step 7: Generating summary report...")
    try:
        summary_path = generate_summary_md(comparison_result, output_dir)
        logger.info("  Generated %s", summary_path)
    except Exception as e:
        logger.warning("  generate_summary_md failed: %s", e)

    # ─── Step 9: Write metadata and raw data ──────────────────────────────
    logger.info("Step 8: Writing metadata and raw data...")
    _write_metadata(output_dir, variants, device)
    _write_raw_data(output_dir, variants, comparison_result, seed_aggregated)

    logger.info("Evaluation complete. Report written to: %s", output_dir)


if __name__ == "__main__":
    main()
