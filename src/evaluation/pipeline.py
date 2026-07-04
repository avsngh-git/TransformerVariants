"""Deep evaluation pipeline module.

Single interface for the full evaluation workflow: load checkpoints, compute
metrics, run probes, perform comparisons, generate visualizations, and write
reports. All orchestration logic concentrates here — callers (CLI, tests,
notebooks, future dashboard) share one interface.

Usage:
    pipeline = EvaluationPipeline(device="cuda", data_dir="data/processed/wikitext-full")
    result = pipeline.run(
        checkpoints=["checkpoints/vanilla_main_s42/", "checkpoints/modern_main_s42/"],
        output_dir="reports/comparison/",
    )
"""

from __future__ import annotations

import csv
import json
import logging
import platform
from dataclasses import dataclass, field
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
from src.evaluation.flops import FLOPBreakdown, compute_mfu, compute_step_flops
from src.evaluation.metrics import (
    MetricsResult,
    compute_per_position_loss,
    compute_perplexity,
    compute_val_loss,
    fit_icl_decay,
)
from src.evaluation.probes import (
    MQARResult,
    StableRankResult,
    CKAResult,
    AttentionEntropyResult,
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
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProbeResults:
    """Collected probe results for all evaluated variants."""

    mqar: dict[str, MQARResult] = field(default_factory=dict)
    stable_rank: dict[str, StableRankResult] = field(default_factory=dict)
    cka: dict[str, CKAResult] = field(default_factory=dict)
    attention_entropy: dict[str, AttentionEntropyResult] = field(default_factory=dict)


@dataclass
class ReportResult:
    """Structured result from a pipeline run.

    Returned alongside file writes so callers can inspect results programmatically
    without re-parsing output files.

    Attributes:
        comparison: Multi-variant comparison results (slicing, Pareto, parity).
        probe_results: Collected probe results for all variants.
        seed_aggregated: Variant name → {metric → (mean, std)} aggregation.
        output_dir: Path to the output report directory.
        generated_files: List of all files written during the run.
        warnings: Non-fatal issues encountered during execution.
        skipped_steps: Steps that were skipped (e.g., probes without data_dir).
    """

    comparison: ComparisonResult
    probe_results: ProbeResults
    seed_aggregated: dict[str, dict[str, tuple[float, float]]]
    output_dir: Path
    generated_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """Deep module for the full evaluation workflow.

    Concentrates orchestration, error handling, seed grouping, and step ordering
    behind a single interface. The CLI script becomes a thin shell; tests and
    notebooks call the same run() method.

    Args:
        device: Device for GPU computations (e.g., "cuda", "cpu").
            If None, auto-detects cuda availability.
        data_dir: Path to validation data directory for probe computations.
            When None, probe steps that require a val_loader are skipped.
    """

    def __init__(
        self,
        device: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        self._device = self._resolve_device(device)
        self._data_dir = data_dir

    def run(
        self,
        checkpoints: list[str | Path],
        output_dir: str | Path,
    ) -> ReportResult:
        """Execute the full evaluation pipeline.

        Steps:
        1. Load variant data from checkpoint directories
        2. Compute per-step FLOPs from configs
        3. Run model-based metrics and probes (if data_dir provided)
        4. Detect seed groups and select representatives
        5. Run comparison analysis (slicing, Pareto, parity)
        6. Aggregate metrics across seeds
        7. Generate all visualizations
        8. Generate summary report
        9. Write metadata and raw data files

        Args:
            checkpoints: Paths to checkpoint directories to evaluate.
            output_dir: Output directory for the report.

        Returns:
            ReportResult with all computed data, paths, and observability info.
        """
        output_dir = Path(output_dir)
        warnings: list[str] = []
        skipped_steps: list[str] = []
        generated_files: list[Path] = []

        logger.info("Evaluation pipeline starting")
        logger.info("  Checkpoints: %s", checkpoints)
        logger.info("  Output: %s", output_dir)
        logger.info("  Device: %s", self._device)
        logger.info("  Data dir: %s", self._data_dir or "(none — probes will be skipped)")

        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)

        # ─── Step 1: Load variant data ────────────────────────────────────
        checkpoint_dirs = [Path(d) for d in checkpoints]
        variants = load_variant_data(checkpoint_dirs)

        if not variants:
            msg = "No valid variants loaded from provided checkpoints."
            warnings.append(msg)
            logger.error(msg)
            return ReportResult(
                comparison=ComparisonResult(),
                probe_results=ProbeResults(),
                seed_aggregated={},
                output_dir=output_dir,
                generated_files=generated_files,
                warnings=warnings,
                skipped_steps=skipped_steps,
            )

        logger.info("Loaded %d variant(s): %s", len(variants), [v.name for v in variants])

        # ─── Step 2: Compute FLOPs ────────────────────────────────────────
        self._compute_flops(variants, warnings)

        # ─── Step 3: Run probes ───────────────────────────────────────────
        probe_results = self._run_probes(variants, warnings, skipped_steps)

        # ─── Step 4: Detect seed groups ───────────────────────────────────
        seed_groups = self._detect_seed_groups(variants)
        representative_variants = [seeds[0] for seeds in seed_groups.values()]

        # ─── Step 5: Comparison analysis ──────────────────────────────────
        comparison = self._run_comparisons(representative_variants, warnings)

        # ─── Step 6: Aggregate across seeds ───────────────────────────────
        seed_aggregated = self._aggregate_seeds(seed_groups, variants, warnings)

        # ─── Step 7: Generate visualizations ──────────────────────────────
        self._generate_plots(
            representative_variants, probe_results, output_dir, warnings, generated_files
        )

        # ─── Step 8: Generate summary ─────────────────────────────────────
        self._generate_summary(comparison, output_dir, warnings, generated_files)

        # ─── Step 9: Write metadata and raw data ──────────────────────────
        self._write_output_files(
            output_dir, variants, comparison, seed_aggregated, warnings, generated_files
        )

        logger.info("Evaluation complete. Report written to: %s", output_dir)

        return ReportResult(
            comparison=comparison,
            probe_results=probe_results,
            seed_aggregated=seed_aggregated,
            output_dir=output_dir,
            generated_files=generated_files,
            warnings=warnings,
            skipped_steps=skipped_steps,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Internal methods
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_device(device_arg: str | None) -> str:
        """Resolve device: use provided arg, or auto-detect cuda."""
        if device_arg is not None:
            return device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _detect_seed_groups(variants: list[VariantData]) -> dict[str, list[VariantData]]:
        """Group variants by name to detect multiple seeds."""
        groups: dict[str, list[VariantData]] = {}
        for v in variants:
            groups.setdefault(v.name, []).append(v)
        return groups

    def _compute_flops(self, variants: list[VariantData], warnings: list[str]) -> None:
        """Compute FLOP breakdowns for all variants with configs."""
        logger.info("Computing FLOP breakdowns...")
        for v in variants:
            if v.config is not None and v.flop_breakdown is None:
                try:
                    v.flop_breakdown = compute_step_flops(v.config)
                    logger.info("  %s: %s FLOPs/step", v.name, f"{v.flop_breakdown.total:,.0f}")
                except Exception as e:
                    msg = f"Failed to compute FLOPs for {v.name}: {e}"
                    warnings.append(msg)
                    logger.warning("  %s", msg)

    def _run_probes(
        self,
        variants: list[VariantData],
        warnings: list[str],
        skipped_steps: list[str],
    ) -> ProbeResults:
        """Run model-based metrics and probes if data_dir is available."""
        probe_results = ProbeResults()

        if not self._data_dir:
            skipped_steps.append("model_probes (no data_dir)")
            logger.info("Skipping model-based probes (no data_dir).")
            return probe_results

        logger.info("Running model-based metrics and probes...")

        for v in variants:
            if v.config is None:
                warnings.append(f"Skipping probes for {v.name}: no config available.")
                continue

            val_loader = self._create_val_loader(v.config.seq_len)
            if val_loader is None:
                warnings.append(f"Skipping probes for {v.name}: could not create val_loader.")
                continue

            model = self._load_model_from_checkpoint(v)
            if model is None:
                warnings.append(f"Skipping probes for {v.name}: could not load model.")
                continue

            try:
                self._run_variant_probes(v, model, val_loader, probe_results, warnings)
            except Exception as e:
                msg = f"Error computing probes for {v.name}: {e}"
                warnings.append(msg)
                logger.warning(msg, exc_info=True)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return probe_results

    def _run_variant_probes(
        self,
        v: VariantData,
        model,
        val_loader,
        probe_results: ProbeResults,
        warnings: list[str],
    ) -> None:
        """Run all probes on a single variant's model."""
        # Per-position loss + ICL decay
        logger.info("  Computing per-position loss for %s...", v.name)
        per_pos_loss = compute_per_position_loss(
            model, val_loader, v.config.seq_len, self._device
        )
        icl_params = fit_icl_decay(per_pos_loss)

        val_loss_value = compute_val_loss(v.log_entries) if v.log_entries else 0.0
        v.metrics = MetricsResult(
            val_loss=val_loss_value,
            perplexity=compute_perplexity(val_loss_value),
            per_position_loss=per_pos_loss,
            icl_exponent=icl_params.get("alpha"),
            icl_fit_params=icl_params,
        )

        # MQAR
        logger.info("  Running MQAR probe for %s...", v.name)
        mqar_result = run_mqar_probe(model, v.config, device=self._device)
        probe_results.mqar[v.name] = mqar_result
        logger.info("    MQAR accuracy: %.3f", mqar_result.accuracy)

        # Stable rank
        logger.info("  Computing stable rank for %s...", v.name)
        srank_result = compute_stable_rank(model, val_loader, n_batches=50, device=self._device)
        probe_results.stable_rank[v.name] = srank_result
        logger.info("    Stable rank mean: %.2f ± %.2f", srank_result.mean, srank_result.std)

        # CKA
        logger.info("  Computing CKA for %s...", v.name)
        cka_result = compute_cka(model, val_loader, n_batches=25, device=self._device)
        probe_results.cka[v.name] = cka_result

        # Attention entropy
        logger.info("  Computing attention entropy for %s...", v.name)
        entropy_result = compute_attention_entropy(model, val_loader, n_batches=25, device=self._device)
        if entropy_result is not None:
            probe_results.attention_entropy[v.name] = entropy_result
            logger.info("    Attention entropy mean: %.3f", entropy_result.per_layer.mean())
        else:
            logger.info("    Attention entropy: N/A (flash-based variant)")

    def _run_comparisons(
        self,
        representative_variants: list[VariantData],
        warnings: list[str],
    ) -> ComparisonResult:
        """Run all comparison analyses."""
        logger.info("Running comparison analysis...")

        fixed_data: dict[str, float] = {}
        fixed_wallclock: dict[str, dict] = {}
        fixed_flops: dict[str, float] = {}
        pareto_front: list[str] = []
        parameter_parity_valid = False
        parameter_counts: dict[str, int] = {}

        try:
            fixed_data = slice_fixed_data(representative_variants)
        except Exception as e:
            warnings.append(f"slice_fixed_data failed: {e}")

        try:
            fixed_wallclock = slice_fixed_wallclock(representative_variants)
        except Exception as e:
            warnings.append(f"slice_fixed_wallclock failed: {e}")

        try:
            fixed_flops = slice_fixed_flops(representative_variants)
        except Exception as e:
            warnings.append(f"slice_fixed_flops failed: {e}")

        try:
            pareto_front = compute_pareto_front(representative_variants)
        except Exception as e:
            warnings.append(f"compute_pareto_front failed: {e}")

        try:
            parameter_parity_valid, parameter_counts = validate_parameter_parity(
                representative_variants
            )
        except Exception as e:
            warnings.append(f"validate_parameter_parity failed: {e}")

        return ComparisonResult(
            fixed_data=fixed_data,
            fixed_wallclock=fixed_wallclock,
            fixed_flops=fixed_flops,
            pareto_front=pareto_front,
            parameter_counts=parameter_counts,
            parameter_parity_valid=parameter_parity_valid,
        )

    def _aggregate_seeds(
        self,
        seed_groups: dict[str, list[VariantData]],
        variants: list[VariantData],
        warnings: list[str],
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Aggregate metrics across seeds for each variant."""
        logger.info("Aggregating metrics across seeds...")
        aggregated: dict[str, dict[str, tuple[float, float]]] = {}

        for variant_name, seeds in seed_groups.items():
            if len(seeds) > 1:
                logger.info("  %s: %d seeds", variant_name, len(seeds))

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

                if v.flop_breakdown is not None:
                    result["step_flops"] = float(v.flop_breakdown.total)

                if v.metrics is not None:
                    result["val_loss"] = v.metrics.val_loss
                    result["perplexity"] = v.metrics.perplexity
                    if v.metrics.icl_exponent is not None:
                        result["icl_alpha"] = v.metrics.icl_exponent

                if result:
                    seed_results.append(result)

            if seed_results:
                try:
                    aggregated[variant_name] = aggregate_across_seeds(seed_results)
                except Exception as e:
                    warnings.append(f"Seed aggregation failed for {variant_name}: {e}")

        return aggregated

    def _generate_plots(
        self,
        representative_variants: list[VariantData],
        probe_results: ProbeResults,
        output_dir: Path,
        warnings: list[str],
        generated_files: list[Path],
    ) -> None:
        """Generate all visualizations."""
        logger.info("Generating visualizations...")

        # Learning curves (3 x-axes)
        for x_axis in ("tokens", "wallclock", "flops"):
            try:
                path = plot_learning_curves(representative_variants, output_dir, x_axis=x_axis)
                generated_files.append(path)
            except Exception as e:
                warnings.append(f"plot_learning_curves ({x_axis}) failed: {e}")

        # Per-position loss
        try:
            path = plot_per_position_loss(representative_variants, output_dir)
            generated_files.append(path)
        except Exception as e:
            warnings.append(f"plot_per_position_loss failed: {e}")

        # MQAR
        if probe_results.mqar:
            try:
                path = plot_mqar_results(probe_results.mqar, output_dir)
                generated_files.append(path)
            except Exception as e:
                warnings.append(f"plot_mqar_results failed: {e}")

        # Stable rank
        if probe_results.stable_rank:
            try:
                path = plot_stable_rank(probe_results.stable_rank, output_dir)
                generated_files.append(path)
            except Exception as e:
                warnings.append(f"plot_stable_rank failed: {e}")

        # CKA
        if probe_results.cka:
            try:
                path = plot_cka_adjacent(probe_results.cka, output_dir)
                generated_files.append(path)
            except Exception as e:
                warnings.append(f"plot_cka_adjacent failed: {e}")

            for variant_name, cka_result in probe_results.cka.items():
                try:
                    path = plot_cka_heatmap(cka_result, variant_name, output_dir)
                    generated_files.append(path)
                except Exception as e:
                    warnings.append(f"plot_cka_heatmap ({variant_name}) failed: {e}")

        # FLOP breakdown
        flop_breakdowns = {
            v.name: v.flop_breakdown
            for v in representative_variants
            if v.flop_breakdown is not None
        }
        if flop_breakdowns:
            try:
                path = plot_flop_breakdown(flop_breakdowns, output_dir)
                generated_files.append(path)
            except Exception as e:
                warnings.append(f"plot_flop_breakdown failed: {e}")

        # Pareto
        try:
            paths = plot_pareto(representative_variants, output_dir)
            if isinstance(paths, list):
                generated_files.extend(paths)
            else:
                generated_files.append(paths)
        except Exception as e:
            warnings.append(f"plot_pareto failed: {e}")

        # Roofline
        try:
            path = plot_roofline(representative_variants, output_dir)
            generated_files.append(path)
        except Exception as e:
            warnings.append(f"plot_roofline failed: {e}")

    def _generate_summary(
        self,
        comparison: ComparisonResult,
        output_dir: Path,
        warnings: list[str],
        generated_files: list[Path],
    ) -> None:
        """Generate summary.md report."""
        logger.info("Generating summary report...")
        try:
            path = generate_summary_md(comparison, output_dir)
            generated_files.append(path)
        except Exception as e:
            warnings.append(f"generate_summary_md failed: {e}")

    def _write_output_files(
        self,
        output_dir: Path,
        variants: list[VariantData],
        comparison: ComparisonResult,
        seed_aggregated: dict[str, dict[str, tuple[float, float]]],
        warnings: list[str],
        generated_files: list[Path],
    ) -> None:
        """Write metadata.json and raw/ data files."""
        logger.info("Writing metadata and raw data...")

        # metadata.json
        try:
            metadata_path = self._write_metadata(output_dir, variants)
            generated_files.append(metadata_path)
        except Exception as e:
            warnings.append(f"Failed to write metadata.json: {e}")

        # raw/metrics.csv and raw/metrics.json
        try:
            csv_path, json_path = self._write_raw_data(
                output_dir, variants, comparison, seed_aggregated
            )
            generated_files.extend([csv_path, json_path])
        except Exception as e:
            warnings.append(f"Failed to write raw data: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # File I/O helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _load_model_from_checkpoint(self, variant_data: VariantData):
        """Load a model from checkpoint weights using the registry."""
        from src.models.registry import build as registry_build, SCALES

        config = variant_data.config
        if config is None:
            return None

        try:
            variant_name = config.variant
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

            model, _ = registry_build(variant_name, scale, activation=config.activation, dtype="float32")
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

        for weight_path in weight_paths:
            if weight_path.exists():
                try:
                    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
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
                    logger.info("Loaded weights for %s from %s", variant_data.name, weight_path.name)
                    break
                except Exception as e:
                    logger.warning("Failed to load weights from %s: %s", weight_path, e)
                    continue

        model = model.to(self._device)
        model.eval()
        return model

    def _create_val_loader(self, seq_len: int):
        """Create a validation data loader."""
        from src.data.dataloader import ShardedDataLoader

        try:
            return ShardedDataLoader(
                data_dir=self._data_dir,
                batch_size=8,
                seq_len=seq_len,
                split="val",
                device=self._device,
            )
        except FileNotFoundError as e:
            logger.warning("Could not create validation loader: %s", e)
            return None

    def _write_metadata(self, output_dir: Path, variants: list[VariantData]) -> Path:
        """Write metadata.json with run information."""
        if self._device.startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            hardware = f"{self._device} ({gpu_name})"
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
            "evaluated_checkpoints": [str(v.checkpoint_dir) for v in variants],
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path

    def _write_raw_data(
        self,
        output_dir: Path,
        variants: list[VariantData],
        comparison: ComparisonResult,
        seed_aggregated: dict[str, dict[str, tuple[float, float]]],
    ) -> tuple[Path, Path]:
        """Write raw/metrics.csv and raw/metrics.json."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        csv_path = raw_dir / "metrics.csv"
        self._write_metrics_csv(csv_path, variants)

        json_path = raw_dir / "metrics.json"
        self._write_metrics_json(json_path, variants, comparison, seed_aggregated)

        return csv_path, json_path

    def _write_metrics_csv(self, csv_path: Path, variants: list[VariantData]) -> None:
        """Write a flat CSV with per-variant, per-seed metric values."""
        seed_groups = self._detect_seed_groups(variants)
        rows: list[dict[str, str | float]] = []

        for variant_name, seeds in seed_groups.items():
            for seed_idx, v in enumerate(seeds):
                row: dict[str, str | float] = {
                    "variant": variant_name,
                    "seed_index": seed_idx,
                    "checkpoint_dir": str(v.checkpoint_dir),
                }

                if v.log_entries:
                    for entry in reversed(v.log_entries):
                        if entry.get("val_loss") is not None:
                            row["val_loss"] = entry["val_loss"]
                            row["perplexity"] = float(np.exp(entry["val_loss"]))
                            break

                if v.flop_breakdown is not None:
                    row["step_flops"] = v.flop_breakdown.total

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
        self,
        json_path: Path,
        variants: list[VariantData],
        comparison: ComparisonResult,
        seed_aggregated: dict[str, dict[str, tuple[float, float]]],
    ) -> None:
        """Write full structured metrics data as JSON."""
        seed_groups = self._detect_seed_groups(variants)

        variants_data: dict[str, list[dict]] = {}
        for variant_name, seeds in seed_groups.items():
            seed_list = []
            for seed_idx, v in enumerate(seeds):
                seed_entry: dict = {
                    "seed_index": seed_idx,
                    "checkpoint_dir": str(v.checkpoint_dir),
                }

                if v.log_entries:
                    for entry in reversed(v.log_entries):
                        if entry.get("val_loss") is not None:
                            seed_entry["val_loss"] = entry["val_loss"]
                            seed_entry["perplexity"] = float(np.exp(entry["val_loss"]))
                            break

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

        # Aggregated section
        aggregated_section: dict[str, dict] = {}
        for variant_name, metrics in seed_aggregated.items():
            aggregated_section[variant_name] = {
                metric_name: {
                    "mean": mean,
                    "std": std if not (isinstance(std, float) and np.isnan(std)) else None,
                }
                for metric_name, (mean, std) in metrics.items()
            }

        # Comparison section
        comparison_section = {
            "fixed_data": comparison.fixed_data,
            "fixed_wallclock": comparison.fixed_wallclock,
            "fixed_flops": comparison.fixed_flops,
            "pareto_front": comparison.pareto_front,
            "parameter_counts": comparison.parameter_counts,
            "parameter_parity_valid": comparison.parameter_parity_valid,
        }

        output = {
            "variants": variants_data,
            "aggregated": aggregated_section,
            "comparison": comparison_section,
        }

        with open(json_path, "w") as f:
            json.dump(output, f, indent=2, default=self._json_default)

    @staticmethod
    def _json_default(obj):
        """JSON serializer for non-standard types."""
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
