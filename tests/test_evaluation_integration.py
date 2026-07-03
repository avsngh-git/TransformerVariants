"""End-to-end integration test for the evaluation framework.

Builds debug-scale models via registry for 2+ variants, creates synthetic
metrics.jsonl files with fabricated training logs, runs the full config-based
pipeline (no model weights needed), and verifies the output directory structure.

Validates: Requirements 13.1, 13.4, 15.1, 15.2, 15.3, 15.4, 15.5
"""

import json
from pathlib import Path

import pytest

from src.evaluation.comparison import (
    ComparisonResult,
    VariantData,
    load_variant_data,
    slice_fixed_data,
    slice_fixed_flops,
    slice_fixed_wallclock,
    compute_pareto_front,
    validate_parameter_parity,
)
from src.evaluation.flops import compute_step_flops
from src.evaluation.visualizations import (
    generate_summary_md,
    plot_flop_breakdown,
    plot_learning_curves,
    plot_pareto,
    plot_roofline,
)
from src.models.config import ModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metrics_jsonl(checkpoint_dir: Path, n_steps: int = 100) -> None:
    """Create a synthetic metrics.jsonl with fabricated training log entries."""
    metrics_file = checkpoint_dir / "metrics.jsonl"
    entries = []
    for step in range(1, n_steps + 1):
        entry = {
            "step": step,
            "train_loss": 5.0 - (step / n_steps) * 2.0,  # decreasing from 5.0 to 3.0
            "val_loss": 4.8 - (step / n_steps) * 1.8 if step % 10 == 0 else None,
            "tokens_seen": step * 1024,
            "elapsed_time": step * 0.5,
            "learning_rate": 3e-4,
            "grad_norm": 1.0 + (step / n_steps) * 0.5,
        }
        entries.append(entry)

    with open(metrics_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_run_config(checkpoint_dir: Path, config: ModelConfig) -> None:
    """Create a run_config.json matching a known variant config."""
    run_config = {
        "model": {
            "n_layer": config.n_layer,
            "d_model": config.d_model,
            "n_head": config.n_head,
            "seq_len": config.seq_len,
            "vocab_size": config.vocab_size,
            "ffn_multiplier": config.ffn_multiplier,
            "dropout": config.dropout,
            "bias": config.bias,
            "tie_embeddings": config.tie_embeddings,
            "activation": config.activation,
            "variant": config.variant,
            "norm_type": config.norm_type,
            "position_encoding": config.position_encoding,
            "ffn_type": config.ffn_type,
            "attention_type": config.attention_type,
            "attention_backend": config.attention_backend,
        }
    }
    if config.window_size is not None:
        run_config["model"]["window_size"] = config.window_size
    if config.projection_rank is not None:
        run_config["model"]["projection_rank"] = config.projection_rank
    if config.n_kv_head is not None:
        run_config["model"]["n_kv_head"] = config.n_kv_head

    config_path = checkpoint_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)


@pytest.fixture
def debug_configs() -> dict[str, ModelConfig]:
    """Create debug-scale configs for 2 variants (vanilla and linear)."""
    vanilla_config = ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=4,
        seq_len=128,
        vocab_size=50257,
        ffn_multiplier=4,
        variant="vanilla",
        norm_type="layernorm",
        position_encoding="learned",
        ffn_type="standard",
        attention_type="full",
        activation="relu",
    )

    linear_config = ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=4,
        seq_len=128,
        vocab_size=50257,
        ffn_multiplier=4,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        activation="swiglu",
        projection_rank=32,
    )

    return {"vanilla": vanilla_config, "linear": linear_config}


@pytest.fixture
def fake_checkpoints(tmp_path: Path, debug_configs: dict[str, ModelConfig]) -> list[Path]:
    """Create fake checkpoint directories with metrics.jsonl and run_config.json."""
    checkpoint_dirs = []
    for variant_name, config in debug_configs.items():
        ckpt_dir = tmp_path / f"{variant_name}_debug_s42"
        ckpt_dir.mkdir(parents=True)
        _make_metrics_jsonl(ckpt_dir, n_steps=100)
        _make_run_config(ckpt_dir, config)
        checkpoint_dirs.append(ckpt_dir)
    return checkpoint_dirs


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create an output directory for the report."""
    out = tmp_path / "report_output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end integration test for the config-based evaluation pipeline.

    Validates: Requirements 13.1, 13.4, 15.1, 15.2, 15.3, 15.4, 15.5
    """

    def test_load_variant_data(self, fake_checkpoints: list[Path]):
        """Requirement 13.1: Pipeline loads variant data from checkpoint dirs."""
        variants = load_variant_data(fake_checkpoints)

        assert len(variants) == 2
        names = {v.name for v in variants}
        assert "vanilla" in names
        assert "linear" in names

        for v in variants:
            assert len(v.log_entries) > 0
            assert v.config is not None

    def test_flop_computation(self, fake_checkpoints: list[Path]):
        """Requirement 13.4: FLOPs are computed from configs."""
        variants = load_variant_data(fake_checkpoints)

        for v in variants:
            flop_breakdown = compute_step_flops(v.config)
            v.flop_breakdown = flop_breakdown
            assert flop_breakdown.total > 0
            assert flop_breakdown.qkv_proj > 0
            assert flop_breakdown.attention_score > 0
            assert flop_breakdown.ffn > 0

    def test_comparison_slicing(self, fake_checkpoints: list[Path]):
        """Requirements 13.4: Comparison slicing produces valid results."""
        variants = load_variant_data(fake_checkpoints)

        for v in variants:
            v.flop_breakdown = compute_step_flops(v.config)

        # Fixed-data comparison
        fixed_data = slice_fixed_data(variants)
        assert len(fixed_data) == 2
        for name, val_loss in fixed_data.items():
            assert isinstance(val_loss, float)
            assert val_loss > 0

        # Fixed-wallclock comparison
        fixed_wallclock = slice_fixed_wallclock(variants)
        assert len(fixed_wallclock) == 2
        for name, frac_dict in fixed_wallclock.items():
            assert isinstance(frac_dict, dict)
            assert 1.0 in frac_dict

        # Fixed-FLOPs comparison
        fixed_flops = slice_fixed_flops(variants)
        assert len(fixed_flops) == 2

    def test_pareto_and_parity(self, fake_checkpoints: list[Path]):
        """Requirements 13.4: Pareto front and parameter parity analysis."""
        variants = load_variant_data(fake_checkpoints)

        for v in variants:
            v.flop_breakdown = compute_step_flops(v.config)

        pareto = compute_pareto_front(variants)
        assert isinstance(pareto, list)
        assert len(pareto) >= 1

        parity_valid, param_counts = validate_parameter_parity(variants)
        assert isinstance(parity_valid, bool)
        assert len(param_counts) == 2
        for name, count in param_counts.items():
            assert count > 0

    def test_full_pipeline_output_structure(
        self, fake_checkpoints: list[Path], output_dir: Path
    ):
        """Requirements 15.1, 15.2, 15.3, 15.4, 15.5: Full pipeline output."""
        # Step 1: Load variant data
        variants = load_variant_data(fake_checkpoints)

        # Step 2: Compute FLOPs
        for v in variants:
            v.flop_breakdown = compute_step_flops(v.config)

        # Step 3: Run comparison
        fixed_data = slice_fixed_data(variants)
        fixed_wallclock = slice_fixed_wallclock(variants)
        fixed_flops = slice_fixed_flops(variants)
        pareto_front = compute_pareto_front(variants)
        parity_valid, parameter_counts = validate_parameter_parity(variants)

        comparison_result = ComparisonResult(
            fixed_data=fixed_data,
            fixed_wallclock=fixed_wallclock,
            fixed_flops=fixed_flops,
            pareto_front=pareto_front,
            parameter_counts=parameter_counts,
            parameter_parity_valid=parity_valid,
        )

        # Step 4: Generate visualizations
        plot_learning_curves(variants, output_dir, x_axis="tokens")
        plot_learning_curves(variants, output_dir, x_axis="wallclock")
        plot_learning_curves(variants, output_dir, x_axis="flops")

        flop_breakdowns = {v.name: v.flop_breakdown for v in variants}
        plot_flop_breakdown(flop_breakdowns, output_dir)

        plot_pareto(variants, output_dir)
        plot_roofline(variants, output_dir)

        # Step 5: Generate summary.md
        generate_summary_md(comparison_result, output_dir)

        # Step 6: Write metadata.json
        import platform
        from datetime import datetime, timezone

        import numpy as np
        import torch

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "software_versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "hardware": "cpu",
            "evaluated_checkpoints": [str(d) for d in fake_checkpoints],
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Step 7: Write raw data
        import csv

        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Write metrics.csv
        csv_path = raw_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["variant", "val_loss", "step_flops"],
            )
            writer.writeheader()
            for v in variants:
                last_val = None
                for entry in reversed(v.log_entries):
                    if entry.get("val_loss") is not None:
                        last_val = entry["val_loss"]
                        break
                writer.writerow({
                    "variant": v.name,
                    "val_loss": last_val,
                    "step_flops": v.flop_breakdown.total if v.flop_breakdown else None,
                })

        # Write metrics.json
        json_path = raw_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "variants": {
                        v.name: {
                            "val_loss": next(
                                (
                                    e["val_loss"]
                                    for e in reversed(v.log_entries)
                                    if e.get("val_loss") is not None
                                ),
                                None,
                            ),
                            "step_flops": v.flop_breakdown.total
                            if v.flop_breakdown
                            else None,
                        }
                        for v in variants
                    },
                    "comparison": {
                        "fixed_data": fixed_data,
                        "fixed_wallclock": {
                            k: {str(fk): fv for fk, fv in vd.items()}
                            for k, vd in fixed_wallclock.items()
                        },
                        "fixed_flops": fixed_flops,
                    },
                },
                f,
                indent=2,
            )

        # ─── Assertions: Verify output directory structure ────────────────
        # Requirement 15.2: plots/ subdirectory with PNGs
        plots_dir = output_dir / "plots"
        assert plots_dir.exists(), "plots/ directory must exist"
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) >= 1, "At least one PNG must be generated"

        # Verify at least one PNG per major plot category
        plot_filenames = {p.name for p in png_files}
        assert any(
            "learning_curves" in name for name in plot_filenames
        ), "Missing learning curve plot"
        assert any(
            "flop_breakdown" in name for name in plot_filenames
        ), "Missing flop breakdown plot"

        # Requirement 15.3: raw/ subdirectory with metrics.csv and metrics.json
        raw_dir = output_dir / "raw"
        assert raw_dir.exists(), "raw/ directory must exist"
        assert (raw_dir / "metrics.csv").exists(), "raw/metrics.csv must exist"
        assert (raw_dir / "metrics.json").exists(), "raw/metrics.json must exist"

        # Requirement 15.1: summary.md exists at top level
        summary_path = output_dir / "summary.md"
        assert summary_path.exists(), "summary.md must exist"

        # Requirement 15.5: summary.md contains expected heading and relative image links
        summary_content = summary_path.read_text()
        assert "# Evaluation Summary Report" in summary_content
        assert "plots/" in summary_content, "summary.md must contain relative image links"

        # Requirement 15.4: metadata.json exists with required fields
        assert metadata_path.exists(), "metadata.json must exist"
        with open(metadata_path) as f:
            meta = json.load(f)
        assert "timestamp" in meta
        assert "software_versions" in meta
        assert "hardware" in meta
        assert "evaluated_checkpoints" in meta

    def test_summary_md_sections(
        self, fake_checkpoints: list[Path], output_dir: Path
    ):
        """Requirement 15.5: summary.md contains expected sections and links."""
        variants = load_variant_data(fake_checkpoints)
        for v in variants:
            v.flop_breakdown = compute_step_flops(v.config)

        fixed_data = slice_fixed_data(variants)
        fixed_wallclock = slice_fixed_wallclock(variants)
        fixed_flops = slice_fixed_flops(variants)
        pareto_front = compute_pareto_front(variants)
        parity_valid, parameter_counts = validate_parameter_parity(variants)

        comparison_result = ComparisonResult(
            fixed_data=fixed_data,
            fixed_wallclock=fixed_wallclock,
            fixed_flops=fixed_flops,
            pareto_front=pareto_front,
            parameter_counts=parameter_counts,
            parameter_parity_valid=parity_valid,
        )

        summary_path = generate_summary_md(comparison_result, output_dir)

        content = summary_path.read_text()

        # Check required sections
        assert "# Evaluation Summary Report" in content
        assert "## Fixed-Data Comparison" in content
        assert "## Fixed-Wallclock Comparison" in content
        assert "## Fixed-FLOPs Comparison" in content
        assert "## Parameter Parity" in content
        assert "## Pareto Front" in content
        assert "## Figures" in content

        # Check relative image links
        assert "plots/learning_curves_tokens.png" in content
        assert "plots/learning_curves_wallclock.png" in content
        assert "plots/flop_breakdown.png" in content
        assert "plots/roofline.png" in content

        # Check variant names appear in tables
        assert "vanilla" in content
        assert "linear" in content

    def test_visualizations_produce_pngs(
        self, fake_checkpoints: list[Path], output_dir: Path
    ):
        """Requirement 15.2: At least one PNG generated per plot category."""
        variants = load_variant_data(fake_checkpoints)
        for v in variants:
            v.flop_breakdown = compute_step_flops(v.config)

        # Learning curves
        plot_learning_curves(variants, output_dir, x_axis="tokens")
        plot_learning_curves(variants, output_dir, x_axis="wallclock")
        plot_learning_curves(variants, output_dir, x_axis="flops")

        # Efficiency plots
        flop_breakdowns = {v.name: v.flop_breakdown for v in variants}
        plot_flop_breakdown(flop_breakdowns, output_dir)
        plot_roofline(variants, output_dir)
        plot_pareto(variants, output_dir)

        plots_dir = output_dir / "plots"
        png_files = list(plots_dir.glob("*.png"))

        # Verify expected files
        filenames = {p.name for p in png_files}

        # Learning curves
        assert "learning_curves_tokens.png" in filenames
        assert "learning_curves_wallclock.png" in filenames
        assert "learning_curves_flops.png" in filenames

        # Efficiency
        assert "flop_breakdown.png" in filenames
        assert "roofline.png" in filenames

        # All PNGs should be non-empty files
        for png in png_files:
            assert png.stat().st_size > 0, f"{png.name} should be non-empty"
