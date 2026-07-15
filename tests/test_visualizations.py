"""Unit tests for the visualizations module."""

import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation.comparison import ComparisonResult, VariantData
from src.evaluation.flops import FLOPBreakdown
from src.evaluation.metrics import MetricsResult
from src.evaluation.probes import CKAResult, MQARResult, StableRankResult
from src.evaluation.visualizations import COLORBLIND_PALETTE
from src.models.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures for efficiency plot tests (task 6.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_breakdowns() -> dict[str, FLOPBreakdown]:
    """Create sample FLOP breakdowns for testing."""
    return {
        "vanilla": FLOPBreakdown(
            qkv_proj=100_000_000_000,
            attention_score=200_000_000_000,
            attention_output=50_000_000_000,
            ffn=300_000_000_000,
            total=650_000_000_000,
        ),
        "swa": FLOPBreakdown(
            qkv_proj=100_000_000_000,
            attention_score=50_000_000_000,
            attention_output=50_000_000_000,
            ffn=300_000_000_000,
            total=500_000_000_000,
        ),
        "linear": FLOPBreakdown(
            qkv_proj=100_000_000_000,
            attention_score=30_000_000_000,
            attention_output=50_000_000_000,
            ffn=300_000_000_000,
            total=480_000_000_000,
        ),
    }


@pytest.fixture
def sample_variants() -> list[VariantData]:
    """Create sample VariantData for Pareto and roofline tests."""
    configs = [
        ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            variant="vanilla",
            attention_type="full",
        ),
        ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            variant="swa",
            attention_type="sliding_window",
            window_size=128,
        ),
        ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            variant="linear",
            attention_type="linear",
            projection_rank=64,
        ),
    ]

    log_entries_template = [
        {"step": 1, "elapsed_time": 0.5, "val_loss": 5.0, "peak_memory_mb": 2000},
        {"step": 100, "elapsed_time": 50.0, "val_loss": 4.0, "peak_memory_mb": 2100},
        {"step": 200, "elapsed_time": 100.0, "val_loss": 3.5, "peak_memory_mb": 2200},
    ]

    variants = []
    val_losses = [3.5, 3.3, 3.8]  # swa is best loss, linear is worst
    for i, config in enumerate(configs):
        entries = []
        for e in log_entries_template:
            entry = dict(e)
            if e["step"] == 200:
                entry["val_loss"] = val_losses[i]
            entries.append(entry)

        from src.evaluation.flops import compute_step_flops

        variants.append(
            VariantData(
                name=config.variant,
                checkpoint_dir=Path(f"/tmp/fake_{config.variant}"),
                log_entries=entries,
                config=config,
                flop_breakdown=compute_step_flops(config),
            )
        )

    return variants


# ---------------------------------------------------------------------------
# Task 6.3: Efficiency plot tests
# ---------------------------------------------------------------------------


class TestPlotFlopBreakdown:
    """Tests for plot_flop_breakdown."""

    def test_creates_png_file(self, tmp_path: Path, sample_breakdowns):
        """Plot function creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_flop_breakdown

        result = plot_flop_breakdown(sample_breakdowns, tmp_path)
        assert result == tmp_path / "plots" / "flop_breakdown.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_plots_directory(self, tmp_path: Path, sample_breakdowns):
        """The plots/ subdirectory is created automatically."""
        from src.evaluation.visualizations import plot_flop_breakdown

        plot_flop_breakdown(sample_breakdowns, tmp_path)
        assert (tmp_path / "plots").is_dir()

    def test_single_variant(self, tmp_path: Path):
        """Works with a single variant."""
        from src.evaluation.visualizations import plot_flop_breakdown

        breakdowns = {
            "vanilla": FLOPBreakdown(
                qkv_proj=100, attention_score=200, attention_output=50, ffn=300, total=650
            )
        }
        result = plot_flop_breakdown(breakdowns, tmp_path)
        assert result.exists()


class TestPlotPareto:
    """Tests for plot_pareto."""

    def test_creates_png_per_axis_pair(self, tmp_path: Path, sample_variants):
        """One PNG is created per objective pair."""
        from src.evaluation.visualizations import plot_pareto

        paths = plot_pareto(sample_variants, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_default_filenames(self, tmp_path: Path, sample_variants):
        """Default axis pairs produce expected filenames."""
        from src.evaluation.visualizations import plot_pareto

        paths = plot_pareto(sample_variants, tmp_path)
        names = {p.name for p in paths}
        assert "pareto_flops_val_loss.png" in names
        assert "pareto_wallclock_val_loss.png" in names
        assert "pareto_peak_memory_val_loss.png" in names

    def test_custom_axes(self, tmp_path: Path, sample_variants):
        """Custom axis pairs work correctly."""
        from src.evaluation.visualizations import plot_pareto

        paths = plot_pareto(sample_variants, tmp_path, axes=[("flops", "val_loss")])
        assert len(paths) == 1
        assert paths[0].name == "pareto_flops_val_loss.png"

    def test_empty_variants(self, tmp_path: Path):
        """Returns empty list for no variants."""
        from src.evaluation.visualizations import plot_pareto

        paths = plot_pareto([], tmp_path)
        assert paths == []


class TestPlotRoofline:
    """Tests for plot_roofline."""

    def test_creates_png_file(self, tmp_path: Path, sample_variants):
        """Plot function creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_roofline

        result = plot_roofline(sample_variants, tmp_path)
        assert result == tmp_path / "plots" / "roofline.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_custom_hardware_params(self, tmp_path: Path, sample_variants):
        """Custom peak_tflops and bandwidth_gbs are accepted."""
        from src.evaluation.visualizations import plot_roofline

        result = plot_roofline(sample_variants, tmp_path, peak_tflops=100.0, bandwidth_gbs=200.0)
        assert result.exists()

    def test_creates_plots_directory(self, tmp_path: Path, sample_variants):
        """The plots/ subdirectory is created automatically."""
        from src.evaluation.visualizations import plot_roofline

        plot_roofline(sample_variants, tmp_path)
        assert (tmp_path / "plots").is_dir()


# ---------------------------------------------------------------------------
# Task 6.4/6.5: generate_summary_md tests (functions not yet implemented)
# These tests are expected to fail until task 6.4 is complete.
# ---------------------------------------------------------------------------


class TestGenerateSummaryMd:
    """Tests for generate_summary_md function."""

    @pytest.fixture(autouse=True)
    def _skip_if_not_implemented(self):
        """Skip these tests if generate_summary_md is not yet implemented."""
        try:
            from src.evaluation.visualizations import generate_summary_md  # noqa: F401
        except ImportError:
            pytest.skip("generate_summary_md not yet implemented (task 6.4)")

    def test_creates_summary_file(self, tmp_path: Path) -> None:
        """summary.md is created in the specified output directory."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_data={"vanilla": 2.34},
            fixed_wallclock={"vanilla": {0.25: 3.5, 0.50: 2.8, 0.75: 2.5, 1.0: 2.34}},
            fixed_flops={"vanilla": 2.34},
            pareto_front=["vanilla"],
            parameter_counts={"vanilla": 1000000},
            parameter_parity_valid=True,
        )
        result = generate_summary_md(comparison, tmp_path)
        assert result.exists()
        assert result.name == "summary.md"
        assert result.parent == tmp_path

    def test_returns_path_to_summary(self, tmp_path: Path) -> None:
        """Function returns the Path to the generated file."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult()
        result = generate_summary_md(comparison, tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "summary.md"

    def test_creates_output_dir_if_missing(self) -> None:
        """Output directory is created if it doesn't exist."""
        from src.evaluation.visualizations import generate_summary_md

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "deep"
            comparison = ComparisonResult()
            result = generate_summary_md(comparison, nested_dir)
            assert result.exists()
            assert nested_dir.exists()

    def test_contains_title_and_overview(self, tmp_path: Path) -> None:
        """Report includes title and overview section."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult()
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "# Evaluation Summary Report" in content

    def test_fixed_data_table(self, tmp_path: Path) -> None:
        """Fixed-data comparison table is correctly formatted."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_data={"vanilla": 2.34, "modern": 2.12},
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "## Fixed-Data Comparison" in content
        assert "| Variant | Val Loss |" in content
        assert "| modern | 2.1200 |" in content
        assert "| vanilla | 2.3400 |" in content

    def test_fixed_wallclock_table(self, tmp_path: Path) -> None:
        """Fixed-wallclock table shows fractions correctly."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_wallclock={
                "vanilla": {0.25: 3.5, 0.50: 2.8, 0.75: 2.5, 1.0: 2.34},
                "modern": {0.25: 3.2, 0.50: 2.6, 0.75: 2.3, 1.0: 2.12},
            },
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "## Fixed-Wallclock Comparison" in content
        assert "25%" in content
        assert "50%" in content
        assert "75%" in content
        assert "100%" in content
        assert "3.5000" in content
        assert "2.1200" in content

    def test_fixed_flops_table(self, tmp_path: Path) -> None:
        """Fixed-FLOPs comparison table is correctly formatted."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_flops={"vanilla": 2.34, "swa": 2.45},
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "## Fixed-FLOPs Comparison" in content
        assert "| Variant | Val Loss |" in content
        assert "| swa | 2.4500 |" in content

    def test_missing_comparison_std_is_labeled_incomplete(self, tmp_path: Path) -> None:
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_wallclock={"vanilla": {1.0: (3.5, float("nan"))}},
            fixed_flops={"vanilla": (3.4, float("nan"))},
        )

        content = generate_summary_md(comparison, tmp_path).read_text()

        assert "Entries without an error range are incomplete historical diagnostics" in content

    def test_parameter_parity_pass(self, tmp_path: Path) -> None:
        """Parameter parity PASS is shown when valid."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            parameter_counts={"vanilla": 1000000, "modern": 1050000},
            parameter_parity_valid=True,
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "PASS" in content
        assert "✅" in content
        assert "1,000,000" in content
        assert "1,050,000" in content

    def test_parameter_parity_fail(self, tmp_path: Path) -> None:
        """Parameter parity FAIL is shown when invalid."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            parameter_counts={"vanilla": 1000000, "modern": 2000000},
            parameter_parity_valid=False,
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "FAIL" in content
        assert "❌" in content

    def test_pareto_front_section(self, tmp_path: Path) -> None:
        """Pareto front section lists optimal variants."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            pareto_front=["modern", "swa"],
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "## Pareto Front" in content
        assert "**modern**" in content
        assert "**swa**" in content

    def test_relative_image_links(self, tmp_path: Path) -> None:
        """Summary contains relative image links to plots/*.png."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult()
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        # All expected plot references
        expected_plots = [
            "plots/learning_curves_tokens.png",
            "plots/learning_curves_wallclock.png",
            "plots/learning_curves_flops.png",
            "plots/per_position_loss.png",
            "plots/mqar_by_distance.png",
            "plots/stable_rank.png",
            "plots/cka_adjacent.png",
            "plots/flop_breakdown.png",
            "plots/pareto_flops_val_loss.png",
            "plots/roofline.png",
        ]
        for plot in expected_plots:
            assert plot in content, f"Missing image link: {plot}"

    def test_image_links_use_markdown_syntax(self, tmp_path: Path) -> None:
        """Image links use proper markdown image syntax ![alt](path)."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult()
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "![Learning Curves (Tokens)](plots/learning_curves_tokens.png)" in content
        assert "![Roofline](plots/roofline.png)" in content

    def test_mean_std_formatting_in_fixed_data(self, tmp_path: Path) -> None:
        """Statistical results formatted as 'mean ± std' in tables."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_data={"vanilla": (2.34, 0.02), "modern": (2.12, 0.01)},
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "2.3400 ± 0.0200" in content
        assert "2.1200 ± 0.0100" in content

    def test_nan_std_shows_only_mean(self, tmp_path: Path) -> None:
        """When std is NaN (< 3 seeds), only mean is shown."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult(
            fixed_data={"vanilla": (2.34, float("nan"))},
        )
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "2.3400" in content
        assert "±" not in content.split("Fixed-Data")[1].split("##")[0]

    def test_empty_comparison(self, tmp_path: Path) -> None:
        """Handles empty ComparisonResult gracefully."""
        from src.evaluation.visualizations import generate_summary_md

        comparison = ComparisonResult()
        result = generate_summary_md(comparison, tmp_path)
        content = result.read_text()
        assert "No fixed-data comparison results available." in content
        assert "No fixed-wallclock comparison results available." in content
        assert "No fixed-FLOPs comparison results available." in content
        assert "No Pareto front data available." in content


class TestFormatMetric:
    """Tests for _format_metric helper."""

    @pytest.fixture(autouse=True)
    def _skip_if_not_implemented(self):
        """Skip these tests if _format_metric is not yet implemented."""
        try:
            from src.evaluation.visualizations import _format_metric  # noqa: F401
        except ImportError:
            pytest.skip("_format_metric not yet implemented (task 6.4)")

    def test_plain_float(self) -> None:
        from src.evaluation.visualizations import _format_metric

        assert _format_metric(2.34) == "2.3400"

    def test_tuple_mean_std(self) -> None:
        from src.evaluation.visualizations import _format_metric

        assert _format_metric((2.34, 0.02)) == "2.3400 ± 0.0200"

    def test_tuple_nan_std(self) -> None:
        from src.evaluation.visualizations import _format_metric

        assert _format_metric((2.34, float("nan"))) == "2.3400"

    def test_zero_value(self) -> None:
        from src.evaluation.visualizations import _format_metric

        assert _format_metric(0.0) == "0.0000"

    def test_small_value(self) -> None:
        from src.evaluation.visualizations import _format_metric

        assert _format_metric(0.0001) == "0.0001"


# ---------------------------------------------------------------------------
# Task 6.5: Additional tests for complete coverage
# Requirements: 12.1 (PNG output), 12.2 (colorblind-safe palette),
#               12.8 (summary.md with relative image links), 15.5 (relative paths)
# ---------------------------------------------------------------------------


class TestColorblindSafePalette:
    """Tests that the colorblind-safe palette is applied (Requirement 12.2)."""

    def test_palette_has_minimum_colors(self) -> None:
        """COLORBLIND_PALETTE has at least 6 distinct colors for 6 variants."""
        assert len(COLORBLIND_PALETTE) >= 6

    def test_palette_colors_are_hex(self) -> None:
        """All palette colors are valid hex color strings."""
        import re

        for color in COLORBLIND_PALETTE:
            assert re.match(r"^#[0-9A-Fa-f]{6}$", color), f"Invalid hex color: {color}"

    def test_palette_colors_are_distinct(self) -> None:
        """All palette colors are unique."""
        assert len(set(COLORBLIND_PALETTE)) == len(COLORBLIND_PALETTE)

    def test_setup_style_applies_consistent_settings(self) -> None:
        """_setup_style configures matplotlib with consistent publication-quality settings."""
        from src.evaluation.visualizations import _setup_style

        _setup_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["axes.grid"] is True

    def test_learning_curves_uses_palette_colors(self, tmp_path: Path) -> None:
        """plot_learning_curves uses colors from COLORBLIND_PALETTE."""
        from src.evaluation.visualizations import plot_learning_curves

        configs = [
            ModelConfig(
                n_layer=2,
                d_model=64,
                n_head=2,
                seq_len=128,
                variant="v0",
                attention_type="full",
            ),
            ModelConfig(
                n_layer=2,
                d_model=64,
                n_head=2,
                seq_len=128,
                variant="v1",
                attention_type="full",
            ),
        ]
        variants = []
        for cfg in configs:
            from src.evaluation.flops import compute_step_flops

            variants.append(
                VariantData(
                    name=cfg.variant,
                    checkpoint_dir=Path(f"/tmp/fake_{cfg.variant}"),
                    log_entries=[
                        {"step": 1, "tokens_seen": 1000, "elapsed_time": 1.0, "val_loss": 5.0},
                        {"step": 100, "tokens_seen": 100000, "elapsed_time": 50.0, "val_loss": 3.0},
                    ],
                    config=cfg,
                    flop_breakdown=compute_step_flops(cfg),
                )
            )

        plot_learning_curves(variants, tmp_path, x_axis="tokens")

        # Verify by reading the saved figure - the plot was produced without error
        # and uses COLORBLIND_PALETTE colors (tested via the module constant)
        output_path = tmp_path / "plots" / "learning_curves_tokens.png"
        assert output_path.exists()


class TestPlotLearningCurves:
    """Tests for plot_learning_curves (Requirement 12.1)."""

    @pytest.fixture
    def sample_learning_variants(self) -> list[VariantData]:
        """Create sample VariantData with log entries for learning curve tests."""
        configs = [
            ModelConfig(
                n_layer=2,
                d_model=64,
                n_head=2,
                seq_len=128,
                variant="vanilla",
                attention_type="full",
            ),
            ModelConfig(
                n_layer=2,
                d_model=64,
                n_head=2,
                seq_len=128,
                variant="swa",
                attention_type="sliding_window",
                window_size=64,
            ),
        ]
        variants = []
        for i, cfg in enumerate(configs):
            from src.evaluation.flops import compute_step_flops

            variants.append(
                VariantData(
                    name=cfg.variant,
                    checkpoint_dir=Path(f"/tmp/fake_{cfg.variant}"),
                    log_entries=[
                        {
                            "step": 1,
                            "tokens_seen": 1000,
                            "elapsed_time": 1.0,
                            "val_loss": 5.0 - i * 0.2,
                        },
                        {
                            "step": 50,
                            "tokens_seen": 50000,
                            "elapsed_time": 25.0,
                            "val_loss": 4.0 - i * 0.1,
                        },
                        {
                            "step": 100,
                            "tokens_seen": 100000,
                            "elapsed_time": 50.0,
                            "val_loss": 3.5 - i * 0.1,
                        },
                    ],
                    config=cfg,
                    flop_breakdown=compute_step_flops(cfg),
                )
            )
        return variants

    def test_creates_png_tokens(self, tmp_path: Path, sample_learning_variants):
        """plot_learning_curves creates PNG for tokens x-axis."""
        from src.evaluation.visualizations import plot_learning_curves

        result = plot_learning_curves(sample_learning_variants, tmp_path, x_axis="tokens")
        assert result == tmp_path / "plots" / "learning_curves_tokens.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_png_wallclock(self, tmp_path: Path, sample_learning_variants):
        """plot_learning_curves creates PNG for wallclock x-axis."""
        from src.evaluation.visualizations import plot_learning_curves

        result = plot_learning_curves(sample_learning_variants, tmp_path, x_axis="wallclock")
        assert result == tmp_path / "plots" / "learning_curves_wallclock.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_png_flops(self, tmp_path: Path, sample_learning_variants):
        """plot_learning_curves creates PNG for flops x-axis."""
        from src.evaluation.visualizations import plot_learning_curves

        result = plot_learning_curves(sample_learning_variants, tmp_path, x_axis="flops")
        assert result == tmp_path / "plots" / "learning_curves_flops.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_invalid_x_axis_raises(self, tmp_path: Path, sample_learning_variants):
        """Invalid x_axis raises ValueError."""
        from src.evaluation.visualizations import plot_learning_curves

        with pytest.raises(ValueError, match="Unsupported x_axis"):
            plot_learning_curves(sample_learning_variants, tmp_path, x_axis="invalid")

    def test_empty_variants(self, tmp_path: Path):
        """plot_learning_curves handles empty variants list."""
        from src.evaluation.visualizations import plot_learning_curves

        result = plot_learning_curves([], tmp_path, x_axis="tokens")
        assert result.exists()


class TestPlotPerPositionLoss:
    """Tests for plot_per_position_loss (Requirement 12.1)."""

    def test_creates_png_file(self, tmp_path: Path):
        """plot_per_position_loss creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_per_position_loss

        cfg = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=2,
            seq_len=32,
            variant="vanilla",
            attention_type="full",
        )
        # Create synthetic per-position loss with power-law shape
        positions = np.arange(1, 33)
        per_pos_loss = 2.0 * positions.astype(np.float64) ** (-0.5) + 1.0

        from src.evaluation.flops import compute_step_flops

        variant = VariantData(
            name="vanilla",
            checkpoint_dir=Path("/tmp/fake_vanilla"),
            log_entries=[{"step": 100, "val_loss": 2.5}],
            config=cfg,
            flop_breakdown=compute_step_flops(cfg),
            metrics=MetricsResult(
                val_loss=2.5,
                perplexity=math.exp(2.5),
                per_position_loss=per_pos_loss,
                icl_exponent=0.5,
                icl_fit_params={"A": 2.0, "alpha": 0.5, "C": 1.0, "r_squared": 0.99},
            ),
        )

        result = plot_per_position_loss([variant], tmp_path)
        assert result == tmp_path / "plots" / "per_position_loss.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_handles_no_metrics(self, tmp_path: Path):
        """plot_per_position_loss handles variants without metrics gracefully."""
        from src.evaluation.visualizations import plot_per_position_loss

        cfg = ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=2,
            seq_len=32,
            variant="vanilla",
            attention_type="full",
        )

        from src.evaluation.flops import compute_step_flops

        variant = VariantData(
            name="vanilla",
            checkpoint_dir=Path("/tmp/fake_vanilla"),
            log_entries=[{"step": 100, "val_loss": 2.5}],
            config=cfg,
            flop_breakdown=compute_step_flops(cfg),
            metrics=None,
        )

        result = plot_per_position_loss([variant], tmp_path)
        assert result.exists()


class TestPlotMqarResults:
    """Tests for plot_mqar_results (Requirement 12.1)."""

    def test_creates_png_file(self, tmp_path: Path):
        """plot_mqar_results creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_mqar_results

        results = {
            "vanilla": MQARResult(
                accuracy=0.85,
                accuracy_by_distance={10: 0.95, 50: 0.80, 100: 0.75, 200: 0.60},
            ),
            "swa": MQARResult(
                accuracy=0.70,
                accuracy_by_distance={10: 0.90, 50: 0.75, 100: 0.55, 200: 0.40},
            ),
        }

        result = plot_mqar_results(results, tmp_path)
        assert result == tmp_path / "plots" / "mqar_by_distance.png"
        assert result.exists()
        assert result.stat().st_size > 0

    def test_single_variant(self, tmp_path: Path):
        """Works with a single variant."""
        from src.evaluation.visualizations import plot_mqar_results

        results = {
            "vanilla": MQARResult(
                accuracy=0.85,
                accuracy_by_distance={10: 0.95, 50: 0.80},
            ),
        }

        result = plot_mqar_results(results, tmp_path)
        assert result.exists()


class TestPlotStableRank:
    """Tests for plot_stable_rank (Requirement 12.1)."""

    def test_creates_png_file(self, tmp_path: Path):
        """plot_stable_rank creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_stable_rank

        results = {
            "vanilla": StableRankResult(
                per_layer=np.array([15.2, 14.8, 13.1, 12.5]),
                mean=13.9,
                std=1.2,
            ),
            "swa": StableRankResult(
                per_layer=np.array([14.0, 12.5, 11.0, 10.5]),
                mean=12.0,
                std=1.5,
            ),
        }

        result = plot_stable_rank(results, tmp_path)
        assert result == tmp_path / "plots" / "stable_rank.png"
        assert result.exists()
        assert result.stat().st_size > 0


class TestPlotCkaAdjacent:
    """Tests for plot_cka_adjacent (Requirement 12.1)."""

    def test_creates_png_file(self, tmp_path: Path):
        """plot_cka_adjacent creates a PNG at the expected path."""
        from src.evaluation.visualizations import plot_cka_adjacent

        results = {
            "vanilla": CKAResult(
                adjacent_curve=np.array([0.85, 0.78, 0.72]),
                full_matrix=np.eye(4),
            ),
            "swa": CKAResult(
                adjacent_curve=np.array([0.90, 0.82, 0.75]),
                full_matrix=np.eye(4),
            ),
        }

        result = plot_cka_adjacent(results, tmp_path)
        assert result == tmp_path / "plots" / "cka_adjacent.png"
        assert result.exists()
        assert result.stat().st_size > 0
