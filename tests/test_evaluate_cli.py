"""Tests for the evaluation pipeline module — task 7.2 functionality.

Covers multi-seed detection, aggregation, metadata.json writing,
and raw data (CSV/JSON) output.
"""

import csv
import json
import math

import pytest

from src.evaluation.comparison import ComparisonResult, VariantData
from src.evaluation.flops import FLOPBreakdown
from src.evaluation.pipeline import EvaluationPipeline
from src.models.config import ModelConfig

# ---------------------------------------------------------------------------
# Helpers — thin wrappers exposing pipeline internals for testing
# ---------------------------------------------------------------------------


def _detect_seed_groups(variants):
    """Expose the static method for test compatibility."""
    return EvaluationPipeline._detect_seed_groups(variants)


def _aggregate_seed_metrics(seed_groups):
    """Run seed aggregation via the pipeline's internal logic."""
    pipeline = EvaluationPipeline(device="cpu")
    return pipeline._aggregate_seeds(seed_groups, [], [])


def _write_metadata(output_dir, variants, device):
    """Run metadata writing via the pipeline."""
    pipeline = EvaluationPipeline(device=device)
    return pipeline._write_metadata(output_dir, variants)


def _write_raw_data(output_dir, variants, comparison, seed_aggregated):
    """Run raw data writing via the pipeline."""
    pipeline = EvaluationPipeline(device="cpu")
    return pipeline._write_raw_data(output_dir, variants, comparison, seed_aggregated)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """A debug-scale model config for testing."""
    return ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=4,
        seq_len=128,
        variant="vanilla",
    )


@pytest.fixture
def sample_flop_breakdown():
    """A simple FLOPBreakdown for testing."""
    return FLOPBreakdown(
        qkv_proj=100_000,
        attention_score=50_000,
        attention_output=30_000,
        ffn=200_000,
        total=380_000,
    )


@pytest.fixture
def sample_log_entries():
    """Synthetic metrics.jsonl entries for testing."""
    return [
        {
            "step": 1,
            "train_loss": 8.0,
            "val_loss": 7.5,
            "tokens_seen": 1024,
            "elapsed_time": 1.0,
            "peak_memory_mb": 512.0,
        },
        {
            "step": 10,
            "train_loss": 6.0,
            "val_loss": 5.5,
            "tokens_seen": 10240,
            "elapsed_time": 10.0,
            "peak_memory_mb": 520.0,
        },
        {
            "step": 50,
            "train_loss": 4.0,
            "val_loss": 3.8,
            "tokens_seen": 51200,
            "elapsed_time": 50.0,
            "peak_memory_mb": 530.0,
        },
        {
            "step": 100,
            "train_loss": 3.5,
            "val_loss": 3.2,
            "tokens_seen": 102400,
            "elapsed_time": 100.0,
            "peak_memory_mb": 540.0,
        },
    ]


@pytest.fixture
def make_variant(sample_config, sample_flop_breakdown, sample_log_entries, tmp_path):
    """Factory for creating VariantData with customizable fields."""

    def _make(name="vanilla", seed=0, val_loss_offset=0.0):
        checkpoint_dir = tmp_path / f"{name}_main_s{42 + seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Offset val_loss for different seeds to make them distinguishable
        entries = []
        for e in sample_log_entries:
            entry = dict(e)
            if entry.get("val_loss") is not None:
                entry["val_loss"] = entry["val_loss"] + val_loss_offset
            entries.append(entry)

        return VariantData(
            name=name,
            checkpoint_dir=checkpoint_dir,
            log_entries=entries,
            config=sample_config,
            flop_breakdown=sample_flop_breakdown,
        )

    return _make


# ---------------------------------------------------------------------------
# Tests: _detect_seed_groups
# ---------------------------------------------------------------------------


class TestDetectSeedGroups:
    """Tests for multi-seed detection logic."""

    def test_single_seed_per_variant(self, make_variant):
        """Each variant with one checkpoint becomes a group of size 1."""
        variants = [make_variant("vanilla", seed=0), make_variant("modern", seed=0)]
        groups = _detect_seed_groups(variants)

        assert len(groups) == 2
        assert "vanilla" in groups
        assert "modern" in groups
        assert len(groups["vanilla"]) == 1
        assert len(groups["modern"]) == 1

    def test_multiple_seeds_detected(self, make_variant):
        """Multiple checkpoints for the same variant are grouped together."""
        variants = [
            make_variant("vanilla", seed=0),
            make_variant("vanilla", seed=1),
            make_variant("vanilla", seed=2),
            make_variant("modern", seed=0),
        ]
        groups = _detect_seed_groups(variants)

        assert len(groups) == 2
        assert len(groups["vanilla"]) == 3
        assert len(groups["modern"]) == 1

    def test_empty_list(self):
        """Empty input returns empty groups."""
        groups = _detect_seed_groups([])
        assert groups == {}


# ---------------------------------------------------------------------------
# Tests: _aggregate_seed_metrics
# ---------------------------------------------------------------------------


class TestAggregateSeedMetrics:
    """Tests for multi-seed metric aggregation."""

    def test_single_seed_nan_std(self, make_variant):
        """With only 1 seed, std should be NaN (insufficient confidence)."""
        variants = [make_variant("vanilla", seed=0)]
        groups = _detect_seed_groups(variants)
        aggregated = _aggregate_seed_metrics(groups)

        assert "vanilla" in aggregated
        metrics = aggregated["vanilla"]
        assert "val_loss" in metrics

        mean, std = metrics["val_loss"]
        assert mean == pytest.approx(3.2, abs=0.01)
        assert math.isnan(std)

    def test_three_seeds_computes_mean_std(self, make_variant):
        """With 3+ seeds, mean and std are computed correctly."""
        variants = [
            make_variant("vanilla", seed=0, val_loss_offset=0.0),
            make_variant("vanilla", seed=1, val_loss_offset=0.1),
            make_variant("vanilla", seed=2, val_loss_offset=-0.1),
        ]
        groups = _detect_seed_groups(variants)
        aggregated = _aggregate_seed_metrics(groups)

        assert "vanilla" in aggregated
        metrics = aggregated["vanilla"]
        # Final val_loss values: 3.2, 3.3, 3.1 → mean=3.2, std≈0.1
        mean, std = metrics["val_loss"]
        assert mean == pytest.approx(3.2, abs=0.01)
        assert std == pytest.approx(0.1, abs=0.01)
        assert not math.isnan(std)

    def test_two_seeds_nan_std(self, make_variant):
        """With 2 seeds (< 3), std should be NaN."""
        variants = [
            make_variant("vanilla", seed=0, val_loss_offset=0.0),
            make_variant("vanilla", seed=1, val_loss_offset=0.2),
        ]
        groups = _detect_seed_groups(variants)
        aggregated = _aggregate_seed_metrics(groups)

        metrics = aggregated["vanilla"]
        _, std = metrics["val_loss"]
        assert math.isnan(std)

    def test_aggregates_multiple_metrics(self, make_variant):
        """Aggregation covers multiple metric types (val_loss, perplexity, etc.)."""
        variants = [
            make_variant("vanilla", seed=0, val_loss_offset=0.0),
            make_variant("vanilla", seed=1, val_loss_offset=0.1),
            make_variant("vanilla", seed=2, val_loss_offset=-0.1),
        ]
        groups = _detect_seed_groups(variants)
        aggregated = _aggregate_seed_metrics(groups)

        metrics = aggregated["vanilla"]
        assert "perplexity" in metrics
        assert "step_flops" in metrics


# ---------------------------------------------------------------------------
# Tests: _write_metadata
# ---------------------------------------------------------------------------


class TestWriteMetadata:
    """Tests for metadata.json generation."""

    def test_writes_valid_json(self, make_variant, tmp_path):
        """metadata.json is valid JSON with required fields."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        path = _write_metadata(output_dir, variants, "cpu")

        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "software_versions" in data
        assert "hardware" in data
        assert data["warnings"] == []
        assert "evaluated_checkpoints" in data

    def test_software_versions_present(self, make_variant, tmp_path):
        """Software versions include python, torch, numpy."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        _write_metadata(output_dir, variants, "cpu")
        with open(output_dir / "metadata.json") as f:
            data = json.load(f)

        versions = data["software_versions"]
        assert "python" in versions
        assert "torch" in versions
        assert "numpy" in versions

    def test_hardware_cpu(self, make_variant, tmp_path):
        """Hardware field shows 'cpu' when device is cpu."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        _write_metadata(output_dir, variants, "cpu")
        with open(output_dir / "metadata.json") as f:
            data = json.load(f)

        assert data["hardware"] == "cpu"

    def test_evaluated_checkpoints_listed(self, make_variant, tmp_path):
        """All checkpoint paths are listed in evaluated_checkpoints."""
        variants = [
            make_variant("vanilla", seed=0),
            make_variant("modern", seed=0),
        ]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        _write_metadata(output_dir, variants, "cpu")
        with open(output_dir / "metadata.json") as f:
            data = json.load(f)

        assert len(data["evaluated_checkpoints"]) == 2

    def test_warnings_are_persisted(self, make_variant, tmp_path):
        output_dir = tmp_path / "report"
        output_dir.mkdir()
        pipeline = EvaluationPipeline(device="cpu")
        pipeline._write_metadata(
            output_dir,
            [make_variant("vanilla", seed=0)],
            ["duplicate seed histories detected"],
        )

        data = json.loads((output_dir / "metadata.json").read_text())
        assert data["warnings"] == ["duplicate seed histories detected"]


# ---------------------------------------------------------------------------
# Tests: _write_raw_data
# ---------------------------------------------------------------------------


class TestWriteRawData:
    """Tests for raw/metrics.csv and raw/metrics.json generation."""

    def test_creates_csv_and_json(self, make_variant, tmp_path):
        """Both metrics.csv and metrics.json are created."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        comparison = ComparisonResult(
            fixed_data={"vanilla": 3.2},
            fixed_wallclock={"vanilla": {0.5: 5.5, 1.0: 3.2}},
            fixed_flops={"vanilla": 3.2},
            pareto_front=["vanilla"],
            parameter_counts={"vanilla": 100000},
            parameter_parity_valid=True,
        )
        seed_aggregated = {"vanilla": {"val_loss": (3.2, float("nan"))}}

        csv_path, json_path = _write_raw_data(output_dir, variants, comparison, seed_aggregated)

        assert csv_path.exists()
        assert json_path.exists()
        assert csv_path.parent.name == "raw"
        assert json_path.parent.name == "raw"

    def test_csv_has_correct_columns(self, make_variant, tmp_path):
        """CSV includes variant, seed_index, checkpoint_dir, and metric columns."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        comparison = ComparisonResult()
        seed_aggregated = {}

        csv_path, _ = _write_raw_data(output_dir, variants, comparison, seed_aggregated)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        assert "variant" in fieldnames
        assert "seed_index" in fieldnames
        assert "checkpoint_dir" in fieldnames
        assert len(rows) == 1
        assert rows[0]["variant"] == "vanilla"

    def test_csv_multi_seed_rows(self, make_variant, tmp_path):
        """CSV has one row per seed per variant."""
        variants = [
            make_variant("vanilla", seed=0),
            make_variant("vanilla", seed=1),
            make_variant("modern", seed=0),
        ]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        comparison = ComparisonResult()
        seed_aggregated = {}

        csv_path, _ = _write_raw_data(output_dir, variants, comparison, seed_aggregated)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        vanilla_rows = [r for r in rows if r["variant"] == "vanilla"]
        assert len(vanilla_rows) == 2

    def test_json_structure(self, make_variant, tmp_path):
        """JSON output has variants, aggregated, and comparison sections."""
        variants = [
            make_variant("vanilla", seed=0),
            make_variant("vanilla", seed=1),
            make_variant("vanilla", seed=2),
        ]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        comparison = ComparisonResult(
            fixed_data={"vanilla": 3.2},
            pareto_front=["vanilla"],
            parameter_counts={"vanilla": 100000},
            parameter_parity_valid=True,
        )
        seed_aggregated = {"vanilla": {"val_loss": (3.2, 0.1)}}

        _, json_path = _write_raw_data(output_dir, variants, comparison, seed_aggregated)

        with open(json_path) as f:
            data = json.load(f)

        assert "variants" in data
        assert "aggregated" in data
        assert "comparison" in data

        # Check variants section
        assert "vanilla" in data["variants"]
        assert len(data["variants"]["vanilla"]) == 3

        # Check aggregated section
        assert "vanilla" in data["aggregated"]
        assert data["aggregated"]["vanilla"]["val_loss"]["mean"] == pytest.approx(3.2)
        assert data["aggregated"]["vanilla"]["val_loss"]["std"] == pytest.approx(0.1)

        # Check comparison section
        assert data["schema_version"] == 2
        assert data["comparison"]["fixed_data"] == {"vanilla": {"mean": 3.2, "std": None, "n": 3}}
        assert data["comparison"]["pareto_front"] == ["vanilla"]

    def test_json_nan_std_is_null(self, make_variant, tmp_path):
        """NaN std values (< 3 seeds) are serialized as null in JSON."""
        variants = [make_variant("vanilla", seed=0)]
        output_dir = tmp_path / "report"
        output_dir.mkdir()

        comparison = ComparisonResult()
        seed_aggregated = {"vanilla": {"val_loss": (3.2, float("nan"))}}

        _, json_path = _write_raw_data(output_dir, variants, comparison, seed_aggregated)

        with open(json_path) as f:
            data = json.load(f)

        assert data["aggregated"]["vanilla"]["val_loss"]["std"] is None
