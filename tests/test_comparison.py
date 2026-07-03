"""Unit tests for the comparison module's data loading and slicing functionality."""

import json
from pathlib import Path

import pytest

from src.evaluation.comparison import (
    ComparisonResult,
    VariantData,
    _infer_variant_name,
    load_variant_data,
    slice_fixed_data,
    slice_fixed_wallclock,
)
from src.models.config import ModelConfig


@pytest.fixture
def valid_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a valid checkpoint directory with metrics.jsonl and run_config.json."""
    run_dir = tmp_path / "vanilla_debug_20240101_1200"
    run_dir.mkdir()

    # Write metrics.jsonl
    metrics = [
        {"type": "train", "step": 1, "train_loss": 5.0, "tokens_processed": 4096, "elapsed_seconds": 1.0},
        {"type": "eval", "step": 100, "val_loss": 4.5, "val_perplexity": 90.0},
        {"type": "train", "step": 200, "train_loss": 3.5, "tokens_processed": 819200, "elapsed_seconds": 100.0},
        {"type": "eval", "step": 200, "val_loss": 3.8, "val_perplexity": 44.7},
    ]
    with open(run_dir / "metrics.jsonl", "w") as f:
        for entry in metrics:
            f.write(json.dumps(entry) + "\n")

    # Write run_config.json with model config
    config = {
        "variant": "vanilla",
        "scale": "debug",
        "model": {
            "n_layer": 4,
            "d_model": 256,
            "n_head": 4,
            "vocab_size": 50257,
            "seq_len": 512,
            "ffn_multiplier": 4,
            "dropout": 0.0,
            "bias": False,
            "tie_embeddings": True,
            "activation": "relu",
            "variant": "vanilla",
            "norm_type": "layernorm",
            "position_encoding": "learned",
            "ffn_type": "standard",
            "attention_type": "full",
        },
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f)

    return run_dir


@pytest.fixture
def modern_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a checkpoint dir for the modern variant."""
    run_dir = tmp_path / "modern_main_20240102_1430"
    run_dir.mkdir()

    metrics = [
        {"type": "train", "step": 1, "train_loss": 4.8, "tokens_processed": 8192, "elapsed_seconds": 2.0},
        {"type": "eval", "step": 100, "val_loss": 4.0, "val_perplexity": 54.6},
    ]
    with open(run_dir / "metrics.jsonl", "w") as f:
        for entry in metrics:
            f.write(json.dumps(entry) + "\n")

    config = {
        "model": {
            "n_layer": 8,
            "d_model": 512,
            "n_head": 8,
            "vocab_size": 50257,
            "seq_len": 1024,
            "ffn_multiplier": 4,
            "dropout": 0.0,
            "bias": False,
            "tie_embeddings": True,
            "activation": "swiglu",
            "variant": "modern",
            "norm_type": "rmsnorm",
            "position_encoding": "rope",
            "ffn_type": "swiglu",
            "attention_type": "flash_sdpa",
        },
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f)

    return run_dir


class TestVariantData:
    """Tests for the VariantData dataclass."""

    def test_fields(self):
        config = ModelConfig()
        vd = VariantData(
            name="vanilla",
            checkpoint_dir=Path("/tmp/test"),
            log_entries=[{"step": 1}],
            config=config,
        )
        assert vd.name == "vanilla"
        assert vd.checkpoint_dir == Path("/tmp/test")
        assert vd.log_entries == [{"step": 1}]
        assert vd.config == config
        assert vd.metrics is None
        assert vd.flop_breakdown is None

    def test_optional_fields(self):
        config = ModelConfig()
        vd = VariantData(
            name="test",
            checkpoint_dir=Path("/tmp"),
            log_entries=[],
            config=config,
            metrics=None,
            flop_breakdown=None,
        )
        assert vd.metrics is None
        assert vd.flop_breakdown is None


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_default_fields(self):
        result = ComparisonResult()
        assert result.fixed_data == {}
        assert result.fixed_wallclock == {}
        assert result.fixed_flops == {}
        assert result.pareto_front == []
        assert result.parameter_counts == {}
        assert result.parameter_parity_valid is False


class TestInferVariantName:
    """Tests for variant name inference from directory names."""

    def test_vanilla_main_seed(self):
        assert _infer_variant_name(Path("vanilla_main_s42")) == "vanilla"

    def test_modern_main_timestamp(self):
        assert _infer_variant_name(Path("modern_main_20240101_1200")) == "modern"

    def test_swa_interleaved(self):
        assert _infer_variant_name(Path("swa_interleaved_debug_20240101")) == "swa_interleaved"

    def test_swa_variant(self):
        assert _infer_variant_name(Path("swa_debug_s1")) == "swa"

    def test_linear_variant(self):
        assert _infer_variant_name(Path("linear_main_s99")) == "linear"

    def test_alibi_variant(self):
        assert _infer_variant_name(Path("alibi_stretch_20240501_0900")) == "alibi"

    def test_unknown_falls_back_to_dir_name(self):
        assert _infer_variant_name(Path("something_weird")) == "something_weird"


class TestLoadVariantData:
    """Tests for load_variant_data function."""

    def test_loads_valid_checkpoint(self, valid_checkpoint_dir: Path):
        variants = load_variant_data([valid_checkpoint_dir])
        assert len(variants) == 1
        vd = variants[0]
        assert vd.name == "vanilla"
        assert vd.checkpoint_dir == valid_checkpoint_dir
        assert len(vd.log_entries) == 4
        assert vd.config.variant == "vanilla"
        assert vd.config.n_layer == 4
        assert vd.config.d_model == 256
        assert vd.flop_breakdown is not None
        assert vd.flop_breakdown.total > 0

    def test_loads_multiple_checkpoints(
        self, valid_checkpoint_dir: Path, modern_checkpoint_dir: Path
    ):
        variants = load_variant_data([valid_checkpoint_dir, modern_checkpoint_dir])
        assert len(variants) == 2
        names = {v.name for v in variants}
        assert "vanilla" in names
        assert "modern" in names

    def test_skips_missing_directory(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        variants = load_variant_data([missing])
        assert len(variants) == 0

    def test_skips_directory_without_metrics(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_variant_debug_s1"
        empty_dir.mkdir()
        variants = load_variant_data([empty_dir])
        assert len(variants) == 0

    def test_handles_corrupt_metrics(self, tmp_path: Path):
        run_dir = tmp_path / "vanilla_debug_s1"
        run_dir.mkdir()
        # Write corrupt metrics.jsonl
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write("not valid json\n")
        variants = load_variant_data([run_dir])
        assert len(variants) == 0

    def test_infers_config_from_variant_name(self, tmp_path: Path):
        """When no config file exists, should infer from variant name."""
        run_dir = tmp_path / "vanilla_debug_s1"
        run_dir.mkdir()
        # Write valid metrics but no config
        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"step": 1, "train_loss": 5.0}) + "\n")
        variants = load_variant_data([run_dir])
        assert len(variants) == 1
        assert variants[0].name == "vanilla"
        assert variants[0].config.variant == "vanilla"

    def test_skips_non_directory_path(self, tmp_path: Path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        variants = load_variant_data([file_path])
        assert len(variants) == 0

    def test_empty_input(self):
        variants = load_variant_data([])
        assert len(variants) == 0

    def test_config_from_config_json(self, tmp_path: Path):
        """Should load config from config.json when run_config.json is absent."""
        run_dir = tmp_path / "gqa_main_s1"
        run_dir.mkdir()

        with open(run_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"step": 1, "train_loss": 4.0}) + "\n")

        config_dict = {
            "n_layer": 8,
            "d_model": 512,
            "n_head": 8,
            "vocab_size": 50257,
            "seq_len": 1024,
            "variant": "gqa",
            "norm_type": "rmsnorm",
            "position_encoding": "rope",
            "ffn_type": "swiglu",
            "attention_type": "flash_gqa",
            "n_kv_head": 2,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_dict, f)

        variants = load_variant_data([run_dir])
        assert len(variants) == 1
        assert variants[0].name == "gqa"
        assert variants[0].config.n_kv_head == 2


# --- Helpers for slicing tests ---


def _make_variant_with_log(
    name: str, log_entries: list[dict], config: ModelConfig | None = None
) -> VariantData:
    """Create a VariantData with synthetic log entries for slicing tests."""
    return VariantData(
        name=name,
        checkpoint_dir=Path(f"/tmp/{name}"),
        log_entries=log_entries,
        config=config or ModelConfig(),
    )


# --- slice_fixed_data tests ---


class TestSliceFixedData:
    """Tests for slice_fixed_data function (Requirement 9.1, 9.5)."""

    def test_empty_variants_returns_empty(self):
        """Empty input returns empty dict."""
        result = slice_fixed_data([])
        assert result == {}

    def test_single_variant_exact_match(self):
        """Single variant with exact token budget match returns that val_loss."""
        log = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.5, "elapsed_time": 20.0},
        ]
        v = _make_variant_with_log("vanilla", log)
        result = slice_fixed_data([v], token_budget=100000)
        assert result == {"vanilla": 3.5}

    def test_single_variant_interpolation(self):
        """Token budget between two log points uses linear interpolation."""
        log = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 20.0},
        ]
        v = _make_variant_with_log("vanilla", log)
        # Budget at 75000 — midpoint between 50000 and 100000
        result = slice_fixed_data([v], token_budget=75000)
        # Linear interpolation: 4.0 + (3.0 - 4.0) * (75000 - 50000) / (100000 - 50000) = 3.5
        assert result["vanilla"] == pytest.approx(3.5)

    def test_auto_budget_uses_min_max_tokens(self):
        """When token_budget is None, uses min of max tokens_seen across variants."""
        log_a = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 20.0},
        ]
        log_b = [
            {"step": 50, "tokens_seen": 40000, "val_loss": 4.5, "elapsed_time": 8.0},
            {"step": 80, "tokens_seen": 80000, "val_loss": 3.2, "elapsed_time": 16.0},
        ]
        v_a = _make_variant_with_log("vanilla", log_a)
        v_b = _make_variant_with_log("modern", log_b)

        # Auto budget = min(100000, 80000) = 80000
        result = slice_fixed_data([v_a, v_b])

        # vanilla at 80000: interpolate between (50000, 4.0) and (100000, 3.0)
        # = 4.0 + (3.0 - 4.0) * (80000 - 50000) / (100000 - 50000) = 4.0 - 0.6 = 3.4
        assert "vanilla" in result
        assert result["vanilla"] == pytest.approx(3.4)

        # modern at 80000: exact match
        assert "modern" in result
        assert result["modern"] == pytest.approx(3.2)

    def test_variant_excluded_when_budget_exceeds_log(self):
        """Variant excluded with warning when it can't reach the token budget."""
        log_a = [
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 20.0},
        ]
        log_b = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
        ]
        v_a = _make_variant_with_log("vanilla", log_a)
        v_b = _make_variant_with_log("modern", log_b)

        # Budget exceeds modern's max tokens
        result = slice_fixed_data([v_a, v_b], token_budget=100000)
        assert "vanilla" in result
        assert "modern" not in result

    def test_two_variants_same_budget_different_losses(self):
        """Two variants at same budget returns correct losses for each."""
        log_a = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 2.5, "elapsed_time": 20.0},
        ]
        log_b = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.2, "elapsed_time": 12.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 24.0},
        ]
        v_a = _make_variant_with_log("vanilla", log_a)
        v_b = _make_variant_with_log("modern", log_b)

        result = slice_fixed_data([v_a, v_b], token_budget=100000)
        assert result["vanilla"] == pytest.approx(2.5)
        assert result["modern"] == pytest.approx(3.0)

    def test_entries_without_val_loss_are_skipped(self):
        """Log entries missing val_loss are skipped during interpolation."""
        log = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 10.0},
            {"step": 75, "tokens_seen": 75000, "elapsed_time": 15.0},  # No val_loss
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 20.0},
        ]
        v = _make_variant_with_log("vanilla", log)
        # Interpolation only uses entries with val_loss present
        result = slice_fixed_data([v], token_budget=75000)
        # Interpolates between (50000, 4.0) and (100000, 3.0) → 3.5
        assert result["vanilla"] == pytest.approx(3.5)


# --- slice_fixed_wallclock tests ---


class TestSliceFixedWallclock:
    """Tests for slice_fixed_wallclock function (Requirement 9.2, 9.3, 9.5)."""

    def test_empty_variants_returns_empty(self):
        """Empty input returns empty dict."""
        result = slice_fixed_wallclock([])
        assert result == {}

    def test_dynamic_budget_is_min_of_max_elapsed(self):
        """Dynamic budget = min(max elapsed_time) across variants."""
        # vanilla runs for 100s, modern runs for 80s → budget = 80s
        log_a = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 50.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 100.0},
        ]
        log_b = [
            {"step": 40, "tokens_seen": 40000, "val_loss": 4.2, "elapsed_time": 40.0},
            {"step": 80, "tokens_seen": 80000, "val_loss": 3.2, "elapsed_time": 80.0},
        ]
        v_a = _make_variant_with_log("vanilla", log_a)
        v_b = _make_variant_with_log("modern", log_b)

        result = slice_fixed_wallclock([v_a, v_b])

        # Dynamic budget = min(100, 80) = 80s
        # Check that fractions are based on 80s budget
        assert "vanilla" in result
        assert "modern" in result

        # At 1.00 fraction (80s): vanilla interpolates between (50, 4.0) and (100, 3.0)
        # = 4.0 + (3.0 - 4.0) * (80 - 50) / (100 - 50) = 4.0 - 0.6 = 3.4
        assert result["vanilla"][1.00] == pytest.approx(3.4)

        # At 1.00 fraction (80s): modern exact match at 80s → 3.2
        assert result["modern"][1.00] == pytest.approx(3.2)

    def test_default_fractions_are_used(self):
        """Default time fractions [0.25, 0.50, 0.75, 1.00] are returned."""
        log = [
            {"step": 25, "tokens_seen": 25000, "val_loss": 4.5, "elapsed_time": 25.0},
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 50.0},
            {"step": 75, "tokens_seen": 75000, "val_loss": 3.5, "elapsed_time": 75.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 100.0},
        ]
        v = _make_variant_with_log("vanilla", log)

        result = slice_fixed_wallclock([v])

        # Budget = 100s. Fractions at 25s, 50s, 75s, 100s
        assert set(result["vanilla"].keys()) == {0.25, 0.50, 0.75, 1.00}
        assert result["vanilla"][0.25] == pytest.approx(4.5)
        assert result["vanilla"][0.50] == pytest.approx(4.0)
        assert result["vanilla"][0.75] == pytest.approx(3.5)
        assert result["vanilla"][1.00] == pytest.approx(3.0)

    def test_custom_fractions(self):
        """Custom time fractions are respected."""
        log = [
            {"step": 50, "tokens_seen": 50000, "val_loss": 4.0, "elapsed_time": 50.0},
            {"step": 100, "tokens_seen": 100000, "val_loss": 3.0, "elapsed_time": 100.0},
        ]
        v = _make_variant_with_log("vanilla", log)

        result = slice_fixed_wallclock([v], time_fractions=[0.50, 1.00])

        # Budget = 100s. Fractions at 50s and 100s
        assert set(result["vanilla"].keys()) == {0.50, 1.00}
        assert result["vanilla"][0.50] == pytest.approx(4.0)
        assert result["vanilla"][1.00] == pytest.approx(3.0)

    def test_interpolates_between_log_points(self):
        """Val loss is interpolated when fraction doesn't exactly match a log point."""
        log = [
            {"step": 40, "tokens_seen": 40000, "val_loss": 4.0, "elapsed_time": 40.0},
            {"step": 80, "tokens_seen": 80000, "val_loss": 3.0, "elapsed_time": 80.0},
        ]
        v = _make_variant_with_log("vanilla", log)

        # Budget = 80s. At 0.75 fraction → 60s
        result = slice_fixed_wallclock([v], time_fractions=[0.75])

        # Interpolate at 60s between (40, 4.0) and (80, 3.0)
        # = 4.0 + (3.0 - 4.0) * (60 - 40) / (80 - 40) = 4.0 - 0.5 = 3.5
        assert result["vanilla"][0.75] == pytest.approx(3.5)

    def test_variant_excluded_at_unreachable_fraction(self):
        """Variant excluded from a fraction it can't reach (log starts late)."""
        log = [
            # Earliest data at 60s — can't reach 0.25 of 80s budget (= 20s)
            {"step": 60, "tokens_seen": 60000, "val_loss": 3.5, "elapsed_time": 60.0},
            {"step": 80, "tokens_seen": 80000, "val_loss": 3.0, "elapsed_time": 80.0},
        ]
        v = _make_variant_with_log("vanilla", log)

        result = slice_fixed_wallclock([v], time_fractions=[0.25, 1.00])

        # 0.25 * 80 = 20s is before the log starts (60s), so excluded
        assert 0.25 not in result["vanilla"]
        # 1.00 * 80 = 80s is reachable
        assert result["vanilla"][1.00] == pytest.approx(3.0)

    def test_multiple_variants_different_budgets(self):
        """With multiple variants, dynamic budget correctly selects minimum."""
        # V_short runs for 60s, V_long runs for 120s → budget = 60s
        log_short = [
            {"step": 30, "tokens_seen": 30000, "val_loss": 4.0, "elapsed_time": 30.0},
            {"step": 60, "tokens_seen": 60000, "val_loss": 3.0, "elapsed_time": 60.0},
        ]
        log_long = [
            {"step": 30, "tokens_seen": 30000, "val_loss": 4.5, "elapsed_time": 30.0},
            {"step": 60, "tokens_seen": 60000, "val_loss": 3.5, "elapsed_time": 60.0},
            {"step": 120, "tokens_seen": 120000, "val_loss": 2.5, "elapsed_time": 120.0},
        ]
        v_short = _make_variant_with_log("short", log_short)
        v_long = _make_variant_with_log("long", log_long)

        result = slice_fixed_wallclock([v_short, v_long], time_fractions=[0.50, 1.00])

        # Budget = min(60, 120) = 60s
        # At 0.50 * 60 = 30s
        assert result["short"][0.50] == pytest.approx(4.0)
        assert result["long"][0.50] == pytest.approx(4.5)

        # At 1.00 * 60 = 60s
        assert result["short"][1.00] == pytest.approx(3.0)
        assert result["long"][1.00] == pytest.approx(3.5)
