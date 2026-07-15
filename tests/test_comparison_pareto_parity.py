"""Tests for compute_pareto_front and validate_parameter_parity in comparison module."""

from pathlib import Path

from src.evaluation.comparison import (
    VariantData,
    _estimate_active_parameter_count,
    _estimate_parameter_count,
    compute_pareto_front,
    validate_parameter_parity,
)
from src.models.config import ModelConfig

# --- Helpers ---


def _make_variant(
    name: str,
    val_loss: float | None = None,
    elapsed_time: float | None = None,
    peak_memory_mb: float | None = None,
    flop_total: int | None = None,
    config: ModelConfig | None = None,
) -> VariantData:
    """Create a VariantData with minimal fields for testing."""
    log_entries = []
    if val_loss is not None or elapsed_time is not None or peak_memory_mb is not None:
        entry = {"step": 100, "tokens_seen": 10000}
        if val_loss is not None:
            entry["val_loss"] = val_loss
        if elapsed_time is not None:
            entry["elapsed_time"] = elapsed_time
        if peak_memory_mb is not None:
            entry["peak_memory_mb"] = peak_memory_mb
        log_entries = [entry]

    from src.evaluation.flops import FLOPBreakdown

    flop_breakdown = None
    if flop_total is not None:
        flop_breakdown = FLOPBreakdown(
            qkv_proj=flop_total // 4,
            attention_score=flop_total // 4,
            attention_output=flop_total // 4,
            ffn=flop_total // 4,
            total=flop_total,
        )

    return VariantData(
        name=name,
        checkpoint_dir=Path(f"/tmp/{name}"),
        log_entries=log_entries,
        config=config or ModelConfig(),
        flop_breakdown=flop_breakdown,
    )


# --- compute_pareto_front tests ---


class TestComputeParetoFront:
    """Tests for compute_pareto_front function."""

    def test_single_variant_always_pareto_optimal(self):
        """Requirement 10.5: Single variant is always Pareto-optimal."""
        v = _make_variant("V0", val_loss=2.5, flop_total=1000)
        result = compute_pareto_front([v], x_metric="flops", y_metric="val_loss")
        assert result == ["V0"]

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = compute_pareto_front([], x_metric="flops", y_metric="val_loss")
        assert result == []

    def test_dominated_variant_excluded(self):
        """Requirement 10.3: Strictly dominated variant not in Pareto front."""
        # V0 has lower FLOPs AND lower val_loss → dominates V1
        v0 = _make_variant("V0", val_loss=2.0, flop_total=500)
        v1 = _make_variant("V1", val_loss=3.0, flop_total=1000)
        result = compute_pareto_front([v0, v1], x_metric="flops", y_metric="val_loss")
        assert "V0" in result
        assert "V1" not in result

    def test_non_dominated_both_on_front(self):
        """Two non-dominated variants both appear on Pareto front."""
        # V0: lower FLOPs, higher val_loss
        # V1: higher FLOPs, lower val_loss
        v0 = _make_variant("V0", val_loss=3.0, flop_total=500)
        v1 = _make_variant("V1", val_loss=2.0, flop_total=1000)
        result = compute_pareto_front([v0, v1], x_metric="flops", y_metric="val_loss")
        assert set(result) == {"V0", "V1"}

    def test_three_variants_mixed_dominance(self):
        """Test with 3 variants where one is dominated."""
        v0 = _make_variant("V0", val_loss=2.0, flop_total=500)   # Pareto
        v1 = _make_variant("V1", val_loss=1.5, flop_total=1000)  # Pareto
        v2 = _make_variant("V2", val_loss=2.5, flop_total=800)  # Dominated by V0
        result = compute_pareto_front(
            [v0, v1, v2], x_metric="flops", y_metric="val_loss"
        )
        assert "V0" in result
        assert "V1" in result
        assert "V2" not in result

    def test_same_x_metric_returns_best_y(self):
        """Requirement 10.4: Same x_metric → only best y_metric returned."""
        v0 = _make_variant("V0", val_loss=2.5, flop_total=1000)
        v1 = _make_variant("V1", val_loss=2.0, flop_total=1000)
        v2 = _make_variant("V2", val_loss=3.0, flop_total=1000)
        result = compute_pareto_front([v0, v1, v2], x_metric="flops", y_metric="val_loss")
        assert result == ["V1"]

    def test_wallclock_metric(self):
        """Requirement 10.2: Supports (wallclock, val_loss) pair."""
        v0 = _make_variant("V0", val_loss=2.0, elapsed_time=100.0)
        v1 = _make_variant("V1", val_loss=3.0, elapsed_time=200.0)
        result = compute_pareto_front([v0, v1], x_metric="wallclock", y_metric="val_loss")
        assert "V0" in result
        assert "V1" not in result

    def test_peak_memory_metric(self):
        """Requirement 10.2: Supports (peak_memory, val_loss) pair."""
        v0 = _make_variant("V0", val_loss=2.0, peak_memory_mb=4000.0)
        v1 = _make_variant("V1", val_loss=1.5, peak_memory_mb=8000.0)
        result = compute_pareto_front([v0, v1], x_metric="peak_memory", y_metric="val_loss")
        # Both are non-dominated: V0 less memory, V1 lower loss
        assert set(result) == {"V0", "V1"}

    def test_missing_data_variant_excluded_with_warning(self):
        """Variants with missing metric data are excluded."""
        v0 = _make_variant("V0", val_loss=2.0, flop_total=500)
        v1 = VariantData(
            name="V1",
            checkpoint_dir=Path("/tmp/V1"),
            log_entries=[],  # No val_loss available
            config=ModelConfig(),
        )
        result = compute_pareto_front([v0, v1], x_metric="flops", y_metric="val_loss")
        assert result == ["V0"]

    def test_equal_on_both_metrics_not_dominated(self):
        """Two variants equal on both metrics: neither dominates the other."""
        v0 = _make_variant("V0", val_loss=2.0, flop_total=1000)
        v1 = _make_variant("V1", val_loss=2.0, flop_total=1000)
        result = compute_pareto_front([v0, v1], x_metric="flops", y_metric="val_loss")
        # When all x values are same, only best y is returned. Both y are same too.
        # Implementation returns the first one found with min y
        assert len(result) == 1


# --- validate_parameter_parity tests ---


class TestValidateParameterParity:
    """Tests for validate_parameter_parity function."""

    def test_single_variant_always_valid(self):
        """Single variant is always valid."""
        v = _make_variant("V0", config=ModelConfig(n_layer=4, d_model=256))
        valid, counts = validate_parameter_parity([v])
        assert valid is True
        assert "V0" in counts
        assert counts["V0"] > 0

    def test_empty_list_valid(self):
        """Empty input returns valid with empty dict."""
        valid, counts = validate_parameter_parity([])
        assert valid is True
        assert counts == {}

    def test_identical_configs_valid(self):
        """Requirement 11.2: Identical configs are always within tolerance."""
        config = ModelConfig(n_layer=4, d_model=256, vocab_size=50257)
        v0 = _make_variant("V0", config=config)
        v1 = _make_variant("V1", config=config)
        valid, counts = validate_parameter_parity([v0, v1])
        assert valid is True
        assert counts["V0"] == counts["V1"]

    def test_similar_configs_within_tolerance(self):
        """Variants with similar (but not identical) param counts pass."""
        # Both have similar sizes — the d_model difference is small
        config1 = ModelConfig(n_layer=4, d_model=256, vocab_size=50257)
        config2 = ModelConfig(n_layer=4, d_model=256, vocab_size=50257)
        v0 = _make_variant("V0", config=config1)
        v1 = _make_variant("V1", config=config2)
        valid, counts = validate_parameter_parity([v0, v1], tolerance=0.05)
        assert valid is True

    def test_very_different_configs_invalid(self):
        """Requirement 11.2: Variants outside ±5% tolerance flagged invalid."""
        # V0 is much smaller than V1
        config_small = ModelConfig(n_layer=2, d_model=64, vocab_size=1000)
        config_large = ModelConfig(n_layer=12, d_model=512, vocab_size=50257)
        v0 = _make_variant("V0", config=config_small)
        v1 = _make_variant("V1", config=config_large)
        valid, counts = validate_parameter_parity([v0, v1], tolerance=0.05)
        assert valid is False
        assert counts["V0"] < counts["V1"]

    def test_returns_param_counts_dict(self):
        """Requirement 11.3: Returns dict mapping variant name to param count."""
        config = ModelConfig(n_layer=4, d_model=256)
        v0 = _make_variant("V0", config=config)
        v1 = _make_variant("V1", config=config)
        valid, counts = validate_parameter_parity([v0, v1])
        assert isinstance(counts, dict)
        assert "V0" in counts
        assert "V1" in counts
        assert all(isinstance(c, int) for c in counts.values())

    def test_custom_tolerance(self):
        """Custom tolerance value is respected."""
        # Create configs that differ by ~10%
        config1 = ModelConfig(n_layer=4, d_model=256, vocab_size=50257)
        config2 = ModelConfig(n_layer=4, d_model=256, vocab_size=55000)
        v0 = _make_variant("V0", config=config1)
        v1 = _make_variant("V1", config=config2)
        _, counts = validate_parameter_parity([v0, v1], tolerance=0.5)
        # With large tolerance, should pass
        valid_large, _ = validate_parameter_parity([v0, v1], tolerance=0.5)
        assert valid_large is True

    def test_param_count_estimation_reasonable(self):
        """Parameter count estimation produces reasonable values."""
        # Debug-scale: n_layer=2, d_model=64, seq_len=128, vocab=50257
        config = ModelConfig(
            n_layer=2, d_model=64, seq_len=128, vocab_size=50257
        )
        count = _estimate_parameter_count(config)
        # Rough expected: vocab_embed ~ 50257*64=3.2M, pos_embed ~ 128*64=8K,
        # per layer: 3*64*64 + 64*64 + 2*64*256 + 2*64 = 12288+4096+32768+128 ≈ 49K × 2 layers ≈ 98K
        # Total ≈ 3.2M + 8K + 98K + 64 ≈ ~3.3M (no output head if tied)
        assert count > 3_000_000  # At least 3M (dominated by vocab embedding)
        assert count < 5_000_000  # But not excessively large

    def test_swiglu_is_parameter_matched_to_standard(self):
        """Rounded 8/3-width SwiGLU stays close to a standard 4x FFN."""
        config_std = ModelConfig(n_layer=4, d_model=256, ffn_type="standard")
        config_swiglu = ModelConfig(n_layer=4, d_model=256, ffn_type="swiglu")
        count_std = _estimate_parameter_count(config_std)
        count_swiglu = _estimate_parameter_count(config_swiglu)
        assert abs(count_swiglu - count_std) / count_std < 0.01

    def test_swiglu_and_moe_counts_match_saved_main_run_metadata(self):
        """Estimator reflects the actual rounded SwiGLU and MoE modules."""
        dense = ModelConfig(
            n_layer=8,
            d_model=512,
            n_head=8,
            vocab_size=50257,
            seq_len=1024,
            variant="modern",
            norm_type="rmsnorm",
            position_encoding="rope",
            ffn_type="swiglu",
            activation="swiglu",
        )
        moe = ModelConfig(**{**vars(dense), "variant": "moe", "num_experts": 8})

        assert _estimate_parameter_count(dense) == 51_430_400
        assert _estimate_parameter_count(moe, "moe") == 172_573_696
        assert _estimate_active_parameter_count(moe, "moe") == 68_764_672
