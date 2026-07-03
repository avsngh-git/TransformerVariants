"""Tests for aggregate_across_seeds in the comparison module."""

import math

import numpy as np
import pytest

from src.evaluation.comparison import aggregate_across_seeds


class TestAggregateAcrossSeeds:
    """Test suite for aggregate_across_seeds function."""

    def test_empty_input_returns_empty(self):
        """Empty seed_results returns empty dict."""
        result = aggregate_across_seeds([])
        assert result == {}

    def test_single_seed_returns_nan_std(self):
        """With 1 seed, mean is the value and std is NaN (lacking confidence)."""
        seed_results = [{"val_loss": 2.34, "perplexity": 10.4}]
        result = aggregate_across_seeds(seed_results)

        assert result["val_loss"][0] == pytest.approx(2.34)
        assert math.isnan(result["val_loss"][1])
        assert result["perplexity"][0] == pytest.approx(10.4)
        assert math.isnan(result["perplexity"][1])

    def test_two_seeds_returns_nan_std(self):
        """With 2 seeds, mean is computed but std is NaN (lacking confidence)."""
        seed_results = [
            {"val_loss": 2.30, "perplexity": 10.0},
            {"val_loss": 2.40, "perplexity": 11.0},
        ]
        result = aggregate_across_seeds(seed_results)

        assert result["val_loss"][0] == pytest.approx(2.35)
        assert math.isnan(result["val_loss"][1])
        assert result["perplexity"][0] == pytest.approx(10.5)
        assert math.isnan(result["perplexity"][1])

    def test_three_seeds_computes_mean_and_std(self):
        """With 3+ seeds, compute proper mean and sample std."""
        seed_results = [
            {"val_loss": 2.30, "perplexity": 10.0, "mfu": 0.30},
            {"val_loss": 2.40, "perplexity": 11.0, "mfu": 0.35},
            {"val_loss": 2.50, "perplexity": 12.0, "mfu": 0.40},
        ]
        result = aggregate_across_seeds(seed_results)

        # Check mean
        assert result["val_loss"][0] == pytest.approx(2.40)
        assert result["perplexity"][0] == pytest.approx(11.0)
        assert result["mfu"][0] == pytest.approx(0.35)

        # Check std (sample std, ddof=1)
        expected_val_loss_std = float(np.std([2.30, 2.40, 2.50], ddof=1))
        assert result["val_loss"][1] == pytest.approx(expected_val_loss_std)

        expected_perplexity_std = float(np.std([10.0, 11.0, 12.0], ddof=1))
        assert result["perplexity"][1] == pytest.approx(expected_perplexity_std)

        expected_mfu_std = float(np.std([0.30, 0.35, 0.40], ddof=1))
        assert result["mfu"][1] == pytest.approx(expected_mfu_std)

    def test_five_seeds_computes_correctly(self):
        """With 5 seeds, compute proper mean and sample std."""
        seed_results = [
            {"val_loss": 2.30},
            {"val_loss": 2.35},
            {"val_loss": 2.40},
            {"val_loss": 2.45},
            {"val_loss": 2.50},
        ]
        result = aggregate_across_seeds(seed_results)

        expected_mean = np.mean([2.30, 2.35, 2.40, 2.45, 2.50])
        expected_std = np.std([2.30, 2.35, 2.40, 2.45, 2.50], ddof=1)
        assert result["val_loss"][0] == pytest.approx(float(expected_mean))
        assert result["val_loss"][1] == pytest.approx(float(expected_std))

    def test_missing_metric_in_some_seeds(self):
        """If a metric is missing from some seeds, aggregate only available values."""
        seed_results = [
            {"val_loss": 2.30, "mfu": 0.30},
            {"val_loss": 2.40},
            {"val_loss": 2.50, "mfu": 0.40},
        ]
        result = aggregate_across_seeds(seed_results)

        # val_loss has 3 values -> mean + std
        assert result["val_loss"][0] == pytest.approx(2.40)
        assert not math.isnan(result["val_loss"][1])

        # mfu has only 2 values available, but n_seeds is still 3
        # so std is computed from the 2 available values (ddof=1)
        assert result["mfu"][0] == pytest.approx(0.35)
        expected_mfu_std = float(np.std([0.30, 0.40], ddof=1))
        assert result["mfu"][1] == pytest.approx(expected_mfu_std)

    def test_returns_dict_mapping_metric_to_tuple(self):
        """Return type is dict[str, tuple[float, float]]."""
        seed_results = [
            {"val_loss": 2.30},
            {"val_loss": 2.40},
            {"val_loss": 2.50},
        ]
        result = aggregate_across_seeds(seed_results)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], float)
            assert isinstance(value[1], float)

    def test_identical_seeds_zero_std(self):
        """When all seeds have the same value, std should be 0."""
        seed_results = [
            {"val_loss": 2.34},
            {"val_loss": 2.34},
            {"val_loss": 2.34},
        ]
        result = aggregate_across_seeds(seed_results)

        assert result["val_loss"][0] == pytest.approx(2.34)
        assert result["val_loss"][1] == pytest.approx(0.0)
