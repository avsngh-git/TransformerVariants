"""Unit tests for the Learning Curves page data extraction logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add dashboard root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages.learning_curves_logic import extract_learning_curve


class TestExtractLearningCurve:
    """Tests for the _extract_learning_curve helper function."""

    def test_single_seed_returns_raw_values(self):
        """Single-seed variant should return raw values without averaging."""
        seeds = [
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.5, "cumulative_flops": 1.2e9, "val_loss": 5.1},
                    {"step": 200, "tokens_seen": 102400, "wallclock": 25.0, "cumulative_flops": 2.4e9, "val_loss": 4.5},
                    {"step": 300, "tokens_seen": 153600, "wallclock": 37.5, "cumulative_flops": 3.6e9, "val_loss": 4.0},
                ]
            }
        ]
        result = extract_learning_curve(seeds, "tokens_seen")

        assert result is not None
        assert result["x_values"] == [51200, 102400, 153600]
        assert result["mean_loss"] == [5.1, 4.5, 4.0]
        assert result["std_loss"] is None
        assert result["num_seeds"] == 1

    def test_multi_seed_averages_val_loss(self):
        """Multi-seed variant should average val_loss across seeds."""
        seeds = [
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.5, "cumulative_flops": 1.2e9, "val_loss": 5.0},
                    {"step": 200, "tokens_seen": 102400, "wallclock": 25.0, "cumulative_flops": 2.4e9, "val_loss": 4.0},
                ]
            },
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.8, "cumulative_flops": 1.2e9, "val_loss": 5.2},
                    {"step": 200, "tokens_seen": 102400, "wallclock": 25.6, "cumulative_flops": 2.4e9, "val_loss": 4.2},
                ]
            },
        ]
        result = extract_learning_curve(seeds, "tokens_seen")

        assert result is not None
        assert result["x_values"] == [51200, 102400]
        assert result["num_seeds"] == 2
        # mean of [5.0, 5.2] = 5.1, mean of [4.0, 4.2] = 4.1
        np.testing.assert_allclose(result["mean_loss"], [5.1, 4.1], atol=1e-10)
        # std of [5.0, 5.2] = 0.1, std of [4.0, 4.2] = 0.1
        np.testing.assert_allclose(result["std_loss"], [0.1, 0.1], atol=1e-10)

    def test_no_log_entries_returns_none(self):
        """Seeds without log_entries should return None."""
        seeds = [{"val_loss": 3.5}]
        result = extract_learning_curve(seeds, "tokens_seen")
        assert result is None

    def test_empty_log_entries_returns_none(self):
        """Seeds with empty log_entries list should return None."""
        seeds = [{"log_entries": []}]
        result = extract_learning_curve(seeds, "tokens_seen")
        assert result is None

    def test_empty_seeds_returns_none(self):
        """Empty seeds list should return None."""
        result = extract_learning_curve([], "tokens_seen")
        assert result is None

    def test_wallclock_x_axis(self):
        """X-axis should use wallclock values when specified."""
        seeds = [
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.5, "cumulative_flops": 1.2e9, "val_loss": 5.1},
                    {"step": 200, "tokens_seen": 102400, "wallclock": 25.0, "cumulative_flops": 2.4e9, "val_loss": 4.5},
                ]
            }
        ]
        result = extract_learning_curve(seeds, "wallclock")

        assert result is not None
        assert result["x_values"] == [12.5, 25.0]

    def test_cumulative_flops_x_axis(self):
        """X-axis should use cumulative_flops values when specified."""
        seeds = [
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.5, "cumulative_flops": 1.2e9, "val_loss": 5.1},
                    {"step": 200, "tokens_seen": 102400, "wallclock": 25.0, "cumulative_flops": 2.4e9, "val_loss": 4.5},
                ]
            }
        ]
        result = extract_learning_curve(seeds, "cumulative_flops")

        assert result is not None
        assert result["x_values"] == [1.2e9, 2.4e9]

    def test_non_dict_seeds_skipped(self):
        """Non-dict entries in seeds list should be safely skipped."""
        seeds = [None, "not a dict", 123]
        result = extract_learning_curve(seeds, "tokens_seen")
        assert result is None

    def test_mixed_valid_invalid_seeds(self):
        """Only seeds with valid log_entries should be considered."""
        seeds = [
            {
                "log_entries": [
                    {"step": 100, "tokens_seen": 51200, "wallclock": 12.5, "cumulative_flops": 1.2e9, "val_loss": 5.0},
                ]
            },
            {"log_entries": []},  # empty
            {"no_log": True},  # missing key
        ]
        result = extract_learning_curve(seeds, "tokens_seen")

        assert result is not None
        assert result["num_seeds"] == 1
        assert result["mean_loss"] == [5.0]
