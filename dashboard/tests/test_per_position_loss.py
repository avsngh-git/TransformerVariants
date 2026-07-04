"""Unit tests for the per-position loss page logic."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dashboard.pages.per_position_logic import (
    build_icl_table,
    compute_icl_curve,
    extract_icl_fit_params,
    extract_per_position_loss,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_metrics_data():
    with open(FIXTURES_DIR / "valid_metrics.json") as f:
        return json.load(f)


class TestExtractPerPositionLoss:
    """Tests for extract_per_position_loss."""

    def test_single_seed(self):
        seeds = [
            {"metrics": {"per_position_loss": [5.0, 4.0, 3.5, 3.0]}}
        ]
        result = extract_per_position_loss(seeds)
        assert result == [5.0, 4.0, 3.5, 3.0]

    def test_multi_seed_averages(self):
        seeds = [
            {"metrics": {"per_position_loss": [5.0, 4.0, 3.0]}},
            {"metrics": {"per_position_loss": [6.0, 5.0, 4.0]}},
        ]
        result = extract_per_position_loss(seeds)
        assert result is not None
        assert len(result) == 3
        assert abs(result[0] - 5.5) < 1e-10
        assert abs(result[1] - 4.5) < 1e-10
        assert abs(result[2] - 3.5) < 1e-10

    def test_no_data_returns_none(self):
        seeds = [{"metrics": {}}]
        assert extract_per_position_loss(seeds) is None

    def test_empty_list_returns_none(self):
        seeds = [{"metrics": {"per_position_loss": []}}]
        assert extract_per_position_loss(seeds) is None

    def test_invalid_seed_skipped(self):
        seeds = [
            "not a dict",
            {"metrics": {"per_position_loss": [5.0, 4.0]}},
        ]
        result = extract_per_position_loss(seeds)
        assert result == [5.0, 4.0]

    def test_missing_metrics_key(self):
        seeds = [{"other_key": "value"}]
        assert extract_per_position_loss(seeds) is None

    def test_fixture_data(self, valid_metrics_data):
        seeds = valid_metrics_data["variants"]["vanilla"]
        result = extract_per_position_loss(seeds)
        assert result is not None
        assert len(result) == 16


class TestExtractIclFitParams:
    """Tests for extract_icl_fit_params."""

    def test_single_seed(self):
        seeds = [
            {
                "metrics": {
                    "icl_fit_params": {
                        "A": 1.2,
                        "alpha": 0.42,
                        "C": 2.8,
                        "r_squared": 0.95,
                    }
                }
            }
        ]
        result = extract_icl_fit_params(seeds)
        assert result is not None
        assert result["A"] == 1.2
        assert result["alpha"] == 0.42
        assert result["C"] == 2.8
        assert result["r_squared"] == 0.95

    def test_multi_seed_averages(self):
        seeds = [
            {
                "metrics": {
                    "icl_fit_params": {
                        "A": 1.0,
                        "alpha": 0.4,
                        "C": 2.0,
                        "r_squared": 0.9,
                    }
                }
            },
            {
                "metrics": {
                    "icl_fit_params": {
                        "A": 2.0,
                        "alpha": 0.6,
                        "C": 3.0,
                        "r_squared": 0.8,
                    }
                }
            },
        ]
        result = extract_icl_fit_params(seeds)
        assert result is not None
        assert abs(result["A"] - 1.5) < 1e-10
        assert abs(result["alpha"] - 0.5) < 1e-10
        assert abs(result["C"] - 2.5) < 1e-10
        assert abs(result["r_squared"] - 0.85) < 1e-10

    def test_no_data_returns_none(self):
        seeds = [{"metrics": {}}]
        assert extract_icl_fit_params(seeds) is None

    def test_partial_params_returns_none(self):
        seeds = [
            {"metrics": {"icl_fit_params": {"A": 1.0, "alpha": 0.4}}}
        ]
        assert extract_icl_fit_params(seeds) is None


class TestComputeIclCurve:
    """Tests for compute_icl_curve."""

    def test_basic_computation(self):
        A, alpha, C = 1.0, 0.5, 2.0
        result = compute_icl_curve(A, alpha, C, seq_len=5)
        assert len(result) == 5
        # Position 1: A * 1^(-0.5) + C = 1.0 + 2.0 = 3.0
        assert abs(result[0] - 3.0) < 1e-10
        # Position 4: A * 4^(-0.5) + C = 1.0 * 0.5 + 2.0 = 2.5
        assert abs(result[3] - 2.5) < 1e-10

    def test_decreasing_sequence(self):
        result = compute_icl_curve(A=2.0, alpha=1.0, C=1.0, seq_len=10)
        # With positive alpha, the curve should be decreasing
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_single_position(self):
        result = compute_icl_curve(A=1.5, alpha=0.3, C=2.0, seq_len=1)
        assert len(result) == 1
        # At t=1: A * 1^(-alpha) + C = A + C
        assert abs(result[0] - 3.5) < 1e-10


class TestBuildIclTable:
    """Tests for build_icl_table."""

    def test_sorted_by_alpha_descending(self):
        variants_data = {
            "low": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.3,
                            "C": 2.0,
                            "r_squared": 0.9,
                        }
                    }
                }
            ],
            "high": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.8,
                            "C": 2.0,
                            "r_squared": 0.95,
                        }
                    }
                }
            ],
            "mid": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.5,
                            "C": 2.0,
                            "r_squared": 0.92,
                        }
                    }
                }
            ],
        }
        rows = build_icl_table(["low", "high", "mid"], variants_data)
        assert len(rows) == 3
        assert rows[0]["variant"] == "high"
        assert rows[1]["variant"] == "mid"
        assert rows[2]["variant"] == "low"

    def test_unavailable_variants_at_bottom(self):
        variants_data = {
            "has_data": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.5,
                            "C": 2.0,
                            "r_squared": 0.9,
                        }
                    }
                }
            ],
            "no_data": [{"metrics": {}}],
        }
        rows = build_icl_table(["has_data", "no_data"], variants_data)
        assert len(rows) == 2
        assert rows[0]["variant"] == "has_data"
        assert rows[0]["has_data"] is True
        assert rows[1]["variant"] == "no_data"
        assert rows[1]["has_data"] is False

    def test_poor_fit_indicator(self):
        variants_data = {
            "good": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.5,
                            "C": 2.0,
                            "r_squared": 0.9,
                        }
                    }
                }
            ],
            "poor": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.3,
                            "C": 2.0,
                            "r_squared": 0.7,
                        }
                    }
                }
            ],
        }
        rows = build_icl_table(["good", "poor"], variants_data)
        good_row = next(r for r in rows if r["variant"] == "good")
        poor_row = next(r for r in rows if r["variant"] == "poor")
        assert good_row["poor_fit"] is False
        assert poor_row["poor_fit"] is True

    def test_r_squared_exactly_0_8_not_poor(self):
        variants_data = {
            "borderline": [
                {
                    "metrics": {
                        "icl_fit_params": {
                            "A": 1.0,
                            "alpha": 0.5,
                            "C": 2.0,
                            "r_squared": 0.8,
                        }
                    }
                }
            ],
        }
        rows = build_icl_table(["borderline"], variants_data)
        assert rows[0]["poor_fit"] is False

    def test_fixture_data(self, valid_metrics_data):
        variants_data = valid_metrics_data["variants"]
        names = sorted(variants_data.keys())
        rows = build_icl_table(names, variants_data)
        assert len(rows) == 3
        # All should have data
        assert all(r["has_data"] for r in rows)
        # Should be sorted by alpha descending
        alphas = [r["alpha"] for r in rows]
        assert alphas == sorted(alphas, reverse=True)
