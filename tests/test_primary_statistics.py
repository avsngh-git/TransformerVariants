"""Tests for prespecified small-sample primary analysis."""

from __future__ import annotations

import pytest

from scripts.analyze_primary_statistics import analyze
from src.evaluation.statistics import paired_difference_summary, sample_summary


def test_five_seed_interval_uses_student_t() -> None:
    result = sample_summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["mean"] == pytest.approx(3.0)
    assert result["std"] == pytest.approx(2.5**0.5)
    assert result["ci95_half_width"] == pytest.approx(2.776 * (2.5**0.5) / (5**0.5))
    assert result["n"] == 5


def test_paired_summary_matches_by_seed_not_list_order() -> None:
    result = paired_difference_summary(
        {137: 2.0, 42: 1.0, 2024: 3.0},
        {2024: 2.5, 42: 1.1, 137: 1.7},
    )

    assert result["matched_seeds"] == [42, 137, 2024]
    assert result["differences"] == pytest.approx([-0.1, 0.3, 0.5])


def test_analysis_uses_only_manifest_prespecified_pairs() -> None:
    metrics = {
        "variants": {
            "vanilla": [
                {"checkpoint_dir": "vanilla_main_500M_s42", "val_loss": 3.0},
                {"checkpoint_dir": "vanilla_main_500M_s137", "val_loss": 2.8},
            ],
            "modern": [
                {"checkpoint_dir": "modern_main_500M_s137", "val_loss": 2.5},
                {"checkpoint_dir": "modern_main_500M_s42", "val_loss": 2.7},
            ],
        }
    }
    manifest = {
        "experiment_id": "main",
        "analysis": {
            "endpoint_key": "val_loss",
            "paired_comparisons": [
                {"baseline": "vanilla", "candidate": "modern", "role": "primary"}
            ],
        },
    }

    result = analyze(metrics, manifest)

    comparison = result["paired_comparisons"][0]
    assert comparison["baseline"] == "vanilla"
    assert comparison["candidate"] == "modern"
    assert comparison["result"]["summary"]["mean"] == pytest.approx(-0.3)


def test_analysis_rejects_duplicate_variant_seed_results() -> None:
    metrics = {
        "variants": {
            "modern": [
                {"checkpoint_dir": "modern_main_500M_s42", "val_loss": 2.7},
                {"checkpoint_dir": "modern_duplicate_s42", "val_loss": 2.6},
            ]
        }
    }

    with pytest.raises(ValueError, match="Duplicate result"):
        analyze(metrics, {"analysis": {"endpoint_key": "val_loss"}})
