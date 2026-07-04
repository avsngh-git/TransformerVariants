"""Unit tests for the Probes page data extraction logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add dashboard root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages.probes_logic import (
    average_lists,
    classify_variants_for_probe,
    get_cka_matrix,
    get_probe_field,
)


class TestAverageLists:
    """Tests for the average_lists helper."""

    def test_empty_input_returns_empty(self):
        assert average_lists([]) == []

    def test_single_list_returns_same(self):
        data = [1.0, 2.0, 3.0]
        assert average_lists([data]) == data

    def test_two_lists_averaged(self):
        result = average_lists([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_three_lists_averaged(self):
        result = average_lists([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        np.testing.assert_allclose(result, [2.0, 2.0])


class TestGetProbeField:
    """Tests for the get_probe_field function."""

    def test_extracts_mqar_accuracies(self):
        variants_data = {
            "vanilla": [
                {"mqar": {"distances": [1, 2, 3], "accuracies": [0.9, 0.8, 0.7]}},
            ]
        }
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == [[0.9, 0.8, 0.7]]

    def test_extracts_multiple_seeds(self):
        variants_data = {
            "vanilla": [
                {"mqar": {"accuracies": [0.9, 0.8]}},
                {"mqar": {"accuracies": [0.85, 0.75]}},
            ]
        }
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == [[0.9, 0.8], [0.85, 0.75]]

    def test_missing_variant_returns_empty(self):
        variants_data = {"vanilla": [{"mqar": {"accuracies": [0.9]}}]}
        result = get_probe_field(variants_data, "nonexistent", "mqar", "accuracies")
        assert result == []

    def test_missing_probe_key_returns_empty(self):
        variants_data = {"vanilla": [{"stable_rank": {"per_layer": [5.0]}}]}
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == []

    def test_missing_field_returns_empty(self):
        variants_data = {"vanilla": [{"mqar": {"distances": [1, 2, 3]}}]}
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == []

    def test_empty_field_list_returns_empty(self):
        variants_data = {"vanilla": [{"mqar": {"accuracies": []}}]}
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == []

    def test_non_dict_seed_skipped(self):
        variants_data = {
            "vanilla": [
                None,
                "not a dict",
                {"mqar": {"accuracies": [0.9, 0.8]}},
            ]
        }
        result = get_probe_field(variants_data, "vanilla", "mqar", "accuracies")
        assert result == [[0.9, 0.8]]

    def test_stable_rank_per_layer(self):
        variants_data = {
            "modern": [
                {"stable_rank": {"per_layer": [6.1, 5.8, 5.5], "mean": 5.8}},
            ]
        }
        result = get_probe_field(variants_data, "modern", "stable_rank", "per_layer")
        assert result == [[6.1, 5.8, 5.5]]

    def test_attention_entropy_per_layer(self):
        variants_data = {
            "vanilla": [
                {"attention_entropy": {"per_layer": [2.1, 2.3, 2.5], "mean": 2.3}},
            ]
        }
        result = get_probe_field(
            variants_data, "vanilla", "attention_entropy", "per_layer"
        )
        assert result == [[2.1, 2.3, 2.5]]


class TestGetCkaMatrix:
    """Tests for the get_cka_matrix function."""

    def test_single_seed_returns_matrix(self):
        matrix = [[1.0, 0.9], [0.9, 1.0]]
        variants_data = {"vanilla": [{"cka": {"full_matrix": matrix}}]}
        result = get_cka_matrix(variants_data, "vanilla")
        assert result == matrix

    def test_multi_seed_averages_matrices(self):
        m1 = [[1.0, 0.8], [0.8, 1.0]]
        m2 = [[1.0, 0.9], [0.9, 1.0]]
        variants_data = {
            "vanilla": [
                {"cka": {"full_matrix": m1}},
                {"cka": {"full_matrix": m2}},
            ]
        }
        result = get_cka_matrix(variants_data, "vanilla")
        np.testing.assert_allclose(result, [[1.0, 0.85], [0.85, 1.0]])

    def test_no_cka_data_returns_none(self):
        variants_data = {"vanilla": [{"stable_rank": {"per_layer": [5.0]}}]}
        result = get_cka_matrix(variants_data, "vanilla")
        assert result is None

    def test_empty_matrix_returns_none(self):
        variants_data = {"vanilla": [{"cka": {"full_matrix": []}}]}
        result = get_cka_matrix(variants_data, "vanilla")
        assert result is None

    def test_missing_variant_returns_none(self):
        variants_data = {"vanilla": [{"cka": {"full_matrix": [[1.0]]}}]}
        result = get_cka_matrix(variants_data, "nonexistent")
        assert result is None


class TestClassifyVariantsForProbe:
    """Tests for classify_variants_for_probe."""

    def test_all_available(self):
        variants_data = {
            "vanilla": [{"mqar": {"accuracies": [0.9, 0.8]}}],
            "modern": [{"mqar": {"accuracies": [0.95, 0.9]}}],
        }
        avail, unavail = classify_variants_for_probe(
            variants_data, ["vanilla", "modern"], "mqar", "accuracies"
        )
        assert avail == ["vanilla", "modern"]
        assert unavail == []

    def test_some_unavailable(self):
        variants_data = {
            "vanilla": [{"mqar": {"accuracies": [0.9, 0.8]}}],
            "modern": [{"stable_rank": {"per_layer": [5.0]}}],
        }
        avail, unavail = classify_variants_for_probe(
            variants_data, ["vanilla", "modern"], "mqar", "accuracies"
        )
        assert avail == ["vanilla"]
        assert unavail == ["modern"]

    def test_none_available(self):
        variants_data = {
            "vanilla": [{"stable_rank": {"per_layer": [5.0]}}],
        }
        avail, unavail = classify_variants_for_probe(
            variants_data, ["vanilla"], "mqar", "accuracies"
        )
        assert avail == []
        assert unavail == ["vanilla"]

    def test_attention_entropy_partial(self):
        """Flash-based variants don't have attention entropy."""
        variants_data = {
            "vanilla": [{"attention_entropy": {"per_layer": [2.1, 2.3]}}],
            "swa": [{"mqar": {"accuracies": [0.9]}}],  # no attention_entropy
        }
        avail, unavail = classify_variants_for_probe(
            variants_data, ["vanilla", "swa"], "attention_entropy", "per_layer"
        )
        assert avail == ["vanilla"]
        assert unavail == ["swa"]
