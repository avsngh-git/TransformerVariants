"""Unit tests for dashboard/components/data_loader.py."""

import json
import os
import pathlib
import sys
from unittest.mock import patch, MagicMock

import pytest

# Add dashboard root to path so we can import components
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Mock streamlit before importing data_loader
mock_st = MagicMock()
mock_st.cache_data = lambda func=None, **kwargs: func if func else (lambda f: f)
sys.modules["streamlit"] = mock_st

from components.data_loader import (
    load_metrics,
    validate_metrics,
    get_variant_names,
    get_seed_count,
    REQUIRED_KEYS,
)


# ---------------------------------------------------------------------------
# Tests for validate_metrics
# ---------------------------------------------------------------------------


class TestValidateMetrics:
    """Tests for the validate_metrics function."""

    def test_all_keys_present(self):
        """No warnings when all required keys are present."""
        data = {"variants": {}, "aggregated": {}, "comparison": {}}
        warnings = validate_metrics(data)
        assert warnings == []

    def test_missing_one_key(self):
        """Reports exactly the one missing key."""
        data = {"variants": {}, "aggregated": {}}
        warnings = validate_metrics(data)
        assert len(warnings) == 1
        assert "comparison" in warnings[0]

    def test_missing_two_keys(self):
        """Reports exactly the two missing keys."""
        data = {"variants": {}}
        warnings = validate_metrics(data)
        assert len(warnings) == 2
        missing_text = " ".join(warnings)
        assert "aggregated" in missing_text
        assert "comparison" in missing_text

    def test_missing_all_keys(self):
        """Reports all three missing keys when dict is empty."""
        data = {}
        warnings = validate_metrics(data)
        assert len(warnings) == 3
        missing_text = " ".join(warnings)
        for key in REQUIRED_KEYS:
            assert key in missing_text

    def test_extra_keys_ignored(self):
        """Extra keys beyond the required ones don't trigger warnings."""
        data = {"variants": {}, "aggregated": {}, "comparison": {}, "extra": "value"}
        warnings = validate_metrics(data)
        assert warnings == []

    def test_no_false_positives(self):
        """Keys that are present are never reported as missing."""
        data = {"variants": {}, "comparison": {}}
        warnings = validate_metrics(data)
        # Only "aggregated" is missing
        assert len(warnings) == 1
        assert "aggregated" in warnings[0]
        assert "variants" not in warnings[0]
        assert "comparison" not in warnings[0]


# ---------------------------------------------------------------------------
# Tests for get_variant_names
# ---------------------------------------------------------------------------


class TestGetVariantNames:
    """Tests for the get_variant_names function."""

    def test_extracts_sorted_names(self):
        """Returns variant names sorted alphabetically."""
        data = {"variants": {"swa": [], "modern": [], "vanilla": []}}
        names = get_variant_names(data)
        assert names == ["modern", "swa", "vanilla"]

    def test_single_variant(self):
        """Works with a single variant."""
        data = {"variants": {"vanilla": [{"seed_index": 0}]}}
        names = get_variant_names(data)
        assert names == ["vanilla"]

    def test_empty_variants_dict(self):
        """Returns empty list for empty variants dict."""
        data = {"variants": {}}
        names = get_variant_names(data)
        assert names == []

    def test_missing_variants_key(self):
        """Returns empty list when 'variants' key is missing."""
        data = {"aggregated": {}}
        names = get_variant_names(data)
        assert names == []

    def test_variants_not_a_dict(self):
        """Returns empty list when 'variants' is not a dict."""
        data = {"variants": "invalid"}
        names = get_variant_names(data)
        assert names == []

    def test_with_fixture_data(self, valid_metrics_data):
        """Works correctly with the full fixture data."""
        names = get_variant_names(valid_metrics_data)
        assert names == ["modern", "swa", "vanilla"]


# ---------------------------------------------------------------------------
# Tests for get_seed_count
# ---------------------------------------------------------------------------


class TestGetSeedCount:
    """Tests for the get_seed_count function."""

    def test_multiple_seeds(self, valid_metrics_data):
        """Returns correct count for variants with multiple seeds."""
        assert get_seed_count(valid_metrics_data, "vanilla") == 2
        assert get_seed_count(valid_metrics_data, "modern") == 2
        assert get_seed_count(valid_metrics_data, "swa") == 2

    def test_single_seed(self, partial_variants_data):
        """Returns 1 for a variant with one seed."""
        assert get_seed_count(partial_variants_data, "vanilla") == 1

    def test_missing_variant(self, valid_metrics_data):
        """Returns 0 for a variant not present in data."""
        assert get_seed_count(valid_metrics_data, "nonexistent") == 0

    def test_missing_variants_key(self):
        """Returns 0 when 'variants' key is missing."""
        data = {"aggregated": {}}
        assert get_seed_count(data, "vanilla") == 0

    def test_variants_not_a_dict(self):
        """Returns 0 when 'variants' is not a dict."""
        data = {"variants": []}
        assert get_seed_count(data, "vanilla") == 0

    def test_variant_value_not_a_list(self):
        """Returns 0 when variant entry is not a list."""
        data = {"variants": {"vanilla": "not_a_list"}}
        assert get_seed_count(data, "vanilla") == 0


# ---------------------------------------------------------------------------
# Tests for load_metrics
# ---------------------------------------------------------------------------


class TestLoadMetrics:
    """Tests for the load_metrics function."""

    def test_successful_load(self, tmp_path):
        """Loads valid JSON from a proper report directory."""
        # Set up report structure
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        metrics = {"variants": {"v0": []}, "aggregated": {}, "comparison": {}}
        (raw_dir / "metrics.json").write_text(json.dumps(metrics))

        mock_st.reset_mock()
        result = load_metrics(str(tmp_path))
        assert result is not None
        assert result["variants"] == {"v0": []}
        # No errors should have been called
        mock_st.error.assert_not_called()

    def test_directory_not_found(self, tmp_path):
        """Displays error when report directory doesn't exist."""
        mock_st.reset_mock()
        result = load_metrics(str(tmp_path / "nonexistent"))
        assert result is None
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "nonexistent" in error_msg

    def test_metrics_file_not_found(self, tmp_path):
        """Displays error when raw/metrics.json doesn't exist."""
        # Directory exists but no raw/metrics.json
        mock_st.reset_mock()
        result = load_metrics(str(tmp_path))
        assert result is None
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "metrics.json" in error_msg

    def test_malformed_json(self, tmp_path):
        """Displays error for malformed JSON."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "metrics.json").write_text("{invalid json content")

        mock_st.reset_mock()
        result = load_metrics(str(tmp_path))
        assert result is None
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "malformed" in error_msg.lower() or "parse" in error_msg.lower()

    def test_missing_keys_shows_warning(self, tmp_path):
        """Shows warning when top-level keys are missing but still returns data."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # Only has "variants" key, missing aggregated and comparison
        metrics = {"variants": {"v0": []}}
        (raw_dir / "metrics.json").write_text(json.dumps(metrics))

        mock_st.reset_mock()
        result = load_metrics(str(tmp_path))
        assert result is not None  # Still returns data (graceful degradation)
        assert result["variants"] == {"v0": []}
        mock_st.warning.assert_called_once()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "aggregated" in warning_msg
        assert "comparison" in warning_msg

    def test_all_keys_present_no_warning(self, tmp_path):
        """No warning when all required keys are present."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        metrics = {"variants": {}, "aggregated": {}, "comparison": {}}
        (raw_dir / "metrics.json").write_text(json.dumps(metrics))

        mock_st.reset_mock()
        result = load_metrics(str(tmp_path))
        assert result is not None
        mock_st.warning.assert_not_called()
        mock_st.error.assert_not_called()

    def test_with_valid_fixture(self, valid_metrics_path):
        """Loads the valid fixture file correctly."""
        mock_st.reset_mock()
        report_dir = valid_metrics_path.parent.parent  # fixtures/ -> tests/
        # Need to create the expected path structure: report_dir/raw/metrics.json
        # The fixture is at tests/fixtures/valid_metrics.json
        # We need to simulate a report_dir that has raw/metrics.json
        # Let's use a tmp setup instead
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = pathlib.Path(tmp) / "raw"
            raw_dir.mkdir()
            shutil.copy(valid_metrics_path, raw_dir / "metrics.json")

            result = load_metrics(tmp)
            assert result is not None
            assert "variants" in result
            assert "modern" in result["variants"]
            assert "vanilla" in result["variants"]
            assert "swa" in result["variants"]
