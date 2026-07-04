"""Tests for streaming pipeline dataclasses and validation logic."""

import pytest
from pathlib import Path

from src.data.streaming_prepare import (
    FilterStats,
    PipelineConfig,
    PipelineResult,
    ResumeState,
)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass and validation."""

    def test_default_values(self):
        """PipelineConfig should have sensible defaults matching the design."""
        config = PipelineConfig()
        assert config.dataset_name == "HuggingFaceFW/fineweb-edu"
        assert config.dataset_config == "sample-10BT"
        assert config.split == "train"
        assert config.max_tokens == 1_000_000_000
        assert config.min_doc_tokens == 50
        assert config.max_doc_tokens == 10_000
        assert config.tokens_per_shard == 10_000_000
        assert config.tokenizer_name == "gpt2"
        assert config.output_dir == Path("data/processed/fineweb-edu")
        assert config.resume is False

    def test_custom_values(self):
        """PipelineConfig should accept custom valid values."""
        config = PipelineConfig(
            dataset_name="custom/dataset",
            min_doc_tokens=100,
            max_doc_tokens=5000,
            tokens_per_shard=5000,
            output_dir=Path("/tmp/test-output"),
        )
        assert config.min_doc_tokens == 100
        assert config.max_doc_tokens == 5000
        assert config.tokens_per_shard == 5000
        assert config.output_dir == Path("/tmp/test-output")

    def test_min_doc_tokens_must_be_at_least_one(self):
        """min_doc_tokens < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_doc_tokens must be >= 1"):
            PipelineConfig(min_doc_tokens=0)

    def test_min_doc_tokens_negative(self):
        """Negative min_doc_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="min_doc_tokens must be >= 1"):
            PipelineConfig(min_doc_tokens=-5)

    def test_max_doc_tokens_must_be_gte_min_doc_tokens(self):
        """max_doc_tokens < min_doc_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="max_doc_tokens.*must be >= min_doc_tokens"):
            PipelineConfig(min_doc_tokens=100, max_doc_tokens=50)

    def test_max_doc_tokens_equal_to_min_is_valid(self):
        """max_doc_tokens == min_doc_tokens is a valid configuration."""
        config = PipelineConfig(min_doc_tokens=100, max_doc_tokens=100)
        assert config.min_doc_tokens == 100
        assert config.max_doc_tokens == 100

    def test_tokens_per_shard_must_be_at_least_1000(self):
        """tokens_per_shard < 1000 should raise ValueError."""
        with pytest.raises(ValueError, match="tokens_per_shard must be >= 1000"):
            PipelineConfig(tokens_per_shard=999)

    def test_tokens_per_shard_exactly_1000_is_valid(self):
        """tokens_per_shard == 1000 is valid (boundary)."""
        config = PipelineConfig(tokens_per_shard=1000)
        assert config.tokens_per_shard == 1000

    def test_output_dir_string_converted_to_path(self):
        """String output_dir should be converted to Path."""
        config = PipelineConfig(output_dir="/tmp/output")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/output")

    def test_max_tokens_none_is_valid(self):
        """max_tokens=None means unlimited processing."""
        config = PipelineConfig(max_tokens=None)
        assert config.max_tokens is None


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_default_values_are_zero(self):
        """All counters should default to zero."""
        stats = FilterStats()
        assert stats.documents_processed == 0
        assert stats.documents_accepted == 0
        assert stats.documents_filtered_short == 0
        assert stats.documents_filtered_long == 0

    def test_custom_values(self):
        """FilterStats should accept custom counter values."""
        stats = FilterStats(
            documents_processed=100,
            documents_accepted=80,
            documents_filtered_short=15,
            documents_filtered_long=5,
        )
        assert stats.documents_processed == 100
        assert stats.documents_accepted == 80
        assert stats.documents_filtered_short == 15
        assert stats.documents_filtered_long == 5


class TestResumeState:
    """Tests for ResumeState dataclass."""

    def test_all_fields_required(self):
        """ResumeState should require all fields."""
        state = ResumeState(
            documents_consumed=1000,
            train_shards_written=5,
            val_shards_written=1,
            train_tokens_written=50_000_000,
            val_tokens_written=500_000,
        )
        assert state.documents_consumed == 1000
        assert state.train_shards_written == 5
        assert state.val_shards_written == 1
        assert state.train_tokens_written == 50_000_000
        assert state.val_tokens_written == 500_000

    def test_missing_field_raises_error(self):
        """Omitting a required field should raise TypeError."""
        with pytest.raises(TypeError):
            ResumeState(documents_consumed=100)  # type: ignore


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_all_fields(self):
        """PipelineResult should store all result values."""
        stats = FilterStats(
            documents_processed=1000,
            documents_accepted=900,
            documents_filtered_short=80,
            documents_filtered_long=20,
        )
        result = PipelineResult(
            output_dir=Path("/tmp/output"),
            train_shards=10,
            val_shards=1,
            train_tokens=100_000_000,
            val_tokens=1_000_000,
            filter_stats=stats,
            documents_consumed=1000,
            processing_time_seconds=3600.5,
        )
        assert result.output_dir == Path("/tmp/output")
        assert result.train_shards == 10
        assert result.val_shards == 1
        assert result.train_tokens == 100_000_000
        assert result.val_tokens == 1_000_000
        assert result.filter_stats is stats
        assert result.documents_consumed == 1000
        assert result.processing_time_seconds == 3600.5
