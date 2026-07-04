"""Unit tests for write_progress() and load_resume_state() helper functions."""

import json
from pathlib import Path

import pytest

from src.data.streaming_prepare import (
    ResumeState,
    ShardBuffer,
    load_resume_state,
    write_progress,
)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def train_buffer(tmp_output_dir: Path) -> ShardBuffer:
    """Create a train shard buffer with some state."""
    buf = ShardBuffer(tmp_output_dir, "train", tokens_per_shard=100, start_shard_idx=0)
    # Simulate having written some tokens (flush a full shard)
    buf.append(list(range(100)))  # triggers one flush
    return buf


@pytest.fixture
def val_buffer(tmp_output_dir: Path) -> ShardBuffer:
    """Create a val shard buffer with some state."""
    buf = ShardBuffer(tmp_output_dir, "val", tokens_per_shard=100, start_shard_idx=0)
    buf.append(list(range(50)))  # partial buffer, no flush
    return buf


class TestWriteProgress:
    """Tests for write_progress()."""

    def test_creates_progress_json(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """write_progress creates a progress.json file."""
        write_progress(tmp_output_dir, 1000, train_buffer, val_buffer)
        assert (tmp_output_dir / "progress.json").exists()

    def test_correct_fields(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """progress.json contains all required fields with correct values."""
        write_progress(tmp_output_dir, 5000, train_buffer, val_buffer)
        data = json.loads((tmp_output_dir / "progress.json").read_text())

        assert data["documents_consumed"] == 5000
        assert data["train_shards_written"] == train_buffer.shards_written
        assert data["val_shards_written"] == val_buffer.shards_written
        assert data["train_tokens_written"] == train_buffer.total_tokens_written
        assert data["val_tokens_written"] == val_buffer.total_tokens_written
        assert data["completed"] is False
        assert "timestamp" in data

    def test_completed_flag(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """completed flag is serialized correctly."""
        write_progress(
            tmp_output_dir, 5000, train_buffer, val_buffer, completed=True
        )
        data = json.loads((tmp_output_dir / "progress.json").read_text())
        assert data["completed"] is True

    def test_timestamp_format(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """Timestamp is ISO-8601 UTC format."""
        write_progress(tmp_output_dir, 100, train_buffer, val_buffer)
        data = json.loads((tmp_output_dir / "progress.json").read_text())
        # Should end with Z and be parseable
        ts = data["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts

    def test_atomic_write_no_tmp_left(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """After write_progress, no .tmp file remains."""
        write_progress(tmp_output_dir, 100, train_buffer, val_buffer)
        assert not (tmp_output_dir / "progress.json.tmp").exists()

    def test_overwrites_existing(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """write_progress overwrites existing progress.json."""
        write_progress(tmp_output_dir, 100, train_buffer, val_buffer)
        write_progress(tmp_output_dir, 200, train_buffer, val_buffer)
        data = json.loads((tmp_output_dir / "progress.json").read_text())
        assert data["documents_consumed"] == 200

    def test_creates_directory_if_needed(
        self, tmp_path: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """write_progress creates the output directory if it doesn't exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        write_progress(new_dir, 100, train_buffer, val_buffer)
        assert (new_dir / "progress.json").exists()


class TestLoadResumeState:
    """Tests for load_resume_state()."""

    def test_returns_none_when_no_file(self, tmp_output_dir: Path) -> None:
        """Returns None when progress.json doesn't exist."""
        result = load_resume_state(tmp_output_dir)
        assert result is None

    def test_loads_valid_progress(self, tmp_output_dir: Path) -> None:
        """Loads a valid progress.json into a ResumeState."""
        progress = {
            "documents_consumed": 5000,
            "train_shards_written": 45,
            "val_shards_written": 1,
            "train_tokens_written": 450000000,
            "val_tokens_written": 4500000,
            "completed": False,
            "timestamp": "2026-07-04T15:30:00Z",
        }
        (tmp_output_dir / "progress.json").write_text(json.dumps(progress))

        result = load_resume_state(tmp_output_dir)
        assert result is not None
        assert isinstance(result, ResumeState)
        assert result.documents_consumed == 5000
        assert result.train_shards_written == 45
        assert result.val_shards_written == 1
        assert result.train_tokens_written == 450000000
        assert result.val_tokens_written == 4500000

    def test_raises_on_malformed_json(self, tmp_output_dir: Path) -> None:
        """Raises ValueError for malformed JSON."""
        (tmp_output_dir / "progress.json").write_text("not valid json {{{")
        with pytest.raises(ValueError, match="malformed"):
            load_resume_state(tmp_output_dir)

    def test_raises_on_missing_fields(self, tmp_output_dir: Path) -> None:
        """Raises ValueError when required fields are missing."""
        progress = {
            "documents_consumed": 100,
            # Missing other required fields
        }
        (tmp_output_dir / "progress.json").write_text(json.dumps(progress))
        with pytest.raises(ValueError, match="missing required fields"):
            load_resume_state(tmp_output_dir)

    def test_raises_on_negative_value(self, tmp_output_dir: Path) -> None:
        """Raises ValueError when a field has a negative value."""
        progress = {
            "documents_consumed": -1,
            "train_shards_written": 0,
            "val_shards_written": 0,
            "train_tokens_written": 0,
            "val_tokens_written": 0,
        }
        (tmp_output_dir / "progress.json").write_text(json.dumps(progress))
        with pytest.raises(ValueError, match="non-negative integer"):
            load_resume_state(tmp_output_dir)

    def test_raises_on_non_integer_value(self, tmp_output_dir: Path) -> None:
        """Raises ValueError when a field has a non-integer value."""
        progress = {
            "documents_consumed": "not a number",
            "train_shards_written": 0,
            "val_shards_written": 0,
            "train_tokens_written": 0,
            "val_tokens_written": 0,
        }
        (tmp_output_dir / "progress.json").write_text(json.dumps(progress))
        with pytest.raises(ValueError, match="non-negative integer"):
            load_resume_state(tmp_output_dir)

    def test_roundtrip_with_write_progress(
        self, tmp_output_dir: Path, train_buffer: ShardBuffer, val_buffer: ShardBuffer
    ) -> None:
        """write_progress output can be loaded by load_resume_state."""
        write_progress(tmp_output_dir, 999, train_buffer, val_buffer)
        result = load_resume_state(tmp_output_dir)
        assert result is not None
        assert result.documents_consumed == 999
        assert result.train_shards_written == train_buffer.shards_written
        assert result.val_shards_written == val_buffer.shards_written
        assert result.train_tokens_written == train_buffer.total_tokens_written
        assert result.val_tokens_written == val_buffer.total_tokens_written

    def test_extra_fields_are_ignored(self, tmp_output_dir: Path) -> None:
        """Extra fields in progress.json are ignored gracefully."""
        progress = {
            "documents_consumed": 100,
            "train_shards_written": 5,
            "val_shards_written": 1,
            "train_tokens_written": 50000,
            "val_tokens_written": 10000,
            "completed": True,
            "timestamp": "2026-07-04T15:30:00Z",
            "extra_field": "should be ignored",
        }
        (tmp_output_dir / "progress.json").write_text(json.dumps(progress))
        result = load_resume_state(tmp_output_dir)
        assert result is not None
        assert result.documents_consumed == 100
