"""Tests for ShardBuffer class in streaming_prepare module."""

import numpy as np
import pytest
from pathlib import Path

from src.data.streaming_prepare import ShardBuffer


class TestShardBufferInit:
    """Tests for ShardBuffer initialization."""

    def test_creates_output_directory(self, tmp_path):
        """ShardBuffer should create output_dir if it doesn't exist."""
        out_dir = tmp_path / "nested" / "output"
        buf = ShardBuffer(out_dir, "train", tokens_per_shard=1000)
        assert out_dir.exists()

    def test_initial_state(self, tmp_path):
        """ShardBuffer should start with zero counters."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=1000)
        assert buf.total_tokens_written == 0
        assert buf.shards_written == 0
        assert buf.buffer_size == 0

    def test_start_shard_idx(self, tmp_path):
        """ShardBuffer should respect start_shard_idx for resumption."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=1000, start_shard_idx=5)
        assert buf.shards_written == 5


class TestShardBufferAppend:
    """Tests for ShardBuffer.append method."""

    def test_append_below_threshold_no_flush(self, tmp_path):
        """Appending fewer tokens than threshold should not flush."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        result = buf.append(list(range(50)))
        assert result == []
        assert buf.buffer_size == 50
        assert buf.total_tokens_written == 0
        assert buf.shards_written == 0

    def test_append_exact_threshold_flushes_one_shard(self, tmp_path):
        """Appending exactly tokens_per_shard should flush one shard."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        tokens = list(range(100))
        result = buf.append(tokens)
        assert result == ["train_000000.bin"]
        assert buf.buffer_size == 0
        assert buf.total_tokens_written == 100
        assert buf.shards_written == 1

    def test_append_above_threshold_flushes_and_keeps_remainder(self, tmp_path):
        """Appending more than threshold should flush and keep remainder."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        tokens = list(range(150))
        result = buf.append(tokens)
        assert result == ["train_000000.bin"]
        assert buf.buffer_size == 50
        assert buf.total_tokens_written == 100

    def test_append_multiple_shards(self, tmp_path):
        """Appending tokens that span multiple shards should flush all."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        tokens = list(range(350))
        result = buf.append(tokens)
        assert result == ["train_000000.bin", "train_000001.bin", "train_000002.bin"]
        assert buf.buffer_size == 50
        assert buf.total_tokens_written == 300
        assert buf.shards_written == 3

    def test_shard_file_content_is_uint16(self, tmp_path):
        """Flushed shards should contain correct uint16 binary data."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=10)
        tokens = list(range(10))
        buf.append(tokens)
        shard_path = tmp_path / "train_000000.bin"
        assert shard_path.exists()
        data = np.fromfile(shard_path, dtype=np.uint16)
        assert list(data) == tokens

    def test_filename_format(self, tmp_path):
        """Filenames should follow {split}_{idx:06d}.bin format."""
        buf = ShardBuffer(tmp_path, "val", tokens_per_shard=10)
        result = buf.append(list(range(30)))
        assert result == ["val_000000.bin", "val_000001.bin", "val_000002.bin"]

    def test_incremental_appends(self, tmp_path):
        """Multiple small appends that eventually reach threshold should flush."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        for i in range(9):
            result = buf.append(list(range(10)))
            assert result == []
        # 10th append pushes to 100
        result = buf.append(list(range(10)))
        assert result == ["train_000000.bin"]
        assert buf.buffer_size == 0


class TestShardBufferFlush:
    """Tests for ShardBuffer.flush method."""

    def test_flush_empty_buffer_returns_none(self, tmp_path):
        """Flushing an empty buffer should return None."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        assert buf.flush() is None

    def test_flush_partial_buffer(self, tmp_path):
        """Flushing a partial buffer should write a smaller-than-threshold shard."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        tokens = list(range(42))
        buf.append(tokens)
        result = buf.flush()
        assert result == "train_000000.bin"
        assert buf.buffer_size == 0
        assert buf.total_tokens_written == 42
        # Verify file content
        data = np.fromfile(tmp_path / "train_000000.bin", dtype=np.uint16)
        assert list(data) == tokens

    def test_flush_after_full_append(self, tmp_path):
        """Flush after an append that already flushed should handle remainder."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        buf.append(list(range(150)))  # Flushes 1 shard, leaves 50 in buffer
        result = buf.flush()
        assert result == "train_000001.bin"
        assert buf.total_tokens_written == 150
        assert buf.shards_written == 2


class TestShardBufferTokenConservation:
    """Tests for the token conservation invariant."""

    def test_invariant_total_tokens_written_plus_buffer_size(self, tmp_path):
        """total_tokens_written + buffer_size == total tokens ever appended."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        total_appended = 0
        for batch_size in [30, 45, 80, 20, 55]:
            buf.append(list(range(batch_size)))
            total_appended += batch_size
            assert buf.total_tokens_written + buf.buffer_size == total_appended

    def test_invariant_after_flush(self, tmp_path):
        """After final flush, all tokens are written and buffer is empty."""
        buf = ShardBuffer(tmp_path, "train", tokens_per_shard=100)
        total = 0
        for batch_size in [30, 45, 80]:
            buf.append(list(range(batch_size)))
            total += batch_size
        buf.flush()
        assert buf.total_tokens_written == total
        assert buf.buffer_size == 0
