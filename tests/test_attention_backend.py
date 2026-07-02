"""Tests for the attention_backend and window_size fields on ModelConfig."""

import pytest

from src.models.config import ModelConfig


def test_default_attention_backend_is_sdpa():
    """ModelConfig defaults attention_backend to 'sdpa'."""
    config = ModelConfig()
    assert config.attention_backend == "sdpa"


def test_explicit_attention_backend_flash_attn():
    """ModelConfig accepts an explicit 'flash_attn' value for attention_backend."""
    config = ModelConfig(attention_backend="flash_attn")
    assert config.attention_backend == "flash_attn"


# --- window_size field tests (Requirements 1.1, 1.4) ---

def test_window_size_defaults_to_none():
    """ModelConfig defaults window_size to None (full causal attention)."""
    config = ModelConfig()
    assert config.window_size is None


def test_window_size_valid_minimum():
    """window_size=1 (minimum valid value) is accepted."""
    config = ModelConfig(seq_len=512, window_size=1)
    assert config.window_size == 1


def test_window_size_valid_equals_seq_len():
    """window_size equal to seq_len (maximum valid value) is accepted."""
    config = ModelConfig(seq_len=512, window_size=512)
    assert config.window_size == 512


def test_window_size_valid_midrange():
    """A typical mid-range window_size (seq_len // 4) is accepted."""
    config = ModelConfig(seq_len=512, window_size=128)
    assert config.window_size == 128


def test_window_size_zero_raises():
    """window_size=0 is below the valid range and raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be between 1 and seq_len"):
        ModelConfig(seq_len=512, window_size=0)


def test_window_size_negative_raises():
    """A negative window_size raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be between 1 and seq_len"):
        ModelConfig(seq_len=512, window_size=-1)


def test_window_size_exceeds_seq_len_raises():
    """window_size greater than seq_len raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be between 1 and seq_len"):
        ModelConfig(seq_len=512, window_size=513)


def test_window_size_error_message_includes_seq_len_and_value():
    """The ValueError message includes the valid range (seq_len) and the bad value."""
    with pytest.raises(ValueError, match=r"seq_len \(64\)") as exc_info:
        ModelConfig(seq_len=64, window_size=100)
    assert "100" in str(exc_info.value)
