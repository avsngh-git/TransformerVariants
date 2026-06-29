"""Tests for the attention_backend field on ModelConfig."""

from src.models.config import ModelConfig


def test_default_attention_backend_is_sdpa():
    """ModelConfig defaults attention_backend to 'sdpa'."""
    config = ModelConfig()
    assert config.attention_backend == "sdpa"


def test_explicit_attention_backend_flash_attn():
    """ModelConfig accepts an explicit 'flash_attn' value for attention_backend."""
    config = ModelConfig(attention_backend="flash_attn")
    assert config.attention_backend == "flash_attn"
