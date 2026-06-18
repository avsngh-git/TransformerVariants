"""Tests for seed setting and RNG state utilities."""

import random

import torch

from src.utils.seed import get_rng_state, set_rng_state, set_seed


class TestSetSeed:
    def test_reproducible_torch(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_reproducible_python_random(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_deterministic_mode(self):
        # Should not raise
        set_seed(42, deterministic=True)
        _ = torch.randn(10)


class TestRngState:
    def test_save_and_restore(self):
        set_seed(42)
        _ = torch.randn(5)  # advance state
        state = get_rng_state()

        # Generate some values
        expected = torch.randn(10)

        # Restore state and generate again
        set_rng_state(state)
        actual = torch.randn(10)

        assert torch.equal(expected, actual)

    def test_state_contains_expected_keys(self):
        set_seed(1)
        state = get_rng_state()
        assert "python" in state
        assert "torch_cpu" in state
