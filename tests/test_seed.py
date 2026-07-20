"""Tests for seed setting and RNG state utilities."""

import random
from unittest.mock import Mock, patch

import pytest
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

    def test_restore_normalizes_cpu_rng_state_after_checkpoint_device_mapping(self):
        """CPU RNG restoration must receive a CPU tensor even after a CUDA map."""
        state = get_rng_state()
        mapped_state = Mock()
        mapped_state.cpu.return_value = state["torch_cpu"]
        state["torch_cpu"] = mapped_state

        with (
            patch("torch.random.set_rng_state") as set_cpu_rng_state,
            patch("torch.cuda.is_available", return_value=False),
        ):
            set_rng_state(state)

        mapped_state.cpu.assert_called_once_with()
        set_cpu_rng_state.assert_called_once_with(mapped_state.cpu.return_value)

    def test_restore_normalizes_cuda_rng_states_after_checkpoint_device_mapping(self):
        """CUDA generator states are CPU byte tensors at the restoration API boundary."""
        state = get_rng_state()
        mapped_cuda_state = Mock()
        normalized_cuda_state = torch.random.get_rng_state()
        mapped_cuda_state.cpu.return_value = normalized_cuda_state
        state["torch_cuda"] = [mapped_cuda_state]

        with (
            patch("torch.random.set_rng_state"),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_rng_state_all") as set_cuda_rng_states,
        ):
            set_rng_state(state)

        mapped_cuda_state.cpu.assert_called_once_with()
        set_cuda_rng_states.assert_called_once_with([normalized_cuda_state])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_restore_checkpoint_state_mapped_to_cuda(self):
        """Reproduce the matrix rollback path with RNG tensors mapped to CUDA."""
        state = get_rng_state()
        state["torch_cpu"] = state["torch_cpu"].cuda()
        state["torch_cuda"] = [cuda_state.cuda() for cuda_state in state["torch_cuda"]]

        set_rng_state(state)
