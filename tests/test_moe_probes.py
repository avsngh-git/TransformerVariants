"""Tests for MoE evaluation probes."""

import torch
import pytest

from src.evaluation.moe_probes import (
    ExpertUtilizationResult,
    RouterEntropyResult,
    ExpertAffinityResult,
    ExpertPairOverlapResult,
    RoutingStabilityResult,
    run_expert_utilization_probe,
    run_router_entropy_probe,
    run_expert_affinity_probe,
    run_expert_pair_overlap_probe,
    run_routing_stability_probe,
)


# --- Fixtures ---


@pytest.fixture
def num_experts():
    return 4


@pytest.fixture
def synthetic_routing_data(num_experts):
    """Create synthetic routing data for testing.

    Simulates 2 batches of routing data for 2 layers (layer 0 and layer 2).
    Each batch has shape (2, 8, 2) for (batch, seq_len, top_k=2).
    """
    batch_size = 2
    seq_len = 8
    top_k = 2

    # Create deterministic routing data
    torch.manual_seed(42)
    data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    for layer_idx in [0, 2]:
        entries = []
        for _ in range(2):  # 2 recorded batches
            # Random expert indices in [0, num_experts)
            expert_indices = torch.randint(
                0, num_experts, (batch_size, seq_len, top_k)
            )
            # Random weights that sum to 1 per token
            raw_weights = torch.rand(batch_size, seq_len, top_k)
            expert_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
            entries.append((expert_indices, expert_weights))
        data[layer_idx] = entries

    return data


# --- Expert Utilization Probe ---


class TestExpertUtilizationProbe:
    def test_returns_correct_type(self, synthetic_routing_data, num_experts):
        result = run_expert_utilization_probe(synthetic_routing_data, num_experts)
        assert isinstance(result, ExpertUtilizationResult)

    def test_fractions_sum_to_one(self, synthetic_routing_data, num_experts):
        result = run_expert_utilization_probe(synthetic_routing_data, num_experts)
        for layer_idx, fractions in result.per_layer.items():
            assert len(fractions) == num_experts
            assert abs(sum(fractions) - 1.0) < 1e-6

    def test_fractions_in_valid_range(self, synthetic_routing_data, num_experts):
        result = run_expert_utilization_probe(synthetic_routing_data, num_experts)
        for layer_idx, fractions in result.per_layer.items():
            for f in fractions:
                assert 0.0 <= f <= 1.0

    def test_empty_input(self, num_experts):
        result = run_expert_utilization_probe({}, num_experts)
        assert isinstance(result, ExpertUtilizationResult)
        assert result.per_layer == {}


# --- Router Entropy Probe ---


class TestRouterEntropyProbe:
    def test_returns_correct_type(self, synthetic_routing_data, num_experts):
        result = run_router_entropy_probe(synthetic_routing_data, num_experts)
        assert isinstance(result, RouterEntropyResult)

    def test_entropy_non_negative(self, synthetic_routing_data, num_experts):
        result = run_router_entropy_probe(synthetic_routing_data, num_experts)
        for layer_idx, entropy in result.per_layer.items():
            assert entropy >= 0.0

    def test_empty_input(self, num_experts):
        result = run_router_entropy_probe({}, num_experts)
        assert isinstance(result, RouterEntropyResult)
        assert result.per_layer == {}


# --- Expert Affinity Probe ---


class TestExpertAffinityProbe:
    def test_returns_correct_type(self, synthetic_routing_data, num_experts):
        result = run_expert_affinity_probe(
            synthetic_routing_data, num_experts, seq_len=8, num_buckets=4
        )
        assert isinstance(result, ExpertAffinityResult)

    def test_rows_sum_to_one(self, synthetic_routing_data, num_experts):
        result = run_expert_affinity_probe(
            synthetic_routing_data, num_experts, seq_len=8, num_buckets=4
        )
        for layer_idx, matrix in result.per_layer.items():
            assert len(matrix) == 4  # num_buckets
            for row in matrix:
                assert len(row) == num_experts
                assert abs(sum(row) - 1.0) < 1e-6

    def test_empty_input(self, num_experts):
        result = run_expert_affinity_probe({}, num_experts, seq_len=8, num_buckets=4)
        assert isinstance(result, ExpertAffinityResult)
        assert result.per_layer == {}


# --- Expert Pair Overlap Probe ---


class TestExpertPairOverlapProbe:
    def test_returns_correct_type(self, synthetic_routing_data, num_experts):
        result = run_expert_pair_overlap_probe(synthetic_routing_data, num_experts)
        assert isinstance(result, ExpertPairOverlapResult)

    def test_matrix_is_symmetric(self, synthetic_routing_data, num_experts):
        result = run_expert_pair_overlap_probe(synthetic_routing_data, num_experts)
        for layer_idx, matrix in result.per_layer.items():
            assert len(matrix) == num_experts
            for i in range(num_experts):
                assert len(matrix[i]) == num_experts
                for j in range(num_experts):
                    assert abs(matrix[i][j] - matrix[j][i]) < 1e-10

    def test_diagonal_is_zero(self, synthetic_routing_data, num_experts):
        result = run_expert_pair_overlap_probe(synthetic_routing_data, num_experts)
        for layer_idx, matrix in result.per_layer.items():
            for i in range(num_experts):
                assert matrix[i][i] == 0.0

    def test_values_in_valid_range(self, synthetic_routing_data, num_experts):
        result = run_expert_pair_overlap_probe(synthetic_routing_data, num_experts)
        for layer_idx, matrix in result.per_layer.items():
            for row in matrix:
                for val in row:
                    assert 0.0 <= val <= 1.0

    def test_empty_input(self, num_experts):
        result = run_expert_pair_overlap_probe({}, num_experts)
        assert isinstance(result, ExpertPairOverlapResult)
        assert result.per_layer == {}


# --- Routing Stability Probe ---


class TestRoutingStabilityProbe:
    def test_returns_correct_type(self, synthetic_routing_data):
        # Use same data for both seeds (should give perfect agreement)
        result = run_routing_stability_probe(
            synthetic_routing_data, synthetic_routing_data
        )
        assert isinstance(result, RoutingStabilityResult)

    def test_same_seed_gives_perfect_agreement(self, synthetic_routing_data):
        result = run_routing_stability_probe(
            synthetic_routing_data, synthetic_routing_data
        )
        for layer_idx, rate in result.per_layer.items():
            assert abs(rate - 1.0) < 1e-6

    def test_agreement_rate_in_valid_range(self, synthetic_routing_data, num_experts):
        # Create different routing data for seed B
        torch.manual_seed(99)
        data_b: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for layer_idx in [0, 2]:
            entries = []
            for _ in range(2):
                expert_indices = torch.randint(0, num_experts, (2, 8, 2))
                raw_weights = torch.rand(2, 8, 2)
                expert_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
                entries.append((expert_indices, expert_weights))
            data_b[layer_idx] = entries

        result = run_routing_stability_probe(synthetic_routing_data, data_b)
        for layer_idx, rate in result.per_layer.items():
            assert 0.0 <= rate <= 1.0

    def test_empty_input_a(self, synthetic_routing_data):
        result = run_routing_stability_probe({}, synthetic_routing_data)
        assert isinstance(result, RoutingStabilityResult)
        assert result.per_layer == {}

    def test_empty_input_b(self, synthetic_routing_data):
        result = run_routing_stability_probe(synthetic_routing_data, {})
        assert isinstance(result, RoutingStabilityResult)
        assert result.per_layer == {}

    def test_both_empty(self):
        result = run_routing_stability_probe({}, {})
        assert isinstance(result, RoutingStabilityResult)
        assert result.per_layer == {}
