"""MoE-specific evaluation probes.

Five probes for diagnosing routing behavior, expert utilization, and
specialization patterns in Mixture of Experts models. Each probe consumes
routing data from model.get_routing_data() and produces a structured
result dataclass.
"""

import math
from dataclasses import dataclass
from itertools import permutations

import torch


@dataclass
class ExpertUtilizationResult:
    """Per-layer expert utilization fractions.

    Attributes:
        per_layer: Mapping from layer index to a list of fractions (one per expert).
            Each fraction is in [0, 1] and the list sums to 1.0 per layer.
    """

    per_layer: dict[int, list[float]]


@dataclass
class RouterEntropyResult:
    """Per-layer Shannon entropy of the router's token-to-expert distribution.

    Attributes:
        per_layer: Mapping from layer index to Shannon entropy value.
            Maximum entropy = log(num_experts) for uniform routing.
    """

    per_layer: dict[int, float]


@dataclass
class ExpertAffinityResult:
    """Per-layer position-bucket × expert co-occurrence matrix.

    Attributes:
        per_layer: Mapping from layer index to a (num_buckets × num_experts) matrix.
            Each row (bucket) sums to 1.0.
    """

    per_layer: dict[int, list[list[float]]]


@dataclass
class ExpertPairOverlapResult:
    """Per-layer symmetric expert co-selection matrix.

    Attributes:
        per_layer: Mapping from layer index to a (num_experts × num_experts) matrix.
            Entry (i, j) = fraction of tokens with both expert i and j selected.
            Diagonal is 0 (can't co-select same expert). Matrix is symmetric.
    """

    per_layer: dict[int, list[list[float]]]


@dataclass
class RoutingStabilityResult:
    """Permutation-aligned held-out cross-seed top-1 agreement per layer.

    Attributes:
        per_layer: Mapping from layer index to agreement rate after learning an
            expert-label permutation on the first half of tokens and evaluating
            it on the held-out second half.
    """

    per_layer: dict[int, float]


def run_expert_utilization_probe(
    routing_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    num_experts: int,
) -> ExpertUtilizationResult:
    """Compute fraction of tokens routed to each expert per layer.

    For each layer, counts how many times each expert appears in expert_indices
    across all recorded batches. Returns fractions that sum to 1.0 per layer.

    Args:
        routing_data: Dict mapping layer index to list of (expert_indices, expert_weights)
            tuples. expert_indices has shape (batch, seq_len, top_k).
        num_experts: Total number of experts in the model.

    Returns:
        ExpertUtilizationResult with per-layer utilization fractions.
    """
    if not routing_data:
        return ExpertUtilizationResult(per_layer={})

    per_layer: dict[int, list[float]] = {}

    for layer_idx, entries in routing_data.items():
        counts = torch.zeros(num_experts)

        for expert_indices, _expert_weights in entries:
            # expert_indices: (batch, seq_len, top_k)
            for expert_id in range(num_experts):
                counts[expert_id] += (expert_indices == expert_id).sum().item()

        total = counts.sum().item()
        if total > 0:
            fractions = (counts / total).tolist()
        else:
            fractions = [0.0] * num_experts

        per_layer[layer_idx] = fractions

    return ExpertUtilizationResult(per_layer=per_layer)


def run_router_entropy_probe(
    routing_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    num_experts: int,
) -> RouterEntropyResult:
    """Compute Shannon entropy of the expert utilization distribution per layer.

    For each layer, computes H = -Σ(p * log(p)) where p is the expert utilization
    fraction. Maximum entropy = log(num_experts) for uniform routing.

    Args:
        routing_data: Dict mapping layer index to list of (expert_indices, expert_weights)
            tuples.
        num_experts: Total number of experts in the model.

    Returns:
        RouterEntropyResult with per-layer entropy values.
    """
    if not routing_data:
        return RouterEntropyResult(per_layer={})

    # First compute utilization fractions, then derive entropy
    utilization = run_expert_utilization_probe(routing_data, num_experts)

    per_layer: dict[int, float] = {}

    for layer_idx, fractions in utilization.per_layer.items():
        entropy = 0.0
        for p in fractions:
            if p > 0:
                entropy -= p * math.log(p)
        per_layer[layer_idx] = entropy

    return RouterEntropyResult(per_layer=per_layer)


def run_expert_affinity_probe(
    routing_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    num_experts: int,
    seq_len: int,
    num_buckets: int = 4,
) -> ExpertAffinityResult:
    """Compute position-bucket × expert co-occurrence matrix per layer.

    Classifies tokens into position quartiles (num_buckets based on sequence
    position). For each (bucket, expert) pair, counts the fraction of tokens
    in that bucket routed to that expert. Each row (bucket) sums to 1.0.

    Args:
        routing_data: Dict mapping layer index to list of (expert_indices, expert_weights)
            tuples. expert_indices has shape (batch, seq_len, top_k).
        num_experts: Total number of experts in the model.
        seq_len: Sequence length used to determine bucket boundaries.
        num_buckets: Number of position buckets (default 4 for quartiles).

    Returns:
        ExpertAffinityResult with per-layer affinity matrices.
    """
    if not routing_data:
        return ExpertAffinityResult(per_layer={})

    # Compute bucket boundaries
    bucket_boundaries = [
        int(seq_len * i / num_buckets) for i in range(num_buckets + 1)
    ]

    per_layer: dict[int, list[list[float]]] = {}

    for layer_idx, entries in routing_data.items():
        # Matrix: (num_buckets, num_experts) counts
        matrix = [[0.0] * num_experts for _ in range(num_buckets)]

        for expert_indices, _expert_weights in entries:
            # expert_indices: (batch, seq_len, top_k)
            batch_size, actual_seq_len, top_k = expert_indices.shape

            for pos in range(actual_seq_len):
                # Determine which bucket this position belongs to
                bucket = -1
                for b in range(num_buckets):
                    if bucket_boundaries[b] <= pos < bucket_boundaries[b + 1]:
                        bucket = b
                        break
                # Handle edge case: pos == seq_len (shouldn't happen, but safety)
                if bucket == -1:
                    bucket = num_buckets - 1

                # Count expert assignments at this position
                pos_indices = expert_indices[:, pos, :]  # (batch, top_k)
                for expert_id in range(num_experts):
                    count = (pos_indices == expert_id).sum().item()
                    matrix[bucket][expert_id] += count

        # Normalize each row (bucket) to sum to 1.0
        for b in range(num_buckets):
            row_sum = sum(matrix[b])
            if row_sum > 0:
                matrix[b] = [v / row_sum for v in matrix[b]]

        per_layer[layer_idx] = matrix

    return ExpertAffinityResult(per_layer=per_layer)


def run_expert_pair_overlap_probe(
    routing_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    num_experts: int,
) -> ExpertPairOverlapResult:
    """Compute symmetric expert co-selection matrix per layer.

    For top-k >= 2 routing, counts how often each pair (i, j) is co-selected.
    Returns a symmetric matrix where entry (i, j) = fraction of tokens with
    both expert i and expert j selected. Diagonal is 0.

    Args:
        routing_data: Dict mapping layer index to list of (expert_indices, expert_weights)
            tuples. expert_indices has shape (batch, seq_len, top_k).
        num_experts: Total number of experts in the model.

    Returns:
        ExpertPairOverlapResult with per-layer overlap matrices.
    """
    if not routing_data:
        return ExpertPairOverlapResult(per_layer={})

    per_layer: dict[int, list[list[float]]] = {}

    for layer_idx, entries in routing_data.items():
        # Co-selection counts: (num_experts, num_experts)
        co_counts = [[0.0] * num_experts for _ in range(num_experts)]
        total_tokens = 0

        for expert_indices, _expert_weights in entries:
            # expert_indices: (batch, seq_len, top_k)
            batch_size, seq_len, top_k = expert_indices.shape

            if top_k < 2:
                # Can't have co-selection with top-1
                total_tokens += batch_size * seq_len
                continue

            # Flatten to (num_tokens, top_k)
            flat_indices = expert_indices.view(-1, top_k)
            num_tokens = flat_indices.shape[0]
            total_tokens += num_tokens

            # For each pair of top-k slots, count co-occurrences
            for k1 in range(top_k):
                for k2 in range(k1 + 1, top_k):
                    experts_k1 = flat_indices[:, k1]  # (num_tokens,)
                    experts_k2 = flat_indices[:, k2]  # (num_tokens,)

                    for t in range(num_tokens):
                        i = experts_k1[t].item()
                        j = experts_k2[t].item()
                        if i != j:
                            co_counts[i][j] += 1
                            co_counts[j][i] += 1

        # Normalize by total tokens
        if total_tokens > 0:
            matrix = [
                [co_counts[i][j] / total_tokens for j in range(num_experts)]
                for i in range(num_experts)
            ]
        else:
            matrix = [[0.0] * num_experts for _ in range(num_experts)]

        # Ensure diagonal is 0
        for i in range(num_experts):
            matrix[i][i] = 0.0

        per_layer[layer_idx] = matrix

    return ExpertPairOverlapResult(per_layer=per_layer)


def run_routing_stability_probe(
    routing_data_a: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    routing_data_b: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
) -> RoutingStabilityResult:
    """Compute permutation-aligned, held-out top-1 agreement per layer.

    Expert identifiers are arbitrary across independently initialized models.
    For each layer, the first half of matched tokens therefore learns the
    one-to-one permutation from seed-B labels to seed-A labels; agreement is
    measured only on the held-out second half.

    Args:
        routing_data_a: Routing data from seed A.
        routing_data_b: Routing data from seed B.

    Returns:
        RoutingStabilityResult with per-layer agreement rates.
    """
    if not routing_data_a or not routing_data_b:
        return RoutingStabilityResult(per_layer={})

    per_layer: dict[int, float] = {}

    # Compare layers that exist in both datasets
    common_layers = set(routing_data_a.keys()) & set(routing_data_b.keys())

    for layer_idx in common_layers:
        entries_a = routing_data_a[layer_idx]
        entries_b = routing_data_b[layer_idx]

        top1_a_parts: list[torch.Tensor] = []
        top1_b_parts: list[torch.Tensor] = []

        # Compare corresponding entries (same batch index)
        n_entries = min(len(entries_a), len(entries_b))

        for i in range(n_entries):
            indices_a, weights_a = entries_a[i]
            indices_b, weights_b = entries_b[i]

            # Get top-1 expert (highest weight) for each token
            # weights shape: (batch, seq_len, top_k)
            top1_slot_a = weights_a.argmax(dim=-1)  # (batch, seq_len)
            top1_slot_b = weights_b.argmax(dim=-1)  # (batch, seq_len)

            # Gather the expert index at the top-1 slot
            top1_expert_a = indices_a.gather(
                dim=-1, index=top1_slot_a.unsqueeze(-1)
            ).squeeze(-1)  # (batch, seq_len)
            top1_expert_b = indices_b.gather(
                dim=-1, index=top1_slot_b.unsqueeze(-1)
            ).squeeze(-1)  # (batch, seq_len)

            length = min(top1_expert_a.numel(), top1_expert_b.numel())
            top1_a_parts.append(top1_expert_a.reshape(-1)[:length].cpu())
            top1_b_parts.append(top1_expert_b.reshape(-1)[:length].cpu())

        if not top1_a_parts:
            per_layer[layer_idx] = 0.0
            continue

        labels_a = torch.cat(top1_a_parts).to(torch.long)
        labels_b = torch.cat(top1_b_parts).to(torch.long)
        if labels_a.numel() < 2:
            per_layer[layer_idx] = float((labels_a == labels_b).float().mean().item())
            continue

        split = labels_a.numel() // 2
        train_a, test_a = labels_a[:split], labels_a[split:]
        train_b, test_b = labels_b[:split], labels_b[split:]
        num_experts = int(torch.cat((labels_a, labels_b)).max().item()) + 1
        contingency = torch.zeros(num_experts, num_experts, dtype=torch.long)
        for expert_a, expert_b in zip(train_a.tolist(), train_b.tolist()):
            contingency[expert_a, expert_b] += 1

        best_mapping = tuple(range(num_experts))
        best_score = -1
        for mapping in permutations(range(num_experts)):
            score = sum(
                int(contingency[mapping[expert_b], expert_b])
                for expert_b in range(num_experts)
            )
            if score > best_score:
                best_score = score
                best_mapping = mapping
        mapping_tensor = torch.tensor(best_mapping, dtype=torch.long)
        aligned_test_b = mapping_tensor[test_b]
        per_layer[layer_idx] = float((test_a == aligned_test_b).float().mean().item())

    return RoutingStabilityResult(per_layer=per_layer)
