"""Contract tests for the cross-seed MoE routing report."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.evaluate_moe_routing import _pairwise_stability, _seed


def _capture(labels: list[int]) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    top1 = torch.tensor(labels).view(1, -1, 1)
    second = ((top1 + 1) % 4).to(torch.long)
    indices = torch.cat((top1, second), dim=-1)
    weights = torch.tensor([0.9, 0.1]).view(1, 1, 2).expand_as(indices).float()
    return {0: [(indices, weights)]}


def test_pairwise_stability_is_permutation_aligned_and_held_out() -> None:
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    permutation = [2, 0, 3, 1]
    permuted = [permutation[label] for label in labels]

    result = _pairwise_stability([(42, _capture(labels)), (137, _capture(permuted))])

    assert result["pairwise"][0]["per_layer"]["0"] == pytest.approx(1.0)
    assert result["per_layer"]["0"]["mean"] == pytest.approx(1.0)


def test_seed_is_read_from_canonical_run_directory() -> None:
    assert _seed(Path("runs/main_500m_5seed/moe_interleaved_s31415")) == 31415
