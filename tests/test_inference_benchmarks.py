"""Contract tests for generation and long-context benchmark helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.evaluation.benchmarks import (
    aggregate_long_context_runs,
    benchmark_generation,
    evaluate_long_context,
    inspect_kv_cache,
    rank_long_context_variants,
)


class CacheModel(nn.Module):
    """Tiny deterministic decoder exposing a real per-layer KV cache."""

    def __init__(self, supports_cache: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(seq_len=32, position_encoding="none")
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.targeted_calls = 0
        self.target_free_calls = 0
        self.supports_cache = supports_cache

    def forward(self, idx, targets=None, kv_cache=None):
        if targets is None:
            self.target_free_calls += 1
        else:
            self.targeted_calls += 1
        hidden = self.embedding(idx)
        logits = self.head(hidden)
        loss = (
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if targets is not None
            else None
        )
        if not self.supports_cache:
            cache = [None]
        else:
            old = kv_cache[0][0] if kv_cache and kv_cache[0] is not None else None
            key = hidden.unsqueeze(1)
            value = key.clone()
            if old is not None:
                key = torch.cat((old, key), dim=2)
                value = torch.cat((kv_cache[0][1], value), dim=2)
            cache = [(key, value)]
        return logits, loss, cache


class TailScoringModel(nn.Module):
    """Decoder whose final two predictions are good and earlier ones are bad."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            seq_len=4,
            position_encoding="rope",
            d_head=2,
            vocab_size=16,
        )
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_inputs: list[torch.Tensor] = []

    def forward(self, idx, targets=None, kv_cache=None):
        self.seen_inputs.append(idx.detach().cpu())
        logits = torch.full((*idx.shape, 16), -10.0, device=idx.device)
        logits[..., 1] = 10.0
        logits[:, -2:, 0] = 10.0
        logits[:, -2:, 1] = -10.0
        return logits, None, [None]


def test_inspect_kv_cache_reports_bytes_or_explicit_unsupported() -> None:
    prompt = torch.tensor([[1, 2, 3, 4]])

    supported = inspect_kv_cache(CacheModel(), prompt)
    unsupported = inspect_kv_cache(CacheModel(supports_cache=False), prompt)

    assert supported["status"] == "ok"
    assert supported["bytes"] > 0
    assert unsupported == {
        "status": "unsupported",
        "reason": "model did not return a reusable prompt cache",
    }


def test_generation_benchmark_keeps_cached_and_uncached_results_separate() -> None:
    result = benchmark_generation(CacheModel(), prompt_length=4, new_tokens=2, repeats=1, warmups=0)

    assert result["uncached"]["status"] == "ok"
    assert result["uncached"]["tokens_per_second"] > 0
    assert result["cached"]["status"] == "ok"
    assert result["kv_cache"]["status"] == "ok"


def test_long_context_records_native_result_and_unsupported_extension() -> None:
    model = CacheModel()
    tokens = torch.arange(9).remainder(16).unsqueeze(0)

    result = evaluate_long_context(
        model,
        tokens,
        context_lengths=[8, 64],
        tail_tokens=2,
    )

    assert result["8"]["status"] == "ok"
    assert result["8"]["prefill_tokens_per_second"]["mean"] > 0
    assert result["8"]["val_loss"]["n"] == 1
    assert result["8"]["perplexity"]["mean"] > 0
    assert result["64"] == {
        "status": "unsupported",
        "reason": "position encoding 'none' has no declared extrapolation path",
    }
    assert model.target_free_calls == 1
    assert model.targeted_calls == 0


def test_long_context_scores_the_same_tail_across_multiple_windows() -> None:
    model = TailScoringModel()
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0, 0, 0],
            [9, 8, 7, 6, 5, 4, 0, 0, 0],
        ]
    )

    result = evaluate_long_context(
        model,
        tokens,
        context_lengths=[4, 8],
        tail_tokens=2,
    )

    for context_length in ("4", "8"):
        measurement = result[context_length]
        assert measurement["status"] == "ok"
        assert measurement["tail_tokens"] == 2
        assert measurement["val_loss"]["n"] == 2
        assert measurement["val_loss"]["mean"] < 1e-6
        assert len(measurement["windows"]) == 2

    assert torch.equal(model.seen_inputs[0], tokens[0:1, 4:8])
    assert torch.equal(model.seen_inputs[1], tokens[1:2, 4:8])
    assert torch.equal(model.seen_inputs[2], tokens[0:1, :8])
    assert torch.equal(model.seen_inputs[3], tokens[1:2, :8])


def test_long_context_rankings_cover_quality_retention_and_throughput() -> None:
    def measurement(perplexity: float, ratio: float, throughput: float) -> dict:
        return {
            "status": "ok",
            "perplexity": {"mean": perplexity, "std": 1.0, "n": 3},
            "perplexity_ratio": {"mean": ratio, "std": 0.1, "n": 3},
            "prefill_tokens_per_second": {
                "mean": throughput,
                "std": 10.0,
                "n": 3,
            },
        }

    variants = {
        "stable": {"long_context": {"4096": measurement(60.0, 1.01, 100.0)}},
        "quality": {"long_context": {"4096": measurement(50.0, 1.20, 80.0)}},
        "fast": {"long_context": {"4096": measurement(70.0, 1.10, 200.0)}},
        "unsupported": {"long_context": {"4096": {"status": "unsupported"}}},
    }

    rankings = rank_long_context_variants(variants, context_length=4096)

    assert [entry["variant"] for entry in rankings["quality"]] == [
        "quality",
        "stable",
        "fast",
    ]
    assert [entry["variant"] for entry in rankings["retention"]] == [
        "stable",
        "fast",
        "quality",
    ]
    assert [entry["variant"] for entry in rankings["throughput"]] == [
        "fast",
        "stable",
        "quality",
    ]


def test_long_context_aggregation_uses_seed_means_and_paired_degradation() -> None:
    def run(seed: int, native_loss: float, extended_loss: float, throughput: float) -> dict:
        return {
            "seed": seed,
            "checkpoint_dir": f"modern_s{seed}",
            "long_context": {
                "4": {
                    "status": "ok",
                    "val_loss": {"mean": native_loss, "std": 0.05, "n": 8},
                    "prefill_tokens_per_second": {
                        "mean": throughput,
                        "std": 1.0,
                        "n": 8,
                    },
                },
                "8": {
                    "status": "ok",
                    "val_loss": {"mean": extended_loss, "std": 0.05, "n": 8},
                    "prefill_tokens_per_second": {
                        "mean": throughput / 2,
                        "std": 1.0,
                        "n": 8,
                    },
                },
            },
        }

    aggregate = aggregate_long_context_runs(
        [
            run(42, 1.0, 1.2, 100.0),
            run(137, 1.2, 1.5, 120.0),
            run(2024, 1.4, 1.8, 110.0),
        ],
        baseline_context=4,
    )

    assert aggregate["4"]["val_loss"] == pytest.approx({"mean": 1.2, "std": 0.2, "n": 3})
    assert aggregate["8"]["val_loss"] == pytest.approx({"mean": 1.5, "std": 0.3, "n": 3})
    assert aggregate["8"]["paired_delta_loss"] == pytest.approx({"mean": 0.3, "std": 0.1, "n": 3})
    assert aggregate["8"]["perplexity_ratio"]["mean"] == pytest.approx(1.3543620878)
