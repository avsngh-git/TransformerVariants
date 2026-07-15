"""Contract tests for generation and long-context benchmark helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.evaluation.benchmarks import (
    benchmark_generation,
    evaluate_long_context,
    inspect_kv_cache,
)


class CacheModel(nn.Module):
    """Tiny deterministic decoder exposing a real per-layer KV cache."""

    def __init__(self, supports_cache: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(seq_len=32, position_encoding="none")
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.supports_cache = supports_cache

    def forward(self, idx, targets=None, kv_cache=None):
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
    result = benchmark_generation(
        CacheModel(), prompt_length=4, new_tokens=2, repeats=1, warmups=0
    )

    assert result["uncached"]["status"] == "ok"
    assert result["uncached"]["tokens_per_second"] > 0
    assert result["cached"]["status"] == "ok"
    assert result["kv_cache"]["status"] == "ok"


def test_long_context_records_native_result_and_unsupported_extension() -> None:
    model = CacheModel()
    tokens = torch.arange(9).remainder(16).unsqueeze(0)

    result = evaluate_long_context(model, tokens, context_lengths=[8, 64])

    assert result["8"]["status"] == "ok"
    assert result["8"]["tokens_per_second"] > 0
    assert result["8"]["perplexity"] > 0
    assert result["64"] == {
        "status": "unsupported",
        "reason": "position encoding 'none' has no declared extrapolation path",
    }
