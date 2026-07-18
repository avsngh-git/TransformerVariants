"""Tests for zero-shot passkey and needle retrieval supplements."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts.evaluate_retrieval import _aggregate
from src.data.tokenizer import get_tokenizer
from src.evaluation.retrieval import build_retrieval_example, run_retrieval_probe


class ConstantAnswerModel(nn.Module):
    def __init__(self, answer_token: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            seq_len=128,
            position_encoding="none",
            vocab_size=50257,
        )
        self.anchor = nn.Parameter(torch.zeros(()))
        self.answer_token = answer_token

    def forward(self, idx, targets=None, kv_cache=None):
        logits = torch.full((*idx.shape, self.config.vocab_size), -10.0)
        logits[..., self.answer_token] = 10.0
        return logits, None, [None]


def test_retrieval_example_has_exact_context_and_distance() -> None:
    tokenizer = get_tokenizer("gpt2")
    example = build_retrieval_example(
        tokenizer,
        task="passkey",
        context_length=128,
        retrieval_distance=32,
        answer=" blue",
    )

    assert len(example.input_ids) == 128
    assert example.retrieval_distance == 32
    assert tokenizer.decode([example.expected_token]) == " blue"


def test_retrieval_probe_reports_exact_and_probability_metrics() -> None:
    tokenizer = get_tokenizer("gpt2")
    answer_token = tokenizer.encode(" blue")[0]
    model = ConstantAnswerModel(answer_token).eval()

    result = run_retrieval_probe(
        model,
        tokenizer,
        task="needle",
        context_lengths=[128],
        distance_fractions=[0.25, 0.75],
        trials=1,
        answers=[" blue"],
        device="cpu",
    )

    assert result["128"]["status"] == "ok"
    assert result["128"]["accuracy"] == 1.0
    assert result["128"]["top5_accuracy"] == 1.0
    assert result["128"]["mean_expected_probability"] > 0.99
    assert len(result["128"]["by_distance"]) == 2
    assert all(
        bucket["mean_negative_log_likelihood"] < 1e-3
        for bucket in result["128"]["by_distance"].values()
    )


def test_cross_seed_retrieval_aggregation_preserves_distance_and_nll() -> None:
    def run(seed: int, accuracy: float, nll: float) -> dict:
        measurement = {
            "status": "ok",
            "accuracy": accuracy,
            "top5_accuracy": 1.0,
            "mean_expected_probability": 0.5,
            "mean_negative_log_likelihood": nll,
            "by_distance": {
                "64": {
                    "accuracy": accuracy,
                    "top5_accuracy": 1.0,
                    "mean_expected_probability": 0.5,
                    "mean_negative_log_likelihood": nll,
                    "n": 2,
                }
            },
        }
        return {
            "variant": "modern",
            "seed": seed,
            "configurations": {"rope_standard": {"passkey": {"128": measurement}}},
        }

    aggregate = _aggregate([run(42, 0.5, 1.0), run(137, 1.0, 0.5)])
    result = aggregate["modern"]["rope_standard"]["passkey"]["128"]

    assert result["mean_negative_log_likelihood"]["mean"] == 0.75
    assert result["by_distance"]["64"]["accuracy"]["mean"] == 0.75
    assert result["by_distance"]["64"]["mean_negative_log_likelihood"]["n"] == 2
