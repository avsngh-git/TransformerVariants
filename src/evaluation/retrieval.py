"""Zero-shot passkey and needle-in-a-haystack retrieval supplements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from src.evaluation.benchmarks import extend_context


class Tokenizer(Protocol):
    """Tokenizer surface needed by the retrieval evaluator."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class RetrievalExample:
    """One exact-token retrieval prompt."""

    input_ids: list[int]
    expected_token: int
    retrieval_distance: int
    answer: str


DEFAULT_ANSWERS = [
    " blue",
    " river",
    " apple",
    " silver",
    " garden",
    " seven",
    " winter",
    " music",
]

_FILLER = (
    " The archive contains ordinary notes about roads, weather, books, and daily events."
    " None of these routine sentences changes the requested hidden fact."
)


def _task_text(task: str, answer: str) -> tuple[str, str]:
    if task == "passkey":
        return (
            f"\nThe secret passkey is{answer}. Remember this passkey.\n",
            "\nQuestion: What is the secret passkey?\nAnswer:",
        )
    if task == "needle":
        return (
            f"\nImportant fact: the code associated with cedar is{answer}.\n",
            "\nQuestion: What code is associated with cedar?\nAnswer:",
        )
    raise ValueError(f"Unknown retrieval task: {task}")


def _repeat_to_length(tokens: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not tokens:
        raise ValueError("Filler text produced no tokens")
    repeats = (length + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:length]


def build_retrieval_example(
    tokenizer: Tokenizer,
    *,
    task: str,
    context_length: int,
    retrieval_distance: int,
    answer: str,
) -> RetrievalExample:
    """Build an exact-length prompt with a fact at a controlled distance."""
    answer_tokens = tokenizer.encode(answer)
    if len(answer_tokens) != 1:
        raise ValueError(f"Answer must encode to exactly one token: {answer!r}")
    needle_text, query_text = _task_text(task, answer)
    needle = tokenizer.encode(needle_text)
    query = tokenizer.encode(query_text)
    query_start = context_length - len(query)
    needle_end = query_start - retrieval_distance
    needle_start = needle_end - len(needle)
    if retrieval_distance < 1 or needle_start < 0:
        raise ValueError(
            f"context_length={context_length} cannot fit task={task!r} at "
            f"retrieval_distance={retrieval_distance}"
        )

    filler = tokenizer.encode(_FILLER)
    input_ids = (
        _repeat_to_length(filler, needle_start)
        + needle
        + _repeat_to_length(filler, retrieval_distance)
        + query
    )
    if len(input_ids) != context_length:
        raise AssertionError(
            f"Retrieval prompt length is {len(input_ids)}, expected {context_length}"
        )
    return RetrievalExample(
        input_ids=input_ids,
        expected_token=answer_tokens[0],
        retrieval_distance=retrieval_distance,
        answer=answer,
    )


def _available_distance(
    tokenizer: Tokenizer,
    *,
    task: str,
    context_length: int,
    answer: str,
) -> int:
    needle_text, query_text = _task_text(task, answer)
    return context_length - len(tokenizer.encode(needle_text)) - len(
        tokenizer.encode(query_text)
    )


@torch.no_grad()
def run_retrieval_probe(
    model: nn.Module,
    tokenizer: Tokenizer,
    *,
    task: str,
    context_lengths: list[int],
    distance_fractions: list[float],
    trials: int = 5,
    answers: list[str] | None = None,
    device: str = "cuda",
    rope_theta: float = 10000.0,
) -> dict[str, dict]:
    """Evaluate exact recall and expected-token probability by distance."""
    if trials < 1:
        raise ValueError("trials must be positive")
    if not distance_fractions or any(not 0 < fraction <= 1 for fraction in distance_fractions):
        raise ValueError("distance_fractions must contain values in (0, 1]")
    answers = answers or DEFAULT_ANSWERS
    if not answers:
        raise ValueError("answers must not be empty")

    model.eval()
    results: dict[str, dict] = {}
    for context_length in sorted(set(context_lengths)):
        supported, reason = extend_context(model, context_length, rope_theta=rope_theta)
        if not supported:
            results[str(context_length)] = {"status": "unsupported", "reason": reason}
            continue

        records: list[dict] = []
        try:
            for trial in range(trials):
                answer = answers[trial % len(answers)]
                available = _available_distance(
                    tokenizer,
                    task=task,
                    context_length=context_length,
                    answer=answer,
                )
                for fraction in distance_fractions:
                    distance = max(1, min(available, int(available * fraction)))
                    example = build_retrieval_example(
                        tokenizer,
                        task=task,
                        context_length=context_length,
                        retrieval_distance=distance,
                        answer=answer,
                    )
                    inputs = torch.tensor([example.input_ids], dtype=torch.long, device=device)
                    logits, _, _ = model(inputs, kv_cache=None)
                    final_logits = logits[0, -1].float()
                    log_probs = torch.log_softmax(final_logits, dim=-1)
                    expected_log_prob = float(log_probs[example.expected_token].item())
                    top5 = final_logits.topk(5).indices.tolist()
                    records.append(
                        {
                            "trial": trial,
                            "distance_fraction": fraction,
                            "retrieval_distance": distance,
                            "answer": answer.strip(),
                            "expected_token": example.expected_token,
                            "predicted_token": int(final_logits.argmax().item()),
                            "correct": int(final_logits.argmax().item())
                            == example.expected_token,
                            "top5_correct": example.expected_token in top5,
                            "negative_log_likelihood": -expected_log_prob,
                            "expected_probability": math.exp(expected_log_prob),
                        }
                    )
        except ValueError as exc:
            results[str(context_length)] = {"status": "unsupported", "reason": str(exc)}
            continue

        by_distance: dict[str, dict] = {}
        for distance in sorted({record["retrieval_distance"] for record in records}):
            bucket = [record for record in records if record["retrieval_distance"] == distance]
            by_distance[str(distance)] = {
                "accuracy": sum(record["correct"] for record in bucket) / len(bucket),
                "top5_accuracy": sum(record["top5_correct"] for record in bucket) / len(bucket),
                "mean_expected_probability": sum(
                    record["expected_probability"] for record in bucket
                )
                / len(bucket),
                "mean_negative_log_likelihood": sum(
                    record["negative_log_likelihood"] for record in bucket
                )
                / len(bucket),
                "n": len(bucket),
            }
        results[str(context_length)] = {
            "status": "ok",
            "accuracy": sum(record["correct"] for record in records) / len(records),
            "top5_accuracy": sum(record["top5_correct"] for record in records) / len(records),
            "mean_negative_log_likelihood": sum(
                record["negative_log_likelihood"] for record in records
            )
            / len(records),
            "mean_expected_probability": sum(
                record["expected_probability"] for record in records
            )
            / len(records),
            "by_distance": by_distance,
            "records": records,
        }
    return results
