"""Inference throughput, KV-cache, and long-context benchmark helpers."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import torch
from torch import nn

from src.models.rope import precompute_rope_frequencies


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_tensors(value: object) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _cache_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _cache_tensors(item)


def _cache_is_reusable(cache: object) -> bool:
    if not isinstance(cache, (list, tuple)) or not cache:
        return False
    return all(item is not None and next(_cache_tensors(item), None) is not None for item in cache)


def _cache_bytes(cache: object) -> int:
    """Count unique tensor storages held by an opaque cache."""
    storages: dict[int, int] = {}
    for tensor in _cache_tensors(cache):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


@torch.no_grad()
def inspect_kv_cache(model: nn.Module, prompt: torch.Tensor) -> dict:
    """Inspect whether a prompt produces a reusable cache and report its bytes."""
    device = _device_of(model)
    prompt = prompt.to(device)
    try:
        _, _, cache = model(prompt, kv_cache=None)
    except (AssertionError, NotImplementedError, RuntimeError) as exc:
        return {"status": "unsupported", "reason": str(exc)}
    if not _cache_is_reusable(cache):
        return {
            "status": "unsupported",
            "reason": "model did not return a reusable prompt cache",
        }
    return {"status": "ok", "bytes": _cache_bytes(cache)}


def _peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


@torch.no_grad()
def _decode_once(
    model: nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    use_cache: bool,
) -> tuple[float, int]:
    device = prompt.device
    idx = prompt
    cache = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(new_tokens):
        input_ids = idx[:, -1:] if use_cache and cache is not None else idx
        logits, _, new_cache = model(input_ids, kv_cache=cache if use_cache else None)
        if use_cache:
            cache = new_cache
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    _synchronize(device)
    return time.perf_counter() - started, _peak_memory(device)


@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    *,
    prompt_length: int = 128,
    new_tokens: int = 32,
    batch_size: int = 1,
    repeats: int = 3,
    warmups: int = 1,
    seed: int = 1234,
) -> dict:
    """Measure end-to-end greedy decode with and without a reusable cache."""
    if prompt_length + new_tokens > model.config.seq_len:
        raise ValueError("prompt_length + new_tokens exceeds model context length")
    device = _device_of(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    vocab_size = getattr(model.config, "vocab_size", 16)
    prompt = torch.randint(
        0,
        vocab_size,
        (batch_size, prompt_length),
        generator=generator,
        device=device,
    )
    cache_result = inspect_kv_cache(model, prompt)

    def measure(use_cache: bool) -> dict:
        for _ in range(warmups):
            _decode_once(model, prompt, new_tokens, use_cache=use_cache)
        samples = [
            _decode_once(model, prompt, new_tokens, use_cache=use_cache) for _ in range(repeats)
        ]
        seconds = [sample[0] for sample in samples]
        return {
            "status": "ok",
            "tokens_per_second": batch_size * new_tokens / (sum(seconds) / len(seconds)),
            "seconds_mean": sum(seconds) / len(seconds),
            "peak_memory_bytes": max(sample[1] for sample in samples),
            "repeats": repeats,
        }

    result = {"uncached": measure(False), "kv_cache": cache_result}
    if cache_result["status"] == "ok":
        result["cached"] = measure(True)
    else:
        result["cached"] = {
            "status": "unsupported",
            "reason": cache_result["reason"],
        }
    return result


def _extend_context(model: nn.Module, context_length: int) -> tuple[bool, str | None]:
    native = int(model.config.seq_len)
    if context_length <= native:
        return True, None
    position_encoding = getattr(model.config, "position_encoding", "unknown")
    if position_encoding not in {"rope", "alibi"}:
        return (
            False,
            f"position encoding '{position_encoding}' has no declared extrapolation path",
        )

    model.config.seq_len = context_length
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "seq_len"):
            config.seq_len = context_length
        if hasattr(module, "seq_len") and isinstance(module.seq_len, int):
            module.seq_len = context_length
        if hasattr(module, "rope_cos") and hasattr(module, "rope_sin"):
            d_head = getattr(module, "d_head", model.config.d_head)
            cos, sin = precompute_rope_frequencies(
                d_head, context_length, device=module.rope_cos.device
            )
            module.rope_cos = cos.to(dtype=module.rope_cos.dtype)
            module.rope_sin = sin.to(dtype=module.rope_sin.dtype)
    return True, None


def _sample_summary(values: list[float]) -> dict[str, float | int]:
    """Return a named estimate with sample standard deviation."""
    count = len(values)
    mean = sum(values) / count
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1)) if count > 1 else 0.0
    )
    return {"mean": mean, "std": std, "n": count}


def aggregate_long_context_runs(
    runs: Iterable[dict],
    *,
    baseline_context: int,
) -> dict[str, dict]:
    """Aggregate window means across independent checkpoint seeds."""
    run_list = list(runs)
    context_keys = sorted(
        {int(context) for run in run_list for context in run.get("long_context", {})}
    )
    baseline_key = str(baseline_context)
    results: dict[str, dict] = {}

    for context_length in context_keys:
        context_key = str(context_length)
        supported: list[tuple[dict, dict]] = []
        unsupported: list[dict] = []
        for run in run_list:
            measurement = run.get("long_context", {}).get(context_key)
            if measurement is not None and measurement.get("status") == "ok":
                supported.append((run, measurement))
            else:
                unsupported.append(
                    {
                        "seed": run.get("seed"),
                        "checkpoint_dir": run.get("checkpoint_dir"),
                        "status": (
                            measurement.get("status") if measurement is not None else "unavailable"
                        ),
                        "reason": (
                            measurement.get("reason", "context result is missing")
                            if measurement is not None
                            else "context result is missing"
                        ),
                    }
                )

        if not supported:
            results[context_key] = {
                "status": "unsupported",
                "supported_checkpoints": 0,
                "total_checkpoints": len(run_list),
                "unsupported": unsupported,
            }
            continue

        seed_losses = [float(measurement["val_loss"]["mean"]) for _, measurement in supported]
        seed_throughputs = [
            float(measurement["prefill_tokens_per_second"]["mean"]) for _, measurement in supported
        ]
        aggregate = {
            "status": "ok" if len(supported) == len(run_list) else "partial",
            "supported_checkpoints": len(supported),
            "total_checkpoints": len(run_list),
            "window_count": sum(int(measurement["val_loss"]["n"]) for _, measurement in supported),
            "seeds": [run.get("seed") for run, _ in supported],
            "val_loss": _sample_summary(seed_losses),
            "perplexity": _sample_summary([math.exp(loss) for loss in seed_losses]),
            "prefill_tokens_per_second": _sample_summary(seed_throughputs),
            "unsupported": unsupported,
        }

        paired_delta_losses: list[float] = []
        for run, measurement in supported:
            baseline = run.get("long_context", {}).get(baseline_key)
            if baseline is None or baseline.get("status") != "ok":
                continue
            paired_delta_losses.append(
                float(measurement["val_loss"]["mean"]) - float(baseline["val_loss"]["mean"])
            )
        if paired_delta_losses:
            aggregate["paired_delta_loss"] = _sample_summary(paired_delta_losses)
            aggregate["perplexity_ratio"] = _sample_summary(
                [math.exp(delta) for delta in paired_delta_losses]
            )

        results[context_key] = aggregate

    return results


def rank_long_context_variants(
    variants: dict[str, dict],
    *,
    context_length: int,
) -> dict[str, list[dict]]:
    """Rank supported variants on quality, retention, and prefill throughput."""
    context_key = str(context_length)
    metric_keys = {
        "quality": "perplexity",
        "retention": "perplexity_ratio",
        "throughput": "prefill_tokens_per_second",
    }
    rankings: dict[str, list[dict]] = {name: [] for name in metric_keys}

    for variant_name, variant_benchmark in variants.items():
        measurement = variant_benchmark.get("long_context", {}).get(context_key, {})
        if measurement.get("status") not in {"ok", "partial"}:
            continue
        for ranking_name, metric_key in metric_keys.items():
            estimate = measurement.get(metric_key)
            if not isinstance(estimate, dict) or estimate.get("mean") is None:
                continue
            rankings[ranking_name].append(
                {
                    "variant": variant_name,
                    "estimate": estimate,
                }
            )

    rankings["quality"].sort(key=lambda entry: float(entry["estimate"]["mean"]))
    rankings["retention"].sort(key=lambda entry: abs(float(entry["estimate"]["mean"]) - 1.0))
    rankings["throughput"].sort(
        key=lambda entry: float(entry["estimate"]["mean"]),
        reverse=True,
    )
    for entries in rankings.values():
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank
    return rankings


@torch.no_grad()
def evaluate_long_context(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    context_lengths: Iterable[int] = (1024, 2048, 4096),
    tail_tokens: int = 256,
) -> dict[str, dict]:
    """Score the same held-out tail under several available context lengths."""
    lengths = sorted(set(context_lengths))
    if not lengths:
        raise ValueError("context_lengths must not be empty")
    if tokens.ndim != 2 or tokens.size(0) == 0:
        raise ValueError("tokens must contain one or more two-dimensional windows")
    if tail_tokens <= 0 or tail_tokens > lengths[0]:
        raise ValueError("tail_tokens must be positive and fit within every context length")

    device = _device_of(model)
    tokens = tokens.to(device)
    results: dict[str, dict] = {}
    for context_length in lengths:
        supported, reason = _extend_context(model, context_length)
        if not supported:
            results[str(context_length)] = {
                "status": "unsupported",
                "reason": reason,
            }
            continue
        if tokens.size(1) < context_length + 1:
            results[str(context_length)] = {
                "status": "unavailable",
                "reason": f"requires {context_length + 1} validation tokens per window",
            }
            continue

        windows: list[dict] = []
        try:
            for window_index in range(tokens.size(0)):
                inputs = tokens[
                    window_index : window_index + 1,
                    -(context_length + 1) : -1,
                ]
                targets = tokens[window_index : window_index + 1, -tail_tokens:]

                _synchronize(device)
                started = time.perf_counter()
                logits, _, _ = model(inputs, targets=None, kv_cache=None)
                _synchronize(device)
                prefill_elapsed = time.perf_counter() - started

                tail_logits = logits[:, -tail_tokens:, :].float()
                loss = torch.nn.functional.cross_entropy(
                    tail_logits.reshape(-1, tail_logits.size(-1)),
                    targets.reshape(-1),
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite tail validation loss")
                val_loss = float(loss.item())
                windows.append(
                    {
                        "window_index": window_index,
                        "val_loss": val_loss,
                        "perplexity": math.exp(val_loss),
                        "prefill_seconds": prefill_elapsed,
                        "prefill_tokens_per_second": context_length / prefill_elapsed,
                    }
                )

            results[str(context_length)] = {
                "status": "ok",
                "tail_tokens": tail_tokens,
                "val_loss": _sample_summary([window["val_loss"] for window in windows]),
                "perplexity": _sample_summary([window["perplexity"] for window in windows]),
                "prefill_seconds": _sample_summary(
                    [window["prefill_seconds"] for window in windows]
                ),
                "prefill_tokens_per_second": _sample_summary(
                    [window["prefill_tokens_per_second"] for window in windows]
                ),
                "windows": windows,
            }
        except (AssertionError, NotImplementedError, RuntimeError) as exc:
            results[str(context_length)] = {
                "status": "unsupported",
                "reason": str(exc),
            }
    return results
