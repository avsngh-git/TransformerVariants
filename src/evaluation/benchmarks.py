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


@torch.no_grad()
def evaluate_long_context(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    context_lengths: Iterable[int] = (1024, 2048, 4096),
) -> dict[str, dict]:
    """Measure one held-out next-token loss per context or explain unsupported cases."""
    device = _device_of(model)
    tokens = tokens.to(device)
    results: dict[str, dict] = {}
    for context_length in sorted(set(context_lengths)):
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
                "reason": f"requires {context_length + 1} validation tokens",
            }
            continue
        inputs = tokens[:, :context_length]
        targets = tokens[:, 1 : context_length + 1]
        try:
            _synchronize(device)
            started = time.perf_counter()
            model(inputs, targets=None, kv_cache=None)
            _synchronize(device)
            prefill_elapsed = time.perf_counter() - started
            _synchronize(device)
            validation_started = time.perf_counter()
            _, loss, _ = model(inputs, targets=targets, kv_cache=None)
            _synchronize(device)
            validation_elapsed = time.perf_counter() - validation_started
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite or missing validation loss")
            val_loss = float(loss.item())
            results[str(context_length)] = {
                "status": "ok",
                "val_loss": val_loss,
                "perplexity": math.exp(val_loss),
                "prefill_seconds": prefill_elapsed,
                "prefill_tokens_per_second": context_length / prefill_elapsed,
                "validation_seconds": validation_elapsed,
            }
        except (AssertionError, NotImplementedError, RuntimeError) as exc:
            results[str(context_length)] = {
                "status": "unsupported",
                "reason": str(exc),
            }
    return results
