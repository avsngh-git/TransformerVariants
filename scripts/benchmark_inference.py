"""Benchmark generation, KV-cache memory, and long-context validation loss."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.data.dataloader import ShardedDataLoader
from src.evaluation.benchmarks import benchmark_generation, evaluate_long_context
from src.evaluation.comparison import load_variant_data
from src.evaluation.pipeline import EvaluationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
    )
    return parser.parse_args()


def _validation_tokens(
    data_dir: str, max_context: int, device: str
) -> torch.Tensor:
    loader = ShardedDataLoader(
        data_dir=data_dir,
        batch_size=1,
        seq_len=max_context,
        split="val",
        device=device,
    )
    inputs, targets = loader.next_batch()
    return torch.cat((inputs, targets[:, -1:]), dim=1)


def main() -> None:
    args = parse_args()
    variants = load_variant_data([Path(path) for path in args.checkpoints])
    if not variants:
        raise SystemExit("No valid checkpoint directories were loaded.")
    tokens = _validation_tokens(
        args.data_dir, max(args.context_lengths), args.device
    )
    pipeline = EvaluationPipeline(device=args.device, data_dir=args.data_dir)
    results: dict[str, dict] = {}

    for variant in variants:
        print(f"Benchmarking {variant.name}: {variant.checkpoint_dir}", flush=True)
        model = pipeline._load_model_from_checkpoint(variant)
        if model is None:
            results[variant.name] = {
                "checkpoint_dir": str(variant.checkpoint_dir),
                "status": "unavailable",
                "reason": "checkpoint could not be loaded",
            }
            continue
        try:
            generation = benchmark_generation(
                model,
                prompt_length=args.prompt_length,
                new_tokens=args.new_tokens,
                repeats=args.repeats,
                warmups=args.warmups,
            )
        except (AssertionError, NotImplementedError, RuntimeError, ValueError) as exc:
            generation = {
                "uncached": {"status": "unsupported", "reason": str(exc)},
                "cached": {"status": "unsupported", "reason": str(exc)},
                "kv_cache": {"status": "unsupported", "reason": str(exc)},
            }

        long_context = evaluate_long_context(
            model, tokens, context_lengths=args.context_lengths
        )
        results[variant.name] = {
            "checkpoint_dir": str(variant.checkpoint_dir),
            "status": "ok",
            "generation": generation,
            "long_context": long_context,
        }
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    hardware = (
        torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "settings": {
            "prompt_length": args.prompt_length,
            "new_tokens": args.new_tokens,
            "batch_size": 1,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "context_lengths": args.context_lengths,
            "long_context_samples_per_variant": 1,
        },
        "limitations": [
            "One representative checkpoint per variant; generation timing is not seed-aggregated.",
            "Long-context loss uses one contiguous validation sample per context length.",
            "Unsupported cache paths are reported explicitly rather than emulated.",
        ],
        "variants": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Benchmark results written to: {output_path}")


if __name__ == "__main__":
    main()
