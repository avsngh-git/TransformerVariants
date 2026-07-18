"""Benchmark serving and paired-tail long-context quality across checkpoint seeds."""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.data.dataloader import ShardedDataLoader
from src.evaluation.benchmarks import (
    aggregate_long_context_runs,
    benchmark_generation_matrix,
    evaluate_long_context,
    extend_context,
    rank_long_context_variants,
)
from src.evaluation.comparison import VariantData, load_variant_data
from src.evaluation.pipeline import EvaluationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[64, 512, 1024, 4096]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--tail-tokens", type=int, default=256)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
    )
    return parser.parse_args()


def _validation_windows(
    data_dir: str,
    max_context: int,
    window_count: int,
    device: str,
) -> torch.Tensor:
    if window_count <= 0:
        raise ValueError("windows must be positive")
    loader = ShardedDataLoader(
        data_dir=data_dir,
        batch_size=window_count,
        seq_len=max_context,
        split="val",
        device=device,
    )
    inputs, targets = loader.next_batch()
    return torch.cat((inputs, targets[:, -1:]), dim=1)


def _infer_seed(checkpoint_dir: Path) -> int | None:
    match = re.search(r"(?:^|_)s(\d+)(?:$|_)", checkpoint_dir.name)
    return int(match.group(1)) if match else None


def _unsupported_generation_cells(
    prompt_length: int,
    batch_sizes: list[int],
    reason: str,
) -> dict[str, dict]:
    return {
        f"prompt_{prompt_length}_batch_{batch_size}": {
            "status": "unsupported",
            "reason": reason,
        }
        for batch_size in batch_sizes
    }


def _representative_checkpoint(
    variants: list[VariantData],
    preferred_seed: int,
) -> VariantData:
    return min(
        variants,
        key=lambda variant: (
            _infer_seed(variant.checkpoint_dir) != preferred_seed,
            str(variant.checkpoint_dir),
        ),
    )


def main() -> None:
    args = parse_args()
    variants = load_variant_data([Path(path) for path in args.checkpoints])
    if not variants:
        raise SystemExit("No valid checkpoint directories were loaded.")
    if args.tail_tokens <= 0 or args.tail_tokens > min(args.context_lengths):
        raise SystemExit("--tail-tokens must be positive and fit the shortest context.")

    tokens = _validation_windows(
        args.data_dir,
        max(args.context_lengths),
        args.windows,
        args.device,
    )
    pipeline = EvaluationPipeline(device=args.device, data_dir=args.data_dir)
    groups: dict[str, list[VariantData]] = {}
    for variant in variants:
        groups.setdefault(variant.name, []).append(variant)

    results: dict[str, dict] = {}
    for variant_name, checkpoints in sorted(groups.items()):
        representative = _representative_checkpoint(checkpoints, args.generation_seed)
        generation: dict[str, dict] = {}
        runs: list[dict] = []

        for variant in sorted(checkpoints, key=lambda item: str(item.checkpoint_dir)):
            seed = _infer_seed(variant.checkpoint_dir)
            print(
                f"Benchmarking {variant.name} seed={seed}: {variant.checkpoint_dir}",
                flush=True,
            )
            model = pipeline.load_model_from_checkpoint(variant)
            if model is None:
                runs.append(
                    {
                        "seed": seed,
                        "checkpoint_dir": str(variant.checkpoint_dir),
                        "status": "unavailable",
                        "reason": "checkpoint could not be loaded",
                        "long_context": {},
                    }
                )
                continue

            if variant is representative:
                for prompt_length in sorted(set(args.prompt_lengths)):
                    required_context = prompt_length + args.new_tokens
                    supported, reason = extend_context(model, required_context)
                    if not supported:
                        generation.update(
                            _unsupported_generation_cells(
                                prompt_length,
                                args.batch_sizes,
                                reason or "context extension is unsupported",
                            )
                        )
                        continue
                    generation.update(
                        benchmark_generation_matrix(
                            model,
                            prompt_lengths=[prompt_length],
                            batch_sizes=args.batch_sizes,
                            new_tokens=args.new_tokens,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            seed=args.generation_seed,
                        )
                    )

            long_context = evaluate_long_context(
                model,
                tokens,
                context_lengths=args.context_lengths,
                tail_tokens=args.tail_tokens,
            )
            runs.append(
                {
                    "seed": seed,
                    "checkpoint_dir": str(variant.checkpoint_dir),
                    "status": "ok",
                    "long_context": long_context,
                }
            )
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        results[variant_name] = {
            "status": "ok" if any(run["status"] == "ok" for run in runs) else "unavailable",
            "checkpoint_dir": str(representative.checkpoint_dir),
            "generation_checkpoint_seed": _infer_seed(representative.checkpoint_dir),
            "generation": generation,
            "long_context": aggregate_long_context_runs(
                runs,
                baseline_context=min(args.context_lengths),
            ),
            "checkpoints": runs,
        }

    hardware = (
        torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    output = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "settings": {
            "prompt_lengths": sorted(set(args.prompt_lengths)),
            "new_tokens": args.new_tokens,
            "generation_batch_sizes": sorted(set(args.batch_sizes)),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "context_lengths": sorted(set(args.context_lengths)),
            "long_context_windows_per_checkpoint": args.windows,
            "long_context_tail_tokens": args.tail_tokens,
            "long_context_checkpoint_counts": {
                name: len(checkpoints) for name, checkpoints in groups.items()
            },
        },
        "long_context_method": {
            "window_sampling": "fixed non-overlapping windows from the validation split",
            "target_alignment": ("the same final tail tokens are scored at every context length"),
            "checkpoint_estimate": "mean across held-out windows",
            "uncertainty_unit": "sample standard deviation across checkpoint seeds",
        },
        "limitations": [
            "Generation and KV-cache timing use one representative checkpoint per variant.",
            (
                "Serving cells that exceed a variant's supported context are explicit, "
                "not extrapolated."
            ),
            "Long-context quality uses fixed windows rather than the entire validation corpus.",
            "Models were trained at 1024 tokens; longer lengths measure extrapolation.",
            "Unsupported cache and context-extension paths are reported rather than emulated.",
        ],
        "long_context_rankings": rank_long_context_variants(
            results,
            context_length=max(args.context_lengths),
        ),
        "variants": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Benchmark results written to: {output_path}")


if __name__ == "__main__":
    main()
