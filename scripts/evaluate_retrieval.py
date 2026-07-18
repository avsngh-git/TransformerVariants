"""Evaluate zero-shot passkey and needle retrieval across checkpoint seeds."""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.data.tokenizer import get_tokenizer
from src.evaluation.comparison import load_variant_data
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.retrieval import run_retrieval_probe
from src.evaluation.statistics import sample_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tasks", nargs="+", default=["passkey", "needle"])
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096]
    )
    parser.add_argument(
        "--distance-fractions", type=float, nargs="+", default=[0.1, 0.5, 0.9]
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--rope-thetas",
        type=float,
        nargs="+",
        default=[10000.0, 100000.0],
        help="Standard and scaled RoPE bases; non-RoPE models run only once",
    )
    return parser.parse_args()


def _seed(path: Path) -> int | None:
    match = re.search(r"(?:^|_)s(\d+)(?:$|_)", path.name)
    return int(match.group(1)) if match else None


def _aggregate(checkpoints: list[dict]) -> dict:
    aggregate: dict[str, dict] = {}
    variants = sorted({run["variant"] for run in checkpoints})
    for variant in variants:
        variant_runs = [run for run in checkpoints if run["variant"] == variant]
        variant_payload: dict[str, dict] = {}
        configurations = sorted(
            {
                configuration
                for run in variant_runs
                for configuration in run.get("configurations", {})
            }
        )
        for configuration in configurations:
            config_payload: dict[str, dict] = {}
            tasks = sorted(
                {
                    task
                    for run in variant_runs
                    for task in run.get("configurations", {}).get(configuration, {})
                }
            )
            for task in tasks:
                task_payload: dict[str, dict] = {}
                contexts = sorted(
                    {
                        context
                        for run in variant_runs
                        for context in run.get("configurations", {})
                        .get(configuration, {})
                        .get(task, {})
                    },
                    key=int,
                )
                for context in contexts:
                    measurements = [
                        run["configurations"][configuration][task][context]
                        for run in variant_runs
                        if run.get("configurations", {})
                        .get(configuration, {})
                        .get(task, {})
                        .get(context, {})
                        .get("status")
                        == "ok"
                    ]
                    if not measurements:
                        task_payload[context] = {"status": "unsupported"}
                        continue
                    task_payload[context] = {
                        "status": (
                            "ok" if len(measurements) == len(variant_runs) else "partial"
                        ),
                        "accuracy": sample_summary(
                            [float(item["accuracy"]) for item in measurements]
                        ),
                        "top5_accuracy": sample_summary(
                            [float(item["top5_accuracy"]) for item in measurements]
                        ),
                        "mean_expected_probability": sample_summary(
                            [float(item["mean_expected_probability"]) for item in measurements]
                        ),
                        "mean_negative_log_likelihood": sample_summary(
                            [
                                float(item["mean_negative_log_likelihood"])
                                for item in measurements
                            ]
                        ),
                        "by_distance": {
                            distance: {
                                metric: sample_summary(
                                    [
                                        float(item["by_distance"][distance][metric])
                                        for item in measurements
                                        if distance in item.get("by_distance", {})
                                    ]
                                )
                                for metric in (
                                    "accuracy",
                                    "top5_accuracy",
                                    "mean_expected_probability",
                                    "mean_negative_log_likelihood",
                                )
                            }
                            for distance in sorted(
                                {
                                    distance
                                    for item in measurements
                                    for distance in item.get("by_distance", {})
                                },
                                key=int,
                            )
                        },
                    }
                config_payload[task] = task_payload
            variant_payload[configuration] = config_payload
        aggregate[variant] = variant_payload
    return aggregate


def main() -> None:
    args = parse_args()
    unknown_tasks = set(args.tasks) - {"passkey", "needle"}
    if unknown_tasks:
        raise SystemExit(f"Unknown retrieval tasks: {', '.join(sorted(unknown_tasks))}")

    variants = load_variant_data([Path(path) for path in args.checkpoints])
    if not variants:
        raise SystemExit("No valid checkpoints were loaded")
    pipeline = EvaluationPipeline(device=args.device)
    tokenizer = get_tokenizer("gpt2")
    runs: list[dict] = []

    for variant in variants:
        print(f"Retrieval evaluation: {variant.checkpoint_dir}", flush=True)
        model = pipeline.load_model_from_checkpoint(variant)
        if model is None:
            runs.append(
                {
                    "variant": variant.name,
                    "seed": _seed(variant.checkpoint_dir),
                    "checkpoint_dir": str(variant.checkpoint_dir),
                    "status": "unavailable",
                    "configurations": {},
                }
            )
            continue

        position_encoding = getattr(model.config, "position_encoding", "unknown")
        theta_values = args.rope_thetas if position_encoding == "rope" else [10000.0]
        configurations: dict[str, dict] = {}
        for theta in theta_values:
            name = "rope_standard" if theta == 10000.0 else f"rope_theta_{theta:g}"
            if position_encoding != "rope":
                name = "native_position_encoding"
            configurations[name] = {
                task: run_retrieval_probe(
                    model,
                    tokenizer,
                    task=task,
                    context_lengths=args.context_lengths,
                    distance_fractions=args.distance_fractions,
                    trials=args.trials,
                    device=args.device,
                    rope_theta=theta,
                )
                for task in args.tasks
            }
        runs.append(
            {
                "variant": variant.name,
                "seed": _seed(variant.checkpoint_dir),
                "checkpoint_dir": str(variant.checkpoint_dir),
                "status": "ok",
                "position_encoding": position_encoding,
                "configurations": configurations,
            }
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        ),
        "software_versions": {"python": platform.python_version(), "torch": torch.__version__},
        "settings": {
            "tasks": args.tasks,
            "context_lengths": args.context_lengths,
            "distance_fractions": args.distance_fractions,
            "trials_per_distance": args.trials,
            "rope_thetas": args.rope_thetas,
        },
        "method": {
            "protocol": "zero-shot exact next-token retrieval",
            "metrics": [
                "top-1 accuracy",
                "top-5 accuracy",
                "expected-token probability",
                "negative log-likelihood",
            ],
            "interpretation": (
                "A supplement to MQAR and paired-tail perplexity; floor effects remain possible "
                "for small base models that were not instruction-tuned."
            ),
        },
        "aggregate": _aggregate(runs),
        "checkpoints": runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Retrieval results written to: {output_path}")


if __name__ == "__main__":
    main()
