"""Evaluate MoE routing utilization, entropy, affinity, overlap, and stability."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import torch

from src.data.dataloader import ShardedDataLoader
from src.evaluation.comparison import load_variant_data
from src.evaluation.moe_probes import (
    run_expert_affinity_probe,
    run_expert_pair_overlap_probe,
    run_expert_utilization_probe,
    run_router_entropy_probe,
    run_routing_stability_probe,
)
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.statistics import sample_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--position-buckets", type=int, default=4)
    return parser.parse_args()


def _seed(path: Path) -> int | None:
    match = re.search(r"(?:^|_)s(\d+)(?:$|_)", path.name)
    return int(match.group(1)) if match else None


def _json_layers(values: dict[int, object]) -> dict[str, object]:
    return {str(layer): value for layer, value in sorted(values.items())}


def _capture_routing(
    model,
    *,
    data_dir: str,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: str,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    for block in model.blocks:
        if hasattr(block.ffn, "record_routing"):
            block.ffn.record_routing = True
    loader = ShardedDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        split="val",
        device=device,
    )
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            inputs, _ = loader.next_batch()
            model(inputs, kv_cache=None)
    captured = model.get_routing_data()
    return {
        layer: [(indices.cpu(), weights.cpu()) for indices, weights in entries]
        for layer, entries in captured.items()
    }


def _pairwise_stability(
    captures: list[tuple[int | None, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]]],
) -> dict:
    pairs: list[dict] = []
    for (seed_a, data_a), (seed_b, data_b) in combinations(captures, 2):
        result = run_routing_stability_probe(data_a, data_b)
        pairs.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "per_layer": _json_layers(result.per_layer),
            }
        )
    layers = sorted(
        {
            int(layer)
            for pair in pairs
            for layer in pair["per_layer"]
        }
    )
    return {
        "pairwise": pairs,
        "per_layer": {
            str(layer): sample_summary(
                [float(pair["per_layer"][str(layer)]) for pair in pairs]
            )
            for layer in layers
        },
    }


def main() -> None:
    args = parse_args()
    if args.batches < 1 or args.batch_size < 1:
        raise SystemExit("--batches and --batch-size must be positive")
    variants = load_variant_data([Path(path) for path in args.checkpoints])
    variants = [variant for variant in variants if variant.config.num_experts is not None]
    if not variants:
        raise SystemExit("No MoE checkpoints were loaded")

    pipeline = EvaluationPipeline(device=args.device, data_dir=args.data_dir)
    runs: list[dict] = []
    captures_by_variant: dict[
        str,
        list[tuple[int | None, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]]],
    ] = {}
    for variant in variants:
        print(f"MoE routing evaluation: {variant.checkpoint_dir}", flush=True)
        model = pipeline.load_model_from_checkpoint(variant)
        if model is None:
            continue
        routing = _capture_routing(
            model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=variant.config.seq_len,
            batches=args.batches,
            device=args.device,
        )
        seed = _seed(variant.checkpoint_dir)
        captures_by_variant.setdefault(variant.name, []).append((seed, routing))
        num_experts = int(variant.config.num_experts)
        utilization = run_expert_utilization_probe(routing, num_experts)
        entropy = run_router_entropy_probe(routing, num_experts)
        affinity = run_expert_affinity_probe(
            routing,
            num_experts,
            seq_len=variant.config.seq_len,
            num_buckets=args.position_buckets,
        )
        overlap = run_expert_pair_overlap_probe(routing, num_experts)
        runs.append(
            {
                "variant": variant.name,
                "seed": seed,
                "checkpoint_dir": str(variant.checkpoint_dir),
                "num_experts": num_experts,
                "top_k": int(variant.config.moe_top_k),
                "tokens_observed": args.batches * args.batch_size * variant.config.seq_len,
                "utilization": _json_layers(utilization.per_layer),
                "router_entropy": _json_layers(entropy.per_layer),
                "maximum_router_entropy": math.log(num_experts),
                "position_affinity": _json_layers(affinity.per_layer),
                "expert_pair_overlap": _json_layers(overlap.per_layer),
                "dropped_tokens": 0,
                "dropped_tokens_reason": (
                    "This implementation has no expert-capacity cutoff; every selected route "
                    "is evaluated."
                ),
            }
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_versions": {"python": platform.python_version(), "torch": torch.__version__},
        "settings": {
            "validation_batches": args.batches,
            "batch_size": args.batch_size,
            "position_buckets": args.position_buckets,
            "same_validation_order_across_seeds": True,
        },
        "runs": runs,
        "cross_seed_routing_stability": {
            "method": (
                "Learn expert-label alignment on the first half of matched validation "
                "tokens; report top-1 agreement on the held-out second half."
            ),
            "expert_ids_compared_directly": False,
            "variants": {
                variant: _pairwise_stability(captures)
                for variant, captures in captures_by_variant.items()
            },
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"MoE routing results written to: {output_path}")


if __name__ == "__main__":
    main()
