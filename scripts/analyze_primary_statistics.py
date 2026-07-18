"""Compute prespecified five-seed summaries and paired recipe differences."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.evaluation.statistics import paired_difference_summary, sample_summary


def _seed(entry: dict) -> int | None:
    if entry.get("seed") is not None:
        return int(entry["seed"])
    name = Path(entry.get("checkpoint_dir", "")).name
    match = re.search(r"(?:^|_)s(\d+)(?:$|_)", name)
    return int(match.group(1)) if match else None


def _metric(entry: dict, endpoint: str) -> float | None:
    metrics = entry.get("metrics") or {}
    value = metrics.get(endpoint, entry.get(endpoint))
    return float(value) if value is not None else None


def analyze(metrics: dict, manifest: dict) -> dict:
    """Analyze only comparisons declared before inspecting the new results."""
    pairs = manifest.get("analysis", {}).get("paired_comparisons", [])
    endpoint = manifest.get("analysis", {}).get("endpoint_key", "val_loss")
    by_variant: dict[str, dict[int, float]] = {}
    for variant, entries in metrics.get("variants", {}).items():
        values: dict[int, float] = {}
        for entry in entries:
            seed = _seed(entry)
            value = _metric(entry, endpoint)
            if seed is not None and value is not None:
                if seed in values:
                    raise ValueError(f"Duplicate result for variant={variant} seed={seed}")
                values[seed] = value
        if values:
            by_variant[variant] = values

    summaries = {
        variant: {
            "seeds": sorted(values),
            endpoint: sample_summary([values[seed] for seed in sorted(values)]),
        }
        for variant, values in sorted(by_variant.items())
    }
    comparisons = []
    for pair in pairs:
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        result = paired_difference_summary(
            by_variant.get(candidate, {}), by_variant.get(baseline, {})
        )
        comparisons.append({**pair, "result": result})

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": manifest.get("experiment_id"),
        "endpoint": endpoint,
        "interval": "two-sided 95% Student-t interval across independent seeds",
        "variant_summaries": summaries,
        "paired_comparisons": comparisons,
        "interpretation": (
            "Intervals and paired effect sizes quantify uncertainty; they do not turn the "
            "bundled recipes into component-level causal ablations."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest_text = args.manifest.read_text(encoding="utf-8")
    manifest = (
        json.loads(manifest_text)
        if args.manifest.suffix == ".json"
        else yaml.safe_load(manifest_text)
    )
    output = analyze(metrics, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Primary statistical analysis written to: {args.output}")


if __name__ == "__main__":
    main()
