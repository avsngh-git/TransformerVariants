#!/usr/bin/env python3
"""Export evaluation data and model internals for a separate static site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataloader import ShardedDataLoader
from src.data.tokenizer import get_tokenizer
from src.evaluation.comparison import load_variant_data
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.site_assets import capture_attention_patterns, export_site_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frontend-agnostic, Jekyll-ready JSON and PNG model-internals assets."
    )
    parser.add_argument(
        "report_dir", type=Path, help="Evaluation report containing raw/metrics.json"
    )
    parser.add_argument("--output-dir", type=Path, help="Destination (default: REPORT/site_assets)")
    parser.add_argument(
        "--with-attention",
        action="store_true",
        help="Load representative checkpoints and export layer/head attention patterns",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        help="Checkpoint directories (default: seed-42 paths from report metadata)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/fineweb-1B"),
        help="Prepared dataset used for the deterministic validation context",
    )
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        help="Layer indices to export (default: first, middle, and final layer)",
    )
    return parser.parse_args()


def _reported_seed42_checkpoints(report_dir: Path) -> list[Path]:
    metadata_path = report_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = [Path(path) for path in metadata.get("evaluated_checkpoints", [])]
    return [path for path in candidates if path.name.endswith("s42")]


def _token_labels(token_ids: torch.Tensor) -> list[str]:
    tokenizer = get_tokenizer("gpt2")
    labels = []
    for token_id in token_ids[0].tolist():
        decoded = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\t", "\\t")
        labels.append(decoded or f"<{token_id}>")
    return labels


def main() -> None:
    args = parse_args()
    if args.context_length < 1:
        raise SystemExit("--context-length must be at least 1")

    patterns = None
    if args.with_attention:
        checkpoint_dirs = args.checkpoints or _reported_seed42_checkpoints(args.report_dir)
        if not checkpoint_dirs:
            raise SystemExit("No checkpoint directories were provided or found in report metadata")
        loader = ShardedDataLoader(
            args.data_dir,
            batch_size=1,
            seq_len=args.context_length,
            split="val",
            device=args.device,
        )
        token_ids, _ = loader.next_batch()
        labels = _token_labels(token_ids)
        pipeline = EvaluationPipeline(data_dir=args.data_dir, device=args.device)
        context_provenance = {
            "data_dir": str(args.data_dir),
            "data_split": "val",
            "batch_index": 0,
            "sequence_index": 0,
            "length": args.context_length,
            "tokenizer": "gpt2",
        }
        patterns = []
        for variant in load_variant_data(checkpoint_dirs):
            model = pipeline.load_model_from_checkpoint(variant)
            if model is None:
                patterns.append(
                    {
                        "variant": variant.name,
                        "status": "error",
                        "reason": "checkpoint load failed",
                        "checkpoint_dir": str(variant.checkpoint_dir),
                        "context": context_provenance,
                    }
                )
                continue
            n_layers = len(model.blocks)
            layers = (
                args.layers if args.layers is not None else sorted({0, n_layers // 2, n_layers - 1})
            )
            capture = capture_attention_patterns(
                model,
                token_ids,
                token_labels=labels,
                variant=variant.name,
                layers=layers,
            )
            capture["checkpoint_dir"] = str(variant.checkpoint_dir)
            capture["context"] = context_provenance
            patterns.append(capture)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = export_site_assets(
        args.report_dir,
        output_dir=args.output_dir,
        attention_patterns=patterns,
    )
    print(f"Exported {len(result.plot_paths)} PNGs and JSON assets to {result.output_dir}")


if __name__ == "__main__":
    main()
