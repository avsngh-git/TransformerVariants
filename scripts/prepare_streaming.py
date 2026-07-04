"""CLI script to run the streaming data preparation pipeline.

Thin wrapper around src.data.streaming_prepare.StreamingPipeline.
Handles argument parsing, config loading, and CLI overrides.

Usage:
    # Default: uses fineweb_edu.yaml
    python scripts/prepare_streaming.py --config configs/data/fineweb_edu.yaml

    # Custom limits
    python scripts/prepare_streaming.py \
      --config configs/data/fineweb_edu.yaml \
      --max-tokens 500000000 \
      --min-doc-tokens 100 \
      --max-doc-tokens 5000 \
      --output-dir data/processed/fineweb-500M

    # Resume interrupted run
    python scripts/prepare_streaming.py \
      --config configs/data/fineweb_edu.yaml \
      --output-dir data/processed/fineweb-1B \
      --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.data.streaming_prepare import PipelineConfig, StreamingPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the streaming pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the streaming data preparation pipeline (FineWeb-Edu → uint16 shards).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g. configs/data/fineweb_edu.yaml)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to produce (overrides config)",
    )
    parser.add_argument(
        "--min-doc-tokens",
        type=int,
        default=None,
        help="Minimum document token count (overrides config)",
    )
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=None,
        help="Maximum document token count (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for shard files (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from progress.json in the output directory",
    )
    return parser.parse_args(argv)


def load_yaml_config(config_path: str) -> dict:
    """Load the data section from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The 'data' section of the config, or the top-level dict if no
        'data' key is present.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}

    # The data config is nested under the "data" key in the YAML
    return raw.get("data", raw)


def build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from YAML config + CLI overrides.

    Priority: CLI args > YAML config > PipelineConfig defaults.
    """
    # Start with YAML config values (if provided)
    if args.config:
        data_cfg = load_yaml_config(args.config)
    else:
        data_cfg = {}

    # Map YAML keys to PipelineConfig field names
    config_kwargs: dict = {}

    # Load from YAML first
    if "dataset_name" in data_cfg:
        config_kwargs["dataset_name"] = data_cfg["dataset_name"]
    if "dataset_config" in data_cfg:
        config_kwargs["dataset_config"] = data_cfg["dataset_config"]
    if "split" in data_cfg:
        config_kwargs["split"] = data_cfg["split"]
    if "max_tokens" in data_cfg:
        config_kwargs["max_tokens"] = data_cfg["max_tokens"]
    if "min_doc_tokens" in data_cfg:
        config_kwargs["min_doc_tokens"] = data_cfg["min_doc_tokens"]
    if "max_doc_tokens" in data_cfg:
        config_kwargs["max_doc_tokens"] = data_cfg["max_doc_tokens"]
    if "tokens_per_shard" in data_cfg:
        config_kwargs["tokens_per_shard"] = data_cfg["tokens_per_shard"]
    if "tokenizer" in data_cfg:
        config_kwargs["tokenizer_name"] = data_cfg["tokenizer"]
    if "output_dir" in data_cfg:
        config_kwargs["output_dir"] = Path(data_cfg["output_dir"])

    # Apply CLI overrides (CLI wins over YAML)
    if args.max_tokens is not None:
        config_kwargs["max_tokens"] = args.max_tokens
    if args.min_doc_tokens is not None:
        config_kwargs["min_doc_tokens"] = args.min_doc_tokens
    if args.max_doc_tokens is not None:
        config_kwargs["max_doc_tokens"] = args.max_doc_tokens
    if args.output_dir is not None:
        config_kwargs["output_dir"] = Path(args.output_dir)
    if args.resume:
        config_kwargs["resume"] = True

    return PipelineConfig(**config_kwargs)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, build config, and run the streaming pipeline."""
    args = parse_args(argv)
    config = build_pipeline_config(args)

    print(f"Streaming pipeline starting")
    print(f"  Dataset: {config.dataset_name}/{config.dataset_config}")
    print(f"  Max tokens: {config.max_tokens:,}" if config.max_tokens else "  Max tokens: unlimited")
    print(f"  Doc filter: [{config.min_doc_tokens}, {config.max_doc_tokens}] tokens")
    print(f"  Output: {config.output_dir}")
    print(f"  Resume: {config.resume}")
    print()

    pipeline = StreamingPipeline(config)
    result = pipeline.run()

    # Print summary
    print()
    print("Pipeline complete!")
    print(f"  Output directory: {result.output_dir}")
    print(f"  Documents consumed: {result.documents_consumed:,}")
    print(f"  Train shards: {result.train_shards} ({result.train_tokens:,} tokens)")
    print(f"  Val shards: {result.val_shards} ({result.val_tokens:,} tokens)")
    print(f"  Total tokens: {result.train_tokens + result.val_tokens:,}")
    print(f"  Filter stats:")
    print(f"    Processed: {result.filter_stats.documents_processed:,}")
    print(f"    Accepted: {result.filter_stats.documents_accepted:,}")
    print(f"    Filtered (short): {result.filter_stats.documents_filtered_short:,}")
    print(f"    Filtered (long): {result.filter_stats.documents_filtered_long:,}")
    print(f"  Processing time: {result.processing_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
