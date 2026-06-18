"""CLI script to prepare a tokenized dataset from a HuggingFace source.

This is a thin wrapper around src.data.prepare.prepare_dataset().
All logic lives in the module — this script just handles argument parsing
and config loading.

Usage:
    python scripts/prepare_data.py --config configs/data/debug.yaml
    python scripts/prepare_data.py --config configs/data/debug.yaml --max-documents 100
"""

import argparse
import sys

from src.data.prepare import prepare_dataset
from src.utils.config import load_config


def main() -> None:
    """Parse CLI arguments, load config, and run the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare a tokenized dataset for training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the data config YAML file (e.g. configs/data/debug.yaml)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Override max_documents from config (useful for quick tests)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()

    # Load the YAML config
    config = load_config(args.config)

    # The data config is nested under the "data" key in the YAML
    data_config = config.get("data", config)

    # Apply CLI overrides (CLI wins over YAML)
    if args.max_documents is not None:
        data_config["max_documents"] = args.max_documents
    if args.max_tokens is not None:
        data_config["max_tokens"] = args.max_tokens
    if args.output_dir is not None:
        data_config["output_dir"] = args.output_dir

    # Run the pipeline
    output_dir = prepare_dataset(data_config)
    print(f"\nDataset written to: {output_dir}")


if __name__ == "__main__":
    main()
