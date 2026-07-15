"""Evaluation CLI entry point — thin shell over EvaluationPipeline.

All orchestration logic lives in src/evaluation/pipeline.py.
This script handles only CLI argument parsing and result display.

Usage:
    python scripts/evaluate.py \
        --checkpoints checkpoints/vanilla_main_s42/ checkpoints/modern_main_s42/ \
        --output reports/main_comparison/ \
        --device cuda \
        --data_dir data/processed/wikitext-full
"""

import argparse
import logging

from src.evaluation.pipeline import EvaluationPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full evaluation suite on Transformer variant checkpoints."
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        required=True,
        help="Paths to checkpoint directories to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the report (plots/, raw/, summary.md, metadata.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for GPU computations (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to validation data directory for probe computations. "
        "If not provided, probe computations requiring a val_loader are skipped.",
    )

    return parser.parse_args()


def main() -> None:
    """Parse args, run pipeline, display results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    pipeline = EvaluationPipeline(device=args.device, data_dir=args.data_dir)
    result = pipeline.run(checkpoints=args.checkpoints, output_dir=args.output)

    # Display warnings
    if result.warnings:
        print(f"\n⚠ {len(result.warnings)} warning(s):")
        for w in result.warnings:
            print(f"  • {w}")

    if result.skipped_steps:
        print(f"\n⏭ Skipped steps: {', '.join(result.skipped_steps)}")

    print(f"\n✓ Report written to: {result.output_dir}")
    print(f"  Files generated: {len(result.generated_files)}")


if __name__ == "__main__":
    main()
