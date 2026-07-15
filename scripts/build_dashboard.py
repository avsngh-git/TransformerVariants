"""Build a self-contained HTML dashboard from an evaluation report."""

from __future__ import annotations

import argparse

from src.viz.html_dashboard import build_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one offline HTML dashboard from raw/metrics.json and plots/."
    )
    parser.add_argument("--report", required=True, help="Evaluation report directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <report>/index.html).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_dashboard(args.report, args.output)
    print(f"Dashboard written to: {output}")


if __name__ == "__main__":
    main()
