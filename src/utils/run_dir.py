"""Run directory creation and management.

Every training run writes outputs to:
    runs/<run_id>/
        config_resolved.yaml
        metrics.jsonl
        summary.json
        logs/
        checkpoints/

This module provides utilities to create and validate that structure.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import yaml


RUNS_ROOT = Path("runs")


def generate_run_id(
    prefix: str = "run",
    timestamp: datetime.datetime | None = None,
) -> str:
    """Generate a unique run ID based on timestamp.

    Format: {prefix}_{YYYYMMDD}_{HHMMSS}

    Args:
        prefix: Short identifier prepended to the timestamp.
        timestamp: Override for testing; defaults to now (UTC).

    Returns:
        A string like 'run_20250115_143022'.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    return f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}"


def create_run_dir(
    run_id: str | None = None,
    prefix: str = "run",
    config: dict[str, Any] | None = None,
    runs_root: str | Path = RUNS_ROOT,
) -> Path:
    """Create a run directory with the standard layout.

    Args:
        run_id: Explicit run ID. If None, one is generated from timestamp.
        prefix: Prefix for auto-generated run IDs.
        config: If provided, write as config_resolved.yaml in the run dir.
        runs_root: Base directory for all runs. Defaults to 'runs/'.

    Returns:
        Path to the created run directory.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    runs_root = Path(runs_root)

    if run_id is None:
        run_id = generate_run_id(prefix=prefix)

    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    # Create directory structure
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "checkpoints").mkdir()

    # Write resolved config if provided
    if config is not None:
        config_path = run_dir / "config_resolved.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create empty metrics file
    (run_dir / "metrics.jsonl").touch()

    return run_dir


def validate_run_dir(run_dir: str | Path) -> bool:
    """Check that a run directory has the expected structure.

    Args:
        run_dir: Path to the run directory.

    Returns:
        True if all expected subdirectories and files exist.
    """
    run_dir = Path(run_dir)
    expected = [
        run_dir / "logs",
        run_dir / "checkpoints",
        run_dir / "metrics.jsonl",
    ]
    return all(p.exists() for p in expected)


def write_summary(run_dir: str | Path, summary: dict[str, Any]) -> Path:
    """Write a summary.json to the run directory.

    Args:
        run_dir: Path to the run directory.
        summary: Dictionary of summary data.

    Returns:
        Path to the written summary.json file.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path
