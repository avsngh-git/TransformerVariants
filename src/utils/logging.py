"""Logging setup for training and evaluation.

Provides structured logging with both console output and file logging.
Metrics are logged separately in JSONL format.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    name: str = "transformer_lab",
) -> logging.Logger:
    """Configure logging with console and optional file output.

    Args:
        log_dir: If provided, also write logs to a file in this directory.
        level: Logging level (default: INFO).
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load all metrics events from a JSONL file.

    Args:
        path: Path to the metrics JSONL file.

    Returns:
        List of metric dictionaries, one per logged event.
    """
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
