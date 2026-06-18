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


class MetricsLogger:
    """Append-only JSONL metrics logger.

    Writes one JSON object per line to a metrics file. Each entry is
    immediately flushed for crash safety.

    Usage:
        metrics = MetricsLogger("runs/my_run/metrics.jsonl")
        metrics.log(step=100, train_loss=5.91, lr=0.0003, tokens_per_sec=18750)
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize metrics logger.

        Args:
            path: Path to the JSONL metrics file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")

    def log(self, **kwargs: Any) -> None:
        """Log a metrics event as a single JSON line.

        Args:
            **kwargs: Metric key-value pairs. Values should be JSON-serializable.
        """
        line = json.dumps(kwargs, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the metrics file."""
        self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


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
