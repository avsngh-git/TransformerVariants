"""Tests for logging and metrics utilities."""

import json
from pathlib import Path

from src.utils.logging import MetricsLogger, load_metrics, setup_logging


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging(name="test_logger_1")
        assert logger.name == "test_logger_1"

    def test_file_logging(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path, name="test_logger_2")
        logger.info("test message")
        log_file = tmp_path / "train.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

    def test_idempotent(self):
        logger1 = setup_logging(name="test_logger_3")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging(name="test_logger_3")
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2


class TestMetricsLogger:
    def test_log_and_read(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as ml:
            ml.log(step=1, loss=5.5)
            ml.log(step=2, loss=5.0)

        events = load_metrics(path)
        assert len(events) == 2
        assert events[0]["step"] == 1
        assert events[0]["loss"] == 5.5
        assert events[1]["step"] == 2

    def test_append_mode(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with MetricsLogger(path) as ml:
            ml.log(step=1, loss=5.5)

        # Open again — should append
        with MetricsLogger(path) as ml:
            ml.log(step=2, loss=5.0)

        events = load_metrics(path)
        assert len(events) == 2

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "metrics.jsonl"
        with MetricsLogger(path) as ml:
            ml.log(step=1, value=42)
        assert path.exists()

    def test_flush_immediately(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        ml = MetricsLogger(path)
        ml.log(step=1, x=1)
        # Read without closing — should be flushed
        content = path.read_text()
        assert '"step": 1' in content
        ml.close()


class TestLoadMetrics:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        assert load_metrics(path) == []

    def test_nonexistent_file(self, tmp_path):
        assert load_metrics(tmp_path / "nope.jsonl") == []
