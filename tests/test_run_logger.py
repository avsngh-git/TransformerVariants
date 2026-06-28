"""Tests for RunLogger directory creation, validation, and summary writing.

Migrated from test_run_dir.py — adapted to use RunLogger interface.
Validates: Requirements 7.1
"""

import json
from pathlib import Path

import pytest

from src.training.run_logger import RunLogger, generate_run_dir


class TestRunLoggerInit:
    """Test that RunLogger.__init__ creates the expected directory structure."""

    def test_creates_run_dir(self, tmp_path):
        run_dir = tmp_path / "test_run"
        RunLogger(run_dir, config={"variant": "test"})
        assert run_dir.is_dir()

    def test_creates_run_config_json(self, tmp_path):
        run_dir = tmp_path / "test_run"
        config = {"model": {"n_layer": 4}, "training": {"lr": 0.001}}
        RunLogger(run_dir, config=config)

        config_path = run_dir / "run_config.json"
        assert config_path.exists()
        loaded = json.loads(config_path.read_text())
        assert loaded == config

    def test_creates_checkpoints_dir(self, tmp_path):
        run_dir = tmp_path / "test_run"
        RunLogger(run_dir, config={})
        assert (run_dir / "checkpoints").is_dir()

    def test_creates_train_log(self, tmp_path):
        run_dir = tmp_path / "test_run"
        RunLogger(run_dir, config={})
        assert (run_dir / "train.log").exists()
        content = (run_dir / "train.log").read_text()
        assert len(content) > 0  # Header written

    def test_creates_metrics_jsonl(self, tmp_path):
        run_dir = tmp_path / "test_run"
        RunLogger(run_dir, config={})
        assert (run_dir / "metrics.jsonl").exists()

    def test_creates_nested_parent_dirs(self, tmp_path):
        run_dir = tmp_path / "nested" / "deep" / "run"
        RunLogger(run_dir, config={})
        assert run_dir.is_dir()


class TestRunLoggerValidate:
    """Test RunLogger.validate() for directory structure checking."""

    def test_valid_directory(self, tmp_path):
        run_dir = tmp_path / "valid_run"
        logger = RunLogger(run_dir, config={"variant": "test"})
        assert logger.validate() is True

    def test_invalid_missing_checkpoints(self, tmp_path):
        run_dir = tmp_path / "broken_run"
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text("{}")
        (run_dir / "train.log").touch()
        (run_dir / "metrics.jsonl").touch()
        # Missing checkpoints/ dir
        logger = RunLogger.__new__(RunLogger)
        logger.run_dir = run_dir
        assert logger.validate() is False

    def test_invalid_missing_train_log(self, tmp_path):
        run_dir = tmp_path / "broken_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "run_config.json").write_text("{}")
        (run_dir / "metrics.jsonl").touch()
        # Missing train.log
        logger = RunLogger.__new__(RunLogger)
        logger.run_dir = run_dir
        assert logger.validate() is False

    def test_invalid_missing_metrics(self, tmp_path):
        run_dir = tmp_path / "broken_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "run_config.json").write_text("{}")
        (run_dir / "train.log").touch()
        # Missing metrics.jsonl
        logger = RunLogger.__new__(RunLogger)
        logger.run_dir = run_dir
        assert logger.validate() is False

    def test_invalid_missing_config(self, tmp_path):
        run_dir = tmp_path / "broken_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "train.log").touch()
        (run_dir / "metrics.jsonl").touch()
        # Missing run_config.json
        logger = RunLogger.__new__(RunLogger)
        logger.run_dir = run_dir
        assert logger.validate() is False


class TestRunLoggerSummary:
    """Test RunLogger.log_summary() writes summary.json correctly."""

    def test_writes_summary_json(self, tmp_path):
        run_dir = tmp_path / "summary_run"
        logger = RunLogger(run_dir, config={"variant": "test"})
        summary = {"final_train_loss": 3.14, "total_tokens": 100_000_000}
        logger.log_summary(summary)

        summary_path = run_dir / "summary.json"
        assert summary_path.exists()
        loaded = json.loads(summary_path.read_text())
        assert loaded == summary

    def test_summary_indent_2(self, tmp_path):
        run_dir = tmp_path / "summary_run"
        logger = RunLogger(run_dir, config={})
        summary = {"key": "value"}
        logger.log_summary(summary)

        content = (run_dir / "summary.json").read_text()
        # indent=2 means multi-line with 2-space indent
        assert "  " in content
        assert content == json.dumps(summary, indent=2)

    def test_summary_appends_to_train_log(self, tmp_path):
        run_dir = tmp_path / "summary_run"
        logger = RunLogger(run_dir, config={})
        summary = {"final_train_loss": 2.5, "final_val_loss": 2.3}
        logger.log_summary(summary)

        log_content = (run_dir / "train.log").read_text()
        assert "Training Complete" in log_content


class TestGenerateRunDir:
    """Test the generate_run_dir() module-level function."""

    def test_basic_naming(self):
        path = generate_run_dir("vanilla", "main", "relu")
        name = path.name
        assert name.startswith("vanilla_relu_main_")

    def test_swiglu_omitted(self):
        path = generate_run_dir("modern", "stretch", "swiglu")
        name = path.name
        # swiglu is the default, should be omitted from name
        assert "swiglu" not in name
        assert name.startswith("modern_stretch_")

    def test_empty_activation_omitted(self):
        path = generate_run_dir("vanilla", "debug", "")
        name = path.name
        assert name.startswith("vanilla_debug_")

    def test_custom_base_dir(self):
        path = generate_run_dir("vanilla", "main", "gelu", base_dir="output/runs")
        assert str(path).startswith("output/runs/")

    def test_timestamp_format(self):
        path = generate_run_dir("vanilla", "main", "relu")
        name = path.name
        # Pattern: variant_activation_scale_YYYYMMDD_HHMM
        parts = name.split("_")
        # Last two parts should be date and time
        date_part = parts[-2]
        time_part = parts[-1]
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 4  # HHMM
        assert date_part.isdigit()
        assert time_part.isdigit()


class TestRunLoggerContextManager:
    """Test RunLogger context manager protocol."""

    def test_context_manager_returns_self(self, tmp_path):
        logger = RunLogger(tmp_path / "ctx_run", config={})
        with logger as ctx:
            assert ctx is logger

    def test_context_manager_callable(self, tmp_path):
        with RunLogger(tmp_path / "ctx_run", config={"variant": "test"}) as logger:
            logger.log_summary({"done": True})
        # No exception means success
        assert (tmp_path / "ctx_run" / "summary.json").exists()
