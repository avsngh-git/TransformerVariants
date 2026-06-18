"""Tests for run directory creation and management."""

import datetime
import json
from pathlib import Path

import pytest
import yaml

from src.utils.run_dir import (
    create_run_dir,
    generate_run_id,
    validate_run_dir,
    write_summary,
)


class TestGenerateRunId:
    def test_default_prefix(self):
        run_id = generate_run_id()
        assert run_id.startswith("run_")

    def test_custom_prefix(self):
        run_id = generate_run_id(prefix="vanilla")
        assert run_id.startswith("vanilla_")

    def test_deterministic_with_timestamp(self):
        ts = datetime.datetime(2025, 1, 15, 14, 30, 22, tzinfo=datetime.timezone.utc)
        run_id = generate_run_id(timestamp=ts)
        assert run_id == "run_20250115_143022"

    def test_format_is_valid(self):
        run_id = generate_run_id()
        parts = run_id.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS


class TestCreateRunDir:
    def test_creates_expected_structure(self, tmp_path):
        run_dir = create_run_dir(run_id="test_run", runs_root=tmp_path)
        assert run_dir == tmp_path / "test_run"
        assert (run_dir / "logs").is_dir()
        assert (run_dir / "checkpoints").is_dir()
        assert (run_dir / "metrics.jsonl").exists()

    def test_writes_config_when_provided(self, tmp_path):
        config = {"model": {"n_layer": 4}, "training": {"lr": 0.001}}
        run_dir = create_run_dir(run_id="cfg_run", config=config, runs_root=tmp_path)
        config_path = run_dir / "config_resolved.yaml"
        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded == config

    def test_no_config_no_yaml(self, tmp_path):
        run_dir = create_run_dir(run_id="no_cfg_run", runs_root=tmp_path)
        assert not (run_dir / "config_resolved.yaml").exists()

    def test_raises_on_duplicate(self, tmp_path):
        create_run_dir(run_id="dup_run", runs_root=tmp_path)
        with pytest.raises(FileExistsError):
            create_run_dir(run_id="dup_run", runs_root=tmp_path)

    def test_auto_generates_id(self, tmp_path):
        run_dir = create_run_dir(runs_root=tmp_path)
        assert run_dir.name.startswith("run_")
        assert run_dir.exists()

    def test_custom_prefix(self, tmp_path):
        run_dir = create_run_dir(prefix="v0_vanilla", runs_root=tmp_path)
        assert run_dir.name.startswith("v0_vanilla_")


class TestValidateRunDir:
    def test_valid_dir(self, tmp_path):
        run_dir = create_run_dir(run_id="valid_run", runs_root=tmp_path)
        assert validate_run_dir(run_dir) is True

    def test_invalid_missing_logs(self, tmp_path):
        run_dir = tmp_path / "broken_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "metrics.jsonl").touch()
        # Missing logs/ dir
        assert validate_run_dir(run_dir) is False

    def test_invalid_missing_metrics(self, tmp_path):
        run_dir = tmp_path / "broken_run2"
        run_dir.mkdir()
        (run_dir / "logs").mkdir()
        (run_dir / "checkpoints").mkdir()
        # Missing metrics.jsonl
        assert validate_run_dir(run_dir) is False


class TestWriteSummary:
    def test_writes_json(self, tmp_path):
        run_dir = create_run_dir(run_id="summary_run", runs_root=tmp_path)
        summary = {"final_loss": 3.14, "total_tokens": 100_000_000}
        path = write_summary(run_dir, summary)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == summary
