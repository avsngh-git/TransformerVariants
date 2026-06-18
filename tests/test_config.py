"""Tests for config loading and merging."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config, load_yaml, merge_configs, resolve_config


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def create_fixtures(tmp_path, monkeypatch):
    """Create temporary YAML fixtures for testing."""
    # We use tmp_path for test isolation
    pass


class TestLoadYaml:
    def test_loads_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  n_layer: 4\n  d_model: 256\n")
        result = load_yaml(cfg_file)
        assert result == {"model": {"n_layer": 4, "d_model": 256}}

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_raises_on_non_yaml_extension(self, tmp_path):
        cfg_file = tmp_path / "test.json"
        cfg_file.write_text("{}")
        with pytest.raises(ValueError, match="Expected a YAML file"):
            load_yaml(cfg_file)

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        result = load_yaml(cfg_file)
        assert result == {}

    def test_raises_on_non_dict_yaml(self, tmp_path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected top-level dict"):
            load_yaml(cfg_file)


class TestMergeConfigs:
    def test_override_scalar(self):
        base = {"model": {"n_layer": 4}}
        override = {"model": {"n_layer": 8}}
        result = merge_configs(base, override)
        assert result["model"]["n_layer"] == 8

    def test_deep_merge_preserves_base_keys(self):
        base = {"model": {"n_layer": 4, "d_model": 256}}
        override = {"model": {"n_layer": 8}}
        result = merge_configs(base, override)
        assert result["model"]["n_layer"] == 8
        assert result["model"]["d_model"] == 256

    def test_add_new_keys(self):
        base = {"model": {"n_layer": 4}}
        override = {"training": {"lr": 0.001}}
        result = merge_configs(base, override)
        assert result["model"]["n_layer"] == 4
        assert result["training"]["lr"] == 0.001

    def test_list_replacement(self):
        base = {"seeds": [1, 2, 3]}
        override = {"seeds": [42]}
        result = merge_configs(base, override)
        assert result["seeds"] == [42]

    def test_does_not_mutate_inputs(self):
        base = {"model": {"n_layer": 4}}
        override = {"model": {"n_layer": 8}}
        merge_configs(base, override)
        assert base["model"]["n_layer"] == 4
        assert override["model"]["n_layer"] == 8

    def test_nested_deep_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 10}}}
        result = merge_configs(base, override)
        assert result == {"a": {"b": {"c": 10, "d": 2}}}


class TestLoadConfig:
    def test_single_file(self, tmp_path):
        cfg = tmp_path / "base.yaml"
        cfg.write_text("model:\n  n_layer: 4\n")
        result = load_config(cfg)
        assert result == {"model": {"n_layer": 4}}

    def test_multiple_files_override(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("model:\n  n_layer: 4\n  d_model: 256\n")
        override = tmp_path / "override.yaml"
        override.write_text("model:\n  n_layer: 8\n")
        result = load_config(base, override)
        assert result["model"]["n_layer"] == 8
        assert result["model"]["d_model"] == 256

    def test_three_layers(self, tmp_path):
        a = tmp_path / "a.yaml"
        a.write_text("x: 1\ny: 2\n")
        b = tmp_path / "b.yaml"
        b.write_text("y: 3\nz: 4\n")
        c = tmp_path / "c.yaml"
        c.write_text("z: 5\n")
        result = load_config(a, b, c)
        assert result == {"x": 1, "y": 3, "z": 5}

    def test_raises_on_no_paths(self):
        with pytest.raises(ValueError, match="At least one config path"):
            load_config()

    def test_loads_project_defaults(self):
        """Smoke test: load the actual project defaults config."""
        project_root = Path(__file__).parent.parent
        defaults_path = project_root / "configs" / "project_defaults.yaml"
        if defaults_path.exists():
            result = load_config(defaults_path)
            assert "project" in result
            assert result["project"]["name"] == "transformer_variant_l4_lab"


class TestResolveConfig:
    def test_passthrough(self):
        config = {"model": {"n_layer": 4}}
        result = resolve_config(config)
        assert result == config

    def test_writes_to_file(self, tmp_path):
        config = {"model": {"n_layer": 4, "d_model": 256}}
        output = tmp_path / "resolved.yaml"
        resolve_config(config, output_path=output)
        assert output.exists()
        loaded = yaml.safe_load(output.read_text())
        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        config = {"x": 1}
        output = tmp_path / "nested" / "dir" / "resolved.yaml"
        resolve_config(config, output_path=output)
        assert output.exists()
