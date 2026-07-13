"""Registry and YAML config tests for causal linear attention."""

from pathlib import Path

import pytest
import yaml

from src.models import registry
from src.models.config import ModelConfig
from src.models.linear_attention import CausalLinearAttention
from src.models.modern_transformer import ModernTransformer
from src.models.registry import VARIANTS


class TestRegistryEntry:
    """Tests for the 'linear' entry in VARIANTS."""

    def test_linear_variant_exists(self):
        """VARIANTS contains a 'linear' key."""
        assert "linear" in VARIANTS

    def test_variant_spec_fields(self):
        """VariantSpec has correct fields for the linear variant."""
        spec = VARIANTS["linear"]
        assert spec.variant == "linear"
        assert spec.attention_type == "linear"
        assert spec.model_class == ModernTransformer
        assert spec.attention_class == CausalLinearAttention
        assert spec.norm_type == "rmsnorm"
        assert spec.position_encoding == "rope"
        assert spec.ffn_type == "swiglu"
        assert spec.requires_bf16 is False


class TestRegistryBuild:
    """Tests for registry.build with the linear variant."""

    @pytest.mark.parametrize("scale", ["debug", "main", "stretch"])
    def test_build_returns_correct_types(self, scale):
        """build('linear', scale) returns (ModernTransformer, ModelConfig)."""
        model, config = registry.build("linear", scale)
        assert isinstance(model, ModernTransformer)
        assert isinstance(config, ModelConfig)

    def test_build_config_has_no_projection_rank(self):
        """Causal linear V5 does not use Linformer's projection rank."""
        _, config = registry.build("linear", "debug")
        assert config.projection_rank is None

    def test_build_invalid_scale_raises(self):
        """build('linear', 'invalid') raises ValueError listing available scales."""
        with pytest.raises(ValueError, match="Available scales"):
            registry.build("linear", "invalid_scale")


class TestParameterCountParity:
    """Parameter count within ±5% of V1 (modern) at each scale."""

    @pytest.mark.parametrize("scale", ["debug", "main", "stretch"])
    def test_param_count_parity(self, scale):
        """Linear variant params are within ±5% of modern variant params."""
        linear_model, _ = registry.build("linear", scale)
        modern_model, _ = registry.build("modern", scale)

        linear_params = sum(p.numel() for p in linear_model.parameters())
        modern_params = sum(p.numel() for p in modern_model.parameters())

        ratio = abs(linear_params - modern_params) / modern_params
        assert ratio <= 0.05, (
            f"Parameter parity failed at {scale}: linear={linear_params}, "
            f"modern={modern_params}, ratio={ratio:.4f}"
        )


class TestYAMLConfig:
    """Tests for configs/model/linear.yaml."""

    @pytest.fixture
    def yaml_config(self):
        """Load and parse the YAML config."""
        path = Path("configs/model/linear.yaml")
        assert path.exists(), "configs/model/linear.yaml does not exist"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_yaml_exists_and_parses(self, yaml_config):
        """YAML file exists and parses without error."""
        assert yaml_config is not None
        assert "model" in yaml_config

    def test_model_fields(self, yaml_config):
        """Model mapping has correct variant fields."""
        model = yaml_config["model"]
        assert model["variant"] == "linear"
        assert model["variant_id"] == "V5"
        assert model["norm_type"] == "rmsnorm"
        assert model["position_encoding"] == "rope"
        assert "projection_rank" not in model
        assert model["ffn_type"] == "swiglu"
        assert model["attention_type"] == "linear"

    def test_shared_fields(self, yaml_config):
        """Shared fields match requirements."""
        model = yaml_config["model"]
        assert model["vocab_size"] == 50257
        assert model["dropout"] == 0.0
        assert model["bias"] is False
        assert model["tie_embeddings"] is True

    @pytest.mark.parametrize(
        "scale,expected",
        [
            ("debug", {"n_layer": 4, "d_model": 256, "n_head": 4, "seq_len": 512}),
            ("main", {"n_layer": 8, "d_model": 512, "n_head": 8, "seq_len": 1024}),
            ("stretch", {"n_layer": 12, "d_model": 768, "n_head": 12, "seq_len": 1024}),
        ],
    )
    def test_scale_fields(self, yaml_config, scale, expected):
        """Each scale has correct dimension fields."""
        model = yaml_config["model"]
        for key, value in expected.items():
            assert model[scale][key] == value
