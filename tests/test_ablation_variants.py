"""Contracts for the secondary Modern-backbone surgical ablations."""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from scripts.train_matrix import load_manifest, token_accounting
from src.models.ffn import FeedForward
from src.models.modern_transformer import ModernTransformer
from src.models.registry import build
from src.models.rmsnorm import RMSNorm
from src.models.swiglu_ffn import SwiGLUFeedForward


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_learned_position_ablation_changes_only_position_mechanism() -> None:
    model, config = build("modern_abspos", "debug")

    assert isinstance(model, ModernTransformer)
    assert config.position_encoding == "learned"
    assert hasattr(model, "pos_emb")
    assert not hasattr(model.blocks[0].attn, "rope_cos")
    assert isinstance(model.blocks[0].ln1, RMSNorm)
    assert isinstance(model.blocks[0].ffn, SwiGLUFeedForward)

    model.eval()
    tokens = torch.randint(0, config.vocab_size, (1, 5))
    full_logits, _, _ = model(tokens)
    cache = None
    cached_logits = []
    for position in range(tokens.shape[1]):
        logits, _, cache = model(tokens[:, position : position + 1], kv_cache=cache)
        cached_logits.append(logits)
    assert torch.allclose(full_logits, torch.cat(cached_logits, dim=1), atol=1e-5)


def test_layernorm_ablation_keeps_rope_and_swiglu() -> None:
    model, config = build("modern_layernorm", "debug")

    assert config.norm_type == "layernorm"
    assert isinstance(model.blocks[0].ln1, nn.LayerNorm)
    assert isinstance(model.ln_f, nn.LayerNorm)
    assert hasattr(model.blocks[0].attn, "rope_cos")
    assert isinstance(model.blocks[0].ffn, SwiGLUFeedForward)


def test_gelu_ablation_exactly_matches_modern_ffn_parameters() -> None:
    modern, modern_config = build("modern", "main")
    gelu, gelu_config = build("modern_gelu", "main")

    assert gelu_config.ffn_type == "standard"
    assert gelu_config.activation == "gelu"
    assert gelu_config.ffn_hidden_dim == 2112
    assert isinstance(gelu.blocks[0].ffn, FeedForward)
    assert _parameter_count(gelu.blocks[0].ffn) == _parameter_count(
        modern.blocks[0].ffn
    )
    assert modern_config.ffn_hidden_dim is None


def test_ablation_yaml_configs_match_registry() -> None:
    for variant in ("modern_abspos", "modern_layernorm", "modern_gelu"):
        payload = yaml.safe_load(Path(f"configs/model/{variant}.yaml").read_text())
        _, config = build(variant, "debug")
        assert payload["model"]["variant"] == config.variant
        assert payload["model"]["norm_type"] == config.norm_type
        assert payload["model"]["position_encoding"] == config.position_encoding
        assert payload["model"]["ffn_type"] == config.ffn_type


def test_ablation_manifest_is_nine_runs_at_exact_budget() -> None:
    manifest = load_manifest("configs/experiment/surgical_ablations.yaml")
    accounting = token_accounting(manifest)

    assert accounting == {
        "tokens_per_step": 65_536,
        "tokens_per_run": 245_760_000,
        "total_runs": 9,
        "total_tokens": 2_211_840_000,
    }
    assert manifest["analysis"]["reference_step"] == 3750
    assert manifest["analysis"]["reference_source"].startswith("primary metrics")
