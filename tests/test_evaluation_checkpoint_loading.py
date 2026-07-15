"""Regression tests for loading real training artifacts during evaluation."""

import json
from pathlib import Path

import torch
from torch import nn

from src.evaluation.comparison import VariantData, load_variant_data
from src.evaluation.pipeline import EvaluationPipeline
from src.models.config import ModelConfig


def _write_metrics(path: Path) -> None:
    path.write_text(json.dumps({"type": "eval", "step": 1, "val_loss": 4.0}) + "\n")


def test_load_variant_data_follows_metrics_symlink_to_training_config(tmp_path: Path):
    """Checkpoint-only directories should recover config from their linked run."""
    run_dir = tmp_path / "runs" / "alibi_main_20260707_1634"
    run_dir.mkdir(parents=True)
    _write_metrics(run_dir / "metrics.jsonl")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "variant": "alibi",
                "scale": "main",
                "model": {
                    "n_layer": 8,
                    "d_model": 512,
                    "n_head": 8,
                    "seq_len": 1024,
                    "vocab_size": 50257,
                    "activation": "swiglu",
                    "bias": False,
                    "dropout": 0.0,
                    "tie_embeddings": True,
                    "total_params": 51430400,
                },
                "training": {"dtype": "bfloat16"},
            }
        )
    )

    checkpoint_dir = tmp_path / "checkpoints" / "alibi_main_1B_s42"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "metrics.jsonl").symlink_to(run_dir / "metrics.jsonl")

    variants = load_variant_data([checkpoint_dir])

    assert len(variants) == 1
    config = variants[0].config
    assert variants[0].name == "alibi"
    assert config.variant == "alibi"
    assert config.n_layer == 8
    assert config.d_model == 512
    assert config.n_head == 8
    assert config.seq_len == 1024
    assert config.norm_type == "rmsnorm"
    assert config.position_encoding == "alibi"
    assert config.ffn_type == "swiglu"
    assert config.attention_type == "flash_alibi"


def test_checkpoint_loader_builds_required_variant_in_bfloat16(tmp_path: Path, monkeypatch):
    """FlashAttention variants must not be reconstructed in float32."""
    checkpoint_dir = tmp_path / "alibi_debug_s42"
    checkpoint_dir.mkdir()
    source_model = nn.Linear(2, 2, bias=False)
    torch.save(
        {"model_state_dict": source_model.state_dict()},
        checkpoint_dir / "checkpoint_latest.pt",
    )

    build_args = {}

    def fake_build(variant_name, scale, activation, dtype):
        build_args.update(variant=variant_name, scale=scale, dtype=dtype)
        model = nn.Linear(2, 2, bias=False)
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        return model, ModelConfig(variant=variant_name)

    monkeypatch.setattr("src.models.registry.build", fake_build)
    variant = VariantData(
        name="alibi",
        checkpoint_dir=checkpoint_dir,
        log_entries=[],
        config=ModelConfig(
            variant="alibi",
            activation="swiglu",
            norm_type="rmsnorm",
            position_encoding="alibi",
            ffn_type="swiglu",
            attention_type="flash_alibi",
        ),
    )

    loaded = EvaluationPipeline(device="cpu").load_model_from_checkpoint(variant)

    assert loaded is not None
    assert build_args["dtype"] == "bfloat16"
    assert next(loaded.parameters()).dtype == torch.bfloat16


def test_checkpoint_loader_returns_none_when_no_weights_load(tmp_path: Path, monkeypatch):
    """A randomly initialized fallback must never be evaluated as a checkpoint."""
    checkpoint_dir = tmp_path / "vanilla_debug_s42"
    checkpoint_dir.mkdir()
    torch.save(
        {"model_state_dict": nn.Linear(3, 3, bias=False).state_dict()},
        checkpoint_dir / "checkpoint_latest.pt",
    )

    def fake_build(variant_name, scale, activation, dtype):
        return nn.Linear(2, 2, bias=False), ModelConfig(variant=variant_name)

    monkeypatch.setattr("src.models.registry.build", fake_build)
    variant = VariantData(
        name="vanilla",
        checkpoint_dir=checkpoint_dir,
        log_entries=[],
        config=ModelConfig(variant="vanilla"),
    )

    loaded = EvaluationPipeline(device="cpu").load_model_from_checkpoint(variant)

    assert loaded is None


def test_checkpoint_loader_rejects_unexpected_architecture_keys(tmp_path: Path, monkeypatch):
    """Old Linformer E/F weights must not load into causal linear V5."""
    checkpoint_dir = tmp_path / "linear_debug_s42"
    checkpoint_dir.mkdir()
    source_model = nn.Linear(2, 2, bias=False)
    state_dict = source_model.state_dict()
    state_dict["legacy_projection.E"] = torch.zeros(2, 1)
    torch.save(
        {"model_state_dict": state_dict},
        checkpoint_dir / "checkpoint_latest.pt",
    )

    def fake_build(variant_name, scale, activation, dtype):
        return nn.Linear(2, 2, bias=False), ModelConfig(variant=variant_name)

    monkeypatch.setattr("src.models.registry.build", fake_build)
    variant = VariantData(
        name="linear",
        checkpoint_dir=checkpoint_dir,
        log_entries=[],
        config=ModelConfig(
            variant="linear",
            norm_type="rmsnorm",
            position_encoding="rope",
            ffn_type="swiglu",
            attention_type="linear",
        ),
    )

    loaded = EvaluationPipeline(device="cpu").load_model_from_checkpoint(variant)
    assert loaded is None
