"""Public-seam tests for the static-site asset exporter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.evaluation.site_assets import capture_attention_patterns, export_site_assets
from src.models.config import ModelConfig
from src.models.linear_attention import CausalLinearAttention
from src.models.modern_transformer import ModernTransformer
from src.models.vanilla_transformer import VanillaTransformer


def _write_report(report_dir: Path) -> None:
    raw_dir = report_dir / "raw"
    raw_dir.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "probes": {
            "aggregated": {
                "vanilla": {
                    "stable_rank": {"per_layer": [2.0, 3.0], "mean": 2.5},
                    "cka": {
                        "adjacent_curve": [0.75],
                        "full_matrix": [[1.0, 0.75], [0.75, 1.0]],
                    },
                    "attention_entropy": {
                        "per_layer": [0.4, 0.7],
                        "per_head": [[0.3, 0.5], [0.6, 0.8]],
                    },
                    "n": 3,
                },
                "linear": {
                    "stable_rank": {"per_layer": [1.5, 2.0], "mean": 1.75},
                    "cka": {
                        "adjacent_curve": [0.5],
                        "full_matrix": [[1.0, 0.5], [0.5, 1.0]],
                    },
                    "attention_entropy": None,
                    "n": 3,
                },
            }
        },
    }
    (raw_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_export_site_assets_writes_frontend_agnostic_bundle(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    output_dir = tmp_path / "site_assets"
    _write_report(report_dir)

    result = export_site_assets(report_dir, output_dir=output_dir)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    internals = json.loads(result.model_internals_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["frontend"] == "agnostic"
    assert manifest["assets"]["model_internals"] == "model_internals.json"
    assert internals["variants"]["vanilla"]["cka"]["full_matrix"][0][1] == 0.75
    assert internals["variants"]["vanilla"]["stable_rank"]["per_layer"] == [2.0, 3.0]
    assert internals["variants"]["vanilla"]["attention_entropy"]["per_head"][1] == [0.6, 0.8]
    assert internals["variants"]["linear"]["attention_entropy"] is None
    assert result.plot_paths
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in result.plot_paths)


def test_export_site_assets_splits_attention_for_lazy_static_site_loading(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    output_dir = tmp_path / "site_assets"
    _write_report(report_dir)
    capture = {
        "variant": "vanilla",
        "status": "supported",
        "checkpoint_dir": "checkpoints/vanilla_main_1B_s42",
        "context": {"data_split": "val", "length": 2},
        "tokens": ["one", "two"],
        "layers": [
            {
                "layer": 0,
                "heads": [{"head": 0, "weights": [[1.0, 0.0], [0.25, 0.75]]}],
                "mean_weights": [[1.0, 0.0], [0.25, 0.75]],
            }
        ],
    }

    result = export_site_assets(report_dir, output_dir=output_dir, attention_patterns=[capture])

    index = json.loads(result.attention_patterns_path.read_text(encoding="utf-8"))
    entry = index["variants"][0]
    assert entry["asset"] == "attention_patterns_vanilla.json"
    assert entry["layers"] == [0]
    variant_payload = json.loads((output_dir / entry["asset"]).read_text(encoding="utf-8"))
    assert variant_payload["layers"][0]["heads"][0]["weights"][1] == [0.25, 0.75]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["assets"]["attention_variants"] == [entry["asset"]]
    assert entry["checkpoint_dir"] == "checkpoints/vanilla_main_1B_s42"
    assert entry["context"] == {"data_split": "val", "length": 2}


def test_capture_attention_patterns_exports_layers_heads_and_causal_probabilities() -> None:
    config = ModelConfig(
        n_layer=2,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=8,
        dropout=0.0,
    )
    model = VanillaTransformer(config).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])

    capture = capture_attention_patterns(
        model,
        token_ids,
        token_labels=["one", "two", "three", "four"],
        variant="vanilla",
    )

    assert capture["status"] == "supported"
    assert capture["tokens"] == ["one", "two", "three", "four"]
    assert len(capture["layers"]) == 2
    matrix = torch.tensor(capture["layers"][0]["heads"][0]["weights"])
    assert matrix.shape == (4, 4)
    assert torch.allclose(matrix.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.count_nonzero(torch.triu(matrix, diagonal=1)) == 0


def test_capture_attention_patterns_marks_linear_attention_unsupported() -> None:
    config = ModelConfig(
        n_layer=1,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=8,
        dropout=0.0,
        attention_type="linear",
    )
    model = VanillaTransformer(config, attention_class=CausalLinearAttention).eval()

    capture = capture_attention_patterns(model, torch.tensor([[1, 2, 3]]), variant="linear")

    assert capture["status"] == "unsupported"
    assert "pairwise softmax" in capture["reason"]


def test_capture_attention_patterns_reconstructs_sdpa_probabilities() -> None:
    config = ModelConfig(
        n_layer=1,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=8,
        dropout=0.0,
        variant="modern",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        activation="swiglu",
        attention_type="flash_sdpa",
    )
    model = ModernTransformer(config).eval()

    capture = capture_attention_patterns(model, torch.tensor([[1, 2, 3, 4]]), variant="modern")

    matrix = torch.tensor(capture["layers"][0]["heads"][1]["weights"])
    assert capture["method"] == "reconstructed_pre_dropout_softmax"
    assert torch.allclose(matrix.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.count_nonzero(torch.triu(matrix, diagonal=1)) == 0


def test_export_site_assets_cli_help_exposes_portable_contract() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/export_site_assets.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = " ".join(completed.stdout.split())
    assert "Jekyll-ready" in help_text
    assert "--output-dir" in help_text
    assert "--with-attention" in help_text
    assert "--context-length" in help_text


@pytest.mark.parametrize("bad_length", [0, -1])
def test_capture_attention_patterns_rejects_empty_context(bad_length: int) -> None:
    config = ModelConfig(n_layer=1, d_model=8, n_head=2, vocab_size=16, seq_len=4)
    model = VanillaTransformer(config).eval()
    with pytest.raises(ValueError, match="at least one token"):
        capture_attention_patterns(model, torch.empty((1, max(0, bad_length)), dtype=torch.long))


def test_export_site_assets_rejects_unsafe_or_colliding_variant_slugs(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    _write_report(report_dir)
    unsafe = {"variant": "../escape", "status": "supported", "tokens": [], "layers": []}
    with pytest.raises(ValueError, match="safe asset identifier"):
        export_site_assets(report_dir, output_dir=tmp_path / "unsafe", attention_patterns=[unsafe])

    duplicate = {"variant": "vanilla", "status": "supported", "tokens": [], "layers": []}
    with pytest.raises(ValueError, match="Duplicate attention asset identifier"):
        export_site_assets(
            report_dir,
            output_dir=tmp_path / "duplicate",
            attention_patterns=[duplicate, duplicate],
        )


def test_export_derives_entropy_for_reconstructed_non_vanilla_attention(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    output_dir = tmp_path / "assets"
    _write_report(report_dir)
    capture = {
        "variant": "modern",
        "status": "supported",
        "tokens": ["one", "two"],
        "layers": [
            {
                "layer": 0,
                "heads": [{"head": 0, "weights": [[1.0, 0.0], [0.5, 0.5]]}],
                "mean_weights": [[1.0, 0.0], [0.5, 0.5]],
            }
        ],
    }

    result = export_site_assets(report_dir, output_dir=output_dir, attention_patterns=[capture])

    internals = json.loads(result.model_internals_path.read_text(encoding="utf-8"))
    entropy = internals["variants"]["modern"]["attention_pattern_entropy"]
    assert entropy["source"] == "reconstructed_pre_dropout_softmax"
    assert entropy["per_head"][0][0] == pytest.approx(0.3465736)
    assert "attention_pattern_entropy.png" in {path.name for path in result.plot_paths}
