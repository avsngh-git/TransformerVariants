"""Contract tests for the canonical multi-run experiment launcher."""

from __future__ import annotations

import json

import pytest

from scripts.train_matrix import (
    build_training_command,
    load_manifest,
    token_accounting,
)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "main_500m_5seed",
        "data_dir": "data/processed/fineweb-1B",
        "scale": "main",
        "variants": ["vanilla", "moe"],
        "seeds": [42, 137],
        "training": {
            "max_steps": 7629,
            "target_tokens_per_run": 500_000_000,
            "sequence_length": 1024,
            "micro_batch_size": 8,
            "grad_accum_steps": 8,
            "max_lr": 0.0003,
            "min_lr": 0.00003,
            "warmup_steps": 250,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "eval_interval": 250,
            "eval_steps": 20,
            "checkpoint_interval": 1250,
            "log_interval": 10,
        },
        "fault_tolerance": {
            "enabled": True,
            "checkpoint_ring_size": 3,
            "resume_latest_verified": True,
        },
        "variant_overrides": {
            "vanilla": {"activation": "gelu", "compile": True},
            "moe": {"compile": False},
        },
        "output": {
            "run_template": "runs/main_500m_5seed/{variant}_s{seed}",
            "checkpoint_template": (
                "runs/main_500m_5seed/{variant}_s{seed}/checkpoints"
            ),
            "resolved_manifest": "reports/500M_5seed/resolved_manifest.json",
        },
    }


def test_manifest_token_accounting_stays_below_declared_budget(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    manifest = load_manifest(path)
    accounting = token_accounting(manifest)

    assert accounting["tokens_per_step"] == 65_536
    assert accounting["tokens_per_run"] == 499_974_144
    assert accounting["total_runs"] == 4
    assert accounting["total_tokens"] == 1_999_896_576


def test_launcher_builds_fault_tolerant_resume_command() -> None:
    command = build_training_command(
        _manifest(),
        variant="vanilla",
        seed=42,
        python_executable="python",
        resume=True,
    )

    assert command[:2] == ["python", "scripts/train.py"]
    assert command[command.index("--max_steps") + 1] == "7629"
    assert command[command.index("--min_lr") + 1] == "3e-05"
    assert command[command.index("--weight_decay") + 1] == "0.1"
    assert command[command.index("--beta2") + 1] == "0.95"
    assert command[command.index("--eval_steps") + 1] == "20"
    assert command[command.index("--activation") + 1] == "gelu"
    assert "--compile" in command
    assert "--fault-tolerant" in command
    assert command[command.index("--resume") + 1] == "latest"
    assert command[command.index("--run-dir") + 1].endswith(
        "main_500m_5seed/vanilla_s42"
    )
    assert command[command.index("--checkpoint_dir") + 1].endswith(
        "main_500m_5seed/vanilla_s42/checkpoints"
    )


def test_manifest_rejects_steps_that_exceed_target_tokens(tmp_path) -> None:
    manifest = _manifest()
    manifest["training"]["max_steps"] = 7630
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds target_tokens_per_run"):
        load_manifest(path)


def test_manifest_rejects_token_accounting_length_drift(tmp_path) -> None:
    manifest = _manifest()
    manifest["training"]["sequence_length"] = 512
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match registry scale"):
        load_manifest(path)
