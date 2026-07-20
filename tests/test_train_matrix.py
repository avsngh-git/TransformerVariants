"""Contract tests for the canonical multi-run experiment launcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.train_matrix import (
    assess_final_checkpoint,
    assess_resume_checkpoint,
    build_training_command,
    final_checkpoint,
    load_manifest,
    main,
    token_accounting,
)
from src.training.checkpoint import AtomicCheckpointWriter, CheckpointRingBuffer


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
            "max_skipped_steps": 0,
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
    assert command[command.index("--max-skipped-steps") + 1] == "0"
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


@pytest.mark.parametrize("value", [None, -1])
def test_manifest_requires_explicit_nonnegative_skip_budget(tmp_path, value) -> None:
    manifest = _manifest()
    if value is None:
        del manifest["training"]["max_skipped_steps"]
    else:
        manifest["training"]["max_skipped_steps"] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="max_skipped_steps"):
        load_manifest(path)


def _write_final_checkpoint(
    path: Path, manifest: dict, *, skipped_steps: int | None = 0
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_state = {
        "tokens_processed": token_accounting(manifest)["tokens_per_run"],
    }
    if skipped_steps is not None:
        training_state["skipped_steps"] = skipped_steps
    AtomicCheckpointWriter().save(
        {
            "step": 7629,
            "training_state": training_state,
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {},
        },
        path,
    )
    return path


def test_final_checkpoint_acceptance_requires_zero_skipped_updates(tmp_path) -> None:
    manifest = _manifest()
    path = _write_final_checkpoint(
        tmp_path / "checkpoint_step_007629.pt", manifest, skipped_steps=25
    )

    assessment = assess_final_checkpoint(manifest, path)

    assert assessment["verified"] is True
    assert assessment["skipped_steps"] == 25
    assert assessment["accepted"] is False
    assert "skipped_steps" in assessment["reasons"]


def test_final_checkpoint_acceptance_checks_step_and_token_budget(tmp_path) -> None:
    manifest = _manifest()
    path = _write_final_checkpoint(tmp_path / "checkpoint_step_007629.pt", manifest)

    assessment = assess_final_checkpoint(manifest, path)

    assert assessment == {
        "accepted": True,
        "verified": True,
        "step": 7629,
        "tokens_processed": 499_974_144,
        "skipped_steps": 0,
        "reasons": [],
    }


@pytest.mark.parametrize("skipped_steps", [None, -1])
def test_final_checkpoint_rejects_missing_or_negative_skip_counter(
    tmp_path, skipped_steps
) -> None:
    manifest = _manifest()
    path = _write_final_checkpoint(
        tmp_path / "checkpoint_step_007629.pt",
        manifest,
        skipped_steps=skipped_steps,
    )

    assessment = assess_final_checkpoint(manifest, path)

    assert assessment["accepted"] is False
    assert "schema" in assessment["reasons"]


def test_resume_assessment_rejects_checkpoint_with_skipped_updates(tmp_path) -> None:
    manifest = _manifest()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "checkpoint_step_000100.pt"
    digest = AtomicCheckpointWriter().save(
        {
            "step": 100,
            "training_state": {
                "tokens_processed": 100 * token_accounting(manifest)["tokens_per_step"],
                "skipped_steps": 1,
            },
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {},
        },
        checkpoint_path,
    )
    CheckpointRingBuffer(checkpoint_dir, capacity=3).register(
        100, checkpoint_path, digest
    )

    assessment = assess_resume_checkpoint(manifest, checkpoint_dir)

    assert assessment["available"] is True
    assert assessment["verified"] is True
    assert assessment["accepted"] is False
    assert assessment["skipped_steps"] == 1
    assert "skipped_steps" in assessment["reasons"]


def _single_run_manifest(tmp_path) -> tuple[dict, Path]:
    manifest = _manifest()
    manifest["variants"] = ["vanilla"]
    manifest["seeds"] = [42]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest["data_dir"] = str(data_dir)
    manifest["output"] = {
        "run_template": str(tmp_path / "runs/{variant}_s{seed}"),
        "checkpoint_template": str(tmp_path / "runs/{variant}_s{seed}/checkpoints"),
        "resolved_manifest": str(tmp_path / "resolved.json"),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


def test_launcher_refuses_to_overwrite_contract_invalid_final(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _single_run_manifest(tmp_path)
    final = final_checkpoint(manifest, "vanilla", 42)
    _write_final_checkpoint(final, manifest, skipped_steps=1)
    subprocess_called = False

    def unexpected_subprocess(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True

    monkeypatch.setattr("scripts.train_matrix.subprocess.run", unexpected_subprocess)
    monkeypatch.setattr("scripts.train_matrix._git_sha", lambda: "test-sha")
    monkeypatch.setattr("scripts.train_matrix.platform.platform", lambda: "test-platform")
    monkeypatch.setattr(sys, "argv", ["train_matrix.py", "--manifest", str(manifest_path)])

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        main()

    assert subprocess_called is False


def test_launcher_rejects_new_run_that_violates_acceptance_contract(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _single_run_manifest(tmp_path)
    final = final_checkpoint(manifest, "vanilla", 42)

    def write_invalid_final(_command, *, check):
        assert check is True
        _write_final_checkpoint(final, manifest, skipped_steps=1)

    monkeypatch.setattr("scripts.train_matrix.subprocess.run", write_invalid_final)
    monkeypatch.setattr("scripts.train_matrix._git_sha", lambda: "test-sha")
    monkeypatch.setattr("scripts.train_matrix.platform.platform", lambda: "test-platform")
    monkeypatch.setattr(sys, "argv", ["train_matrix.py", "--manifest", str(manifest_path)])

    with pytest.raises(RuntimeError, match=r"reasons=\['skipped_steps'\]"):
        main()

    resolved = json.loads((tmp_path / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["resolved"]["runs"][0]["status"] == "failed_acceptance"
