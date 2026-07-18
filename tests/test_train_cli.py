"""Command-level contract tests for training fault tolerance."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.training.run_config_builder import RunConfigBuilder


def test_train_help_exposes_fault_tolerant_training_contract() -> None:
    """Operators can discover fault tolerance and verified-latest resume from the CLI."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = " ".join(result.stdout.split())
    assert "--fault-tolerant" in help_text
    assert "--checkpoint-ring-size" in help_text
    assert "latest verified checkpoint" in help_text


def test_run_config_records_fault_tolerance_policy() -> None:
    args = Namespace(
        variant="vanilla",
        scale="debug",
        seed=42,
        activation="relu",
        max_steps=1,
        checkpoint_dir="checkpoints/test",
        max_lr=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        dtype="float32",
        log_interval=1,
        eval_interval=1,
        eval_steps=2,
        checkpoint_interval=1,
        compile=False,
        data_dir="data/test",
        resume="latest",
        fault_tolerant=True,
        checkpoint_ring_size=5,
    )
    config = ModelConfig(n_layer=1, d_model=8, n_head=2, vocab_size=16, seq_len=4)

    bundle = RunConfigBuilder.from_args(args, config, VanillaTransformer(config))

    assert bundle.run_config["training"] == {
        "max_steps": 1,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 1,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
        "grad_clip": 1.0,
        "dtype": "float32",
        "compiled": False,
        "eval_interval": 1,
        "eval_steps": 2,
        "checkpoint_interval": 1,
    }
    assert bundle.run_config["fault_tolerance"] == {
        "enabled": True,
        "checkpoint_ring_size": 5,
        "integrity": "sha256",
        "checkpoint_write": "async_atomic",
        "health_monitor": True,
    }


def test_run_config_records_legacy_policy_when_fault_tolerance_is_disabled() -> None:
    args = Namespace(
        variant="vanilla",
        scale="debug",
        seed=42,
        activation="relu",
        max_steps=1,
        checkpoint_dir="checkpoints/test",
        max_lr=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        dtype="float32",
        log_interval=1,
        eval_interval=1,
        eval_steps=2,
        checkpoint_interval=1,
        compile=False,
        data_dir="data/test",
        resume=None,
        fault_tolerant=False,
        checkpoint_ring_size=3,
    )
    config = ModelConfig(n_layer=1, d_model=8, n_head=2, vocab_size=16, seq_len=4)

    bundle = RunConfigBuilder.from_args(args, config, VanillaTransformer(config))

    assert bundle.run_config["fault_tolerance"] == {
        "enabled": False,
        "checkpoint_ring_size": None,
        "integrity": None,
        "checkpoint_write": "legacy_torch_save",
        "health_monitor": False,
    }
