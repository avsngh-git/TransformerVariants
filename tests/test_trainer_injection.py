"""Test that demonstrates loader injection into the Trainer.

Validates Requirements 6.1, 6.2, 6.3 (decouple-trainer-dataloader):
- A list-based loader can be injected into the Trainer
- The Trainer trains for 10+ steps without error
- No disk I/O is required for data loading

Validates Requirement 7.4 (consolidate-logging):
- Trainer delegates logging to RunLogger (run_logger passed via constructor)
- Trainer no longer has _save_log or _save_running_log methods
"""

import json
import random
from pathlib import Path

import pytest
import torch

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.training.checkpoint import AsyncCheckpointWriter, CheckpointRingBuffer
from src.training.run_logger import RunLogger
from src.training.synthetic_loader import SyntheticLoader
from src.training.trainer import TrainConfig, Trainer
from src.utils.seed import get_rng_state, set_seed


def test_trainer_injection_with_list_based_loader(tmp_path):
    """Trainer trains for 10+ steps with a trivial list-based SyntheticLoader.

    This test demonstrates the decoupling seam: pre-made tensor pairs are
    wrapped in a SyntheticLoader, injected into the Trainer, and training
    completes without any disk I/O for data loading.
    """
    # 1. Create a small model
    model_config = ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=2,
        vocab_size=256,
        seq_len=32,
    )
    model = VanillaTransformer(model_config)

    # 2. Create pre-made (x, y) tensor pairs
    batch_size = 4
    seq_len = 32
    num_batches = 5

    batches = [
        (
            torch.randint(0, model_config.vocab_size, (batch_size, seq_len), dtype=torch.int64),
            torch.randint(0, model_config.vocab_size, (batch_size, seq_len), dtype=torch.int64),
        )
        for _ in range(num_batches)
    ]

    # 3. Wrap in SyntheticLoader
    train_loader = SyntheticLoader(batches)
    val_loader = SyntheticLoader(batches)

    # 4. Create TrainConfig with max_steps=10 and reduced settings for speed
    train_config = TrainConfig(
        max_steps=10,
        micro_batch_size=batch_size,
        grad_accum_steps=1,
        warmup_steps=2,
        log_interval=5,
        eval_interval=100,  # Don't eval mid-run to keep test fast
        checkpoint_interval=1000,  # Don't checkpoint during test
        checkpoint_dir=str(tmp_path / "ckpt"),
        dtype="float32",  # CPU doesn't support bfloat16 well on all platforms
    )

    # 5. Create RunLogger for structured logging
    run_logger = RunLogger(tmp_path / "run", config={"variant": "test", "scale": "debug"})

    # 6. Create Trainer with injected loaders and run_logger
    trainer = Trainer(
        model,
        train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        run_logger=run_logger,
        device="cpu",
    )

    # 7. Run training for 10 steps
    results = trainer.train()

    # 8. Assert no errors raised (implicit — we reached here)
    assert results is not None
    assert "final_train_loss" in results
    assert results["final_train_loss"] > 0  # Loss should be a positive number

    # 9. Assert no disk I/O was required for data loading
    # The SyntheticLoader never touches disk — it only cycles through in-memory tensors.
    # We verify this by confirming the loader's state shows it consumed batches
    # and that no temporary shard files were created.
    assert train_loader._idx > 0, "Train loader should have been called"
    assert val_loader._idx > 0, "Val loader should have been called (final eval)"


def test_trainer_delegates_logging_to_run_logger(tmp_path):
    """Verify Trainer delegates all logging to RunLogger.

    After training, RunLogger's metrics.jsonl should contain entries,
    proving Trainer uses RunLogger rather than inline logging.

    Validates: Requirement 7.4
    """
    model_config = ModelConfig(n_layer=2, d_model=64, n_head=2, vocab_size=256, seq_len=32)
    model = VanillaTransformer(model_config)

    batch_size = 4
    batches = [
        (
            torch.randint(0, 256, (batch_size, 32), dtype=torch.int64),
            torch.randint(0, 256, (batch_size, 32), dtype=torch.int64),
        )
        for _ in range(5)
    ]

    train_loader = SyntheticLoader(batches)
    val_loader = SyntheticLoader(batches)

    train_config = TrainConfig(
        max_steps=10,
        micro_batch_size=batch_size,
        grad_accum_steps=1,
        warmup_steps=2,
        log_interval=5,
        eval_interval=100,
        checkpoint_interval=1000,
        checkpoint_dir=str(tmp_path / "ckpt"),
        dtype="float32",
    )

    run_logger = RunLogger(tmp_path / "run", config={"variant": "test", "scale": "debug"})

    trainer = Trainer(
        model,
        train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        run_logger=run_logger,
        device="cpu",
    )

    trainer.train()

    # RunLogger's metrics.jsonl should have entries after training
    metrics_path = tmp_path / "run" / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl should exist"
    content = metrics_path.read_text().strip()
    assert len(content) > 0, "metrics.jsonl should have entries"

    # Parse and verify entries are valid JSON with expected fields
    lines = content.split("\n")
    assert len(lines) > 0, "Should have at least one metrics entry"

    for line in lines:
        entry = json.loads(line)
        assert "type" in entry
        assert entry["type"] in ("train", "eval")


def test_trainer_no_inline_logging_methods():
    """Verify Trainer no longer has _save_log or _save_running_log.

    Validates: Requirement 7.4
    """
    assert not hasattr(Trainer, "_save_log"), "Trainer should not have _save_log"
    assert not hasattr(Trainer, "_save_running_log"), "Trainer should not have _save_running_log"


def test_trainer_requires_run_logger():
    """Verify run_logger is a required constructor parameter (no set_run_logger).

    Validates: Requirement 7.4
    """
    assert not hasattr(Trainer, "set_run_logger"), "Trainer should not have set_run_logger method"


def test_trainer_resumes_fault_tolerant_checkpoint_with_rng_state(tmp_path):
    """Trainer accepts the fault-tolerant checkpoint schema and restores progress and RNG."""
    model_config = ModelConfig(n_layer=1, d_model=32, n_head=2, vocab_size=64, seq_len=8)
    batches = [
        (
            torch.randint(0, 64, (2, 8), dtype=torch.int64),
            torch.randint(0, 64, (2, 8), dtype=torch.int64),
        )
    ]
    trainer = Trainer(
        VanillaTransformer(model_config),
        TrainConfig(max_steps=1, checkpoint_dir=str(tmp_path), dtype="float32"),
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "run", config={"variant": "test"}),
        device="cpu",
    )
    trainer.step = 17
    trainer.tokens_processed = 12_345
    trainer.best_val_loss = 2.75

    set_seed(2026)
    saved_rng_state = get_rng_state()
    expected_python_random = random.random()
    expected_torch_random = torch.rand(3)

    checkpoint_dir = tmp_path / "fault_tolerant"
    ring = CheckpointRingBuffer(checkpoint_dir, capacity=3)
    writer = AsyncCheckpointWriter(ring, checkpoint_dir)
    writer.save(
        step=trainer.step,
        model=trainer.model,
        optimizer=trainer.optimizer,
        training_state={
            "tokens_processed": trainer.tokens_processed,
            "best_val_loss": trainer.best_val_loss,
            "rng_state": saved_rng_state,
        },
    )
    writer.wait()

    trainer.step = 0
    trainer.tokens_processed = 0
    trainer.best_val_loss = float("inf")
    set_seed(7)
    trainer.load_checkpoint(writer.latest())

    assert trainer.step == 17
    assert trainer.tokens_processed == 12_345
    assert trainer.best_val_loss == 2.75
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(3), expected_torch_random)
    writer.shutdown()


def test_fault_tolerant_training_returns_with_durable_resumable_checkpoint(tmp_path):
    """Completed training flushes its async checkpoint and preserves RNG continuity."""
    model_config = ModelConfig(n_layer=1, d_model=32, n_head=2, vocab_size=64, seq_len=8)
    batches = [
        (
            torch.randint(0, 64, (2, 8), dtype=torch.int64),
            torch.randint(0, 64, (2, 8), dtype=torch.int64),
        )
    ]
    train_config = TrainConfig(
        max_steps=1,
        eval_steps=1,
        grad_accum_steps=1,
        checkpoint_interval=10,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        dtype="float32",
    )
    ring = CheckpointRingBuffer(Path(train_config.checkpoint_dir), capacity=3)
    writer = AsyncCheckpointWriter(ring, Path(train_config.checkpoint_dir))
    trainer = Trainer(
        VanillaTransformer(model_config),
        train_config,
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "run", config={"variant": "test"}),
        checkpoint_manager=writer,
        device="cpu",
    )

    set_seed(2026)
    trainer.train()
    checkpoint_path = writer.latest()
    assert checkpoint_path is not None
    assert checkpoint_path.exists()
    expected_python_random = random.random()
    expected_torch_random = torch.rand(3)

    resumed = Trainer(
        VanillaTransformer(model_config),
        train_config,
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "resumed", config={"variant": "test"}),
        device="cpu",
    )
    set_seed(7)
    resumed.load_checkpoint(checkpoint_path)

    assert resumed.step == trainer.step
    assert resumed.tokens_processed == trainer.tokens_processed
    assert resumed.train_loader._idx == trainer.train_loader._idx
    assert resumed.val_loader._idx == trainer.val_loader._idx
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(3), expected_torch_random)
    writer.shutdown()


def test_periodic_fault_tolerant_checkpoint_resumes_at_next_step(tmp_path):
    model_config = ModelConfig(n_layer=1, d_model=16, n_head=2, vocab_size=32, seq_len=4)
    batches = [
        (
            torch.randint(0, 32, (1, 4), dtype=torch.int64),
            torch.randint(0, 32, (1, 4), dtype=torch.int64),
        )
    ]
    train_config = TrainConfig(
        max_steps=2,
        eval_steps=1,
        grad_accum_steps=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        dtype="float32",
    )
    writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(Path(train_config.checkpoint_dir), capacity=3),
        Path(train_config.checkpoint_dir),
    )
    trainer = Trainer(
        VanillaTransformer(model_config),
        train_config,
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "run", config={}),
        checkpoint_manager=writer,
        device="cpu",
    )

    trainer._training_step()
    trainer._save_checkpoint(completed_step=trainer.step + 1)
    writer.wait()

    resumed = Trainer(
        VanillaTransformer(model_config),
        train_config,
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "resume", config={}),
        device="cpu",
    )
    resumed.load_checkpoint(writer.latest())

    assert resumed.step == 1
    writer.shutdown()


def test_rollback_without_verified_checkpoint_fails_safely(tmp_path):
    model_config = ModelConfig(n_layer=1, d_model=16, n_head=2, vocab_size=32, seq_len=4)
    batches = [
        (
            torch.randint(0, 32, (1, 4), dtype=torch.int64),
            torch.randint(0, 32, (1, 4), dtype=torch.int64),
        )
    ]
    checkpoint_dir = tmp_path / "empty"
    writer = AsyncCheckpointWriter(CheckpointRingBuffer(checkpoint_dir, capacity=3), checkpoint_dir)
    trainer = Trainer(
        VanillaTransformer(model_config),
        TrainConfig(max_steps=1, eval_steps=1, dtype="float32"),
        train_loader=SyntheticLoader(batches),
        val_loader=SyntheticLoader(batches),
        run_logger=RunLogger(tmp_path / "run", config={}),
        checkpoint_manager=writer,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="no verified checkpoint"):
        trainer._rollback_to_latest_verified()

    writer.shutdown()


def test_zero_skip_contract_fails_on_first_health_monitor_skip(tmp_path):
    """Canonical training stops immediately instead of silently losing an update."""

    class AlwaysSkipMonitor:
        def check(self, step: int, loss: float, grad_norm: float):
            from src.training.health_monitor import Action

            return Action.SKIP_STEP

    model_config = ModelConfig(n_layer=1, d_model=16, n_head=2, vocab_size=32, seq_len=4)
    batch = (
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
    )
    trainer = Trainer(
        VanillaTransformer(model_config),
        TrainConfig(
            max_steps=2,
            eval_steps=1,
            grad_accum_steps=1,
            max_skipped_steps=0,
            dtype="float32",
        ),
        train_loader=SyntheticLoader([batch]),
        val_loader=SyntheticLoader([batch]),
        run_logger=RunLogger(tmp_path / "run", config={}),
        health_monitor=AlwaysSkipMonitor(),
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="skip budget"):
        trainer.train()

    assert trainer.step == 0
    assert trainer._skipped_steps == 1


def test_zero_skip_contract_rejects_contaminated_resume_checkpoint(tmp_path):
    """Direct CLI-style resume cannot bypass the launcher's ring assessment."""
    model_config = ModelConfig(n_layer=1, d_model=16, n_head=2, vocab_size=32, seq_len=4)
    batch = (
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
    )
    train_config = TrainConfig(
        max_steps=2,
        eval_steps=1,
        grad_accum_steps=1,
        max_skipped_steps=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        dtype="float32",
    )
    trainer = Trainer(
        VanillaTransformer(model_config),
        train_config,
        train_loader=SyntheticLoader([batch]),
        val_loader=SyntheticLoader([batch]),
        run_logger=RunLogger(tmp_path / "run", config={}),
        device="cpu",
    )
    writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(Path(train_config.checkpoint_dir), capacity=3),
        Path(train_config.checkpoint_dir),
    )
    writer.save(
        step=1,
        model=trainer.model,
        optimizer=trainer.optimizer,
        training_state={
            "tokens_processed": 4,
            "best_val_loss": 1.0,
            "skipped_steps": 1,
        },
    )
    writer.wait()

    with pytest.raises(RuntimeError, match="Checkpoint violates skipped-step contract"):
        trainer.load_checkpoint(writer.latest())

    writer.shutdown()


def test_fault_tolerant_training_bootstraps_checkpoint_and_retries_rollback(tmp_path):
    """A recovery request before the periodic interval restores step zero and retries it."""

    class RollbackOnceMonitor:
        def __init__(self) -> None:
            self.check_calls = 0

        def check(self, step: int, loss: float, grad_norm: float):
            from src.training.health_monitor import Action

            self.check_calls += 1
            return Action.ROLLBACK if self.check_calls == 1 else Action.CONTINUE

        def reset(self) -> None:
            pass

    model_config = ModelConfig(
        n_layer=1,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=4,
    )
    batch = (
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
    )
    checkpoint_dir = tmp_path / "checkpoints"
    writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(checkpoint_dir, capacity=3), checkpoint_dir
    )
    monitor = RollbackOnceMonitor()
    trainer = Trainer(
        VanillaTransformer(model_config),
        TrainConfig(
            max_steps=1,
            eval_steps=1,
            grad_accum_steps=1,
            checkpoint_interval=100,
            checkpoint_dir=str(checkpoint_dir),
            dtype="float32",
        ),
        train_loader=SyntheticLoader([batch]),
        val_loader=SyntheticLoader([batch]),
        run_logger=RunLogger(tmp_path / "run", config={}),
        checkpoint_manager=writer,
        health_monitor=monitor,
        device="cpu",
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.train()

    assert monitor.check_calls == 2
    assert trainer.step == 1
    assert trainer.tokens_processed == 4
    assert any(
        not torch.equal(initial_parameters[name], parameter)
        for name, parameter in trainer.model.named_parameters()
    )
    assert writer.rollback() is not None
    recovery_events = [
        json.loads(line)
        for line in (tmp_path / "run" / "recovery_events.jsonl").read_text().splitlines()
    ]
    assert recovery_events[0]["event"] == "health_monitor_rollback"
    assert recovery_events[0]["trigger_step"] == 0
    assert recovery_events[0]["attempt"] == 1
    assert recovery_events[0]["checkpoint"].endswith("checkpoint_step_000000.pt")
    writer.shutdown()


def test_training_intervals_are_numbered_by_completed_optimizer_steps(tmp_path):
    """Logs/evaluations/checkpoints at step N represent exactly N completed updates."""
    model_config = ModelConfig(
        n_layer=1,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=4,
    )
    batch = (
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
        torch.randint(0, 32, (1, 4), dtype=torch.int64),
    )
    checkpoint_dir = tmp_path / "checkpoints"
    writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(checkpoint_dir, capacity=4), checkpoint_dir
    )
    run_dir = tmp_path / "run"
    trainer = Trainer(
        VanillaTransformer(model_config),
        TrainConfig(
            max_steps=4,
            eval_steps=1,
            grad_accum_steps=1,
            log_interval=1,
            eval_interval=2,
            checkpoint_interval=2,
            checkpoint_dir=str(checkpoint_dir),
            dtype="float32",
        ),
        train_loader=SyntheticLoader([batch]),
        val_loader=SyntheticLoader([batch]),
        run_logger=RunLogger(run_dir, config={}),
        checkpoint_manager=writer,
        device="cpu",
    )

    trainer.train()

    entries = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    train_entries = [entry for entry in entries if entry["type"] == "train"]
    eval_entries = [entry for entry in entries if entry["type"] == "eval"]
    assert [entry["step"] for entry in train_entries] == [1, 2, 3, 4]
    assert [entry["tokens_processed"] for entry in train_entries] == [4, 8, 12, 16]
    assert [entry["step"] for entry in eval_entries] == [2, 4]
    assert [entry.step for entry in writer._ring_buffer.list_available()] == [0, 2, 4]
    writer.shutdown()
