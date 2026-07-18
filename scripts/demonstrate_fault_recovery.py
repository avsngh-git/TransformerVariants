"""Produce a deterministic corruption/rollback/resume recovery demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.training.checkpoint import (
    AsyncCheckpointWriter,
    AtomicCheckpointWriter,
    CheckpointRingBuffer,
)
from src.training.run_logger import RunLogger
from src.training.synthetic_loader import SyntheticLoader
from src.training.trainer import TrainConfig, Trainer
from src.utils.seed import set_seed


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7001)
    return [
        (
            torch.randint(0, 32, (1, 4), generator=generator),
            torch.randint(0, 32, (1, 4), generator=generator),
        )
        for _ in range(4)
    ]


def _trainer(root: Path, name: str, writer: AsyncCheckpointWriter | None = None) -> Trainer:
    set_seed(2026)
    config = ModelConfig(
        n_layer=1,
        d_model=16,
        n_head=2,
        vocab_size=32,
        seq_len=4,
        dropout=0.0,
    )
    return Trainer(
        VanillaTransformer(config),
        TrainConfig(
            max_steps=4,
            warmup_steps=1,
            eval_steps=1,
            micro_batch_size=1,
            grad_accum_steps=1,
            dtype="float32",
            checkpoint_dir=str(root / "checkpoints"),
        ),
        train_loader=SyntheticLoader(_batches()),
        val_loader=SyntheticLoader(_batches()),
        run_logger=RunLogger(root / name, config={"variant": "recovery_demo"}),
        checkpoint_manager=writer,
        device="cpu",
    )


def _advance(trainer: Trainer, stop_step: int, *, checkpoint_each_step: bool = False) -> None:
    while trainer.step < stop_step:
        trainer._training_step()
        if checkpoint_each_step:
            trainer._save_checkpoint(completed_step=trainer.step + 1)
            if trainer.checkpoint_manager is not None:
                trainer.checkpoint_manager.wait()
        trainer.step += 1


def _fixed_loss(trainer: Trainer) -> float:
    inputs, targets = _batches()[0]
    trainer.model.eval()
    with torch.no_grad():
        _, loss, _ = trainer.model(inputs, targets)
    return float(loss.item())


def run_demo(work_dir: Path) -> dict:
    """Run uninterrupted and corrupted/resumed paths and compare their result."""
    work_dir.mkdir(parents=True, exist_ok=True)

    uninterrupted = _trainer(work_dir, "uninterrupted")
    _advance(uninterrupted, 4)
    uninterrupted_loss = _fixed_loss(uninterrupted)

    checkpoint_dir = work_dir / "checkpoints"
    writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(checkpoint_dir, capacity=3), checkpoint_dir
    )
    interrupted = _trainer(work_dir, "interrupted", writer)
    _advance(interrupted, 2, checkpoint_each_step=True)
    writer.shutdown()

    newest = checkpoint_dir / "checkpoint_step_000002.pt"
    corrupted_bytes = bytearray(newest.read_bytes())
    corrupted_bytes[len(corrupted_bytes) // 2] ^= 0xFF
    newest.write_bytes(corrupted_bytes)
    corrupted_detected = not AtomicCheckpointWriter.verify_trusted(newest)

    recovery_writer = AsyncCheckpointWriter(
        CheckpointRingBuffer(checkpoint_dir, capacity=3), checkpoint_dir
    )
    recovered_checkpoint = recovery_writer.rollback()
    if recovered_checkpoint is None:
        raise RuntimeError("No verified checkpoint survived the corruption test")

    resumed = _trainer(work_dir, "resumed")
    resumed.load_checkpoint(recovered_checkpoint)
    recovered_step = resumed.step
    _advance(resumed, 4)
    resumed_loss = _fixed_loss(resumed)
    recovery_writer.shutdown()

    parameter_max_abs_diff = max(
        float((left - right).abs().max().item())
        for left, right in zip(
            uninterrupted.model.state_dict().values(), resumed.model.state_dict().values()
        )
    )
    return {
        "schema_version": 1,
        "scenario": "corrupt newest checkpoint, roll back, and deterministically replay",
        "corrupted_checkpoint": newest.name,
        "corruption_detected": corrupted_detected,
        "recovered_checkpoint": recovered_checkpoint.name,
        "recovered_step": recovered_step,
        "uninterrupted_final_loss": uninterrupted_loss,
        "resumed_final_loss": resumed_loss,
        "absolute_loss_difference": abs(uninterrupted_loss - resumed_loss),
        "parameter_max_absolute_difference": parameter_max_abs_diff,
        "equivalent": (
            corrupted_detected
            and abs(uninterrupted_loss - resumed_loss) < 1e-8
            and parameter_max_abs_diff < 1e-8
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/transformer_recovery_demo"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["equivalent"]:
        raise SystemExit("Recovery demonstration did not reproduce the uninterrupted result")
    print(f"Recovery demonstration written to: {args.output}")


if __name__ == "__main__":
    main()
