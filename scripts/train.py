"""Training script entry point.

Usage:
    python scripts/train.py --data_dir data/wikitext --scale debug
    python scripts/train.py --data_dir data/wikitext --scale main --max_steps 5000
    python scripts/train.py --data_dir data/wikitext --variant modern --scale main
    python scripts/train.py --resume checkpoints/checkpoint_latest.pt

This creates the model, data loader, and trainer, then runs the training loop.
"""

import argparse
from pathlib import Path

import torch

from src.data.dataloader import ShardedDataLoader
from src.models.registry import SCALES, VARIANTS
from src.models.registry import build as registry_build
from src.training.checkpoint import (
    AsyncCheckpointWriter,
    AtomicCheckpointWriter,
    CheckpointRingBuffer,
)
from src.training.health_monitor import HealthMonitor
from src.training.run_config_builder import RunConfigBuilder
from src.training.run_logger import RunLogger, generate_run_dir
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer variant")

    # Model
    parser.add_argument(
        "--variant",
        type=str,
        default="vanilla",
        choices=VARIANTS.keys(),
        help="Model variant (vanilla=V0, modern=V1 LLaMA-style)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="debug",
        choices=SCALES.keys(),
        help="Model scale tier (debug/main/stretch)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="FFN activation function (vanilla only: relu or gelu)",
    )

    # Training
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Training steps (default: 2000 debug, 5000 main/stretch)",
    )
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for ~15-25%% speedup (requires PyTorch 2.0+)",
    )

    # Precision
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/wikitext-full",
        help="Path to directory with binary shard files",
    )

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint dir. Default auto-generates from variant+scale.",
    )
    parser.add_argument(
        "--fault-tolerant",
        action="store_true",
        help="Enable async atomic checkpoints, integrity verification, health checks, and rollback",
    )
    parser.add_argument(
        "--checkpoint-ring-size",
        type=int,
        default=3,
        help="Verified checkpoints retained in fault-tolerant mode (default: 3)",
    )

    # Seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint, or 'latest' for the latest verified checkpoint",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set random seed for reproducibility
    from src.utils.seed import set_seed

    set_seed(args.seed)

    # Build model (registry handles activation override, dtype casting, compile)
    model, model_config = registry_build(
        args.variant,
        args.scale,
        activation=args.activation if args.variant == "vanilla" else None,
        dtype=args.dtype,
        compile_model=args.compile,
    )

    # Build all config artifacts
    bundle = RunConfigBuilder.from_args(args, model_config, model)
    if args.checkpoint_ring_size < 2:
        raise ValueError("--checkpoint-ring-size must be at least 2")

    # Print model info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\nModel: {args.variant} ({args.scale})")
    print(f"  Parameters: {n_params:,}")
    print(
        f"  Layers: {model_config.n_layer}, d_model: {model_config.d_model}, "
        f"heads: {model_config.n_head}"
    )
    print(f"  Seq len: {model_config.seq_len}")
    print(f"  Activation: {bundle.activation_label}")
    print()

    # Create data loaders
    train_loader = ShardedDataLoader(
        data_dir=args.data_dir,
        batch_size=bundle.train_config.micro_batch_size,
        seq_len=model_config.seq_len,
        split="train",
        device=device,
    )
    val_loader = ShardedDataLoader(
        data_dir=args.data_dir,
        batch_size=bundle.train_config.micro_batch_size,
        seq_len=model_config.seq_len,
        split="val",
        device=device,
    )

    # Create logger and trainer
    run_dir = generate_run_dir(
        variant=args.variant,
        scale=args.scale,
        activation=bundle.activation_label,
    )
    run_logger = RunLogger(run_dir, bundle.run_config)
    print(f"  Run dir: {run_dir}")

    checkpoint_manager = None
    health_monitor = None
    if args.fault_tolerant:
        checkpoint_dir = Path(bundle.checkpoint_dir)
        checkpoint_ring = CheckpointRingBuffer(checkpoint_dir, capacity=args.checkpoint_ring_size)
        checkpoint_manager = AsyncCheckpointWriter(checkpoint_ring, checkpoint_dir)
        health_monitor = HealthMonitor()
        print(f"  Fault tolerance: enabled (verified checkpoint ring={args.checkpoint_ring_size})")

    trainer = Trainer(
        model,
        bundle.train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        run_logger=run_logger,
        device=device,
        checkpoint_manager=checkpoint_manager,
        health_monitor=health_monitor,
    )

    try:
        if args.resume:
            resume_path: str | Path = args.resume
            if args.resume == "latest":
                if checkpoint_manager is None:
                    raise ValueError("--resume latest requires --fault-tolerant")
                verified = checkpoint_manager.rollback()
                if verified is None:
                    raise FileNotFoundError(
                        f"No verified checkpoint is available in {bundle.checkpoint_dir}"
                    )
                resume_path = verified
            elif args.fault_tolerant and not AtomicCheckpointWriter.verify_trusted(
                Path(resume_path)
            ):
                raise ValueError(f"Checkpoint integrity verification failed: {resume_path}")
            trainer.load_checkpoint(resume_path)

        results = trainer.train()
    finally:
        if checkpoint_manager is not None:
            checkpoint_manager.shutdown()

    # Print summary
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss:   {results['final_val_loss']:.4f}")
    print(f"  Best val loss:    {results['best_val_loss']:.4f}")
    print(f"  Total tokens:     {results['total_tokens']:,}")
    print(f"  Total time:       {results['total_time']:.1f}s")
    print(f"  Avg tok/s:        {results['total_tokens'] / results['total_time']:,.0f}")


if __name__ == "__main__":
    main()
