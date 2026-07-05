"""Training script entry point.

Usage:
    python scripts/train.py --data_dir data/wikitext --scale debug
    python scripts/train.py --data_dir data/wikitext --scale main --max_steps 5000
    python scripts/train.py --data_dir data/wikitext --variant modern --scale main
    python scripts/train.py --resume checkpoints/checkpoint_latest.pt

This creates the model, data loader, and trainer, then runs the training loop.
"""

import argparse

import torch

from src.models.registry import build as registry_build, SCALES, VARIANTS
from src.data.dataloader import ShardedDataLoader
from src.training.trainer import Trainer
from src.training.run_config_builder import RunConfigBuilder
from src.training.run_logger import RunLogger, generate_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer variant")

    # Model
    parser.add_argument("--variant", type=str, default="vanilla", choices=VARIANTS.keys(),
                        help="Model variant (vanilla=V0, modern=V1 LLaMA-style)")
    parser.add_argument("--scale", type=str, default="debug", choices=SCALES.keys(),
                        help="Model scale tier (debug/main/stretch)")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"],
                        help="FFN activation function (vanilla only: relu or gelu)")

    # Training
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Training steps (default: 2000 debug, 5000 main/stretch)")
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for ~15-25%% speedup (requires PyTorch 2.0+)")

    # Precision
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])

    # Data
    parser.add_argument("--data_dir", type=str, default="data/processed/wikitext-full",
                        help="Path to directory with binary shard files")

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint dir. Default auto-generates from variant+scale.")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set random seed for reproducibility
    from src.utils.seed import set_seed
    set_seed(args.seed)

    # Build model (registry handles activation override, dtype casting, compile)
    model, model_config = registry_build(
        args.variant, args.scale,
        activation=args.activation if args.variant == "vanilla" else None,
        dtype=args.dtype,
        compile_model=args.compile,
    )

    # Build all config artifacts
    bundle = RunConfigBuilder.from_args(args, model_config, model)

    # Print model info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\nModel: {args.variant} ({args.scale})")
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: {model_config.n_layer}, d_model: {model_config.d_model}, heads: {model_config.n_head}")
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

    trainer = Trainer(
        model, bundle.train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        run_logger=run_logger,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    results = trainer.train()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss:   {results['final_val_loss']:.4f}")
    print(f"  Best val loss:    {results['best_val_loss']:.4f}")
    print(f"  Total tokens:     {results['total_tokens']:,}")
    print(f"  Total time:       {results['total_time']:.1f}s")
    print(f"  Avg tok/s:        {results['total_tokens'] / results['total_time']:,.0f}")


if __name__ == "__main__":
    main()
