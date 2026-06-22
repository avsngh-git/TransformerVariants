"""Training script entry point.

Usage:
    python scripts/train.py --data_dir data/wikitext --scale debug
    python scripts/train.py --data_dir data/wikitext --scale main --max_steps 5000
    python scripts/train.py --resume checkpoints/checkpoint_latest.pt

This creates the model, data loader, and trainer, then runs the training loop.
"""

import argparse

import torch

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.training.trainer import Trainer, TrainConfig


# Pre-defined model scales (matching configs/model/vanilla.yaml)
MODEL_SCALES = {
    "debug": {"n_layer": 4, "d_model": 256, "n_head": 4, "seq_len": 512},
    "main": {"n_layer": 8, "d_model": 512, "n_head": 8, "seq_len": 1024},
    "stretch": {"n_layer": 12, "d_model": 768, "n_head": 12, "seq_len": 1024},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vanilla Transformer")

    # Model
    parser.add_argument("--scale", type=str, default="debug", choices=MODEL_SCALES.keys(),
                        help="Model scale tier (debug/main/stretch)")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"],
                        help="FFN activation function (relu=vanilla, gelu=GPT-2)")

    # Training
    parser.add_argument("--max_steps", type=int, default=1000)
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
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with binary shard files")

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint dir. Default auto-generates from variant+scale.")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model config from scale
    scale_params = MODEL_SCALES[args.scale]
    model_config = ModelConfig(
        n_layer=scale_params["n_layer"],
        d_model=scale_params["d_model"],
        n_head=scale_params["n_head"],
        seq_len=scale_params["seq_len"],
        vocab_size=50257,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        activation=args.activation,
    )

    # Create model
    model = VanillaTransformer(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: vanilla V0 ({args.scale})")
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: {model_config.n_layer}, d_model: {model_config.d_model}, heads: {model_config.n_head}")
    print(f"  Seq len: {model_config.seq_len}")
    print(f"  Activation: {model_config.activation}")

    # torch.compile — fuses operations, reduces kernel launches, 15-25% faster
    if args.compile:
        print(f"  Compiling model with torch.compile...")
        model = torch.compile(model)
        print(f"  Compiled!")

    print()

    # Auto-generate checkpoint directory if not specified
    # Format: checkpoints/vanilla_{activation}_{scale}/
    if args.checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/vanilla_{args.activation}_{args.scale}"
    else:
        checkpoint_dir = args.checkpoint_dir

    # Create training config
    train_config = TrainConfig(
        max_lr=args.max_lr,
        min_lr=args.max_lr * 0.1,  # standard: min_lr = 10% of max
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        data_dir=args.data_dir,
        seq_len=model_config.seq_len,
    )

    # Create trainer
    trainer = Trainer(model, train_config, device=device)

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
