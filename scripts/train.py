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

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer
from src.training.trainer import Trainer, TrainConfig
from src.training.run_logger import RunLogger, generate_run_dir


# Pre-defined model scales (matching configs/model/vanilla.yaml)
MODEL_SCALES = {
    "debug": {"n_layer": 4, "d_model": 256, "n_head": 4, "seq_len": 512, "default_steps": 2000},
    "main": {"n_layer": 8, "d_model": 512, "n_head": 8, "seq_len": 1024, "default_steps": 5000},
    "stretch": {"n_layer": 12, "d_model": 768, "n_head": 12, "seq_len": 1024, "default_steps": 5000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer variant")

    # Model
    parser.add_argument("--variant", type=str, default="vanilla", choices=["vanilla", "modern"],
                        help="Model variant (vanilla=V0, modern=V1 LLaMA-style)")
    parser.add_argument("--scale", type=str, default="debug", choices=MODEL_SCALES.keys(),
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

    # Apply default max_steps from scale if not explicitly set
    if args.max_steps is None:
        args.max_steps = scale_params["default_steps"]

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

    # Create model based on variant
    if args.variant == "vanilla":
        model = VanillaTransformer(model_config)
        variant_label = f"vanilla_{args.activation}"
    elif args.variant == "modern":
        model = ModernTransformer(model_config)
        variant_label = "modern"
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.variant} ({args.scale})")
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: {model_config.n_layer}, d_model: {model_config.d_model}, heads: {model_config.n_head}")
    print(f"  Seq len: {model_config.seq_len}")
    if args.variant == "vanilla":
        print(f"  Activation: {model_config.activation}")
    else:
        print(f"  Components: RoPE, RMSNorm, SwiGLU, Flash Attention")

    # torch.compile — fuses operations, reduces kernel launches, 15-25% faster
    if args.compile:
        print(f"  Compiling model with torch.compile...")
        model = torch.compile(model)
        print(f"  Compiled!")

    print()

    # Auto-generate checkpoint directory if not specified
    # Format: checkpoints/{variant_label}_{scale}/
    if args.checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{variant_label}_{args.scale}"
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

    # Set up structured run logging
    activation_label = args.activation if args.variant == "vanilla" else "swiglu"
    run_dir = generate_run_dir(
        variant=args.variant,
        scale=args.scale,
        activation=activation_label,
    )

    run_config = {
        "variant": args.variant,
        "scale": args.scale,
        "model": {
            "n_layer": model_config.n_layer,
            "d_model": model_config.d_model,
            "n_head": model_config.n_head,
            "seq_len": model_config.seq_len,
            "vocab_size": model_config.vocab_size,
            "activation": activation_label,
            "bias": model_config.bias,
            "dropout": model_config.dropout,
            "tie_embeddings": model_config.tie_embeddings,
            "total_params": n_params,
        },
        "training": {
            "max_steps": args.max_steps,
            "max_lr": args.max_lr,
            "min_lr": args.max_lr * 0.1,
            "warmup_steps": args.warmup_steps,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "grad_clip": args.grad_clip,
            "dtype": args.dtype,
            "compiled": args.compile,
        },
        "data": {
            "data_dir": args.data_dir,
            "seq_len": model_config.seq_len,
        },
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device == "cuda" else 0,
        },
        "resumed_from": args.resume,
    }

    run_logger = RunLogger(run_dir, run_config)
    trainer.set_run_logger(run_logger)
    print(f"  Run dir: {run_dir}")

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
