"""Run configuration builder.

Encapsulates all training configuration assembly that was previously scattered
across scripts/train.py: variant label formatting, default steps resolution,
checkpoint directory generation, TrainConfig construction, and run_config dict
assembly.

Usage:
    bundle = RunConfigBuilder.from_args(args, model_config, model)
    # bundle.train_config — TrainConfig for Trainer
    # bundle.run_config   — dict for RunLogger serialization
    # bundle.checkpoint_dir — resolved checkpoint path
    # bundle.variant_label — formatted label (e.g., "vanilla_gelu", "modern")
    # bundle.activation_label — activation name for logging
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.registry import VARIANTS
from src.training.trainer import TrainConfig


@dataclass
class RunConfigBundle:
    """All configuration artifacts produced from CLI args + model config."""

    train_config: TrainConfig
    run_config: dict
    checkpoint_dir: str
    variant_label: str
    activation_label: str


class RunConfigBuilder:
    """Assembles training configuration from CLI args and model config.

    Extracts all policy decisions about config assembly out of the train script:
    - Default steps resolution from VariantSpec
    - Variant label formatting (vanilla special case)
    - Activation label determination
    - Checkpoint directory generation
    - TrainConfig construction
    - run_config dict assembly with hardware detection
    """

    @staticmethod
    def from_args(
        args: argparse.Namespace,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> RunConfigBundle:
        """Build all config artifacts from parsed CLI args and model config.

        Args:
            args: Parsed argparse namespace with all CLI flags.
            model_config: ModelConfig returned by registry.build().
            model: The constructed model (needed for parameter counting).

        Returns:
            RunConfigBundle with train_config, run_config, checkpoint_dir,
            variant_label, and activation_label.
        """
        spec = VARIANTS[args.variant]

        # Resolve max_steps: use explicit value or fall back to variant's default
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else spec.default_steps[args.scale]
        )

        # Format labels
        activation_label = args.activation if args.variant == "vanilla" else "swiglu"
        variant_label = (
            f"vanilla_{args.activation}" if args.variant == "vanilla" else args.variant
        )

        # Resolve checkpoint directory
        checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else f"checkpoints/{variant_label}_{args.scale}"
        )

        # Build TrainConfig
        train_config = TrainConfig(
            max_lr=args.max_lr,
            min_lr=args.max_lr * 0.1,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            dtype=args.dtype,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
        )

        # Detect hardware
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name()
            gpu_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
        else:
            gpu_name = "cpu"
            gpu_memory_gb = 0

        # Count model parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Assemble run_config dict
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
                "max_steps": max_steps,
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
                "gpu": gpu_name,
                "gpu_memory_gb": gpu_memory_gb,
            },
            "resumed_from": args.resume,
        }

        return RunConfigBundle(
            train_config=train_config,
            run_config=run_config,
            checkpoint_dir=checkpoint_dir,
            variant_label=variant_label,
            activation_label=activation_label,
        )
