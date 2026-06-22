"""Training loop for the Transformer model.

This module implements the core training logic:
- Mixed precision (bf16) for speed and memory efficiency
- Gradient accumulation to simulate larger batch sizes
- Gradient clipping to prevent exploding gradients
- Periodic evaluation on validation data
- Checkpointing (save/resume)
- Logging of loss, learning rate, throughput

The training loop follows the standard recipe:
    for each step:
        1. Get batch from DataLoader
        2. Forward pass (in bf16 for speed)
        3. Compute loss
        4. Backward pass (accumulate gradients)
        5. Every N steps: clip gradients, optimizer step, zero gradients
        6. Log metrics
        7. Periodically evaluate and checkpoint
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataloader import ShardedDataLoader
from src.training.scheduler import get_lr


@dataclass
class TrainConfig:
    """Training hyperparameters.

    These match project_defaults.yaml but are kept in code for type safety.
    """

    # Optimization
    max_lr: float = 3e-4
    min_lr: float = 3e-5          # 10% of max_lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Batching
    micro_batch_size: int = 8
    grad_accum_steps: int = 8     # effective batch = micro * accum * seq_len tokens

    # Schedule
    warmup_steps: int = 100
    max_steps: int = 1000         # total training steps (override per experiment)

    # Precision
    dtype: str = "bfloat16"       # "bfloat16", "float16", or "float32"

    # Logging & eval
    log_interval: int = 10        # log every N steps
    eval_interval: int = 100      # evaluate every N steps
    eval_steps: int = 20          # number of eval batches per evaluation

    # Checkpointing
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"

    # Data
    data_dir: str = "data"
    seq_len: int = 512


class Trainer:
    """Handles the training loop, evaluation, and checkpointing.

    Args:
        model: The Transformer model to train.
        train_config: TrainConfig with all hyperparameters.
        device: Device to train on ("cuda" or "cpu").
    """

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.config = train_config
        self.device = device

        # Set up precision context
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[train_config.dtype]
        self.use_amp = train_config.dtype != "float32"

        # Set up optimizer (AdamW with weight decay only on 2D params)
        self.optimizer = self._create_optimizer()

        # Set up data loaders
        self.train_loader = ShardedDataLoader(
            data_dir=train_config.data_dir,
            batch_size=train_config.micro_batch_size,
            seq_len=train_config.seq_len,
            split="train",
            device=device,
        )
        self.val_loader = ShardedDataLoader(
            data_dir=train_config.data_dir,
            batch_size=train_config.micro_batch_size,
            seq_len=train_config.seq_len,
            split="val",
            device=device,
        )

        # Training state
        self.step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with proper weight decay grouping.

        Weight decay is only applied to 2D parameters (weight matrices).
        1D parameters (biases, LayerNorm weights) are NOT decayed.

        Why? Weight decay is a regularizer that pushes weights toward zero.
        For biases and norms, this would hurt performance — they need to be
        whatever value works best, not pushed toward zero.
        """
        # Separate parameters into "decay" and "no_decay" groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 2D params (linear weights, embeddings) get weight decay
            # 1D params (biases, LayerNorm) do not
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
        )
        return optimizer

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dictionary with final metrics (train_loss, val_loss, tokens_processed, etc.)
        """
        self.model.train()
        log_history = []
        t_start = time.time()

        print(f"Starting training for {self.config.max_steps} steps")
        print(f"  Micro batch size: {self.config.micro_batch_size}")
        print(f"  Grad accumulation: {self.config.grad_accum_steps}")
        print(f"  Effective batch tokens: {self.config.micro_batch_size * self.config.grad_accum_steps * self.config.seq_len:,}")
        print(f"  Precision: {self.config.dtype}")
        print(f"  Device: {self.device}")
        print()

        while self.step < self.config.max_steps:
            step_metrics = self._training_step()

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = self.tokens_processed / elapsed if elapsed > 0 else 0
                print(
                    f"step {self.step:>6d} | "
                    f"loss {step_metrics['loss']:.4f} | "
                    f"lr {step_metrics['lr']:.2e} | "
                    f"tok/s {tokens_per_sec:,.0f} | "
                    f"tokens {self.tokens_processed:,}"
                )
                entry = {
                    "step": self.step,
                    "loss": step_metrics["loss"],
                    "lr": step_metrics["lr"],
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_processed": self.tokens_processed,
                }
                log_history.append(entry)

                # Append to live log file (JSONL format — one JSON object per line)
                log_dir = Path(self.config.checkpoint_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / "train_log.jsonl", "a") as f:
                    f.write(json.dumps(entry) + "\n")
                # Write running log so progress is visible during training
                self._save_running_log(log_history)

            # Evaluation
            if self.step % self.config.eval_interval == 0 and self.step > 0:
                val_loss = self._evaluate()
                print(f"  → val_loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0 and self.step > 0:
                self._save_checkpoint()

            self.step += 1

        # Final evaluation and checkpoint
        val_loss = self._evaluate()
        print(f"\nTraining complete. Final val_loss: {val_loss:.4f}")
        self._save_checkpoint()

        results = {
            "final_train_loss": step_metrics["loss"],
            "final_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "total_tokens": self.tokens_processed,
            "total_time": time.time() - t_start,
            "log_history": log_history,
        }

        # Save training log to checkpoint directory
        self._save_log(results)

        return results

    def _training_step(self) -> dict:
        """Execute one optimizer step (possibly with gradient accumulation).

        With gradient accumulation, we do multiple forward/backward passes
        and accumulate gradients before doing one optimizer step. This
        simulates a larger batch size without needing more GPU memory.

        Returns:
            Dict with 'loss' and 'lr' for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Update learning rate for this step
        lr = get_lr(
            self.step,
            max_lr=self.config.max_lr,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.max_steps,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Accumulate gradients over multiple micro-batches
        total_loss = 0.0
        for micro_step in range(self.config.grad_accum_steps):
            x, y = self.train_loader.next_batch()

            # Mixed precision forward pass
            with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self.use_amp):
                logits, loss, _ = self.model(x, y)

            # Scale loss by accumulation steps (so gradients average correctly)
            scaled_loss = loss / self.config.grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.item()
            self.tokens_processed += x.numel()

        # Gradient clipping (prevents exploding gradients)
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        # Optimizer step (update weights)
        self.optimizer.step()

        avg_loss = total_loss / self.config.grad_accum_steps
        return {"loss": avg_loss, "lr": lr}

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run evaluation on the validation set.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        for _ in range(self.config.eval_steps):
            x, y = self.val_loader.next_batch()
            with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self.use_amp):
                _, loss, _ = self.model(x, y)
            total_loss += loss.item()

        self.model.train()
        return total_loss / self.config.eval_steps

    def _save_checkpoint(self) -> None:
        """Save model, optimizer, and training state."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        # Save as step-numbered file
        path = ckpt_dir / f"checkpoint_step_{self.step:06d}.pt"
        torch.save(checkpoint, path)

        # Also save a "latest" symlink/copy for easy resume
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"  → Saved checkpoint at step {self.step}")

    def _save_log(self, results: dict) -> None:
        """Save training log as JSON for later analysis.

        Writes to checkpoint_dir/train_log.json with all metrics from the run.
        """
        log_dir = Path(self.config.checkpoint_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "train_log.json"
        log_data = {
            "final_train_loss": results["final_train_loss"],
            "final_val_loss": results["final_val_loss"],
            "best_val_loss": results["best_val_loss"],
            "total_tokens": results["total_tokens"],
            "total_time_seconds": results["total_time"],
            "steps": results["log_history"],
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"  → Saved training log to {log_path}")

    def _save_running_log(self, log_history: list) -> None:
        """Write a running log file updated every log_interval steps.

        This way you can monitor progress even while training is ongoing.
        Overwrites the file each time (keeps the full history).
        """
        log_dir = Path(self.config.checkpoint_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "running_log.json"
        log_data = {
            "status": "running",
            "current_step": self.step,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "steps": log_history,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.tokens_processed = checkpoint["tokens_processed"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed from step {self.step} ({self.tokens_processed:,} tokens)")
