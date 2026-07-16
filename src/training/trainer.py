"""Training loop for the Transformer model.

This module implements the core training logic:
- Mixed precision (bf16) for speed and memory efficiency
- Gradient accumulation to simulate larger batch sizes
- Gradient clipping to prevent exploding gradients
- Periodic evaluation on validation data
- Checkpointing (save/resume)
- Logging delegated to RunLogger

The training loop follows the standard recipe:
    for each step:
        1. Get batch from DataLoader
        2. Forward pass (in bf16 for speed)
        3. Compute loss
        4. Backward pass (accumulate gradients)
        5. Every N steps: clip gradients, optimizer step, zero gradients
        6. Log metrics via RunLogger
        7. Periodically evaluate and checkpoint
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.training.protocols import DataLoader
from src.training.run_logger import RunLogger
from src.training.scheduler import get_lr

if TYPE_CHECKING:
    from src.training.checkpoint import AsyncCheckpointWriter
    from src.training.health_monitor import HealthMonitor


@dataclass
class TrainConfig:
    """Training hyperparameters.

    These match project_defaults.yaml but are kept in code for type safety.
    """

    # Optimization
    max_lr: float = 3e-4
    min_lr: float = 3e-5  # 10% of max_lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Batching
    micro_batch_size: int = 8
    grad_accum_steps: int = 8  # effective batch = micro * accum * seq_len tokens

    # Schedule
    warmup_steps: int = 100
    max_steps: int = 1000  # total training steps (override per experiment)

    # Precision
    dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"

    # Logging & eval
    log_interval: int = 10  # log every N steps
    eval_interval: int = 100  # evaluate every N steps
    eval_steps: int = 20  # number of eval batches per evaluation

    # Checkpointing
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"


class Trainer:
    """Handles the training loop, evaluation, and checkpointing.

    Args:
        model: The Transformer model to train.
        train_config: TrainConfig with all hyperparameters.
        train_loader: A DataLoader providing training batches via next_batch().
        val_loader: A DataLoader providing validation batches via next_batch().
        device: Device to train on ("cuda" or "cpu").
    """

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_logger: RunLogger,
        device: str = "cuda",
        checkpoint_manager: AsyncCheckpointWriter | None = None,
        health_monitor: HealthMonitor | None = None,
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

        # Store injected data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Run logger (required — all logging delegated here)
        self.run_logger = run_logger

        # Fault tolerance (optional)
        self.checkpoint_manager = checkpoint_manager
        self.health_monitor = health_monitor

        # Training state
        self.step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")
        self._skipped_steps = 0

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
        t_start = time.time()

        print(f"Starting training for {self.config.max_steps} steps")
        print(f"  Micro batch size: {self.config.micro_batch_size}")
        print(f"  Grad accumulation: {self.config.grad_accum_steps}")
        # Note: seq_len is determined by the data loader, not TrainConfig
        effective_batch_size = self.config.micro_batch_size * self.config.grad_accum_steps
        print(f"  Effective batch size: {effective_batch_size}")
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
                    f"grad_norm {step_metrics['grad_norm']:.2f} | "
                    f"tok/s {tokens_per_sec:,.0f} | "
                    f"tokens {self.tokens_processed:,}"
                )

                # Delegate all structured logging to RunLogger
                self.run_logger.log_step(
                    step=self.step,
                    train_loss=step_metrics["loss"],
                    lr=step_metrics["lr"],
                    grad_norm=step_metrics["grad_norm"],
                    tokens_per_sec=tokens_per_sec,
                    tokens_processed=self.tokens_processed,
                )

            # Evaluation
            if self.step % self.config.eval_interval == 0 and self.step > 0:
                eval_start = time.time()
                val_loss = self._evaluate()
                eval_time = time.time() - eval_start
                print(f"  → val_loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                self.run_logger.log_eval(self.step, val_loss, eval_time)

            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0 and self.step > 0:
                self._save_checkpoint(completed_step=self.step + 1)

            self.step += 1

        # Final evaluation and checkpoint
        eval_start = time.time()
        val_loss = self._evaluate()
        eval_time = time.time() - eval_start
        print(f"\nTraining complete. Final val_loss: {val_loss:.4f}")
        self._save_checkpoint()
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.wait()

        total_time = time.time() - t_start
        results = {
            "final_train_loss": step_metrics["loss"],
            "final_val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "total_tokens": self.tokens_processed,
            "total_time": total_time,
            "avg_tokens_per_sec": self.tokens_processed / total_time if total_time > 0 else 0,
        }

        # Delegate summary writing to RunLogger
        self.run_logger.log_eval(self.step, val_loss, eval_time)
        self.run_logger.log_summary(results)

        return results

    def _training_step(self) -> dict:
        """Execute one optimizer step (possibly with gradient accumulation).

        With gradient accumulation, we do multiple forward/backward passes
        and accumulate gradients before doing one optimizer step. This
        simulates a larger batch size without needing more GPU memory.

        Returns:
            Dict with 'loss', 'lr', and 'grad_norm' for this step.
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
        total_aux_loss = 0.0
        for micro_step in range(self.config.grad_accum_steps):
            x, y = self.train_loader.next_batch()

            # Mixed precision forward pass
            with torch.autocast(device_type=self.device, dtype=self.dtype, enabled=self.use_amp):
                logits, loss, _ = self.model(x, y)

            # Get aux loss (zero for dense models, non-zero for MoE)
            if hasattr(self.model, "get_aux_loss"):
                aux_loss = self.model.get_aux_loss()
            else:
                aux_loss = torch.tensor(0.0, device=self.device)

            # Combine cross-entropy loss with auxiliary loss
            combined_loss = loss + aux_loss

            # Scale loss by accumulation steps (so gradients average correctly)
            scaled_loss = combined_loss / self.config.grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.item()
            total_aux_loss += aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            self.tokens_processed += x.numel()

        avg_loss = total_loss / self.config.grad_accum_steps
        avg_aux = total_aux_loss / self.config.grad_accum_steps

        # clip_grad_norm_ returns the pre-clip norm. Reject non-finite norms
        # before it can multiply otherwise finite gradients by a NaN coefficient.
        max_grad_norm = self.config.grad_clip if self.config.grad_clip > 0 else float("inf")
        clip_error = None
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
                error_if_nonfinite=True,
            ).item()
        except RuntimeError as error:
            grad_norm = float("nan")
            clip_error = error

        metrics = {
            "loss": avg_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "aux_loss": avg_aux,
        }
        nonfinite_reason = None
        if clip_error is not None:
            nonfinite_reason = f"Non-finite gradient norm at step {self.step}"
        elif not math.isfinite(avg_loss):
            nonfinite_reason = f"Non-finite loss at step {self.step}"

        if nonfinite_reason is not None:
            self.optimizer.zero_grad(set_to_none=True)
            if self.health_monitor is not None:
                from src.training.health_monitor import Action

                action = self.health_monitor.check(self.step, avg_loss, grad_norm)
                if action == Action.ROLLBACK:
                    self._rollback_to_latest_verified()
                    return metrics
            raise FloatingPointError(f"{nonfinite_reason}; optimizer step aborted") from clip_error

        # Health check uses the same unclipped norm that is reported in logs.
        if self.health_monitor is not None:
            from src.training.health_monitor import Action

            action = self.health_monitor.check(self.step, avg_loss, grad_norm)

            if action == Action.SKIP_STEP:
                self.optimizer.zero_grad(set_to_none=True)
                self._skipped_steps += 1
                return metrics

            if action == Action.ROLLBACK:
                self.optimizer.zero_grad(set_to_none=True)
                self._rollback_to_latest_verified()
                return metrics

        self.optimizer.step()

        return metrics

    def _rollback_to_latest_verified(self) -> None:
        """Restore the newest trusted checkpoint or fail without advancing."""
        if self.checkpoint_manager is None:
            raise RuntimeError("Rollback requested but no verified checkpoint manager is available")
        self.checkpoint_manager.wait()
        rollback_path = self.checkpoint_manager.rollback()
        if rollback_path is None:
            raise RuntimeError("Rollback requested but no verified checkpoint is available")
        self.load_checkpoint(rollback_path)
        if self.health_monitor is not None:
            self.health_monitor.reset()

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

    def _save_checkpoint(self, completed_step: int | None = None) -> None:
        """Save model, optimizer, loader cursors, RNG, and training progress.

        completed_step is the next scheduler/loop step after weights already
        updated by a periodic save. Final saves use the current loop step.
        """
        from src.utils.seed import get_rng_state

        checkpoint_step = self.step if completed_step is None else completed_step

        def loader_state(loader):
            state_dict = getattr(loader, "state_dict", None)
            return state_dict() if callable(state_dict) else None

        training_state = {
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "skipped_steps": self._skipped_steps,
            "rng_state": get_rng_state(),
            "train_loader_state": loader_state(self.train_loader),
            "val_loader_state": loader_state(self.val_loader),
        }

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save(
                step=checkpoint_step,
                model=self.model,
                optimizer=self.optimizer,
                training_state=training_state,
            )
            print(f"  → Queued async checkpoint at step {checkpoint_step}")
            return

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Strip _orig_mod. prefix from compiled models so checkpoints are portable
        model_state = self.model.state_dict()
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}

        checkpoint = {
            "step": checkpoint_step,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
            "skipped_steps": self._skipped_steps,
            "rng_state": training_state["rng_state"],
            "train_loader_state": training_state["train_loader_state"],
            "val_loader_state": training_state["val_loader_state"],
            "model_state_dict": cleaned_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        # Save as step-numbered file
        path = ckpt_dir / f"checkpoint_step_{checkpoint_step:06d}.pt"
        torch.save(checkpoint, path)
        # Also save a "latest" symlink/copy for easy resume
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"  → Saved checkpoint at step {checkpoint_step}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint.

        Handles checkpoints saved from both compiled and non-compiled models
        by stripping the '_orig_mod.' prefix that torch.compile adds.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle torch.compile prefix mismatch
        # Checkpoints are saved with clean keys (no _orig_mod. prefix).
        # If loading into a compiled model, its state_dict expects "_orig_mod." prefix.
        model_state = checkpoint["model_state_dict"]

        # Check if the current model expects the _orig_mod. prefix
        current_keys = set(self.model.state_dict().keys())
        needs_prefix = any(k.startswith("_orig_mod.") for k in current_keys)

        cleaned_state = {}
        for k, v in model_state.items():
            # Strip any existing prefix first (normalize)
            clean_key = k.replace("_orig_mod.", "")
            # Add prefix if the compiled model expects it
            if needs_prefix:
                cleaned_state[f"_orig_mod.{clean_key}"] = v
            else:
                cleaned_state[clean_key] = v

        self.model.load_state_dict(cleaned_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        training_state = checkpoint.get("training_state", checkpoint)
        self.step = checkpoint["step"]
        self.tokens_processed = training_state["tokens_processed"]
        self.best_val_loss = training_state["best_val_loss"]
        self._skipped_steps = training_state.get("skipped_steps", 0)
        for loader, state_key in (
            (self.train_loader, "train_loader_state"),
            (self.val_loader, "val_loader_state"),
        ):
            loader_state = training_state.get(state_key)
            if loader_state is None:
                continue
            load_state_dict = getattr(loader, "load_state_dict", None)
            if not callable(load_state_dict):
                raise RuntimeError(
                    f"Checkpoint contains {state_key}, but loader cannot restore state"
                )
            load_state_dict(loader_state)
        if rng_state := training_state.get("rng_state"):
            from src.utils.seed import set_rng_state

            set_rng_state(rng_state)

        print(f"Resumed from step {self.step} ({self.tokens_processed:,} tokens)")
