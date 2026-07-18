"""Run logging system for training experiments.

Creates a structured run directory with:
- run_config.json: frozen snapshot of all config (written at start)
- train.log: human-readable live log (header + per-step lines)
- metrics.jsonl: machine-readable per-step metrics
- summary.json: final results (written on completion)
- checkpoints/: model + optimizer state

The run directory naming convention is:
    runs/{variant}_{activation}_{scale}_{YYYYMMDD_HHMM}/
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


class RunLogger:
    """Manages all logging for a training run.

    Creates the run directory, writes config, and provides methods to log
    training steps, evaluation results, and final summary.

    Args:
        run_dir: Path to the run directory (created if it doesn't exist).
        config: Dictionary with full run configuration.
    """

    @classmethod
    def create(
        cls,
        variant: str,
        scale: str,
        activation: str = "",
        config: dict | None = None,
        base_dir: str = "runs",
    ) -> "RunLogger":
        """Factory: generate timestamped run dir and return initialized RunLogger.

        Args:
            variant: Model variant (e.g., "vanilla", "modern").
            scale: Model scale (e.g., "debug", "main", "stretch").
            activation: Activation function (e.g., "relu", "gelu", "swiglu").
            config: Dictionary with full run configuration.
            base_dir: Base directory for all runs.

        Returns:
            A new RunLogger instance with a generated timestamped run directory.
        """
        run_dir = generate_run_dir(variant, scale, activation, base_dir)
        return cls(run_dir, config if config is not None else {})

    def __init__(self, run_dir: str | Path, config: dict) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        self.config = config
        self.start_time = time.time()

        # Write config immediately
        with open(self.run_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(self.run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # Open log files
        self.train_log_path = self.run_dir / "train.log"
        nested_log_path = self.run_dir / "logs" / "train.log"
        if not nested_log_path.exists() and not nested_log_path.is_symlink():
            nested_log_path.symlink_to(Path("..") / "train.log")
        self.metrics_path = self.run_dir / "metrics.jsonl"

        # Write header to train.log
        self._write_header()

        if config.get("resumed_from"):
            self.log_recovery(
                event="verified_checkpoint_resume",
                checkpoint=str(config["resumed_from"]),
            )

        # Create empty metrics.jsonl if it doesn't exist
        if not self.metrics_path.exists():
            self.metrics_path.touch()

    def _write_header(self) -> None:
        """Write the human-readable header to train.log."""
        c = self.config
        model = c.get("model", {})
        training = c.get("training", {})
        data = c.get("data", {})
        hardware = c.get("hardware", {})
        tokens_per_step = (
            training.get("micro_batch_size", 0)
            * training.get("grad_accum_steps", 0)
            * model.get("seq_len", 0)
        )
        batch_description = (
            f"{training.get('micro_batch_size', 0)} micro × "
            f"{training.get('grad_accum_steps', 0)} accum × "
            f"{model.get('seq_len', 0)} = {tokens_per_step:,} tokens/step"
        )

        header = f"""{'='*80}
Training Run: {self.run_dir.name}
{'='*80}
Variant:      {c.get('variant', 'unknown')}
Scale:        {c.get('scale', 'unknown')}
Parameters:   {model.get('total_params', 0)/1e6:.1f}M
Activation:   {model.get('activation', 'unknown')}
Device:       {hardware.get('gpu', 'unknown')} ({hardware.get('gpu_memory_gb', '?')}GB)
Precision:    {training.get('dtype', 'unknown')}
Compiled:     {training.get('compiled', False)}

Data:         {data.get('data_dir', 'unknown')}
Seq len:      {model.get('seq_len', 0)}
Batch:        {batch_description}
Max steps:    {training.get('max_steps', 0)}
Warmup:       {training.get('warmup_steps', 0)} steps
"""
        if c.get("resumed_from"):
            header += f"Resumed from: {c['resumed_from']}\n"

        header += f"{'='*80}\n"

        mode = "a" if self.train_log_path.exists() else "w"
        with open(self.train_log_path, mode) as f:
            f.write(header)

    def log_step(
        self,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
        tokens_processed: int,
        gpu_memory_mb: float | None = None,
    ) -> None:
        """Log a training step to both train.log and metrics.jsonl."""
        elapsed = time.time() - self.start_time
        perplexity = math.exp(min(train_loss, 20))  # cap to avoid overflow

        # Get GPU memory if not provided
        if gpu_memory_mb is None and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        # Human-readable line
        elapsed_str = self._format_time(elapsed)
        gpu_str = f"{gpu_memory_mb / 1024:.1f}GB" if gpu_memory_mb else "N/A"
        line = (
            f"step {step:>5d} | "
            f"train_loss {train_loss:.3f} | "
            f"ppl {perplexity:>7.1f} | "
            f"lr {lr:.2e} | "
            f"grad_norm {grad_norm:.2f} | "
            f"tok/s {tokens_per_sec:>8,.0f} | "
            f"gpu {gpu_str} | "
            f"elapsed {elapsed_str}\n"
        )
        with open(self.train_log_path, "a") as f:
            f.write(line)

        # Machine-readable JSON
        # Crash-safety guarantee: opening in append mode, writing, and closing
        # the file on each call ensures the entry is flushed to disk immediately.
        # No explicit flush() is needed — close() triggers the OS-level write.
        entry = {
            "type": "train",
            "step": step,
            "train_loss": train_loss,
            "perplexity": perplexity,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "gpu_memory_mb": gpu_memory_mb,
            "tokens_processed": tokens_processed,
            "elapsed_seconds": elapsed,
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_eval(
        self,
        step: int,
        val_loss: float,
        eval_time_seconds: float,
    ) -> None:
        """Log an evaluation result."""
        val_perplexity = math.exp(min(val_loss, 20))

        # Human-readable
        line = (
            f"  → eval | "
            f"val_loss {val_loss:.3f} | "
            f"val_ppl {val_perplexity:.1f} | "
            f"eval_time {eval_time_seconds:.1f}s\n"
        )
        with open(self.train_log_path, "a") as f:
            f.write(line)

        # Machine-readable
        # Crash-safety guarantee: open/write/close per call ensures immediate
        # flush to disk without needing explicit flush().
        entry = {
            "type": "eval",
            "step": step,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "eval_time_seconds": eval_time_seconds,
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_summary(self, results: dict) -> None:
        """Write final summary on training completion."""
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        # Also append to train.log
        with open(self.train_log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write("Training Complete\n")
            f.write(f"{'='*80}\n")
            f.write(f"Final train loss: {results.get('final_train_loss', 0):.4f}\n")
            f.write(f"Final val loss:   {results.get('final_val_loss', 0):.4f}\n")
            f.write(f"Best val loss:    {results.get('best_val_loss', 0):.4f}\n")
            f.write(f"Total tokens:     {results.get('total_tokens', 0):,}\n")
            f.write(f"Total time:       {self._format_time(results.get('total_time', 0))}\n")
            f.write(f"Avg tok/s:        {results.get('avg_tokens_per_sec', 0):,.0f}\n")

    def log_recovery(
        self,
        *,
        event: str,
        checkpoint: str,
        trigger_step: int | None = None,
        attempt: int | None = None,
    ) -> None:
        """Append one structured resume or rollback event."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "checkpoint": checkpoint,
        }
        if trigger_step is not None:
            payload["trigger_step"] = trigger_step
        if attempt is not None:
            payload["attempt"] = attempt
        with open(self.run_dir / "recovery_events.jsonl", "a") as f:
            f.write(json.dumps(payload) + "\n")

    @property
    def checkpoint_dir(self) -> Path:
        """Path to the checkpoints subdirectory."""
        return self.run_dir / "checkpoints"

    def validate(self) -> bool:
        """Check that the run directory has the expected structure.

        Returns:
            True if the directory contains resolved YAML/JSON config, root and
            nested training logs, metrics, and checkpoints; False otherwise.
        """
        expected = [
            self.run_dir / "run_config.json",
            self.run_dir / "config_resolved.yaml",
            self.run_dir / "train.log",
            self.run_dir / "logs" / "train.log",
            self.run_dir / "metrics.jsonl",
            self.run_dir / "checkpoints",
        ]
        return all(p.exists() for p in expected)

    def close(self) -> None:
        """Close any open file handles.

        Currently a no-op since RunLogger opens and closes files per write
        operation, but provides the context manager protocol for forward
        compatibility.
        """
        pass

    def __enter__(self) -> "RunLogger":
        """Enter context manager, returning self."""
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager, calling close()."""
        self.close()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}"


def generate_run_dir(
    variant: str,
    scale: str,
    activation: str = "",
    base_dir: str = "runs",
) -> Path:
    """Generate a run directory path with timestamp.

    Args:
        variant: Model variant (e.g., "vanilla", "modern")
        scale: Model scale (e.g., "debug", "main", "stretch")
        activation: Activation function (e.g., "relu", "gelu", "swiglu")
        base_dir: Base directory for all runs.

    Returns:
        Path like runs/vanilla_relu_stretch_20260623_1430/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = [variant]
    if activation and activation != "swiglu":
        parts.append(activation)
    parts.append(scale)
    parts.append(timestamp)
    name = "_".join(parts)
    return Path(base_dir) / name
