"""Metrics extraction and computation from training logs and model checkpoints."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit


@dataclass
class MetricsResult:
    """Aggregated loss-based metrics for a single checkpoint."""

    val_loss: float
    perplexity: float
    per_position_loss: np.ndarray | None  # shape: (seq_len,), None if not computed yet
    icl_exponent: float | None  # α from power-law fit
    icl_fit_params: dict | None  # {"A": float, "alpha": float, "C": float, "r_squared": float}


def load_metrics_log(checkpoint_dir: Path) -> list[dict]:
    """Load metrics.jsonl from a checkpoint directory.

    Parses the file line by line, returning a list of dicts with fields
    including step, train_loss, val_loss, tokens_seen, and elapsed_time.

    Args:
        checkpoint_dir: Path to a checkpoint directory containing metrics.jsonl.

    Returns:
        List of parsed dicts, one per line in metrics.jsonl.

    Raises:
        FileNotFoundError: If metrics.jsonl does not exist in checkpoint_dir.
    """
    metrics_path = Path(checkpoint_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.jsonl not found in checkpoint directory: {checkpoint_dir}"
        )

    entries: list[dict] = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_val_loss(log_entries: list[dict]) -> float:
    """Extract final validation loss from log entries.

    Finds the last entry that has a non-None val_loss field and returns
    that value.

    Args:
        log_entries: List of parsed log entry dicts from load_metrics_log.

    Returns:
        The final validation loss value.

    Raises:
        ValueError: If no entry contains a non-None val_loss field.
    """
    for entry in reversed(log_entries):
        val_loss = entry.get("val_loss")
        if val_loss is not None:
            return float(val_loss)
    raise ValueError("No entry with a non-None val_loss found in log entries.")


def compute_perplexity(val_loss: float) -> float:
    """Compute perplexity from validation loss.

    Args:
        val_loss: Validation loss value.

    Returns:
        exp(val_loss)
    """
    return math.exp(val_loss)


def compute_per_position_loss(
    model: nn.Module,
    val_loader,
    seq_len: int,
    device: str = "cuda",
) -> np.ndarray:
    """Compute validation loss at each token position, averaged over batches.

    Runs the model in eval/no-grad mode on batches from val_loader, computing
    cross-entropy loss at each of the seq_len token positions. Results are
    averaged across all batches.

    Args:
        model: A causal language model that takes idx (B, T) and returns
            (logits, loss, kv_cache) where logits has shape (B, T, vocab_size).
        val_loader: A data loader with a next_batch() method returning (x, y)
            tensors of shape (batch_size, seq_len).
        seq_len: The sequence length (number of positions to compute loss for).
        device: Device to run computation on.

    Returns:
        A numpy array of shape (seq_len,) containing average per-position loss.
        All values are non-negative.
    """
    model.eval()
    position_loss_sum = torch.zeros(seq_len, device=device)
    n_batches = 0

    with torch.no_grad():
        # Use a reasonable number of batches from the validation set
        # We iterate until the loader cycles or we have enough data
        n_eval_batches = 50  # default number of evaluation batches
        for _ in range(n_eval_batches):
            x, y = val_loader.next_batch()
            x = x.to(device)
            y = y.to(device)

            # Forward pass: model returns (logits, loss, kv_cache)
            logits, _, _ = model(x)

            # Compute per-position cross-entropy loss
            # logits: (B, T, vocab_size), y: (B, T)
            B, T, V = logits.shape
            # Clamp T to seq_len in case loader provides different length
            T_eff = min(T, seq_len)

            # Reshape for cross_entropy: (B*T, V) vs (B*T,)
            loss_per_token = F.cross_entropy(
                logits[:, :T_eff, :].reshape(-1, V),
                y[:, :T_eff].reshape(-1),
                reduction="none",
            )  # shape: (B * T_eff,)

            # Reshape back to (B, T_eff) and average over batch dim
            loss_per_token = loss_per_token.view(B, T_eff)
            batch_avg = loss_per_token.mean(dim=0)  # shape: (T_eff,)

            position_loss_sum[:T_eff] += batch_avg
            n_batches += 1

    # Average across batches
    per_position_loss = position_loss_sum / max(n_batches, 1)

    return per_position_loss.cpu().numpy()


def _power_law(t: np.ndarray, A: float, alpha: float, C: float) -> np.ndarray:
    """Power-law decay model: L(t) = A * t^(-alpha) + C."""
    return A * t ** (-alpha) + C


def fit_icl_decay(per_position_loss: np.ndarray) -> dict:
    """Fit power-law L(t) = A * t^(-α) + C to per-position loss.

    Uses nonlinear least-squares optimization (scipy.optimize.curve_fit) to
    estimate parameters of the ICL decay model. Positions are indexed from
    1..seq_len to avoid division by zero at t=0.

    Args:
        per_position_loss: 1D array of shape (seq_len,) containing average
            loss at each token position.

    Returns:
        Dict with keys "A", "alpha", "C", "r_squared". On convergence failure,
        returns NaN for A, alpha, C and 0.0 for r_squared.

    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
    """
    seq_len = len(per_position_loss)
    # Use positions 1..seq_len to avoid t=0 singularity
    t = np.arange(1, seq_len + 1, dtype=np.float64)
    y = per_position_loss.astype(np.float64)

    # If data is essentially constant (no decay), return alpha=0 directly.
    # This avoids the optimizer wandering when A≈0 makes alpha irrelevant.
    y_range = np.max(y) - np.min(y)
    y_mean = np.mean(y)
    if y_mean > 0 and y_range / y_mean < 1e-6:
        return {
            "A": 0.0,
            "alpha": 0.0,
            "C": float(y_mean),
            "r_squared": 1.0,
        }

    # Initial guesses: A~1.0, alpha~0.5, C~final loss value
    p0 = [1.0, 0.5, float(y[-1])]
    # Bounds: A > 0, alpha > 0, C >= 0
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(
            _power_law, t, y, p0=p0, bounds=bounds, maxfev=10000
        )
        A, alpha, C = popt

        # If fitted A is negligible relative to signal, treat as no decay
        if A < 1e-10:
            alpha = 0.0

        # Compute R²
        y_pred = _power_law(t, A, alpha, C)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "A": float(A),
            "alpha": float(alpha),
            "C": float(C),
            "r_squared": float(r_squared),
        }
    except (RuntimeError, ValueError):
        # Convergence failure or invalid input
        return {
            "A": float("nan"),
            "alpha": float("nan"),
            "C": float("nan"),
            "r_squared": 0.0,
        }
