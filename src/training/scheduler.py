"""Learning rate scheduler: linear warmup + cosine decay.

The learning rate schedule used by GPT-2/3, LLaMA, and most modern LLMs:

1. Warmup phase: linearly ramp LR from 0 to max over `warmup_steps` steps.
   Why? At the start, the model's weights are random and gradients are noisy.
   A large LR on random gradients can destabilize training. Warmup lets the
   optimizer "find its bearings" with gentle updates first.

2. Cosine decay phase: smoothly decrease LR from max to min following a cosine curve.
   Why cosine? It decays slowly at first (when you're still learning a lot),
   then faster in the middle, then slowly again at the end (gentle convergence).
   Empirically better than linear decay or step decay for LLMs.

The final LR (min_lr) is typically 10% of max_lr — not zero, because you
still want some learning to happen at the end of training.
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Compute learning rate for a given step.

    Args:
        step: Current training step (0-indexed).
        max_lr: Peak learning rate (reached at end of warmup).
        min_lr: Minimum learning rate (at end of training).
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.

    Returns:
        Learning rate for this step.
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 3: After total_steps, just return min_lr
    if step >= total_steps:
        return min_lr

    # Phase 2: Cosine decay from max_lr to min_lr
    # Progress through the decay phase: 0.0 at start of decay, 1.0 at end
    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    # Cosine goes from 1 to -1 over [0, π], so (1 + cos(progress*π))/2 goes from 1 to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Interpolate between max_lr and min_lr
    return min_lr + coeff * (max_lr - min_lr)
