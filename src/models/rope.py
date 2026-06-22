"""Rotary Position Embeddings (RoPE).

RoPE encodes position information by rotating Q and K vectors in 2D subspaces.
Instead of ADDING a position vector (learned embeddings), we ROTATE Q and K
by an angle proportional to their position.

Key insight: if Q at position m is rotated by angle m*θ, and K at position n
is rotated by angle n*θ, then Q·K depends on (m-n)*θ — the RELATIVE distance.
The model naturally learns relative position without explicit relative attention.

Why RoPE won over learned embeddings:
1. Better length generalization — the rotation pattern extrapolates to unseen lengths
2. Relative position is built into the dot product (no extra parameters)
3. Decays naturally with distance (far-away tokens have less influence)
4. Used in LLaMA, Mistral, GPT-NeoX, and most modern open-source LLMs

How it works:
- Split each head's d_head dimensions into pairs: (d0, d1), (d2, d3), ...
- Each pair is a 2D subspace that gets rotated by a position-dependent angle
- Different pairs rotate at different frequencies (like Fourier features)
- Apply to Q and K only (not V — values don't need position info)
"""

import torch


def precompute_rope_frequencies(
    d_head: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos and sin tables for RoPE.

    These are computed once and cached — they don't depend on the input,
    only on position and dimension.

    Args:
        d_head: Dimension per attention head (must be even).
        seq_len: Maximum sequence length to precompute for.
        theta: Base frequency (10000 is the standard from the paper).
        device: Target device.

    Returns:
        Tuple of (cos, sin), each shape (seq_len, d_head).
    """
    assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"

    # Frequency for each pair of dimensions
    # Pairs at lower indices rotate faster (high frequency),
    # pairs at higher indices rotate slower (low frequency).
    # This is like having multiple "clocks" ticking at different speeds.
    dim_indices = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (dim_indices / d_head))  # shape: (d_head // 2,)

    # Position indices
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: angles[pos, dim_pair] = pos * freq[dim_pair]
    angles = torch.outer(positions, freqs)  # shape: (seq_len, d_head // 2)

    # We need cos and sin for each position and each dimension pair
    # Shape: (seq_len, d_head // 2) — one value per pair
    cos = torch.cos(angles)  # (seq_len, d_head // 2)
    sin = torch.sin(angles)  # (seq_len, d_head // 2)

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a tensor.

    The rotation is applied to pairs of dimensions:
    For dimensions (x0, x1): rotate by angle θ
        x0_new = x0 * cos(θ) - x1 * sin(θ)
        x1_new = x0 * sin(θ) + x1 * cos(θ)

    This is equivalent to multiplying by a rotation matrix in each 2D subspace.

    Args:
        x: Input tensor of shape (batch, n_head, seq_len, d_head).
        cos: Cosine table, shape (seq_len, d_head) or broadcastable.
        sin: Sine table, shape (seq_len, d_head) or broadcastable.

    Returns:
        Rotated tensor of same shape.
    """
    # x shape: (B, n_head, T, d_head)
    # cos/sin shape: (T, d_head) — need to reshape for broadcasting
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head)

    # Split into pairs: first half and second half of d_head
    # This is the "interleaved" RoPE approach used in LLaMA
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    # Rotate: for each pair (x1, x2), apply 2D rotation
    # x1_new = x1 * cos - x2 * sin
    # x2_new = x1 * sin + x2 * cos
    cos_half = cos[..., :d_half]
    sin_half = sin[..., :d_half]

    x1_rot = x1 * cos_half - x2 * sin_half
    x2_rot = x1 * sin_half + x2 * cos_half

    return torch.cat([x1_rot, x2_rot], dim=-1)
