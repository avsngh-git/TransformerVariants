"""Component-level FLOP accounting for Transformer variants.

Computes per-step training FLOPs broken down by component (QKV projections,
attention score, output projection, FFN) with variant-aware attention cost
modeling. Pure arithmetic from config values — no model instantiation needed.
"""

from dataclasses import dataclass

from src.models.config import ModelConfig


@dataclass
class MFUResult:
    """Result of Model FLOPs Utilization computation.

    Attributes:
        achieved_tflops: Actual TFLOPS achieved during the training step.
        peak_tflops: Hardware peak TFLOPS (default: 242.0 for L4 BF16).
        mfu: Model FLOPs Utilization ratio (achieved / peak).
        arithmetic_intensity: FLOPs per byte of memory traffic (set to 0.0
            until estimate_bytes_transferred is implemented in task 2.3).
    """

    achieved_tflops: float
    peak_tflops: float
    mfu: float
    arithmetic_intensity: float


@dataclass
class FLOPBreakdown:
    """Per-step training FLOP breakdown by component.

    All values include the 3× training multiplier (forward + backward ≈ 3× forward).

    Attributes:
        qkv_proj: FLOPs for Q, K, V projections across all layers.
        attention_score: FLOPs for attention score computation (variant-dependent).
        attention_output: FLOPs for attention output projection across all layers.
        ffn: FLOPs for feed-forward network across all layers.
        total: Sum of all components.
    """

    qkv_proj: int
    attention_score: int
    attention_output: int
    ffn: int
    total: int


def compute_step_flops(config: ModelConfig) -> FLOPBreakdown:
    """Compute per-training-step FLOPs from model configuration.

    Pure function — uses only arithmetic on config values. Each matmul of
    shape (M, K) × (K, N) costs 2×M×K×N FLOPs.

    Includes a 3× training multiplier (forward + backward ≈ 3× forward FLOPs).

    Args:
        config: Model configuration specifying architecture and dimensions.

    Returns:
        FLOPBreakdown with per-component and total FLOP counts.
    """
    seq_len = config.seq_len
    d_model = config.d_model
    n_head = config.n_head
    n_layer = config.n_layer
    d_head = config.d_head
    d_ff = config.d_ff

    # --- Per-layer FLOP components ---

    # QKV projections: 3 matmuls of (seq_len, d_model) × (d_model, d_model)
    # Each costs 2 × seq_len × d_model × d_model
    qkv_proj_per_layer = 3 * 2 * seq_len * d_model * d_model

    # Attention score: variant-dependent
    if config.attention_type in ("full", "flash_sdpa"):
        # Q @ K^T: (seq_len, d_head) × (d_head, seq_len) per head
        # = 2 × n_head × seq_len × seq_len × d_head
        attention_score_per_layer = 2 * n_head * seq_len * seq_len * d_head
    elif config.attention_type == "sliding_window":
        # Each query attends to min(window_size, seq_len) keys
        window = min(config.window_size, seq_len) if config.window_size else seq_len
        attention_score_per_layer = 2 * n_head * seq_len * window * d_head
    elif config.attention_type == "linear":
        # Causal linear attention has two dominant per-token operations:
        # update prefix state with phi(K) @ V^T, then query it with phi(Q).
        # The denominator dot product is lower order but included.
        state_update = 2 * n_head * seq_len * d_head * d_head
        state_query = 2 * n_head * seq_len * d_head * d_head
        normalization = 2 * n_head * seq_len * d_head
        attention_score_per_layer = state_update + state_query + normalization
    else:
        # Fallback to full attention cost
        attention_score_per_layer = 2 * n_head * seq_len * seq_len * d_head

    # Attention output projection: (seq_len, d_model) × (d_model, d_model)
    attention_output_per_layer = 2 * seq_len * d_model * d_model

    # FFN: depends on ffn_type
    if config.ffn_type == "swiglu":
        # SwiGLU: 3 projections (gate, up, down)
        # gate: (seq_len, d_model) × (d_model, d_ff) = 2 × seq_len × d_model × d_ff
        # up:   (seq_len, d_model) × (d_model, d_ff) = 2 × seq_len × d_model × d_ff
        # down: (seq_len, d_ff) × (d_ff, d_model)    = 2 × seq_len × d_model × d_ff
        ffn_per_layer = 3 * 2 * seq_len * d_model * d_ff
    else:
        # Standard FFN: 2 projections (up, down)
        # up:   (seq_len, d_model) × (d_model, d_ff) = 2 × seq_len × d_model × d_ff
        # down: (seq_len, d_ff) × (d_ff, d_model)    = 2 × seq_len × d_model × d_ff
        ffn_per_layer = 2 * 2 * seq_len * d_model * d_ff

    # --- Scale by n_layer ---
    qkv_proj_total = qkv_proj_per_layer * n_layer
    attention_score_total = attention_score_per_layer * n_layer
    attention_output_total = attention_output_per_layer * n_layer
    ffn_total = ffn_per_layer * n_layer

    # --- Apply 3× training multiplier (forward + backward) ---
    training_multiplier = 3
    qkv_proj_final = qkv_proj_total * training_multiplier
    attention_score_final = attention_score_total * training_multiplier
    attention_output_final = attention_output_total * training_multiplier
    ffn_final = ffn_total * training_multiplier

    total = qkv_proj_final + attention_score_final + attention_output_final + ffn_final

    return FLOPBreakdown(
        qkv_proj=qkv_proj_final,
        attention_score=attention_score_final,
        attention_output=attention_output_final,
        ffn=ffn_final,
        total=total,
    )


def estimate_bytes_transferred(config: ModelConfig) -> int:
    """Estimate memory traffic (bytes) per training step for roofline analysis.

    Accounts for:
    - Parameter reads: all model weights are read once per forward pass (BF16 = 2 bytes each).
    - Activation reads/writes: per-layer activations flowing through attention and FFN.
    - Training multiplier: forward + backward pass doubles the memory traffic
      (backward reads activations and writes gradients).

    All elements are assumed BF16 (2 bytes per element).

    Args:
        config: Model configuration specifying architecture and dimensions.

    Returns:
        Estimated total bytes transferred per training step.
    """
    bytes_per_element = 2  # BF16

    seq_len = config.seq_len
    d_model = config.d_model
    n_layer = config.n_layer
    d_ff = config.d_ff

    # --- Parameter bytes (weights read during forward pass) ---
    # QKV projections: 3 weight matrices of shape (d_model, d_model) per layer
    qkv_param_bytes = n_layer * 3 * d_model * d_model * bytes_per_element

    # Output projection: (d_model, d_model) per layer
    output_proj_param_bytes = n_layer * d_model * d_model * bytes_per_element

    # FFN weights per layer
    if config.ffn_type == "swiglu":
        # SwiGLU: gate (d_model, d_ff) + up (d_model, d_ff) + down (d_ff, d_model)
        ffn_param_bytes = n_layer * 3 * d_model * d_ff * bytes_per_element
    else:
        # Standard: up (d_model, d_ff) + down (d_ff, d_model)
        ffn_param_bytes = n_layer * 2 * d_model * d_ff * bytes_per_element

    total_param_bytes = qkv_param_bytes + output_proj_param_bytes + ffn_param_bytes

    # --- Activation bytes (read/write per layer) ---
    # Each layer reads input activations and writes output activations.
    # Input/output activations have shape (seq_len, d_model).
    # Per layer: read input + write output for attention block + FFN block.

    # Attention block: read input (seq_len × d_model), write Q/K/V (3 × seq_len × d_model),
    # write attention output (seq_len × d_model)
    attn_activation_bytes_per_layer = (
        # Input read
        seq_len * d_model
        # Q, K, V intermediate activations (written then read)
        + 3 * seq_len * d_model
        # Attention output (written)
        + seq_len * d_model
    ) * bytes_per_element

    # FFN block: read input (seq_len × d_model), write hidden (seq_len × d_ff),
    # read hidden, write output (seq_len × d_model)
    ffn_activation_bytes_per_layer = (
        # Input read
        seq_len * d_model
        # Hidden activation (written then read)
        + seq_len * d_ff
        # Output (written)
        + seq_len * d_model
    ) * bytes_per_element

    total_activation_bytes = n_layer * (
        attn_activation_bytes_per_layer + ffn_activation_bytes_per_layer
    )

    # --- Training multiplier ---
    # Forward pass: read params + read/write activations
    # Backward pass: read params again + read activations + write gradients
    # Approximately 2× the forward-pass memory traffic for a full training step.
    training_multiplier = 2

    total_bytes = (total_param_bytes + total_activation_bytes) * training_multiplier

    return total_bytes


def compute_arithmetic_intensity(config: ModelConfig) -> float:
    """Compute arithmetic intensity: FLOPs per byte transferred.

    Used for roofline positioning — determines whether a variant is
    compute-bound or memory-bound relative to hardware limits.

    Args:
        config: Model configuration specifying architecture and dimensions.

    Returns:
        Arithmetic intensity (FLOPs/byte). Higher values indicate more
        compute-bound behavior.
    """
    total_flops = compute_step_flops(config).total
    total_bytes = estimate_bytes_transferred(config)
    return total_flops / total_bytes


def compute_mfu(
    step_flops: int,
    step_time_seconds: float,
    peak_tflops: float = 242.0,
) -> MFUResult:
    """Compute Model FLOPs Utilization for a training step.

    MFU measures how efficiently a model uses available hardware compute,
    expressed as the ratio of achieved TFLOPS to hardware peak TFLOPS.

    Args:
        step_flops: Total floating-point operations for one training step.
        step_time_seconds: Wall-clock time for the step in seconds.
        peak_tflops: Hardware peak TFLOPS (default 242.0 for L4 BF16).

    Returns:
        MFUResult with achieved_tflops, peak_tflops, mfu ratio, and
        arithmetic_intensity (set to 0.0 until task 2.3 is implemented).

    Raises:
        ValueError: If step_time_seconds is zero or negative.
    """
    if step_time_seconds <= 0:
        raise ValueError(f"step_time_seconds must be positive, got {step_time_seconds}")

    achieved_tflops = step_flops / (step_time_seconds * 1e12)
    mfu = achieved_tflops / peak_tflops

    return MFUResult(
        achieved_tflops=achieved_tflops,
        peak_tflops=peak_tflops,
        mfu=mfu,
        arithmetic_intensity=0.0,  # Populated by task 2.3
    )
