"""Synthetic diagnostic probes for model evaluation.

Probes measure retrieval capacity, representation health, and attention behavior
on synthetic sequences — no real data needed. Each probe returns a structured
result dataclass with metrics that can be compared across variants.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.models.config import ModelConfig


@dataclass
class MQARResult:
    """Result of the Multi-Query Associative Recall probe.

    Attributes:
        accuracy: P(correct recall) across all query positions and sequences.
        accuracy_by_distance: Mapping from retrieval distance to accuracy.
            Distance = query_position - key_position for each association.
    """

    accuracy: float
    accuracy_by_distance: dict[int, float]


def _generate_mqar_sequences(
    n_sequences: int,
    n_associations: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[list[int]], list[list[int]], list[list[int]]]:
    """Generate synthetic MQAR sequences with planted key-value associations.

    Sequence format:
        [key₁ val₁ key₂ val₂ ... keyₙ valₙ ... filler ... query₁ ... queryₙ]

    Keys and values are drawn from disjoint token ranges within the vocabulary.
    Filler tokens fill remaining positions between the pairs region and the query region.
    At query position i, the correct output (next token) is the value associated with key_i.

    Args:
        n_sequences: Number of sequences to generate.
        n_associations: Number of key-value pairs per sequence (n).
        seq_len: Maximum sequence length.
        vocab_size: Vocabulary size for token sampling.
        device: Device for tensor allocation.

    Returns:
        Tuple of (tokens, key_positions, query_positions, expected_values):
        - tokens: shape (n_sequences, seq_len)
        - key_positions: list of lists, each with n_associations key positions
        - query_positions: list of lists, each with n_associations query positions
        - expected_values: list of lists, each with n_associations expected value tokens
    """
    # Reserve token ranges:
    # - Keys: tokens in [1, vocab_size // 3)
    # - Values: tokens in [vocab_size // 3, 2 * vocab_size // 3)
    # - Filler: token 0
    key_range_start = 1
    key_range_end = vocab_size // 3
    val_range_start = vocab_size // 3
    val_range_end = 2 * vocab_size // 3
    filler_token = 0

    # Layout: pairs region takes 2 * n_associations positions,
    # queries take n_associations positions at the end.
    # The rest is filler between pairs and queries.
    pairs_end = 2 * n_associations  # positions [0, pairs_end) hold key-val pairs
    query_start = seq_len - n_associations  # queries at the end

    assert pairs_end <= query_start, (
        f"Not enough room for {n_associations} associations in seq_len={seq_len}. "
        f"Need at least {3 * n_associations} positions."
    )

    tokens = torch.full((n_sequences, seq_len), filler_token, dtype=torch.long, device=device)

    all_key_positions: list[list[int]] = []
    all_query_positions: list[list[int]] = []
    all_expected_values: list[list[int]] = []

    for seq_idx in range(n_sequences):
        # Sample unique keys and values for this sequence
        keys = torch.randint(key_range_start, key_range_end, (n_associations,), device=device)
        values = torch.randint(val_range_start, val_range_end, (n_associations,), device=device)

        key_positions = []
        query_positions = []
        expected_values = []

        # Place key-value pairs in the first part of the sequence
        for i in range(n_associations):
            key_pos = 2 * i
            val_pos = 2 * i + 1
            tokens[seq_idx, key_pos] = keys[i]
            tokens[seq_idx, val_pos] = values[i]
            key_positions.append(key_pos)

        # Place query tokens (keys repeated) at the end of the sequence
        for i in range(n_associations):
            q_pos = query_start + i
            tokens[seq_idx, q_pos] = keys[i]
            query_positions.append(q_pos)
            expected_values.append(values[i].item())

        all_key_positions.append(key_positions)
        all_query_positions.append(query_positions)
        all_expected_values.append(expected_values)

    return tokens, all_key_positions, all_query_positions, all_expected_values


def run_mqar_probe(
    model: nn.Module,
    config: ModelConfig,
    vocab_size: int = 50257,
    n_associations: int = 8,
    n_sequences: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> MQARResult:
    """Run MQAR (Multi-Query Associative Recall) probe on a model checkpoint.

    Generates synthetic sequences with planted key-value associations and measures
    the model's ability to recall the correct value when queried with the key.

    The probe measures P(correct recall) overall and broken down by retrieval
    distance (distance = query_position - key_position).

    Args:
        model: The model to evaluate (must be in eval mode).
        config: ModelConfig for the model (provides seq_len).
        vocab_size: Vocabulary size for token generation.
        n_associations: Number of key-value pairs per sequence.
        n_sequences: Number of sequences to evaluate.
        batch_size: Inference micro-batch size. Kept deliberately small because
            each forward materializes ``batch × seq_len × vocab_size`` logits.
        device: Device for computation.

    Returns:
        MQARResult with overall accuracy and per-distance breakdown.
    """
    model.eval()
    seq_len = config.seq_len
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    tokens, all_key_positions, all_query_positions, all_expected_values = _generate_mqar_sequences(
        n_sequences=n_sequences,
        n_associations=n_associations,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
    )

    # Run batched inference
    batch_size = min(batch_size, n_sequences)
    total_correct = 0
    total_queries = 0
    distance_correct: dict[int, int] = {}
    distance_total: dict[int, int] = {}

    with torch.no_grad():
        for batch_start in range(0, n_sequences, batch_size):
            batch_end = min(batch_start + batch_size, n_sequences)
            batch_tokens = tokens[batch_start:batch_end]

            # Model forward: returns (logits, loss, kv_cache)
            output = model(batch_tokens)
            # Handle both (logits, loss, kv_cache) and (logits, kv_cache) patterns
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Check predictions at query positions
            for seq_offset in range(batch_end - batch_start):
                seq_idx = batch_start + seq_offset
                query_positions = all_query_positions[seq_idx]
                key_positions = all_key_positions[seq_idx]
                expected_values = all_expected_values[seq_idx]

                for i, q_pos in enumerate(query_positions):
                    # The model predicts the next token at position q_pos
                    # So we check logits[seq_offset, q_pos] for the predicted token
                    predicted_token = logits[seq_offset, q_pos].argmax(dim=-1).item()
                    expected_token = expected_values[i]

                    correct = predicted_token == expected_token
                    total_correct += int(correct)
                    total_queries += 1

                    # Compute distance from key to query
                    distance = q_pos - key_positions[i]

                    if distance not in distance_correct:
                        distance_correct[distance] = 0
                        distance_total[distance] = 0
                    distance_correct[distance] += int(correct)
                    distance_total[distance] += 1

            del logits, output

    # Compute overall accuracy
    accuracy = total_correct / total_queries if total_queries > 0 else 0.0

    # Compute per-distance accuracy
    accuracy_by_distance: dict[int, float] = {}
    for dist in sorted(distance_total.keys()):
        if distance_total[dist] > 0:
            accuracy_by_distance[dist] = distance_correct[dist] / distance_total[dist]

    return MQARResult(accuracy=accuracy, accuracy_by_distance=accuracy_by_distance)


@dataclass
class StableRankResult:
    """Result of the stable rank computation across model layers.

    Attributes:
        per_layer: Stable rank for each layer, shape (n_layer,).
        mean: Mean stable rank across layers.
        std: Standard deviation of stable rank across layers.
    """

    per_layer: np.ndarray  # shape: (n_layer,)
    mean: float
    std: float


def compute_stable_rank(
    model: nn.Module,
    val_loader,
    n_batches: int = 100,
    device: str = "cuda",
) -> StableRankResult:
    """Compute stable rank per layer: srank(H) = ||H||²_F / σ₁².

    Registers forward hooks on each transformer block to capture hidden states
    (the output of each block, i.e. the residual stream after the block).
    Computes SVD of the hidden states reshaped to (batch_size * seq_len, d_model),
    then averages stable rank over n_batches validation batches.

    Args:
        model: The transformer model to evaluate (must be in eval mode).
            Expected to have a `.blocks` attribute (nn.ModuleList of transformer blocks).
        val_loader: A data loader with a next_batch() method returning (x, y) tuples.
        n_batches: Number of validation batches to average over.
        device: Device for computation.

    Returns:
        StableRankResult with per_layer shape (n_layer,), mean, and std.
        Values are guaranteed to be in [1.0, d_model].
    """
    model.eval()

    # Identify the transformer blocks
    blocks = model.blocks

    n_layers = len(blocks)
    # Accumulator: sum of stable ranks per layer across batches
    srank_accum = np.zeros(n_layers, dtype=np.float64)

    # Storage for captured hidden states per forward pass
    hidden_states: list[torch.Tensor | None] = [None] * n_layers

    # Register forward hooks on each block to capture the output (residual stream)
    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output is (x, new_kv_cache) tuple from the block's forward
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0].detach()
            else:
                hidden_states[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for i, block in enumerate(blocks):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, _y = val_loader.next_batch()
                x = x.to(device)

                # Forward pass to trigger hooks
                model(x)

                # Compute stable rank for each layer from captured hidden states
                for layer_idx in range(n_layers):
                    H = hidden_states[layer_idx]
                    # H shape: (batch_size, seq_len, d_model)
                    # Reshape to (batch_size * seq_len, d_model) for SVD
                    B, T, D = H.shape
                    # Linear algebra analysis runs in float32 even when model
                    # inference uses bf16/fp16 (required by FlashAttention).
                    H_flat = H.reshape(B * T, D).float()

                    # Compute singular values only (no need for U, V)
                    sv = torch.linalg.svdvals(H_flat)

                    # srank(H) = ||H||²_F / σ₁²
                    # ||H||²_F = sum of squared singular values
                    frobenius_sq = (sv * sv).sum().item()
                    sigma1_sq = (sv[0] * sv[0]).item()

                    if sigma1_sq > 0:
                        srank = frobenius_sq / sigma1_sq
                    else:
                        srank = 1.0  # Degenerate case: all zeros

                    # Clamp to valid range [1.0, d_model]
                    srank = max(1.0, min(float(D), srank))
                    srank_accum[layer_idx] += srank

                    # Clear reference to free memory
                    hidden_states[layer_idx] = None
    finally:
        # Always remove hooks to avoid leaking state
        for h in hooks:
            h.remove()

    # Average over batches
    per_layer = srank_accum / n_batches
    mean = float(per_layer.mean())
    std = float(per_layer.std())

    return StableRankResult(per_layer=per_layer, mean=mean, std=std)


@dataclass
class CKAResult:
    """Result of the Centered Kernel Alignment computation between layers.

    Attributes:
        adjacent_curve: CKA similarity between adjacent layers, shape (n_layer - 1,).
            adjacent_curve[i] = CKA(layer_i, layer_{i+1}).
        full_matrix: Full pairwise CKA matrix, shape (n_layer, n_layer).
            full_matrix[i, j] = CKA(layer_i, layer_j).
    """

    adjacent_curve: np.ndarray  # shape: (n_layer - 1,)
    full_matrix: np.ndarray  # shape: (n_layer, n_layer)


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Linear CKA formula:
        CKA(X, Y) = ||Y^T X||²_F / (||X^T X||_F × ||Y^T Y||_F)

    where X and Y are centered representation matrices of shape (N, D).

    Args:
        X: Centered representation matrix, shape (N, D1).
        Y: Centered representation matrix, shape (N, D2).

    Returns:
        CKA similarity value in [0.0, 1.0].
    """
    # Compute cross-covariance and self-covariances
    # YtX = Y^T @ X, shape (D2, D1)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    # Frobenius norms squared
    numerator = (YtX * YtX).sum().item()  # ||Y^T X||²_F
    denom_x = (XtX * XtX).sum().item()  # ||X^T X||²_F
    denom_y = (YtY * YtY).sum().item()  # ||Y^T Y||²_F

    denominator = (denom_x * denom_y) ** 0.5

    if denominator == 0.0:
        return 0.0

    cka_val = numerator / denominator

    # Clamp to [0, 1] for numerical stability
    return max(0.0, min(1.0, cka_val))


def compute_cka(
    model: nn.Module,
    val_loader,
    n_batches: int = 50,
    device: str = "cuda",
) -> CKAResult:
    """Compute linear CKA between all pairs of layer representations.

    Registers forward hooks on each transformer block to capture hidden states.
    Accumulates representations across n_batches for more stable CKA estimates,
    then computes the pairwise CKA matrix.

    Args:
        model: The transformer model to evaluate (must be in eval mode).
            Expected to have a `.blocks` attribute (nn.ModuleList of transformer blocks).
        val_loader: A data loader with a next_batch() method returning (x, y) tuples.
        n_batches: Number of validation batches to accumulate representations over.
        device: Device for computation.

    Returns:
        CKAResult with:
        - adjacent_curve: shape (n_layer - 1,), CKA between consecutive layers
        - full_matrix: shape (n_layer, n_layer), symmetric with diagonal = 1.0
    """
    model.eval()

    # Identify the transformer blocks
    blocks = model.blocks
    n_layers = len(blocks)

    # Storage for captured hidden states per forward pass
    hidden_states: list[torch.Tensor | None] = [None] * n_layers

    # Register forward hooks on each block to capture the output
    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output is (x, new_kv_cache) tuple from the block's forward
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0].detach()
            else:
                hidden_states[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for i, block in enumerate(blocks):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Accumulate representations across batches
    # Each layer gets a list of flattened representation tensors
    all_representations: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, _y = val_loader.next_batch()
                x = x.to(device)

                # Forward pass to trigger hooks
                model(x)

                # Collect representations from each layer
                for layer_idx in range(n_layers):
                    H = hidden_states[layer_idx]
                    # H shape: (batch_size, seq_len, d_model)
                    # Reshape to (batch_size * seq_len, d_model)
                    B, T, D = H.shape
                    H_flat = H.reshape(B * T, D)
                    # Keep representation statistics in float32 for numerical
                    # stability and consistent support across devices.
                    all_representations[layer_idx].append(H_flat.float().cpu())

                    # Clear reference to free GPU memory
                    hidden_states[layer_idx] = None
    finally:
        # Always remove hooks to avoid leaking state
        for h in hooks:
            h.remove()

    # Concatenate all batch representations per layer: (N_total, d_model)
    layer_reps: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        concat = torch.cat(all_representations[layer_idx], dim=0)
        # Center the representations: X_c = X - mean(X, dim=0)
        concat = concat - concat.mean(dim=0, keepdim=True)
        layer_reps.append(concat)

    # Free accumulated raw representations
    del all_representations

    # Compute full pairwise CKA matrix
    full_matrix = np.zeros((n_layers, n_layers), dtype=np.float64)

    for i in range(n_layers):
        # Diagonal is always 1.0 (self-similarity)
        full_matrix[i, i] = 1.0
        for j in range(i + 1, n_layers):
            cka_val = _linear_cka(layer_reps[i], layer_reps[j])
            full_matrix[i, j] = cka_val
            full_matrix[j, i] = cka_val  # Symmetric

    # Extract adjacent-layer CKA curve
    adjacent_curve = np.array(
        [full_matrix[i, i + 1] for i in range(n_layers - 1)], dtype=np.float64
    )

    return CKAResult(adjacent_curve=adjacent_curve, full_matrix=full_matrix)


@dataclass
class AttentionEntropyResult:
    """Result of the attention entropy computation across model layers and heads.

    Attributes:
        per_layer: Mean Shannon entropy per layer, averaged over heads and batches.
            Shape: (n_layer,).
        per_head: Mean Shannon entropy per head per layer, averaged over batches.
            Shape: (n_layer, n_head).
    """

    per_layer: np.ndarray  # shape: (n_layer,)
    per_head: np.ndarray  # shape: (n_layer, n_head)


def compute_attention_entropy(
    model: nn.Module,
    val_loader,
    n_batches: int = 50,
    device: str = "cuda",
) -> "AttentionEntropyResult | None":
    """Compute Shannon entropy of attention weight distributions per layer and head.

    Only applicable to V0, where softmax attention weights are explicitly
    materialized. Causal linear attention maintains prefix statistics rather
    than a probability matrix, so Shannon entropy is not directly defined.

    All other variants return None.

    Shannon entropy: H = -Σ p·log(p) where p is the attention probability distribution.
    Values of p=0 are handled safely (0·log(0) = 0).

    Args:
        model: The transformer model to evaluate (must be in eval mode).
            Expected to have a `.config` attribute with `attention_type` and
            a `.blocks` attribute (nn.ModuleList of transformer blocks), where
            each block has an `.attn` attribute with an `.attn_dropout` submodule.
        val_loader: A data loader with a next_batch() method returning (x, y) tuples.
        n_batches: Number of validation batches to average entropy over.
        device: Device for computation.

    Returns:
        AttentionEntropyResult with per_layer shape (n_layer,) and per_head shape
        (n_layer, n_head), or None if the model uses flash attention.
    """
    model.eval()

    # Check if attention weights are accessible for this variant
    attention_type = model.config.attention_type
    if attention_type != "full":
        return None

    blocks = model.blocks
    n_layers = len(blocks)
    n_heads = model.config.n_head

    # Accumulator for entropy: shape (n_layer, n_head)
    entropy_accum = np.zeros((n_layers, n_heads), dtype=np.float64)

    # Storage for captured attention weights per forward pass
    attn_weights_store: list[torch.Tensor | None] = [None] * n_layers

    # Register a forward hook on each attention module's attn_dropout.
    # The attn_dropout receives attention weights as input: dropout(weights).
    # We capture the input to attn_dropout, which is the post-softmax attention weights.
    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # input is a tuple; input[0] is the attention weights tensor
            # For CausalSelfAttention: shape (B, n_head, T, S)
            attn_weights_store[layer_idx] = input[0].detach()

        return hook_fn

    hooks = []
    for i, block in enumerate(blocks):
        attn_module = block.attn
        h = attn_module.attn_dropout.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, _y = val_loader.next_batch()
                x = x.to(device)

                # Forward pass to trigger hooks
                model(x)

                # Compute entropy for each layer from captured attention weights
                for layer_idx in range(n_layers):
                    weights = attn_weights_store[layer_idx]
                    # weights shape: (B, n_head, T, seq_len)

                    # Shannon entropy: H = -Σ p·log(p)
                    # Clamp to avoid log(0); where p=0, p·log(p) = 0
                    p = weights.clamp(min=1e-12)
                    log_p = torch.log(p)
                    # Entropy per query position: -sum over key dimension
                    # Shape: (B, n_head, T)
                    entropy_per_pos = -(weights * log_p).sum(dim=-1)

                    # Average over batch and positions → shape (n_head,)
                    entropy_per_head = entropy_per_pos.mean(dim=(0, 2))

                    entropy_accum[layer_idx] += entropy_per_head.cpu().numpy()

                    # Clear reference to free memory
                    attn_weights_store[layer_idx] = None
    finally:
        # Always remove hooks to avoid leaking state
        for h in hooks:
            h.remove()

    # Average over batches
    per_head = entropy_accum / n_batches  # shape: (n_layer, n_head)
    per_layer = per_head.mean(axis=1)  # shape: (n_layer,)

    return AttentionEntropyResult(per_layer=per_layer, per_head=per_head)
