"""Comparison module for multi-variant evaluation.

Loads training logs and configs from checkpoint directories, slices along
controlled comparison axes (fixed-data, fixed-wallclock, fixed-FLOPs), and
performs Pareto analysis.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.flops import FLOPBreakdown, compute_step_flops
from src.evaluation.metrics import MetricsResult, load_metrics_log
from src.models.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class VariantData:
    """Loaded data for a single variant checkpoint.

    Attributes:
        name: Variant name (e.g., "vanilla", "modern", "swa").
        checkpoint_dir: Path to the checkpoint/run directory.
        log_entries: Parsed metrics.jsonl entries.
        config: Model configuration for this variant.
        metrics: Aggregated loss-based metrics (None if not yet computed).
        flop_breakdown: Per-step FLOP breakdown (None if not yet computed).
    """

    name: str
    checkpoint_dir: Path
    log_entries: list[dict]
    config: ModelConfig
    metrics: MetricsResult | None = None
    flop_breakdown: FLOPBreakdown | None = None


@dataclass
class ComparisonResult:
    """Result of a full multi-variant comparison.

    Attributes:
        fixed_data: Variant name → val_loss at token budget.
        fixed_wallclock: Variant name → {time_fraction → val_loss}.
        fixed_flops: Variant name → val_loss at FLOP budget.
        pareto_front: List of Pareto-optimal variant names.
        parameter_counts: Variant name → total trainable param count.
        parameter_parity_valid: True if all variants within ±5% of mean.
    """

    fixed_data: dict[str, float] = field(default_factory=dict)
    fixed_wallclock: dict[str, dict] = field(default_factory=dict)
    fixed_flops: dict[str, float] = field(default_factory=dict)
    pareto_front: list[str] = field(default_factory=list)
    parameter_counts: dict[str, int] = field(default_factory=dict)
    parameter_parity_valid: bool = False


def _infer_variant_name(directory: Path) -> str:
    """Infer variant name from directory name.

    Supports naming conventions:
    - `{variant}_{scale}_s{seed}` (e.g., "vanilla_main_s42" → "vanilla")
    - `{variant}_{activation}_{scale}_{timestamp}`
      (e.g., "modern_swiglu_main_20240101_1200" → "modern")
    - `{variant}_{scale}_{timestamp}` (e.g., "vanilla_debug_20240101_1200" → "vanilla")

    Falls back to the full directory name if no pattern matches.
    """
    dir_name = directory.name

    # Known variant names from the registry
    known_variants = [
        "vanilla", "modern", "alibi", "gqa", "swa", "swa_interleaved", "linear"
    ]

    # Try matching known variant names at the start of the directory name
    for variant in sorted(known_variants, key=len, reverse=True):
        if dir_name.startswith(variant + "_") or dir_name == variant:
            return variant

    # Fallback: take everything before the first underscore followed by a scale or seed pattern
    match = re.match(r"^([a-zA-Z_]+?)_(?:debug|main|stretch|s\d+)", dir_name)
    if match:
        return match.group(1)

    # Last resort: return full directory name
    return dir_name


def _load_config_from_dir(checkpoint_dir: Path) -> ModelConfig | None:
    """Attempt to load a ModelConfig from a checkpoint/run directory.

    Looks for:
    1. run_config.json (training run format with nested model config)
    2. config.json (direct ModelConfig serialization)

    Returns None if no config file is found or parsing fails.
    """
    search_dirs = [checkpoint_dir]
    metrics_path = checkpoint_dir / "metrics.jsonl"
    if metrics_path.is_symlink():
        try:
            run_dir = metrics_path.resolve(strict=True).parent
            if run_dir != checkpoint_dir:
                search_dirs.append(run_dir)
        except OSError as e:
            logger.warning("Failed to resolve metrics symlink in %s: %s", checkpoint_dir, e)

    for config_dir in search_dirs:
        # Try run_config.json first (standard training run output)
        run_config_path = config_dir / "run_config.json"
        if run_config_path.exists():
            try:
                with open(run_config_path, "r") as f:
                    run_config = json.load(f)
                return _run_config_to_model_config(run_config)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse run_config.json in {config_dir}: {e}"
                )

        # Try config.json (direct serialization)
        config_path = config_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                return _dict_to_model_config(config_dict)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse config.json in {config_dir}: {e}"
                )

    return None


def _dict_to_model_config(d: dict) -> ModelConfig:
    """Convert a dictionary to a ModelConfig, using only known fields.

    Filters out any keys not accepted by ModelConfig to avoid TypeErrors.
    """
    # Get ModelConfig field names
    valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}

    # Filter to only valid fields
    filtered = {k: v for k, v in d.items() if k in valid_fields}

    return ModelConfig(**filtered)


def _run_config_to_model_config(run_config: dict) -> ModelConfig:
    """Reconstruct a complete model config from a saved training run config."""
    model_dict = run_config.get("model", run_config)
    if not isinstance(model_dict, dict):
        raise TypeError("run config 'model' must be an object")

    variant_name = model_dict.get("variant") or run_config.get("variant")
    scale = run_config.get("scale")
    config = _config_from_variant_name(variant_name, scale=scale) if variant_name else None

    if config is None:
        return _dict_to_model_config(model_dict)

    valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    merged = vars(config).copy()
    merged.update({k: v for k, v in model_dict.items() if k in valid_fields})
    merged["variant"] = variant_name
    return ModelConfig(**merged)


def _config_from_variant_name(
    variant_name: str, scale: str | None = None
) -> ModelConfig | None:
    """Attempt to create a default ModelConfig from a known variant name.

    Uses the registry's known defaults. Returns None if the variant is unknown.
    """
    # Import here to avoid circular imports at module level
    try:
        from src.models.registry import SCALES, VARIANTS

        if variant_name not in VARIANTS:
            return None

        spec = VARIANTS[variant_name]
        scale_name = scale if scale in SCALES else "debug"
        dims = SCALES[scale_name]

        config = ModelConfig(
            n_layer=dims["n_layer"],
            d_model=dims["d_model"],
            n_head=dims["n_head"],
            seq_len=dims["seq_len"],
            variant=spec.variant,
            norm_type=spec.norm_type,
            position_encoding=spec.position_encoding,
            ffn_type=spec.ffn_type,
            attention_type=spec.attention_type,
            attention_backend="sdpa",
            activation=spec.default_activation,
        )

        return spec.config_overrides(config, dims)
    except ImportError:
        return None


def load_variant_data(checkpoint_dirs: list[Path]) -> list[VariantData]:
    """Load log entries and configs for each variant checkpoint.

    For each checkpoint directory:
    1. Load metrics.jsonl via load_metrics_log
    2. Load model config from run_config.json or config.json
    3. Infer variant name from config or directory name
    4. Compute FLOP breakdown from config

    Missing or corrupt checkpoints are logged as warnings and skipped.

    Args:
        checkpoint_dirs: List of paths to checkpoint/run directories.

    Returns:
        List of successfully loaded VariantData objects.

    Validates: Requirements 9.1, 9.6, 13.7
    """
    variants: list[VariantData] = []

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            logger.warning(
                f"Checkpoint directory does not exist, skipping: {checkpoint_dir}"
            )
            continue

        if not checkpoint_dir.is_dir():
            logger.warning(
                f"Path is not a directory, skipping: {checkpoint_dir}"
            )
            continue

        # Load metrics.jsonl
        try:
            log_entries = load_metrics_log(checkpoint_dir)
        except FileNotFoundError:
            logger.warning(
                f"metrics.jsonl not found in {checkpoint_dir}, skipping variant"
            )
            continue
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Failed to load metrics.jsonl from {checkpoint_dir}: {e}, skipping variant"
            )
            continue

        # Load config
        config = _load_config_from_dir(checkpoint_dir)

        # Infer variant name
        if config is not None and hasattr(config, "variant") and config.variant:
            variant_name = config.variant
        else:
            variant_name = _infer_variant_name(checkpoint_dir)

        # If no config found from files, try to infer from variant name
        if config is None:
            config = _config_from_variant_name(variant_name)
            if config is not None:
                logger.warning(
                    f"No config file found in {checkpoint_dir}, "
                    f"using default config for variant '{variant_name}'"
                )
            else:
                logger.warning(
                    f"No config file found and unable to infer config for "
                    f"variant '{variant_name}' in {checkpoint_dir}, skipping"
                )
                continue

        # Compute FLOP breakdown
        try:
            flop_breakdown = compute_step_flops(config)
        except Exception as e:
            logger.warning(
                f"Failed to compute FLOPs for {variant_name} in {checkpoint_dir}: {e}"
            )
            flop_breakdown = None

        variant_data = VariantData(
            name=variant_name,
            checkpoint_dir=checkpoint_dir,
            log_entries=log_entries,
            config=config,
            metrics=None,  # Computed later by metrics module
            flop_breakdown=flop_breakdown,
        )
        variants.append(variant_data)

    return variants


def aggregate_across_seeds(
    seed_results: list[dict],
) -> dict[str, tuple[float, float]]:
    """Compute mean ± std across seeds for each metric.

    Args:
        seed_results: List of dicts, each mapping metric names to float values.
            Each dict represents results from one seed run.

    Returns:
        Dict mapping metric name to (mean, std) tuples.
        If fewer than 3 seeds are provided, std is reported as NaN
        to flag lack of statistical confidence.

    Validates: Requirements 14.1, 14.2, 14.3, 14.4
    """
    if not seed_results:
        return {}

    # Collect all metric keys across all seed results
    all_keys: set[str] = set()
    for result in seed_results:
        all_keys.update(result.keys())

    n_seeds = len(seed_results)
    aggregated: dict[str, tuple[float, float]] = {}

    for key in sorted(all_keys):
        # Gather values for this metric across seeds (skip missing)
        values = [result[key] for result in seed_results if key in result]

        if not values:
            continue

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))

        if n_seeds >= 3:
            # Sufficient seeds for statistical confidence
            std = float(np.std(arr, ddof=1))  # sample std (unbiased)
        else:
            # Fewer than 3 seeds: flag as lacking confidence
            std = float("nan")
            logger.warning(
                "Metric '%s' has only %d seed(s) — insufficient for "
                "statistical confidence. Std reported as NaN.",
                key,
                n_seeds,
            )

        aggregated[key] = (mean, std)

    return aggregated



def _get_x_metric_value(variant: VariantData, x_metric: str) -> float | None:
    """Extract the x-axis metric value for a variant.

    Args:
        variant: The variant data.
        x_metric: One of "flops", "wallclock", "peak_memory".

    Returns:
        The metric value, or None if unavailable.
    """
    if x_metric == "flops":
        if variant.flop_breakdown is not None:
            return float(variant.flop_breakdown.total)
        # Fallback: compute from config
        return float(compute_step_flops(variant.config).total)

    elif x_metric == "wallclock":
        # Use max elapsed_time from log_entries
        if not variant.log_entries:
            return None
        times = [
            entry.get("elapsed_time", 0.0)
            for entry in variant.log_entries
            if entry.get("elapsed_time") is not None
        ]
        return max(times) if times else None

    elif x_metric == "peak_memory":
        # Use max peak_memory_mb from log_entries
        if not variant.log_entries:
            return None
        memories = [
            entry.get("peak_memory_mb")
            for entry in variant.log_entries
            if entry.get("peak_memory_mb") is not None
        ]
        return max(memories) if memories else None

    else:
        raise ValueError(
            f"Unsupported x_metric: {x_metric!r}. "
            f"Supported: 'flops', 'wallclock', 'peak_memory'"
        )


def _get_y_metric_value(variant: VariantData, y_metric: str) -> float | None:
    """Extract the y-axis metric value for a variant.

    Args:
        variant: The variant data.
        y_metric: Currently only "val_loss" is supported.

    Returns:
        The metric value, or None if unavailable.
    """
    if y_metric == "val_loss":
        # Try metrics result first
        if variant.metrics is not None:
            return variant.metrics.val_loss
        # Fallback: get final val_loss from log_entries
        if not variant.log_entries:
            return None
        for entry in reversed(variant.log_entries):
            val_loss = entry.get("val_loss")
            if val_loss is not None:
                return float(val_loss)
        return None
    else:
        raise ValueError(
            f"Unsupported y_metric: {y_metric!r}. Supported: 'val_loss'"
        )


def compute_pareto_front(
    variants: list[VariantData],
    x_metric: str = "flops",
    y_metric: str = "val_loss",
) -> list[str]:
    """Identify Pareto-optimal variants (non-dominated set).

    A variant A dominates variant B if A is strictly better on both objectives
    (lower x_metric AND lower y_metric). The Pareto front is the set of
    variants not dominated by any other variant.

    Special cases:
    - Single variant is always Pareto-optimal.
    - When all variants have the same x_metric value, returns only the one
      with the best (lowest) y_metric.

    Supported objective pairs:
    - (flops, val_loss)
    - (wallclock, val_loss)
    - (peak_memory, val_loss)

    Args:
        variants: List of VariantData with populated log_entries and/or metrics.
        x_metric: X-axis metric — "flops", "wallclock", or "peak_memory".
        y_metric: Y-axis metric — currently "val_loss".

    Returns:
        List of variant names on the Pareto front.

    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
    """
    if not variants:
        return []

    # Single variant is always Pareto-optimal
    if len(variants) == 1:
        return [variants[0].name]

    # Extract (x, y) values for each variant, skip those with missing data
    points: list[tuple[str, float, float]] = []
    for v in variants:
        x_val = _get_x_metric_value(v, x_metric)
        y_val = _get_y_metric_value(v, y_metric)
        if x_val is None or y_val is None:
            logger.warning(
                "Variant %r missing data for (%s, %s), excluded from Pareto computation.",
                v.name,
                x_metric,
                y_metric,
            )
            continue
        points.append((v.name, x_val, y_val))

    if not points:
        return []

    if len(points) == 1:
        return [points[0][0]]

    # Check if all x_metric values are the same
    x_values = [p[1] for p in points]
    if all(x == x_values[0] for x in x_values):
        # All same x — return only the variant with best (lowest) y_metric
        best = min(points, key=lambda p: p[2])
        return [best[0]]

    # General Pareto front computation
    # A point is non-dominated if no other point is strictly better on BOTH objectives
    # (lower is better for both x and y)
    pareto_names: list[str] = []
    for i, (name_i, x_i, y_i) in enumerate(points):
        dominated = False
        for j, (name_j, x_j, y_j) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j is <= on both and strictly < on at least one
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            pareto_names.append(name_i)

    return pareto_names


def _estimate_parameter_count(config: ModelConfig) -> int:
    """Estimate total trainable parameter count from model config.

    Accounts for:
    - Token embeddings: vocab_size × d_model
    - Position embeddings (if learned): seq_len × d_model
    - Per-layer:
      - QKV projections: 3 × d_model × d_model (or adjusted for GQA)
      - Output projection: d_model × d_model
      - FFN: depends on ffn_type (standard: 2 × d_model × d_ff, swiglu: 3 × d_model × d_ff)
      - Layer norms: 2 × d_model (weight only, or weight + bias)
    - Final layer norm: d_model
    - Output head: vocab_size × d_model (unless tied with embeddings)

    Args:
        config: Model configuration.

    Returns:
        Estimated total trainable parameter count.
    """
    d_model = config.d_model
    n_layer = config.n_layer
    vocab_size = config.vocab_size
    d_ff = config.d_ff

    # Token embeddings
    params = vocab_size * d_model

    # Position embeddings (only for learned position encoding)
    if config.position_encoding == "learned":
        params += config.seq_len * d_model

    # Per-layer parameters
    for _ in range(n_layer):
        # QKV projections
        if config.n_kv_head is not None and config.n_kv_head < config.n_head:
            # Grouped Query Attention: Q uses n_head, K/V use n_kv_head
            q_params = d_model * d_model  # Q projection
            kv_dim = config.n_kv_head * config.d_head
            kv_params = 2 * d_model * kv_dim  # K and V projections
            params += q_params + kv_params
        else:
            # Standard MHA: all 3 projections are d_model × d_model
            params += 3 * d_model * d_model

        # Output projection
        params += d_model * d_model

        # FFN
        if config.ffn_type == "swiglu":
            # gate + up + down: 3 × d_model × d_ff
            params += 3 * d_model * d_ff
        else:
            # up + down: 2 × d_model × d_ff
            params += 2 * d_model * d_ff

        # Layer norms (2 per layer: pre-attention and pre-FFN)
        if config.bias:
            params += 2 * 2 * d_model  # weight + bias for 2 norms
        else:
            params += 2 * d_model  # weight only for 2 norms

    # Final layer norm
    if config.bias:
        params += 2 * d_model
    else:
        params += d_model

    # Output head (unembedding)
    if not config.tie_embeddings:
        params += vocab_size * d_model

    # Bias terms for linear layers (if applicable)
    if config.bias:
        for _ in range(n_layer):
            # QKV bias
            if config.n_kv_head is not None and config.n_kv_head < config.n_head:
                kv_dim = config.n_kv_head * config.d_head
                params += d_model + 2 * kv_dim  # Q bias + K/V bias
            else:
                params += 3 * d_model
            # Output projection bias
            params += d_model
            # FFN biases
            if config.ffn_type == "swiglu":
                params += 2 * d_ff + d_model  # gate_bias + up_bias + down_bias
            else:
                params += d_ff + d_model  # up_bias + down_bias

    return params


def validate_parameter_parity(
    variants: list[VariantData],
    tolerance: float = 0.05,
) -> tuple[bool, dict[str, int]]:
    """Check that all variants have parameter counts within ±tolerance of mean.

    Computes the total trainable parameter count for each variant using the
    model config, then checks whether all counts fall within ±tolerance
    (default 5%) of the mean count.

    Args:
        variants: List of VariantData with populated config fields.
        tolerance: Maximum allowed fractional deviation from mean (default 0.05 = 5%).

    Returns:
        Tuple of (valid, param_counts) where:
        - valid: True if all variants are within ±tolerance of the mean.
        - param_counts: Dict mapping variant name to parameter count.

    Validates: Requirements 11.1, 11.2, 11.3, 11.4
    """
    if not variants:
        return (True, {})

    # Compute parameter counts for each variant
    param_counts: dict[str, int] = {}
    for v in variants:
        count = _estimate_parameter_count(v.config)
        param_counts[v.name] = count

    # Single variant is always valid
    if len(param_counts) == 1:
        return (True, param_counts)

    # Compute mean
    counts = list(param_counts.values())
    mean_count = float(np.mean(counts))

    if mean_count == 0:
        # All zeros — trivially valid
        return (True, param_counts)

    # Check each variant is within ±tolerance of mean
    valid = all(
        abs(count - mean_count) / mean_count <= tolerance for count in counts
    )

    return (valid, param_counts)


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------


def _interpolate_val_loss(
    entries: list[dict], budget_value: float, budget_key: str
) -> float | None:
    """Interpolate val_loss at a target budget value using linear interpolation.

    Only entries with non-None val_loss are considered. If the budget_value
    is exactly at a logged point, that val_loss is returned directly. If it
    falls between two logged points, linear interpolation is used. If it's
    outside the range of available data, None is returned.

    Args:
        entries: Log entries (must contain budget_key and val_loss fields).
        budget_value: The target value of the budget metric to interpolate at.
        budget_key: The dict key to use as the x-axis (e.g. "tokens_seen",
            "elapsed_time", or "cumulative_flops").

    Returns:
        Interpolated val_loss, or None if interpolation is not possible.
    """
    # Filter to entries that have both the budget key and a non-None val_loss
    valid = [
        e for e in entries
        if e.get(budget_key) is not None and e.get("val_loss") is not None
    ]

    if not valid:
        return None

    # Sort by budget_key value for safe interpolation
    valid.sort(key=lambda e: e[budget_key])

    # Check bounds — budget must be within the range of available data
    if budget_value < valid[0][budget_key] or budget_value > valid[-1][budget_key]:
        return None

    # Exact match check
    for e in valid:
        if e[budget_key] == budget_value:
            return float(e["val_loss"])

    # Find bracketing entries and interpolate
    for i in range(len(valid) - 1):
        x0 = valid[i][budget_key]
        x1 = valid[i + 1][budget_key]
        if x0 <= budget_value <= x1:
            y0 = valid[i]["val_loss"]
            y1 = valid[i + 1]["val_loss"]
            # Linear interpolation
            if x1 == x0:
                return float(y0)
            t = (budget_value - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    return None


# ---------------------------------------------------------------------------
# Slicing functions
# ---------------------------------------------------------------------------


def slice_fixed_data(
    variants: list[VariantData],
    token_budget: int | None = None,
) -> dict[str, float]:
    """Compare val_loss at the same token budget across variants.

    If token_budget is None, uses the minimum of (max tokens_seen) across all
    variants as the budget — this ensures every variant can reach the budget.

    Uses linear interpolation when the exact token count doesn't appear in
    the log. Variants that cannot reach the budget are excluded with a
    logged warning.

    Args:
        variants: List of VariantData to compare.
        token_budget: Target token count. If None, uses min of max
            tokens_seen across all variants.

    Returns:
        Dict mapping variant name to interpolated val_loss at the token budget.

    Validates: Requirements 9.1, 9.5, 9.6
    """
    if not variants:
        return {}

    # Determine token budget
    if token_budget is None:
        max_tokens_per_variant: list[int] = []
        for v in variants:
            tokens_values = [
                e["tokens_seen"] for e in v.log_entries
                if e.get("tokens_seen") is not None
            ]
            if tokens_values:
                max_tokens_per_variant.append(max(tokens_values))
        if not max_tokens_per_variant:
            return {}
        token_budget = min(max_tokens_per_variant)

    results: dict[str, float] = {}
    for v in variants:
        val_loss = _interpolate_val_loss(v.log_entries, token_budget, "tokens_seen")
        if val_loss is not None:
            results[v.name] = val_loss
        else:
            logger.warning(
                "Variant '%s' does not extend to token budget %d; "
                "excluding from fixed-data comparison.",
                v.name,
                token_budget,
            )

    return results


def slice_fixed_wallclock(
    variants: list[VariantData],
    time_fractions: list[float] | None = None,
) -> dict[str, dict[float, float]]:
    """Compare val_loss at same wall-clock budgets across variants.

    Determines a dynamic wall-clock budget as min(total_elapsed_time) across
    all variants, then returns val_loss at each fraction of that budget.

    Args:
        variants: List of VariantData to compare.
        time_fractions: Fractions of the dynamic budget at which to measure.
            Defaults to [0.25, 0.50, 0.75, 1.00].

    Returns:
        Dict mapping variant name to a dict of {fraction → val_loss}.
        Variants that cannot provide val_loss at a given fraction are
        excluded from that fraction with a logged warning.

    Validates: Requirements 9.2, 9.3, 9.5, 9.6
    """
    if time_fractions is None:
        time_fractions = [0.25, 0.50, 0.75, 1.00]

    if not variants:
        return {}

    # Determine dynamic wall-clock budget = min(max elapsed_time) across variants
    max_times: list[float] = []
    for v in variants:
        time_values = [
            e["elapsed_time"] for e in v.log_entries
            if e.get("elapsed_time") is not None
        ]
        if time_values:
            max_times.append(max(time_values))

    if not max_times:
        return {}

    dynamic_budget = min(max_times)

    results: dict[str, dict[float, float]] = {}
    for v in variants:
        fraction_results: dict[float, float] = {}
        for frac in time_fractions:
            target_time = dynamic_budget * frac
            val_loss = _interpolate_val_loss(
                v.log_entries, target_time, "elapsed_time"
            )
            if val_loss is not None:
                fraction_results[frac] = val_loss
            else:
                logger.warning(
                    "Variant '%s' cannot provide val_loss at %.0f%% of "
                    "wall-clock budget (%.1fs); excluding from this fraction.",
                    v.name,
                    frac * 100,
                    target_time,
                )
        if fraction_results:
            results[v.name] = fraction_results

    return results


def slice_fixed_flops(
    variants: list[VariantData],
    flop_budget: int | None = None,
) -> dict[str, float]:
    """Compare val_loss at the same cumulative FLOP budget across variants.

    Cumulative FLOPs for a log entry are computed as:
        step_flops (from variant's flop_breakdown or computed from config) × step_number

    If flop_budget is None, uses the minimum of (max cumulative FLOPs) across
    all variants as the budget.

    Args:
        variants: List of VariantData to compare.
        flop_budget: Target cumulative FLOPs. If None, uses min of max
            cumulative FLOPs across variants.

    Returns:
        Dict mapping variant name to interpolated val_loss at the FLOP budget.

    Validates: Requirements 9.4, 9.5, 9.6
    """
    if not variants:
        return {}

    # Compute per-step FLOPs for each variant and annotate entries with cumulative_flops
    variant_entries: dict[str, list[dict]] = {}
    max_flops_per_variant: list[int] = []

    for v in variants:
        # Get per-step FLOP count
        if v.flop_breakdown is not None:
            step_flops = v.flop_breakdown.total
        else:
            step_flops = compute_step_flops(v.config).total

        # Annotate each entry with cumulative_flops = step_flops × step
        annotated: list[dict] = []
        for e in v.log_entries:
            step = e.get("step")
            if step is not None:
                entry_copy = dict(e)
                entry_copy["cumulative_flops"] = step_flops * step
                annotated.append(entry_copy)

        variant_entries[v.name] = annotated

        # Track max cumulative flops for this variant
        if annotated:
            max_flops_per_variant.append(
                max(ae["cumulative_flops"] for ae in annotated)
            )

    if not max_flops_per_variant:
        return {}

    # Determine FLOP budget
    if flop_budget is None:
        flop_budget = min(max_flops_per_variant)

    results: dict[str, float] = {}
    for v in variants:
        entries = variant_entries.get(v.name, [])
        val_loss = _interpolate_val_loss(entries, flop_budget, "cumulative_flops")
        if val_loss is not None:
            results[v.name] = val_loss
        else:
            logger.warning(
                "Variant '%s' does not extend to FLOP budget %d; "
                "excluding from fixed-FLOPs comparison.",
                v.name,
                flop_budget,
            )

    return results
