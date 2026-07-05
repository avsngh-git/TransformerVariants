"""Variant registry — maps (variant_name, scale) to (model, config).

Central place for all variant definitions and scale dimensions.
Adding a new variant means adding one entry to the VARIANTS dict with its
own config_overrides callable — no edits to build() required.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Type

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.attention import CausalSelfAttention
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer
from src.models.modern_attention import ModernAttention
from src.models.flash_attention import FlashAttention
from src.models.alibi_attention import ALiBiAttention
from src.models.gqa_attention import GQAAttention
from src.models.linear_attention import LinformerAttention


SCALES: dict[str, dict[str, int]] = {
    "debug":   {"n_layer": 4,  "d_model": 256, "n_head": 4,  "seq_len": 512},
    "main":    {"n_layer": 8,  "d_model": 512, "n_head": 8,  "seq_len": 1024},
    "stretch": {"n_layer": 12, "d_model": 768, "n_head": 12, "seq_len": 1024},
}


# --- Config override functions (variant-specific construction knowledge) ---


def _identity_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """Default: no overrides needed."""
    return config


def _swa_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """SWA: set window_size and force flash_attn backend."""
    window_size = dims["seq_len"] // 4
    if window_size < 1:
        raise ValueError(
            f"seq_len ({dims['seq_len']}) is too small for sliding window attention. "
            f"Minimum seq_len is 4 (window_size = seq_len // 4 must be >= 1)."
        )
    config.window_size = window_size
    config.attention_backend = "flash_attn"
    return config


def _gqa_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """GQA: set n_kv_head to n_head // 4."""
    config.n_kv_head = dims["n_head"] // 4
    return config


def _linear_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """Linformer: set projection_rank."""
    config.projection_rank = 64
    return config


def _swa_interleaved_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """SWA interleaved: set window_size and force flash_attn backend.

    The per-layer configs are handled by per_layer_config_builder, but
    the base config still needs the backend override for attention class selection.
    Window size is NOT set on the base config (model-wide components don't need it).
    """
    config.attention_backend = "flash_attn"
    return config


def _moe_overrides(config: ModelConfig, dims: dict) -> ModelConfig:
    """MoE: set num_experts and routing hyperparameters on base config."""
    config.num_experts = 8
    config.moe_top_k = 2
    config.aux_loss_alpha = 0.01
    config.z_loss_beta = 0.001
    return config


# --- Per-layer config builders ---


def _swa_interleaved_per_layer(config: ModelConfig, dims: dict) -> list[ModelConfig]:
    """Build per-layer configs alternating full attention (even) and SWA (odd)."""
    window_size = dims["seq_len"] // 4
    return [
        replace(config, window_size=None if i % 2 == 0 else window_size)
        for i in range(dims["n_layer"])
    ]


def _moe_interleaved_per_layer(config: ModelConfig, dims: dict) -> list[ModelConfig]:
    """Odd layers get MoE, even layers stay dense."""
    return [
        replace(config, num_experts=8 if i % 2 == 1 else None)
        for i in range(dims["n_layer"])
    ]


def _moe_deep_per_layer(config: ModelConfig, dims: dict) -> list[ModelConfig]:
    """Second half of layers get MoE, first half dense.

    If n_layer is odd, the extra middle layer goes to the dense half.
    """
    split = dims["n_layer"] // 2  # dense layers: [0, split), MoE: [split, n_layer)
    return [
        replace(config, num_experts=8 if i >= split else None)
        for i in range(dims["n_layer"])
    ]


# --- VariantSpec dataclass ---


@dataclass(frozen=True)
class VariantSpec:
    """Everything needed to construct a variant beyond scale dimensions.

    Each variant's construction knowledge lives in its config_overrides callable
    rather than in conditional branches inside build(). Adding a new variant
    means providing a VariantSpec with its own overrides — no edits to build().

    Attributes:
        model_class: The transformer model class to instantiate.
        variant: Variant identifier string (e.g., "vanilla", "modern").
        norm_type: Normalization type ("layernorm" or "rmsnorm").
        position_encoding: Position encoding method ("learned", "rope", "alibi", "none").
        ffn_type: FFN architecture ("standard" or "swiglu").
        attention_type: Attention mechanism identifier.
        attention_class: Default attention module class for this variant.
        default_activation: Default FFN activation.
        requires_bf16: Whether the variant requires bf16 casting.
        default_steps: Default training steps per scale.
        config_overrides: Callable that applies variant-specific config mutations.
            Signature: (config: ModelConfig, dims: dict) -> ModelConfig.
        per_layer_config_builder: Optional callable that produces per-layer configs.
            Only needed for variants with heterogeneous layer configurations.
            Signature: (config: ModelConfig, dims: dict) -> list[ModelConfig].
    """

    model_class: Type[nn.Module]
    variant: str
    norm_type: str
    position_encoding: str
    ffn_type: str
    attention_type: str
    attention_class: Type[nn.Module] = ModernAttention
    default_activation: str = "relu"
    requires_bf16: bool = False
    default_steps: dict[str, int] = field(default_factory=lambda: {
        "debug": 2000, "main": 5000, "stretch": 5000
    })
    config_overrides: Callable[[ModelConfig, dict], ModelConfig] = _identity_overrides
    per_layer_config_builder: Callable[[ModelConfig, dict], list[ModelConfig]] | None = None


VARIANTS: dict[str, VariantSpec] = {
    "vanilla": VariantSpec(
        model_class=VanillaTransformer,
        variant="vanilla",
        norm_type="layernorm",
        position_encoding="learned",
        ffn_type="standard",
        attention_type="full",
        attention_class=CausalSelfAttention,
        default_activation="relu",
        requires_bf16=False,
    ),
    "modern": VariantSpec(
        model_class=ModernTransformer,
        variant="modern",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_class=ModernAttention,
        default_activation="swiglu",
        requires_bf16=False,
    ),
    "alibi": VariantSpec(
        model_class=ModernTransformer,
        variant="alibi",
        norm_type="rmsnorm",
        position_encoding="alibi",
        ffn_type="swiglu",
        attention_type="flash_alibi",
        attention_class=ALiBiAttention,
        default_activation="swiglu",
        requires_bf16=True,
    ),
    "gqa": VariantSpec(
        model_class=ModernTransformer,
        variant="gqa",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_gqa",
        attention_class=GQAAttention,
        default_activation="swiglu",
        requires_bf16=True,
        config_overrides=_gqa_overrides,
    ),
    "swa": VariantSpec(
        model_class=ModernTransformer,
        variant="swa",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="sliding_window",
        attention_class=FlashAttention,
        default_activation="swiglu",
        requires_bf16=True,
        config_overrides=_swa_overrides,
    ),
    "swa_interleaved": VariantSpec(
        model_class=ModernTransformer,
        variant="swa_interleaved",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="sliding_window",
        attention_class=FlashAttention,
        default_activation="swiglu",
        requires_bf16=True,
        config_overrides=_swa_interleaved_overrides,
        per_layer_config_builder=_swa_interleaved_per_layer,
    ),
    "linear": VariantSpec(
        model_class=ModernTransformer,
        variant="linear",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="linear",
        attention_class=LinformerAttention,
        default_activation="swiglu",
        requires_bf16=False,
        config_overrides=_linear_overrides,
    ),
    "moe": VariantSpec(
        model_class=ModernTransformer,
        variant="moe",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_class=ModernAttention,
        default_activation="swiglu",
        requires_bf16=False,
        config_overrides=_moe_overrides,
    ),
    "moe_interleaved": VariantSpec(
        model_class=ModernTransformer,
        variant="moe_interleaved",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_class=ModernAttention,
        default_activation="swiglu",
        requires_bf16=False,
        config_overrides=_moe_overrides,
        per_layer_config_builder=_moe_interleaved_per_layer,
    ),
    "moe_deep": VariantSpec(
        model_class=ModernTransformer,
        variant="moe_deep",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_class=ModernAttention,
        default_activation="swiglu",
        requires_bf16=False,
        config_overrides=_moe_overrides,
        per_layer_config_builder=_moe_deep_per_layer,
    ),
}


def build(
    variant_name: str,
    scale: str,
    attention_backend: str | None = None,
    activation: str | None = None,
    dtype: str = "bfloat16",
    compile_model: bool = False,
) -> tuple[nn.Module, ModelConfig]:
    """Construct a model and its config from variant name and scale tier.

    This is now a generic dispatcher. Variant-specific config logic lives in
    each VariantSpec's config_overrides callable — adding a new variant requires
    no edits here.

    Args:
        variant_name: One of the registered variant keys (e.g., "vanilla", "modern").
        scale: One of the registered scale keys ("debug", "main", "stretch").
        attention_backend: Optional override for the attention kernel dispatch path.
            When "flash_attn", uses FlashAttention instead of the variant's default
            attention class. When None, uses the variant's default (ModernAttention
            for "modern" variant, which dispatches via PyTorch SDPA).
        activation: Optional override for the FFN activation function. When None,
            uses the variant's default_activation from its VariantSpec.
        dtype: Precision for model casting ("bfloat16", "float16", or "float32").
            Used to determine whether to cast model when bf16 is required.
        compile_model: Whether to apply torch.compile to the model.

    Returns:
        A tuple of (model_instance, config) where model is ready for training.

    Raises:
        ValueError: If variant_name or scale is not registered.
    """
    if variant_name not in VARIANTS:
        available = ", ".join(sorted(VARIANTS.keys()))
        raise ValueError(f"Unknown variant '{variant_name}'. Available variants: {available}")
    if scale not in SCALES:
        available = ", ".join(sorted(SCALES.keys()))
        raise ValueError(f"Unknown scale '{scale}'. Available scales: {available}")

    spec = VARIANTS[variant_name]
    dims = SCALES[scale]

    # Determine effective activation and backend
    effective_activation = activation if activation is not None else spec.default_activation
    effective_backend = attention_backend if attention_backend is not None else "sdpa"

    # Construct base config from scale dimensions and spec metadata
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
        attention_backend=effective_backend,
        activation=effective_activation,
    )

    # Apply variant-specific config overrides
    config = spec.config_overrides(config, dims)

    # Update effective_backend after overrides (SWA variants force flash_attn)
    effective_backend = config.attention_backend

    # Build per-layer configs if the variant needs them
    per_layer_configs = None
    if spec.per_layer_config_builder is not None:
        per_layer_configs = spec.per_layer_config_builder(config, dims)
        # Base config should NOT have window_size set for interleaved variants
        config.window_size = None

    # Attention class selection: override with FlashAttention when backend is "flash_attn"
    # Only override if the variant uses the default ModernAttention — specialized
    # attention classes (e.g., ALiBiAttention, GQAAttention) should not be replaced.
    if effective_backend == "flash_attn" and spec.attention_class == ModernAttention:
        attention_class = FlashAttention
    else:
        attention_class = spec.attention_class

    # Construct the model
    if per_layer_configs is not None:
        model = spec.model_class(config, attention_class=attention_class, per_layer_configs=per_layer_configs)
    else:
        model = spec.model_class(config, attention_class=attention_class)

    # bf16/fp16 casting: apply when variant requires it or flash_attn backend is used
    if spec.requires_bf16 or effective_backend == "flash_attn":
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_dtype = dtype_map.get(dtype, torch.bfloat16)
        if model_dtype in (torch.bfloat16, torch.float16):
            model = model.to(model_dtype)

    # torch.compile for kernel fusion and speedup
    if compile_model:
        moe_variants = {"moe", "moe_interleaved", "moe_deep"}
        if variant_name in moe_variants:
            pass  # Skip compile for MoE (dynamic routing conflicts with tracing)
        else:
            model = torch.compile(model)

    return model, config
