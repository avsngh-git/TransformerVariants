"""Variant registry — maps (variant_name, scale) to (model, config).

Central place for all variant definitions and scale dimensions.
Adding a new variant means adding one entry to the VARIANTS dict.
"""

from dataclasses import dataclass, field, replace
from typing import Type

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


@dataclass(frozen=True)
class VariantSpec:
    """Everything needed to construct a variant beyond scale dimensions."""
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

    Handles all model construction policy:
    - Activation override (uses override if provided, else spec.default_activation)
    - bf16/fp16 casting (when spec.requires_bf16 or flash_attn backend)
    - torch.compile (when requested)

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

    # Determine effective activation: use override if provided, else spec default
    effective_activation = activation if activation is not None else spec.default_activation

    # Determine effective attention backend
    effective_backend = attention_backend if attention_backend is not None else "sdpa"

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

    # Compute window_size for SWA variants (sliding_window attention_type)
    if spec.attention_type == "sliding_window":
        window_size = dims["seq_len"] // 4
        if window_size < 1:
            raise ValueError(
                f"seq_len ({dims['seq_len']}) is too small for sliding window attention. "
                f"Minimum seq_len is 4 (window_size = seq_len // 4 must be >= 1)."
            )
        config.window_size = window_size
        # SWA requires flash_attn kernel — set backend in config if not explicitly overridden
        if attention_backend is None:
            config.attention_backend = "flash_attn"
            effective_backend = "flash_attn"

    # Set n_kv_head for GQA variants (grouped-query attention needs fewer KV heads)
    if spec.attention_type == "flash_gqa":
        config.n_kv_head = dims["n_head"] // 4

    # Set projection_rank for Linformer variants (linear attention needs projection rank)
    if spec.attention_type == "linear":
        config.projection_rank = 64

    # Build per-layer configs for swa_interleaved variant
    # Let the existing sliding_window block run first (it sets window_size and effective_backend),
    # then override with per-layer logic for the interleaved pattern.
    per_layer_configs = None

    if variant_name == "swa_interleaved":
        window_size = dims["seq_len"] // 4
        per_layer_configs = [
            replace(config, window_size=None if i % 2 == 0 else window_size)
            for i in range(dims["n_layer"])
        ]
        # Base config should NOT have window_size set — model-wide components don't need it
        config.window_size = None

    # Select attention class: override with FlashAttention when backend is "flash_attn"
    # Only override if the variant uses the default ModernAttention — specialized
    # attention classes (e.g., ALiBiAttention) should not be replaced.
    if effective_backend == "flash_attn" and spec.attention_class == ModernAttention:
        attention_class = FlashAttention
    else:
        attention_class = spec.attention_class

    if per_layer_configs is not None:
        model = spec.model_class(config, attention_class=attention_class, per_layer_configs=per_layer_configs)
    else:
        model = spec.model_class(config, attention_class=attention_class)

    # bf16/fp16 casting: apply when variant requires it or flash_attn backend is used,
    # AND the requested dtype is bfloat16 or float16
    if spec.requires_bf16 or effective_backend == "flash_attn":
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_dtype = dtype_map.get(dtype, torch.bfloat16)
        if model_dtype in (torch.bfloat16, torch.float16):
            model = model.to(model_dtype)

    # torch.compile for kernel fusion and speedup
    if compile_model:
        model = torch.compile(model)

    return model, config
