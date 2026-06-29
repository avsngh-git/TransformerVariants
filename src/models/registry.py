"""Variant registry — maps (variant_name, scale) to (model, config).

Central place for all variant definitions and scale dimensions.
Adding a new variant means adding one entry to the VARIANTS dict.
"""

import inspect
from dataclasses import dataclass
from typing import Type

import torch.nn as nn

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer
from src.models.modern_attention import ModernAttention
from src.models.flash_attention import FlashAttention
from src.models.alibi_attention import ALiBiAttention
from src.models.gqa_attention import GQAAttention


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


VARIANTS: dict[str, VariantSpec] = {
    "vanilla": VariantSpec(
        model_class=VanillaTransformer,
        variant="vanilla",
        norm_type="layernorm",
        position_encoding="learned",
        ffn_type="standard",
        attention_type="full",
    ),
    "modern": VariantSpec(
        model_class=ModernTransformer,
        variant="modern",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_class=ModernAttention,
    ),
    "alibi": VariantSpec(
        model_class=ModernTransformer,
        variant="alibi",
        norm_type="rmsnorm",
        position_encoding="alibi",
        ffn_type="swiglu",
        attention_type="flash_alibi",
        attention_class=ALiBiAttention,
    ),
    "gqa": VariantSpec(
        model_class=ModernTransformer,
        variant="gqa",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_gqa",
        attention_class=GQAAttention,
    ),
}


def build(variant_name: str, scale: str, attention_backend: str | None = None) -> tuple[nn.Module, ModelConfig]:
    """Construct a model and its config from variant name and scale tier.

    Args:
        variant_name: One of the registered variant keys (e.g., "vanilla", "modern").
        scale: One of the registered scale keys ("debug", "main", "stretch").
        attention_backend: Optional override for the attention kernel dispatch path.
            When "flash_attn", uses FlashAttention instead of the variant's default
            attention class. When None, uses the variant's default (ModernAttention
            for "modern" variant, which dispatches via PyTorch SDPA).

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
    )

    # Set n_kv_head for GQA variants (grouped-query attention needs fewer KV heads)
    if spec.attention_type == "flash_gqa":
        config.n_kv_head = dims["n_head"] // 4

    # Select attention class: override with FlashAttention when backend is "flash_attn"
    # Only override if the variant uses the default ModernAttention — specialized
    # attention classes (e.g., ALiBiAttention) should not be replaced.
    if effective_backend == "flash_attn" and spec.attention_class == ModernAttention:
        attention_class = FlashAttention
    else:
        attention_class = spec.attention_class

    sig = inspect.signature(spec.model_class.__init__)
    if "attention_class" in sig.parameters:
        model = spec.model_class(config, attention_class=attention_class)
    else:
        model = spec.model_class(config)
    return model, config
