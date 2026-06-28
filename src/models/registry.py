"""Variant registry — maps (variant_name, scale) to (model, config).

Central place for all variant definitions and scale dimensions.
Adding a new variant means adding one entry to the VARIANTS dict.
"""

from dataclasses import dataclass
from typing import Type

import torch.nn as nn

from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer
from src.models.modern_transformer import ModernTransformer


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
    ),
}


def build(variant_name: str, scale: str) -> tuple[nn.Module, ModelConfig]:
    """Construct a model and its config from variant name and scale tier.

    Args:
        variant_name: One of the registered variant keys (e.g., "vanilla", "modern").
        scale: One of the registered scale keys ("debug", "main", "stretch").

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
    )

    model = spec.model_class(config)
    return model, config
