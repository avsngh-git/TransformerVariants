"""ProbeTarget protocol and adapter for decoupling probes from model internals.

Defines a structural protocol that probes program against, plus a concrete
adapter that wraps VanillaTransformer/ModernTransformer instances. This seam
lets probes be tested with simple fakes (canned tensors, no GPU, no model)
and insulates them from model archaeology (hook registration, .blocks access).

Two adapters justify this seam: the real ModelProbeAdapter (production) and
test fakes (canned tensors for unit tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from src.models.config import ModelConfig


@dataclass
class ProbeInternals:
    """Bundled internal model state from a forward pass with instrumentation.

    Attributes:
        logits: Model output logits, shape (B, T, vocab_size).
        layer_outputs: Residual stream output per layer, each shape (B, T, d_model).
        attention_weights: Post-softmax attention weights per layer, or None for
            flash-attention variants where weights are never materialized.
            When present, each tensor has shape (B, n_head, T, key_dim).
    """

    logits: torch.Tensor
    layer_outputs: list[torch.Tensor]
    attention_weights: list[torch.Tensor] | None


@runtime_checkable
class ProbeTarget(Protocol):
    """Protocol for any object that probes can evaluate.

    Probes program against this interface rather than reaching into model
    internals. Two methods separate the lightweight path (logits only, for
    MQAR) from the heavy path (full internals, for stable rank / CKA / entropy).

    Attributes:
        config: ModelConfig providing metadata (seq_len, attention_type, etc.).
    """

    @property
    def config(self) -> ModelConfig:
        """Model configuration metadata."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits only.

        Used by MQAR probe which doesn't need layer internals.

        Args:
            x: Input token IDs, shape (B, T).

        Returns:
            Logits tensor, shape (B, T, vocab_size).
        """
        ...

    def forward_with_internals(self, x: torch.Tensor) -> ProbeInternals:
        """Run a forward pass capturing per-layer hidden states and attention weights.

        Used by stable rank, CKA, and attention entropy probes.

        Args:
            x: Input token IDs, shape (B, T).

        Returns:
            ProbeInternals with logits, layer_outputs, and optional attention_weights.
        """
        ...


class ModelProbeAdapter:
    """Wraps a VanillaTransformer or ModernTransformer as a ProbeTarget.

    Concentrates all hook registration logic in one place. Probes no longer
    need to know about model.blocks, attn_dropout hooks, or attention_type
    checks — the adapter handles it all.

    Args:
        model: A transformer model with a `.blocks` attribute (nn.ModuleList)
            and a `.config` attribute (ModelConfig).
        device: Device for computation.
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self._model = model
        self._device = device
        self._model.eval()

    @property
    def config(self) -> ModelConfig:
        """Model configuration metadata."""
        return self._model.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass, return logits only.

        Args:
            x: Input token IDs, shape (B, T).

        Returns:
            Logits tensor, shape (B, T, vocab_size).
        """
        x = x.to(self._device)
        with torch.no_grad():
            output = self._model(x)
            # Handle both (logits, loss, kv_cache) and (logits, kv_cache) patterns
            if isinstance(output, tuple):
                return output[0]
            return output

    def forward_with_internals(self, x: torch.Tensor) -> ProbeInternals:
        """Run forward pass with hooks capturing layer outputs and attention weights.

        Registers temporary forward hooks on each block (and optionally on
        attn_dropout for attention weights), runs the forward pass, then
        removes all hooks.

        Args:
            x: Input token IDs, shape (B, T).

        Returns:
            ProbeInternals with logits, layer_outputs, and attention_weights.
        """
        x = x.to(self._device)
        blocks = self._model.blocks
        n_layers = len(blocks)

        # Storage for captured states
        layer_outputs: list[torch.Tensor | None] = [None] * n_layers
        attn_weights: list[torch.Tensor | None] = [None] * n_layers

        # Determine if attention weights are accessible
        attention_type = self.config.attention_type
        capture_attn = attention_type in ("full", "linear")

        # --- Register hooks ---
        hooks: list[torch.utils.hooks.RemovableHook] = []

        # Layer output hooks
        def make_layer_hook(idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[idx] = output[0].detach()
                else:
                    layer_outputs[idx] = output.detach()
            return hook_fn

        for i, block in enumerate(blocks):
            h = block.register_forward_hook(make_layer_hook(i))
            hooks.append(h)

        # Attention weight hooks (only for non-flash variants)
        if capture_attn:
            def make_attn_hook(idx: int):
                def hook_fn(module, input, output):
                    # input[0] is the post-softmax attention weights
                    attn_weights[idx] = input[0].detach()
                return hook_fn

            for i, block in enumerate(blocks):
                attn_module = block.attn
                h = attn_module.attn_dropout.register_forward_hook(make_attn_hook(i))
                hooks.append(h)

        # --- Run forward pass ---
        try:
            with torch.no_grad():
                output = self._model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
        finally:
            # Always remove hooks
            for h in hooks:
                h.remove()

        # --- Assemble result ---
        final_layer_outputs = [lo for lo in layer_outputs if lo is not None]
        final_attn_weights: list[torch.Tensor] | None = None
        if capture_attn:
            collected = [aw for aw in attn_weights if aw is not None]
            final_attn_weights = collected if collected else None

        return ProbeInternals(
            logits=logits,
            layer_outputs=final_layer_outputs,
            attention_weights=final_attn_weights,
        )
