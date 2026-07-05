# ADR 0006: MoE Design Decisions

## Status: Accepted

## Context

Adding a Mixture of Experts variant (V6) to the Transformer Variant Lab. The MoE variant demonstrates conditional computation — a fundamentally different scaling approach from the dense variants (V0–V5).

## Decisions

### 1. Architecture: Mixtral-style, 8 experts, top-2 routing

- 8 SwiGLU expert FFNs per MoE layer
- Top-2 gating: each token is processed by exactly 2 experts, outputs weighted by router softmax
- Fair comparison: match **active parameters** (~51M at main scale) against V1 dense baseline
- Total parameters will be much larger (~170M main, ~520M stretch) — all fit in 24GB L4

### 2. Three sub-variants via per-layer configuration

| Variant | Registry name | MoE layers |
|---------|--------------|------------|
| V6-full | `moe` | All layers |
| V6-interleaved | `moe_interleaved` | Odd layers only (even = dense) |
| V6-deep | `moe_deep` | Last half of layers only |

All three use the same `MoEFeedForward` module — only the `per_layer_config_builder` differs.

### 3. Load balancing: Switch-style aux loss + z-loss

- Aux loss: `α * Σ(f_i * P_i)` with α=0.01
- Z-loss: `β * mean(log(Σ exp(logits))²)` with β=0.001
- No capacity-based token dropping (single-GPU, not needed)

### 4. Integration pattern: config-driven FFN selection + internal aux loss accumulation

- `ModelConfig` gains fields: `num_experts`, `moe_top_k`, `aux_loss_alpha`, `z_loss_beta`
- `ModernTransformerBlock` checks `config.num_experts`: if set, uses `MoEFeedForward`; otherwise `SwiGLUFeedForward`
- `MoEFeedForward` stores aux loss in `self._aux_loss` during forward pass
- `ModernTransformer.get_aux_loss()` sums across all MoE layers, clears buffers
- Trainer unconditionally calls `model.get_aux_loss()` — dense models return 0

### 5. No new model shell

`ModernTransformer` serves all MoE variants. The only new module is `src/models/moe_ffn.py`. MoE is a configuration of the existing shell, not a separate class.

### 6. torch.compile disabled for MoE

MoE routing involves dynamic scatter/gather that may fight with `torch.compile`. Since this is a training-comparison project, we simply skip compilation for MoE variants. The `--compile` flag is already optional.

### 7. Generation supported

KV-cache generation works naturally because MoE only changes the FFN — attention is identical to V1. The router runs per-token during generation the same as during training.

### 8. Weight initialization: standard (no MoE-specific init)

Existing `_init_weights` handles MoE correctly via name-suffix matching:
- Router (`nn.Linear`): standard N(0, 0.02)
- Expert `w_down` layers: residual-scaled N(0, 0.02/√(2n))

### 9. Routing data capture for evaluation probes

`MoEFeedForward` has a `record_routing: bool` flag. When True, stores `(expert_indices, expert_weights)` per forward pass. `model.get_routing_data()` returns and clears the buffer.

## Consequences

- 4 new fields on `ModelConfig` (all default to None/0, backward compatible)
- One new method on `ModernTransformer` (`get_aux_loss`)
- One new conditional in `ModernTransformerBlock.__init__` (FFN selection)
- One line change in `Trainer._training_step` (add aux loss to total)
- One new module: `src/models/moe_ffn.py`
- Three new config files: `configs/model/moe.yaml`, `moe_interleaved.yaml`, `moe_deep.yaml`
- Three new registry entries with `per_layer_config_builder` for interleaved/deep
- Five new MoE-specific evaluation probes (expert utilization, router entropy, token affinity, pair overlap, seed stability)
