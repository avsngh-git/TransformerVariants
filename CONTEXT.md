# Domain Context — Transformer Variant Lab

The ubiquitous language for this project. Every term below has a single canonical meaning. Use these terms consistently in code, docs, and discussion.

---

## Core Concepts

### Model

The decoder-only Transformer architecture. The shared structural skeleton that all variants build upon: token embedding → N transformer blocks → output head. The "model" is the family; variants are specific instantiations.

### Variant

A coherent design philosophy applied to the Model. Defined by a published recipe (e.g., GPT-2, LLaMA) that may swap architectural components, introduce new mechanisms, add compute optimizations, or any combination. Each variant is identified by an ID (V0–V5).

A variant is NOT defined by which "slots" it fills — it may introduce entirely new concepts that don't map to prior variants.

| ID | Name | Design Philosophy | Base |
|----|------|-------------------|------|
| V0 | Vanilla Transformer | GPT-2 recipe | — |
| V1 | Modern Baseline | LLaMA recipe | — |
| V2 | ALiBi | Swap RoPE → ALiBi (linear position biases) | V1 |
| V3 | GQA | Swap independent heads → grouped-query attention | V1 |
| V4 | SWA (Sliding Window Attention) | Fixed-proportion sliding window, no global tokens | V1 |
| V5 | Linear (Linformer) | Low-rank K/V projection, O(n) attention complexity | V1 |

### Sub-variant

A single component swap within a variant's recipe, with everything else unchanged. Example: V0-GELU is a sub-variant of V0 where only the activation function changes (ReLU → GELU).

Known sub-variants:
- **V4-interleaved** — alternates local (SWA) and global (full attention) layers, Gemma2-style. Even layers attend to the full sequence; odd layers use window_size=W. Isolates the question: "does periodic full-context access recover information lost by windowing?"

### Architectural Component

A module that changes the mathematical behavior of the model. Swapping one produces different outputs given the same inputs. Components live in named slots but variants may introduce new slots.

Known component types:
- **Position encoding** — how position information enters the model (Learned, Sinusoidal, RoPE, ALiBi)
- **Normalization** — how activations are scaled (LayerNorm, RMSNorm)
- **FFN activation** — the nonlinearity in the feed-forward block (ReLU, GELU, SwiGLU)
- **Attention pattern** — which tokens can attend to which (full causal, sliding window, linear approximation)
- **Attention structure** — how heads share parameters (independent heads, grouped-query, multi-query)

### Compute Optimization

A technique that changes speed or memory usage without altering the model's mathematical behavior. Same inputs → same outputs, faster or cheaper.

Examples: Flash Attention, KV-cache, torch.compile, mixed precision.

A variant's recipe may include compute optimizations alongside architectural components.

### Attention Backend

A config-level choice of kernel dispatch for the attention computation. Does not change mathematical behavior — only speed and memory characteristics. Controlled via `ModelConfig.attention_backend`.

| Backend | Implementation | Notes |
|---------|---------------|-------|
| sdpa | `F.scaled_dot_product_attention` | PyTorch built-in, auto-selects best kernel |
| flash_attn | `flash_attn` library (Dao AI Lab) | Explicit Flash Attention 2 kernel, supports ALiBi slopes natively |

Within a controlled experiment, all compared variants use the same backend.

### Scale

A size configuration that determines the model's width and depth. Applies to any variant.

| Scale | Role | Used in comparisons? |
|-------|------|---------------------|
| debug | Fast iteration, correctness testing | No |
| main | Primary benchmark (~51M params) | Yes |
| stretch | Near-memory-limit exploration (~124M params) | Yes |

Parameter counts across variants at the same scale need not be identical — the ±5% tolerance rule applies. Dimensions (d_model, n_layer, n_head) are fixed per scale; parameter count is whatever falls out of the variant's architecture.

### Run

A single execution of the training script. Produces one checkpoint directory, one training log. Identified by variant + scale + seed + timestamp.

### Experiment

A controlled comparison: multiple runs across variants and/or seeds under the same protocol (same data, token budget, optimizer, batch size, precision). Governed by the experiment contract.

### Sliding Window Attention (SWA)

An attention pattern where each query token attends only to the W tokens immediately preceding it (plus itself), rather than the full sequence. W is the window size. In this project, W is a fixed proportion of seq_len (W = seq_len // 4), constant across all layers. SWA applies during training only — generation uses the full KV cache.

The flash_attn kernel supports SWA natively via its `window_size` parameter. No custom masks or block-sparse patterns are needed.

SWA isolates the variable "attention span" while keeping all other components (RoPE, projections, FFN, normalization) identical to V1.

### Linear Attention (Linformer)

An attention mechanism that approximates full self-attention by projecting the Key and Value matrices from length T down to a fixed rank r=64, giving O(n·r) complexity instead of O(n²). Each transformer layer has two shared projection matrices E and F of shape `(seq_len, r)` — shared across all heads in that layer.

RoPE is applied to Q and K before the low-rank projection. The attention computation is: `softmax(Q · (EK)^T / sqrt(d_head)) · (FV)`.

V5 uses `ModernTransformer` as its model shell (RMSNorm, SwiGLU) and a standalone `LinearAttention(nn.Module)` class. Because the E and F projection matrices are tied to a fixed `seq_len`, autoregressive KV-cache generation is not supported — V5 is a training-comparison-only variant.

### Shard

A fixed-size chunk of consecutive tokenized data stored as a binary file (uint16). The data pipeline splits the full corpus into shards for memory-mapped loading. Shards have no semantic boundary alignment — they're purely a storage/IO mechanism.

---

## Training & Data Concepts

### Residual Stream

The main data pathway through the Transformer. Each block *adds* its output to this stream rather than replacing it: `x = x + block_output`. The stream accumulates information layer by layer. All components (attention, FFN) read from and write to the residual stream.

### Teacher Forcing

The training regime where the model receives the ground-truth previous tokens as input at every position, rather than its own predictions. All positions are processed in parallel. This is why KV-cache is not used during training — there's no sequential generation.

### Token Budget

The total number of tokens a run consumes during training. A fixed token budget is one of the controlled variables in an experiment — all compared runs see the same number of tokens regardless of how fast they process them.

### Next-Token Prediction

The training objective. Given tokens at positions 0..t, predict the token at position t+1. The loss is cross-entropy between the model's predicted distribution and the actual next token. Every position in a sequence provides a training signal simultaneously (via teacher forcing).

### Mixed Precision

Training with lower-precision numerics (bfloat16) for speed while keeping critical state (optimizer moments, master weights) in float32 for stability. In this project: bfloat16 for forward/backward matmuls, float32 for reductions and optimizer state.

### Gradient Accumulation

Simulating a larger effective batch size by accumulating gradients over multiple micro-batches before taking an optimizer step. Effective batch = micro_batch_size × gradient_accumulation_steps × seq_len tokens.

---

## Invariants

- All variants in a comparison share: same tokenized data in same order, same token budget, same optimizer hyperparameters, same effective batch size, same precision (bf16).
- Parameter counts across compared variants must be within ±5%. Exception: variants whose mechanism inherently reduces parameters (e.g., GQA) may exceed this tolerance — the difference is noted in results rather than compensated.
- Results are reported over 3+ random seeds.
- Comparisons happen at main and stretch scale only. Debug scale is never part of formal comparisons.

---

## Open Questions

- Evaluation framework: what the primary comparison axis is (fixed compute vs fixed data vs Pareto) — to be decided after all variants are implemented.
- ALiBi extrapolation experiment: train-short/infer-long capability. Deferred until after the controlled comparison at fixed seq_len is complete.
- KV-Cache unification: ModernAttention (SDPA) uses a concat-based 2-tuple cache; FlashAttentionBase uses a pre-allocated 3-tuple cache. These are incompatible seam contracts but `generate.py` currently works by passing cache opaquely. Unifying into a single `KVCache` abstraction is deferred until all variants (V4, V5) are implemented — it's a design smell, not a blocking bug today.