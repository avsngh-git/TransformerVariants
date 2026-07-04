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
- Results are reported as mean ± standard deviation over 3+ random seeds. Additional seeds may be added selectively for pairs with overlapping intervals.
- Comparisons happen at main and stretch scale only. Debug scale is never part of formal comparisons.
- All three comparison axes (fixed-data, fixed-compute/wall-clock, fixed-compute/FLOPs) are reported for every experiment.

---

## Evaluation Concepts

### Comparison Axis

A controlled perspective from which variants are compared. Three axes are used simultaneously:

| Axis | Control variable | Comparison metric |
|------|-----------------|-------------------|
| Fixed-data | Same token budget across all variants | Final val loss at budget exhaustion |
| Fixed-compute (wall-clock) | Same wall-clock duration | Val loss at the time threshold |
| Fixed-compute (FLOPs) | Same cumulative FLOPs | Val loss at the FLOP threshold |

Fixed-compute comparisons use post-hoc slicing of training logs — no separate training runs are needed. The wall-clock budget is determined dynamically as the minimum total training time across all variants at a given scale.

### Component-Level FLOP Accounting

Per-step FLOP estimation that sums actual operations per layer: projection matmuls, attention score computation (variant-dependent), FFN matmuls. Not the coarse 6NT approximation — captures that V4-SWA does O(T·W·d) attention vs V1's O(T²·d).

### Model FLOPs Utilization (MFU)

The ratio of achieved FLOPs (from component-level accounting) to the hardware's theoretical peak (L4: 242 TFLOPS BF16). Distinguishes compute-bound from memory-bandwidth-bound regimes.

### Stable Rank

A measure of the effective dimensionality of a hidden state matrix H: srank(H) = ||H||²_F / ||H||²_2. A low stable rank in deep layers indicates representation collapse. Measured at the final checkpoint, averaged over ~100 validation batches.

### Multi-Query Associative Recall (MQAR)

A synthetic evaluation probe that tests a model's ability to bind and recall key-value associations across context. Sequences contain planted token patterns; the metric is whether the model assigns high probability to the correct recall token. Isolates retrieval capacity — variants with limited context (V4-SWA, V5-Linformer) should show measurable degradation.

### ICL Loss Decay Exponent (α)

The power-law exponent of per-position validation loss: L(t) = A·t^(-α) + C. A steep α indicates effective context utilization; flattening at large t reveals hard information-processing limits imposed by position encoding or attention masking.

### Centered Kernel Alignment (CKA)

A similarity measure between representations at different layers. High CKA between non-adjacent layers indicates redundant computation. Presented as an adjacent-layer curve (primary) and full L×L heatmap (supplementary).

### Roofline Diagram

A plot of achieved performance (TFLOPS/s) vs arithmetic intensity (FLOPs/byte). The hardware boundary (L4: 242 TFLOPS peak, 300 GB/s bandwidth, ridge at ~807 FLOPs/byte) separates memory-bound from compute-bound regimes. Each variant is plotted as a point at varying sequence lengths.

---

## Infrastructure Concepts

### Evaluation Pipeline

The single-interface module (`src/evaluation/pipeline.py`) that orchestrates the full post-hoc evaluation workflow: load checkpoints → compute FLOPs → run probes → slice comparisons → aggregate seeds → generate visualizations → write reports. Callers (CLI, tests, notebooks, future dashboard) share one interface: `EvaluationPipeline.run(checkpoints, output_dir) → ReportResult`.

### ProbeTarget

A structural protocol (`src/evaluation/probe_target.py`) that decouples diagnostic probes from model internals. Probes program against `ProbeTarget` rather than reaching into `model.blocks` or registering hooks directly. Two methods: `forward(x) → logits` (lightweight, for MQAR) and `forward_with_internals(x) → ProbeInternals` (heavy, for stable rank / CKA / entropy). The `ModelProbeAdapter` wraps real models to satisfy this protocol.

### AttentionModule

A structural protocol (`src/models/attention_protocol.py`) formalizing the informal contract all attention variants already satisfy: `forward(x, kv_cache=None) → (output, new_kv_cache)`. Purely documentary — does not enforce cache shape compatibility. Used as a type annotation in `TransformerBlock` and `ModernTransformerBlock`.

---

## Open Questions

- ALiBi extrapolation experiment: train-short/infer-long capability. Deferred until after the controlled comparison at fixed seq_len is complete.
- KV-Cache unification: ModernAttention (SDPA) uses a concat-based 2-tuple cache; FlashAttentionBase uses a pre-allocated 3-tuple cache. These are incompatible seam contracts but `generate.py` currently works by passing cache opaquely. Unifying into a single `KVCache` abstraction is deferred — all variants exist now, but the formal `AttentionModule` Protocol deliberately uses `Any` for cache type to respect this deferral.