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
| V5 | Causal Linear Attention | ELU+1 feature-map prefix state, O(n) sequence complexity | V1 |
| V6 | MoE (Mixture of Experts) | Mixtral-style top-2 routing, 8 SwiGLU experts per MoE layer | V1 |

### Sub-variant

A single component swap within a variant's recipe, with everything else unchanged. Example: V0-GELU is a sub-variant of V0 where only the activation function changes (ReLU → GELU).

Known sub-variants:
- **V4-interleaved** — alternates local (SWA) and global (full attention) layers, Gemma2-style. Even layers attend to the full sequence; odd layers use window_size=W. Isolates the question: "does periodic full-context access recover information lost by windowing?"
- **V6-interleaved** — alternates dense FFN (even layers) and MoE (odd layers). Reduces total parameters while retaining expert specialization in half the layers. Isolates: "do you need experts at every layer, or can periodic dense layers maintain quality?"
- **V6-deep** — first half of layers use dense FFN, second half use MoE. Tests the hypothesis that early layers learn general representations (don't need specialization) while deeper layers benefit from expert routing.

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

Dense variants at the same scale target active-parameter counts within ±5%. Dimensions (d_model, n_layer, n_head) remain fixed; inherent reducers such as GQA and capacity expanders such as the completed MoE recipes are labeled non-conforming when they fall outside tolerance, with active and total counts reported rather than silently compensated.

### Run

A single execution of the training script. Produces one checkpoint directory, one training log. Identified by variant + scale + seed + timestamp.

### Experiment

A controlled comparison: multiple runs across variants and/or seeds under the same protocol (same data, token budget, optimizer, batch size, precision). Governed by the experiment contract.

### Sliding Window Attention (SWA)

An attention pattern where each query token attends only to the W tokens immediately preceding it (plus itself), rather than the full sequence. W is the window size. In this project, W is a fixed proportion of seq_len (W = seq_len // 4), constant across all layers. SWA applies during training only — generation uses the full KV cache.

The flash_attn kernel supports SWA natively via its `window_size` parameter. No custom masks or block-sparse patterns are needed.

SWA isolates the variable "attention span" while keeping all other components (RoPE, projections, FFN, normalization) identical to V1.

### Causal Linear Attention

An autoregressive attention mechanism using the positive feature map `phi(x) = ELU(x) + 1`. Instead of materializing a T-by-T matrix, each position reads cumulative key and key-value statistics from its prefix. The result is strictly causal and has O(T * d_head^2) complexity instead of O(T^2 * d_head).

ELU+1 is applied first. Following RoFormer equation 19, RoPE rotates the positive Q and K features used by the numerator, while the denominator uses unrotated positive features. Stability-sensitive recurrence products and prefix states use float32. The implementation processes fixed-size chunks: a small triangular matrix handles causal interactions within each chunk, while accumulated state summarizes all earlier chunks. This is algebraically equivalent to the token-wise recurrence.

V5 uses `ModernTransformer` as its model shell (RMSNorm, SwiGLU) and `CausalLinearAttention`. It supports full-sequence validation, long-context diagnostics, and uncached generation. Recurrent generation state is not yet exposed through the shared KV-cache interface, so reusable cached serving remains unsupported.

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
- Dense comparison recipes target active parameter counts within ±5%. Mechanisms that inherently reduce or expand active parameters are explicitly labeled non-conforming when outside tolerance; active and total counts are reported and the mismatch is not compensated post hoc.
- Primary training and checkpoint-quality results are reported as mean ± sample standard deviation over 3+ independent random-seed records. An axis whose authentic per-seed history is unavailable must be labeled incomplete/non-statistical, not presented as a primary result. Clearly labeled representative-checkpoint serving and long-context diagnostics are capability measurements, not statistical main results.
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

A synthetic evaluation probe that tests a model's ability to bind and recall key-value associations across context. Sequences contain planted token patterns; the metric is whether the model assigns high probability to the correct recall token. Isolates retrieval capacity — variants with limited or compressed context (V4-SWA, V5 causal linear) should show measurable degradation.

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

### Dashboard

A self-contained static report generated by `src/viz/html_dashboard.py` and `scripts/build_dashboard.py`. It consumes `raw/metrics.json`, optional `raw/benchmarks.json`, metadata, and publication plots, then embeds all JSON, CSS, JavaScript, and PNG assets into one `index.html`. It needs no server, CDN, Streamlit, or internet connection. The evaluation pipeline builds it automatically; the CLI rebuilds it after separate inference benchmarks. The former `dashboard/` Streamlit implementation is retained only as legacy reference.

---

## Large-Scale Data Pipeline Concepts

### Streaming Preparation

A data ingestion pattern where documents are consumed from an iterator one at a time, tokenized immediately, and flushed to binary shards when a buffer fills. Memory usage is constant regardless of corpus size. The source iterator is never materialized in full — only one document is in memory at a time.

### Document Filter

A lightweight guard applied to each document during streaming preparation. Current filters: minimum token count (skip short docs), maximum token count (truncate overlong docs). Filters are defensive — the source corpus (FineWeb-Edu) is pre-curated, so these catch edge cases rather than performing primary quality selection.

### Hash-Based Split

A deterministic train/val routing decision made per-document using content hashing. SHA-256 of the first 256 bytes, modulo 100; values < 1 route to validation, ≥ 1 route to training. Produces a reproducible ~1% val split independent of stream ordering.

### Resumption Checkpoint

A small JSON file (`progress.json`) written alongside shards after each flush, recording how many documents have been consumed from the stream. If the pipeline is interrupted, it resumes by skipping that many documents on restart.

---

## Fault-Tolerance Concepts

### Atomic Checkpoint Write

A crash-safe persistence pattern: serialize to a temporary file, call `fsync` on the file descriptor, then `os.rename` to the final path. Because `rename` is atomic on POSIX filesystems, the checkpoint is either fully present or absent — never half-written. SHA-256 integrity hashes are stored alongside for post-crash validation.

### Checkpoint Ring Buffer

A rotation scheme that keeps the last N (default 3) verified checkpoints. A new checkpoint is only promoted to the ring after its integrity hash is validated. The oldest entry is deleted only after the new one is confirmed good. Prevents the failure mode where the only checkpoint is the one currently being overwritten.

### Async Background Save

A performance optimization where the training loop snapshots model and optimizer state_dicts to CPU memory (fast, blocks training briefly), then a background thread handles serialization and disk I/O. Training resumes immediately after the CPU snapshot. The background thread performs the atomic write protocol independently.

### Health Monitor

An injected dependency in the Trainer that inspects grad_norm and loss after each step. Maintains a rolling window (default 100 steps) and computes z-scores. Returns one of three actions: CONTINUE (normal), SKIP_STEP (discard the current gradient update, don't step the optimizer), or ROLLBACK (reload the most recent checkpoint from the ring buffer and resume from there).

### Fault Injection Test

A test that deliberately introduces a failure mode — corrupted checkpoint (bit-flip), process kill mid-write, NaN injection into gradients, loss spike beyond threshold — and asserts that the recovery mechanism handles it correctly. These tests prove the fault-tolerance system works, not just that it exists.

---

## Mixture of Experts (MoE) Concepts

### Mixture of Experts (MoE)

A conditional computation architecture where each transformer block's FFN is replaced by N parallel expert FFNs and a learned router. Each token is processed by only the top-k experts (k << N), meaning total parameters are much larger than per-token active parameters. This project uses Mixtral-style MoE: 8 SwiGLU experts per MoE layer, top-2 routing per token.

### Router

A small linear projection `nn.Linear(d_model, num_experts, bias=False)` that produces per-token expert affinity scores. The softmax over these scores determines which experts process each token and with what weight. The router is the only component that "decides" — experts themselves are standard SwiGLU FFNs.

### Active Parameters

The number of parameters used to process a single token. In MoE, this is: shared parameters (embeddings, attention, norms) + top-k expert FFNs. For fair comparison with dense models, active parameters must match — not total parameters.

### Load-Balancing Auxiliary Loss

A regularization term `α * Σ(f_i * P_i)` added to the training loss to prevent expert collapse (all tokens routing to the same few experts). `f_i` = fraction of tokens assigned to expert i in a batch, `P_i` = mean router probability for expert i. Coefficient α is typically 0.01.

### Z-Loss

A stabilization term `β * mean(log(Σ exp(router_logits))²)` that prevents router logits from growing unboundedly large, which could cause numerical instability. Coefficient β is typically 0.001.

### Expert Collapse

A failure mode where the router converges to sending all/most tokens to 1-2 experts, leaving the rest untrained. Load-balancing aux loss prevents this. Detectable via the expert utilization histogram: healthy routing shows near-uniform distribution.

### Routing Data

Per-token routing decisions captured during evaluation: which experts were selected, with what weights, at each layer. Stored in an internal buffer when `record_routing=True` is set, then exposed via `model.get_routing_data()` for post-hoc analysis by MoE-specific probes.

---

## Open Questions

- ALiBi extrapolation experiment: train-short/infer-long capability. Deferred until after the controlled comparison at fixed seq_len is complete.
- KV-Cache unification: ModernAttention (SDPA) uses a concat-based 2-tuple cache; FlashAttentionBase uses a pre-allocated 3-tuple cache. These are incompatible seam contracts but `generate.py` currently works by passing cache opaquely. Unifying into a single `KVCache` abstraction is deferred — all variants exist now, but the formal `AttentionModule` Protocol deliberately uses `Any` for cache type to respect this deferral.