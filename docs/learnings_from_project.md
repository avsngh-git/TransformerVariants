# Learnings from the Project

A running document of concepts, design decisions, and explanations I've picked up while building this project.

---

## Learned vs. Calculated Position Embeddings

There are two flavors of position embeddings:

**Calculated (sinusoidal)** — the original "Attention Is All You Need" paper. You compute fixed sine/cosine waves at different frequencies for each position. They're not learned — they're deterministic mathematical functions. The idea was that the model could generalize to longer sequences because the pattern is systematic.

**Learned** — you create an `nn.Embedding(max_seq_len, d_model)` and let the model learn what each position "means" during training. GPT-2 uses this approach. The positions are just another embedding table (like the token embeddings), initialized randomly and updated via backpropagation.

**For our vanilla V0 model:** Learned position embeddings are the correct choice. They match the original GPT-2 architecture (which IS the "vanilla" decoder-only Transformer). In Phase 5, we'll switch to RoPE (rotary position embeddings) — a more modern approach that's neither purely calculated nor purely learned. The progression is intentional: V0 shows the classic approach, V1 shows the modern one.

Yes, in both cases the position embedding is **added** to the token embedding before entering the Transformer layers.

---

## Why No Dropout?

This is a modern best practice for pretraining. The reasoning:

1. **Dropout was designed for small-data regimes** to prevent overfitting. When you're training on hundreds of millions of tokens (more data than the model can memorize), overfitting isn't really the problem — underfitting is.

2. **Modern LLMs (LLaMA, GPT-3, Chinchilla) all train with dropout=0.** The empirical finding is that at sufficient data scale, dropout hurts convergence speed without helping generalization.

3. **Fair comparison.** If some variants use dropout and others don't, you can't tell if performance differences come from the architecture or the regularization.

However — if you *wanted* dropout for the debug model (small data, risk of overfitting), that's reasonable. The config supports it (`dropout: 0.0` is explicit so you can change it). For main/stretch runs on 100M+ tokens, keep it at 0.

---

## Why No Bias?

The `bias: false` in your config means linear layers are `nn.Linear(in, out, bias=False)` — no additive bias term.

**Why skip it:**

1. **LLaMA/modern LLMs drop bias.** Empirically, at scale, the bias terms add parameters without measurably helping loss. They're just extra numbers the optimizer has to track.

2. **Cleaner interaction with normalization.** When you have LayerNorm before every linear layer, the normalization already handles centering (shifting the mean). A bias after that is redundant — it's trying to shift something that just got normalized.

3. **Fewer parameters = slightly faster** — not a huge deal, but it's free.

4. **Simplifies some variants.** When you later implement RoPE, biases in the Q/K projections would interfere with the rotary math.

**When you WOULD use bias:** in very small models or fine-tuning scenarios where every bit of capacity helps. For pretraining at 40M+ params on plenty of data, it's unnecessary.


---

## Pre-LayerNorm vs. Post-LayerNorm

In the original "Attention Is All You Need" paper, LayerNorm is applied **after** the residual addition:

```
x = x + Attention(x)
x = LayerNorm(x)          ← Post-LN
```

GPT-2 and almost all modern models switched to **pre-normalization** — applying LayerNorm **before** the sublayer:

```
x = x + Attention(LayerNorm(x))    ← Pre-LN
```

**Why Pre-LN won:**

1. **Training stability.** Post-LN models are notoriously hard to train — they often diverge without careful learning rate warmup and gradient clipping. Pre-LN makes gradients flow more smoothly through the residual stream.

2. **No need for a careful warmup schedule.** Post-LN basically requires warmup or the model blows up early in training. Pre-LN is much more forgiving.

3. **GPT-2 uses Pre-LN.** Since our V0 is "vanilla GPT-2 style," Pre-LN is the matching choice.

There's also a final LayerNorm after all layers (before the output projection) in the Pre-LN pattern. This is standard.

**Our choice:** Pre-LN for V0. It's more stable, matches GPT-2, and is what you'll find in every modern implementation. Post-LN is only interesting historically.


---

## Weight Initialization

Neural networks need their weights initialized to some values before training starts. The choice matters a lot for training stability — bad initialization can cause gradients to explode or vanish before the model even starts learning.

**Options:**

- **Xavier/Glorot uniform** — the original Transformer paper's choice. Scales weights based on fan-in and fan-out so that variance stays roughly constant across layers.
- **GPT-2 style** — normal distribution with std `0.02` for most layers, and a special scaled initialization for the output projection of attention and FFN: `0.02 / sqrt(2 * n_layers)`. This ensures that deeper models don't have exploding activations early in training.
- **Small init for embeddings** — some implementations init embeddings with smaller std.

**Why GPT-2 style initialization matters:**

The key insight is the `1/sqrt(2*n_layers)` scaling on residual projections. Every layer adds its output to the residual stream:

```
x = x + layer_output
```

If each `layer_output` has variance 1, then after `n` layers, the residual has variance `n`. That grows with depth. By scaling the output projection's init by `1/sqrt(2*n_layers)`, you keep the variance growth under control at initialization, so the model starts in a stable regime.

The "2" in `2*n_layers` accounts for the fact that each Transformer layer has **two** residual additions: one for attention and one for FFN.

**Our choice:** GPT-2 style — `N(0, 0.02)` for most weights, residual projections scaled by `1/sqrt(2*n_layers)`. Matches V0's identity as a vanilla GPT-2 model.


---

## Decoding Strategies (How Models Generate Text)

Once a model produces a probability distribution over the next token, you need a **decoding strategy** to pick which token to actually emit.

### Greedy Decoding

```python
next_token = torch.argmax(probs, dim=-1)  # always pick the highest probability
```

**Pro:** Deterministic, fast, consistent.
**Con:** Often produces repetitive, bland text. The model always makes the "safest" choice, which tends to be common tokens.

### Modern Approaches (in order of adoption)

**1. Nucleus Sampling (Top-p) — the current industry standard**

- Sort tokens by probability
- Keep the top tokens until their cumulative probability exceeds `p` (e.g., 0.9)
- Sample from only those tokens, discard the rest
- This removes the "tail" of low-probability tokens that would introduce randomness

**Why it won:** Balances diversity with coherence. The model can still be creative (by sampling), but only among high-confidence options. Used in GPT-3, Claude, etc.

**2. Top-k Sampling**

- Keep only the top-k highest probability tokens, sample from those
- Simpler than nucleus but less adaptive (k is fixed)

**3. Beam Search**

- Explore multiple candidate sequences in parallel (beams)
- Keep the most likely ones at each step
- Final answer is the highest-probability beam

**Pro:** More globally optimal than greedy.
**Con:** Much slower, can produce repetitive results (length bias), rarely used in modern LLMs at generation time.

**4. Greedy with penalties (modern variant)**

- Still pick argmax, but penalize tokens that were already generated (reduce-repeat penalty)
- Used in some RLHF-tuned models

### Our V0 Implementation

We use greedy decoding with temperature scaling for now. We'll add nucleus sampling (top-p) later when we implement the full generation utilities.


---

## KV-Cache (Key-Value Cache)

The KV-cache is the single biggest optimization for Transformer generation speed.

### The Problem

During autoregressive generation, we produce one token at a time. To generate token at position `t`, we need to:
1. Run the full forward pass over positions `0..t`
2. Get the logit for position `t`
3. Pick the next token
4. Repeat for position `t+1`

Without caching, generating a 100-token sequence means 100 forward passes, each processing the entire growing sequence. That's O(T²) total computation across all generation steps.

### The Insight

In causal attention, positions `0..t-1` can't see position `t`. So the K and V vectors for positions `0..t-1` **never change** regardless of what comes later. We're recomputing the exact same K and V values on every generation step.

### The Solution

Cache the K and V tensors from prior positions. On each new generation step:
- Only compute Q, K, V for the **new** token (position `t`)
- Concatenate the new K, V with the cached K, V from positions `0..t-1`
- Compute attention: new Q against all cached K's
- Update the cache

This turns each generation step from O(T) to O(1) in the compute dimension (one new token's Q attending to the growing cache). Total generation cost goes from O(T²) to O(T).

### The Shape

```
Without KV-cache (step t):
  Q, K, V all computed for full sequence: (B, n_head, t, d_head)
  Attention: (B, n_head, t, t) — wasteful, we only need the last row

With KV-cache (step t):
  Q computed for new token only: (B, n_head, 1, d_head)
  K, V retrieved from cache: (B, n_head, t-1, d_head)
  New K, V appended: (B, n_head, t, d_head)
  Attention: (B, n_head, 1, t) — just one row, much cheaper
```

### Memory Trade-off

The cache takes memory: `2 × n_layer × batch × n_head × seq_len × d_head` (factor of 2 for K and V). For our debug model: 2 × 4 × 1 × 4 × 512 × 64 = 1MB. For large models (70B parameters, long sequences), the KV-cache can be several GB — this is why "context window" is limited and why techniques like GQA (grouped-query attention) exist to reduce cache size.

### When It's Used

- **Training:** NO KV-cache. All positions process in parallel (teacher forcing), so there's no sequential generation. Every position sees its K/V computed fresh.
- **Generation/Inference:** YES KV-cache. We generate one token at a time, so caching prior K/V avoids redundant work.

---

## Activation Functions — All Variants We Use

Activation functions are the nonlinearities between linear layers. Without them, stacking linear layers is pointless (two linear layers = one linear layer). The activation is what gives the network the ability to learn non-linear relationships.

### ReLU (Rectified Linear Unit) — Our V0 Baseline

```
ReLU(x) = max(0, x)
```

The simplest activation. A hard gate: negative values become 0, positive values pass through unchanged.

```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
```

**Pros:**
- Dead simple, fast to compute
- Sparse activations (many zeros) — some argue this helps interpretability
- The original Transformer paper uses ReLU

**Cons:**
- "Dead neuron" problem: if a neuron's output goes negative, gradient is 0, and it never recovers
- Discontinuous gradient at x=0 (jumps from 0 to 1)
- Kills negative information entirely

**Used in:** Our V0-vanilla (pure textbook Transformer)

---

### GELU (Gaussian Error Linear Unit) — Our V0-GPT2 Variant

```
GELU(x) = x · Φ(x)
```

Where Φ(x) is the CDF of the standard normal distribution. Approximated in practice as:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

A smooth, probabilistic gate. Instead of hard-killing negatives, it scales values by their "probability of being positive."

```
Input:  [-2,   -1,   0,    1,    2  ]
Output: [-0.05, -0.16, 0, 0.84, 1.96]  ← small negatives partially survive
```

**Pros:**
- Smooth gradient everywhere (better optimization)
- No dead neurons — small negatives get small but nonzero gradient
- Empirically better loss curves than ReLU at scale

**Cons:**
- Slightly slower to compute than ReLU
- The approximation vs exact question (use `approximate="tanh"` for GPT-2 match)

**Used in:** Our V0-GPT2 variant, BERT, GPT-2, GPT-3

**In PyTorch:**
```python
self.act = nn.GELU(approximate="tanh")  # matches GPT-2
```

---

### SwiGLU (Swish-Gated Linear Unit) — Future Variant (LLaMA style)

```
SwiGLU(x) = Swish(x · W₁) ⊙ (x · W₂)
```

Where Swish(x) = x · σ(x) and ⊙ is element-wise multiplication.

This is a **gated** activation — it uses two parallel linear projections. One goes through the Swish activation, the other stays linear, and they're multiplied together. The "gate" controls how much information flows through.

```
Standard FFN:  x → Linear(d_model, d_ff) → Activation → Linear(d_ff, d_model)
SwiGLU FFN:    x → [Linear₁(d_model, d_ff) → Swish] ⊙ Linear₂(d_model, d_ff) → Linear(d_ff, d_model)
```

**Key difference:** SwiGLU has THREE weight matrices instead of two. To keep parameter count comparable, the hidden dimension is typically `8/3 × d_model` instead of `4 × d_model`.

**Pros:**
- Best empirical results among all activations at scale
- The gating mechanism lets the model learn "what to let through"
- Used in LLaMA, Mistral, PaLM — the current state of the art

**Cons:**
- Three weight matrices = more complex implementation
- Different hidden dimension ratio (8/3x vs 4x) makes parameter matching trickier

**Used in:** Our future LLaMA-style variant

**In PyTorch:**
```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # up projection
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # down projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

---

### Summary Table

| Activation | Formula | Used In | Dead Neurons? | Gated? |
|-----------|---------|---------|---------------|--------|
| ReLU | max(0, x) | V0-vanilla | Yes | No |
| GELU | x·Φ(x) | V0-GPT2, BERT | No | No |
| SwiGLU | Swish(xW₁) ⊙ xW₂ | LLaMA variant | No | Yes |

**The progression in our project:**
- V0-vanilla: ReLU (textbook baseline)
- V0-GPT2: GELU (GPT-2 match, smoother)
- Future variant: SwiGLU (current SOTA, gated)

Each step is a strict improvement in empirical loss at scale. The trade-off is always slightly more compute per activation.

### Our Config-Driven Approach

```python
# In ModelConfig:
activation: str = "relu"  # "relu", "gelu", or "swiglu" (future)

# In FFN __init__:
if config.activation == "gelu":
    self.act = nn.GELU(approximate="tanh")
elif config.activation == "relu":
    self.act = nn.ReLU()
```

This lets us swap activations via config without changing any model code.


---

## torch.compile — Free Training Speedup

`torch.compile(model)` is a one-line optimization that gives ~15-25% training speedup without changing the model's behavior at all. The outputs are numerically identical.

### The Problem It Solves

Without compile, PyTorch runs in "eager mode" — each operation executes immediately as a separate GPU kernel:

```python
x = self.ln1(x)        # Launch GPU kernel 1, wait
x = self.qkv_proj(x)   # Launch GPU kernel 2, wait
q, k, v = x.chunk(3)   # Launch GPU kernel 3, wait
q = q.view(...)         # Launch GPU kernel 4, wait
q = q.transpose(...)    # Launch GPU kernel 5, wait
```

Each kernel launch costs ~5-10μs of CPU overhead. For tiny operations like `transpose` that take 2μs on the GPU, the overhead is bigger than the work itself. Plus, each kernel reads data from GPU memory, processes it, writes it back — and the next kernel reads it again. Wasteful.

### What Compile Does

1. **Traces the computation graph** — records what operations happen instead of executing immediately
2. **Fuses operations** — combines many small kernels into fewer large ones
3. **Eliminates memory round-trips** — intermediate tensors stay in fast GPU registers instead of being written to memory and read back
4. **Optimizes memory access patterns** — rearranges how data is read for maximum throughput

### Before vs After (Attention Block)

Eager mode (~19 kernel launches):
```
Kernel 1: LayerNorm
Kernel 2: QKV projection
Kernel 3: Chunk into Q,K,V
Kernel 4-6: Reshape Q,K,V
Kernel 7-9: Transpose Q,K,V
Kernel 10: Q @ K^T
Kernel 11: Scale
Kernel 12: Mask
Kernel 13: Softmax
Kernel 14: Dropout
Kernel 15: Attn @ V
Kernel 16-18: Transpose, reshape, output proj
Kernel 19: Dropout
```

After compile (~3 fused kernels):
```
Fused Kernel 1: LayerNorm + QKV proj + reshape + transpose
Fused Kernel 2: Q@K^T + scale + mask + softmax
Fused Kernel 3: Attn@V + transpose + reshape + out_proj + dropout
```

### Why 15-25% (Not 5x)

The big matmuls (linear projections, attention) are already near-optimal via cuBLAS. Compile can't speed those up. The gains come from eliminating overhead on the small operations *between* matmuls — the element-wise ops, reshapes, normalizations. These are "memory-bound" operations where the bottleneck is reading/writing data, not computation.

### Trade-offs

- **First step is slow** — compilation takes 10-60 seconds (one-time cost)
- **Dynamic shapes break it** — if tensor shapes change between steps, it recompiles. Training has fixed shapes (good). Generation with KV-cache has growing shapes (bad — don't compile for generation).
- **Debugging is harder** — error messages point to compiled code, not your source

### Our Usage

```bash
# Training: use --compile for real runs
python scripts/train.py --data_dir data/... --compile

# Generation: don't compile (KV-cache has dynamic shapes)
model.generate(...)  # runs uncompiled, uses KV-cache instead
```

The two optimizations complement each other: `torch.compile` speeds up training (fixed shapes, many iterations), KV-cache speeds up generation (dynamic shapes, sequential tokens).


---

## bfloat16 Mixed Precision — Why Less Precision Is Fine

Float32 is more precise than bfloat16, but for training neural networks the extra precision is almost always wasted. The speed gain (2x) far outweighs the negligible quality difference (none measurable).

### The Formats

```
float32:  1 sign + 8 exponent + 23 mantissa = 32 bits
bfloat16: 1 sign + 8 exponent +  7 mantissa = 16 bits
float16:  1 sign + 5 exponent + 10 mantissa = 16 bits
```

bfloat16 keeps the same exponent range as float32 (can represent the same range of numbers) but sacrifices mantissa precision (fewer decimal places).

### Why Precision Doesn't Matter Much for Training

Neural network gradients are inherently noisy. You're computing gradients on a random mini-batch — that noise is far larger than the rounding error from 7 vs 23 mantissa bits. The model doesn't need to know a gradient is 0.000342847 vs 0.000343 — both push the weight the same direction.

What DOES matter is **range** — if numbers overflow to infinity or underflow to zero, training breaks. bfloat16's 8-bit exponent handles range identically to float32. float16's 5-bit exponent has smaller range, which is why it needs a "gradient scaler."

### Speed Gains

1. **Half the memory per tensor** — fit double the batch size or model in the same GPU RAM
2. **Faster matmuls** — NVIDIA tensor cores do bf16 at 2x throughput vs fp32
3. **Less memory bandwidth** — reading/writing 16 bits is twice as fast as 32 bits

### Mixed Precision (What Actually Happens)

`torch.autocast` selectively uses bf16 where it's safe:

- **Matmuls** (linear layers, attention): bf16 — biggest speed benefit
- **Reductions** (softmax, layernorm, loss): fp32 — sensitive to precision
- **Optimizer state** (Adam moments): fp32 — need precision for tiny weight updates
- **Master weights**: fp32 — the "true" weights; bf16 copies used for forward/backward

### bfloat16 vs float16

| Property | bfloat16 | float16 |
|----------|---------|---------|
| Range | Same as fp32 | Much smaller |
| Precision | Lower (7 mantissa) | Medium (10 mantissa) |
| Needs gradient scaler? | No | Yes |
| Hardware | Ampere+ (A100, L4, H100) | All modern GPUs |

bfloat16 is strictly simpler — same range as fp32, no scaler needed, just works.

### Our Result

Same loss curve, 1.75x faster. Every major LLM trains in bf16.
