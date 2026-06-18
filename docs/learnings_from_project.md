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
