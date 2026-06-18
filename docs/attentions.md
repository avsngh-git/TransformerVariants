# Attention Mechanisms

A reference document for the attention variants used in this project.

---

## Vanilla Attention (Standard Multi-Head Self-Attention)

The classic attention mechanism from GPT-2. Every head computes its own Q, K, V from the input, scores all token pairs, masks future tokens, and produces a weighted sum of values.

### The Forward — What Happens at Runtime

1. **Project:** x → Q, K, V via one matmul
2. **Reshape:** split d_model into n_head × d_head
3. **Score:** Q·Kᵀ / √d_head
4. **Mask:** set future positions to -∞
5. **Softmax:** turn scores into probabilities (0-1, sum to 1)
6. **Aggregate:** weighted sum of V
7. **Merge heads:** concatenate and project

### The Shape Journey

For a debug model (B=2, T=512, d_model=256, n_head=4, d_head=64):

```
x:           (2, 512, 256)      ← input
qkv:         (2, 512, 768)      ← 3 * 256 (combined projection)
q, k, v:     (2, 512, 256)      ← split into three
q reshaped:  (2, 4, 512, 64)    ← per head
attn_scores: (2, 4, 512, 512)   ← T×T attention matrix
out:         (2, 4, 512, 64)    ← weighted values
merged:      (2, 512, 256)      ← heads concatenated
```

### Key Properties

- **Complexity:** O(T²·d) — quadratic in sequence length
- **Memory:** stores the full T×T attention matrix per head
- **Causal mask:** lower-triangular, so position i can only attend to positions 0..i
- **Each head is independent:** heads share no parameters and compute attention separately
