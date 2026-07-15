# V5 Causal Linear Attention: Literature and Stability Audit

Date: 2026-07-14

## Question and conclusion

The causal prefix recurrence in V5 is a valid implementation of the core
linear-attention equation from Katharopoulos et al., and its chunked evaluation
is an established exact rearrangement in real arithmetic. However, the complete
V5 mechanism is **not** the mechanism proposed in that paper or in the RoFormer
extension:

1. Its handwritten `torch.where` feature map is not backward-equivalent to the
   official `F.elu(x) + 1` and produces NaN gradients for large positive inputs.
2. V5 applies RoPE to `Q` and `K` **before** `ELU+1`.
3. It evaluates important recurrence products under bf16 autocast.
4. It uses the original row-sum division even though subsequent work identified
   that normalization as a source of unbounded gradients.

The leading immediate NaN mechanism is the feature-map backward bug. The known
unbounded-gradient behavior of normalized `ELU+1` attention is a plausible
reason a projected query or key eventually crosses the overflow threshold, and
mixed-precision recurrence may amplify the instability. Pre-feature-map RoPE is a real
literature deviation and should be removed or reformulated, but the literature
does not establish it as a direct cause of NaNs.

This is a literature-backed diagnosis, not yet a causal proof. The completed
runs only logged every ten steps and have no checkpoint before divergence, so
they cannot identify the first non-finite tensor.

## Observed failure

All three seeds transition abruptly from ordinary finite loss and gradient norm
to `NaN` at nearly the same training stage:

| Seed | Last logged finite step | First logged NaN | Learning rate |
|---:|---:|---:|---:|
| 42 | 1,620 | 1,630 | `2.96e-4` |
| 137 | 1,690 | 1,700 | `2.95e-4` |
| 2024 | 1,670 | 1,680 | `2.96e-4` |

The finite gradient norms immediately beforehand are unremarkable, generally
about `0.65–0.85`. This is consistent with a single batch producing a
non-finite forward value or a sharp attention-gradient singularity; it does not
look like a slowly visible global gradient-norm explosion. Because logging is
sparse, this distinction remains an inference.

The run uses AdamW, bf16 autocast, peak learning rate `3e-4`, and 500 warmup
steps for a 15,000-step run (3.33%). The learning rate had already been near
its peak for roughly 1,100 steps when the failure appeared.

## Equation-by-equation comparison

### What Katharopoulos et al. proposed

For `phi(x) = ELU(x) + 1`, causal linear attention is

\[
o_i =
\frac{\phi(q_i)^\top \sum_{j\leq i}\phi(k_j)v_j^\top}
     {\phi(q_i)^\top \sum_{j\leq i}\phi(k_j)}.
\]

The paper defines the prefix states

\[
S_i=\sum_{j\leq i}\phi(k_j)v_j^\top,\qquad
z_i=\sum_{j\leq i}\phi(k_j),
\]

and updates both additively. It specifically chooses `ELU+1` because the
similarity must be non-negative and ELU retains a nonzero gradient on negative
inputs. These are equations 7 and 9–12 of
[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html).

The official implementation follows the same equation, adds `1e-6` to the
denominator, and even contains a TODO asking whether `Q` and `K` should be
divided by a large norm to avoid denominator instability
([official causal attention source](https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/attention/causal_linear_attention.py)).

### The feature-map implementation is not backward-equivalent

The paper and official source use `torch.nn.functional.elu(x) + 1`. V5 instead
implements the same forward piecewise formula as
`torch.where(x >= 0, x + 1, torch.exp(x))`. The two expressions return the same
finite forward value, but PyTorch autograd still evaluates the derivative of
the unselected exponential branch. At a positive float32/bfloat16 input around
89, `exp(x)` overflows; its masked gradient becomes `0 * inf = NaN`.

A minimal reproduction against the repository code gives:

| Input | V5 `torch.where` gradient | Official `F.elu + 1` gradient |
|---:|---:|---:|
| 80 | 1.0 | 1.0 |
| 89 | NaN | 1.0 |
| 90 | NaN | 1.0 |
| 100 | NaN | 1.0 |

This is not merely a precision concern or a theoretical conditioning issue. It
is a deterministic NaN-producing backward path in the exact V5 feature map.
The missing pre-divergence checkpoint prevents confirmation that a projected
query or key exceeded the threshold in the failed runs, but this is the most
direct candidate for the abrupt transition.

The trainer then makes the blast radius global. `clip_grad_norm_` is called
with its default `error_if_nonfinite=False`. Once one parameter has a NaN
gradient, the total norm is NaN and the clip coefficient is NaN, which turns
otherwise finite gradients into NaNs before AdamW updates every parameter. A
two-parameter minimal reproduction confirmed that one NaN gradient changed a
separate finite gradient from `2.0` to NaN. This explains why the logged global
gradient norm and subsequent losses are wholly NaN rather than identifying the
original tensor.

### What V5 computes

V5 computes

\[
\tilde q_i=\phi(R_iq_i),\qquad
\tilde k_j=\phi(R_jk_j),
\]

then substitutes `tilde q` and `tilde k` into the Katharopoulos recurrence. Its
chunked form splits the prefix into prior chunks plus a lower-triangular current
chunk. That decomposition is algebraically exact.

The chunk construction is not novel or suspect by itself. Yang et al. describe
non-overlapping chunks with an inter-chunk recurrent state and an intra-chunk
parallel calculation, and explicitly state that the ungated setting recovers
linear attention's chunkwise form
([Gated Linear Attention Transformers](https://arxiv.org/abs/2312.06635)). The
official Flash Linear Attention reference also uses a 64-token chunk and the
same `previous-state + triangular-current-chunk` decomposition
([reference implementation](https://github.com/fla-org/flash-linear-attention/blob/ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0/fla/ops/linear_attn/naive.py)).

## The RoPE deviation

RoFormer does propose RoPE for linear attention, but not in V5's order of
operations. It says to rotate the **outputs of the non-negative feature maps**
in the numerator:

\[
(R_i\phi(q_i))^\top(R_j\phi(k_j))
=\phi(q_i)^\top R_{j-i}\phi(k_j).
\]

It deliberately leaves the denominator unrotated to avoid division by zero;
the numerator may consequently contain negative terms. See the “RoPE with
linear attention” subsection of
[RoFormer](https://arxiv.org/abs/2104.09864), especially its equations 18–19.

V5 instead computes `phi(R_i q_i)`. Since `ELU+1` is nonlinear, in general

\[
\phi(R_iq_i) \neq R_i\phi(q_i).
\]

The resulting similarity cannot be reduced to a function of only `j-i` by the
RoPE rotation identity. In other words, V5's construction is positive and
causal, but it does not have the relative-position derivation claimed for RoPE.
This exact `RoPE -> ELU+1 -> normalized positive recurrence` combination was
not proposed by either cited paper.

This is a high-confidence architecture finding. Its connection to NaNs is only
medium-to-low confidence: rotation preserves vector norm, and `ELU+1` itself
cannot overflow on its positive branch and is at most one on its negative
exponential branch. The ordering changes optimization geometry and can drive
small similarities, but it is not an obvious direct overflow operation.

## Known gradient instability in normalized linear attention

Qin et al. analyze the row-sum division used by kernel linear attention and
show that its attention gradients are not bounded as similarities approach
zero. Their experiments report substantially more variable gradients for the
`1+ELU` mechanism than for softmax attention. They replace the division with an
RMSNorm applied after the unnormalized attention output and prove a finite
gradient bound for that construction
([The Devil in Linear Transformer](https://aclanthology.org/2022.emnlp-main.473/),
sections 3.1 and 4.2).

This result applies directly to V5's normalized positive feature-map
attention. `denominator.clamp_min(1e-6)` only places a floor on the final row
denominator. It does not make the normalized attention's full Jacobian behave
like softmax, and at the floor it changes the derivative discontinuously from
the official implementation's smooth `denominator + eps`.

This is the strongest literature match to the observed failure: three seeds
reach the same optimization regime and then fail abruptly, while the model
uses precisely the mechanism identified as having unstable gradients.

It is not yet proof. A decisive trace must show that the first non-finite value
or gradient originates in a V5 attention normalization rather than elsewhere.

## Precision and accumulation

The original authors' causal-product CPU and CUDA kernels are hard-coded for
32-bit `float` inputs, states, outputs, and gradients
([official CUDA kernel](https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/causal_product/causal_product_cuda.cu)).
The original paper does not report mixed-precision training.

A local dtype probe under the actual V5 autocast context found:

| Quantity | Runtime dtype |
|---|---|
| RoPE output and `phi(Q/K)` | float32 |
| values | bfloat16 |
| per-chunk `K^T V` update | bfloat16 output |
| history numerator product | bfloat16 output |
| local score matrix | bfloat16 output |
| stored Python prefix state | float32 after addition |
| denominator and normalized output | float32 |

Thus, the state tensor is stored as float32, but the important outer products
and reads from that state are dispatched through autocast and rounded to bf16.
The official modern FLA reference first casts `Q`, `K`, `V`, and the recurrent
state to float32, and its optimized recurrent/chunk kernels declare float32
state accumulators
([reference recurrence](https://github.com/fla-org/flash-linear-attention/blob/ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0/fla/ops/linear_attn/naive.py),
[chunk-state kernel](https://github.com/fla-org/flash-linear-attention/blob/ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0/fla/ops/common/chunk_h.py)).

This makes V5's blanket-autocast recurrence a meaningful implementation
deviation. It is a plausible amplifier of an unstable normalization, although
bf16 has the same exponent range as float32, so a simple reduced-range overflow
explanation would be incorrect. The concern is loss of precision in cumulative
products, cancellation in the value-weighted state, and gradients near a
poorly conditioned division.

## Training recipe comparison

The original paper's realistic 8- and 9-layer experiments used RAdam with a
learning rate of `1e-4`; its `1e-3` example was a small four-layer, length-128
copy task. The RoFormer Performer experiment at sequence length 1,024 also used
`1e-4` (while using RoFormer's different rotation formula).

There is no universal evidence that `3e-4` is intrinsically too high. In the
later TransNormer study, the controlled autoregressive WikiText-103 experiment
used Adam with peak learning rate `5e-4`, but it warmed up for 8,000 of 100,000
updates and used sequence length 512
([experiment configuration](https://aclanthology.org/2022.emnlp-main.473.pdf)).
V5 warms up for 500 of 15,000 updates, roughly 3.33% rather than 8%.

The short warmup and use of a shared softmax-model recipe are therefore
plausible contributors, but they rank below the attention equation and numeric
precision. The failure occurs long after warmup, and learning rate alone does
not explain why only this architecture fails under the shared protocol.

## Other differences that are not likely root causes

- **Chunk size 64:** established by modern linear-attention implementations;
  the decomposition is exact in real arithmetic.
- **Causal ordering:** prior chunks plus a lower-triangular local chunk includes
  exactly `j <= i`.
- **Feature-map equation:** V5's piecewise formula is forward-equivalent to
  `ELU(x)+1`, although its `torch.where` implementation is not
  backward-equivalent and is a leading root-cause candidate, as shown above.
- **No `1/sqrt(d)` scaling:** the Katharopoulos `ELU+1` equation and official
  implementation do not require softmax's dot-product scaling.
- **No recurrent decay/gate:** this matches the 2020 mechanism, so it is not an
  implementation error. Later successful language models add fixed or
  data-dependent forgetting and per-head output normalization; GLA describes
  these as important for performance and stable scalable training. Adopting
  them would define a different V5 mechanism rather than repair the original
  one.

## Ranked hypotheses and discriminating experiments

### 1. The handwritten feature map produces a non-finite backward pass

**Evidence:** the exact repository function deterministically returns a NaN
gradient for sufficiently large positive float32/bfloat16 inputs, while the
official `F.elu(x) + 1` remains finite.

**Prediction:** immediately before the first NaN, a post-RoPE projected query
or key will cross the exponential overflow threshold. Replacing only the
implementation with the official primitive will eliminate this path without
changing the intended forward equation.

### 2. Normalized `ELU+1` attention reaches its known unstable-gradient regime

**Evidence:** direct theoretical and empirical match from Qin et al.; the
original official source itself flags denominator stability; abrupt failure in
all seeds.

**Prediction:** immediately before the first NaN, an attention layer will show
extreme `Q/K` features, small/poorly conditioned normalization, or the first
non-finite gradient in `q_proj`/`k_proj`. Replacing row-sum division with the
published NormAttention output normalization should remove this failure mode,
though it changes the architecture.

### 3. bf16 autocast of recurrence products amplifies the instability

**Evidence:** V5's observed dtypes differ from the original float32-only code
and the float32 reference/state accumulators in modern FLA.

**Prediction:** keeping projections in bf16 but forcing feature maps, prefix
state updates, numerator, denominator, and their backward path to float32 will
stay finite on the same batches and schedule. A paired bf16 run will reproduce
the failure earlier or exhibit much larger numeric error.

### 4. Pre-feature-map RoPE creates an unvalidated and poorly conditioned kernel

**Evidence:** exact mismatch with the RoFormer linear-attention derivation;
`phi` and rotation do not commute.

**Prediction:** an otherwise identical model with a published positional
construction—absolute/sinusoidal input positions for the original
Katharopoulos mechanism, or RoFormer's `R(phi(Q/K))` numerator with an
unrotated denominator—will remain finite or show materially healthier feature
and gradient statistics.

### 5. The shared learning-rate schedule is too aggressive for V5

**Evidence:** 3.33% warmup; closest original/RoFormer large-sequence recipes
used `1e-4`; normalized linear attention is known to have noisier gradients.

**Prediction:** `1e-4` peak learning rate or an approximately 8% warmup prevents
the failure without changing attention statistics at equal weights. This
should be tested after the float32 and position controls so that a lower
learning rate does not merely hide a mathematical/numeric defect.

## Recommended diagnostic order

1. Add per-layer finite checks at the boundaries `Q/K/V`, post-RoPE,
   post-feature-map, prefix states, numerator, denominator, normalized output,
   and their gradients. Log extrema and the first offending layer only.
2. Run a paired short reproduction with the whole V5 attention calculation in
   float32 versus current autocast, replaying the same data batches.
3. Run the no-RoPE/absolute-position control that most closely matches the 2020
   proposal.
4. Only then test lower peak learning rate and a materially longer warmup.
5. If the original normalized mechanism remains unstable in float32, choose
   explicitly between preserving it and using the published NormAttention (or
   a gated/decayed modern linear-attention variant). Record that as a new
   architecture decision because it changes what V5 measures.

## Bottom line

Yes, V5 currently does something that was not proposed: it rotates `Q/K` before
the nonlinear positive feature map and then treats the result as standard
normalized positive linear attention. The published linear-RoPE construction
uses the opposite order for the numerator and an unrotated denominator.

The most likely immediate NaN mechanism, however, is the known unbounded-gradient
behavior of normalized kernel linear attention, with bf16 recurrence products
as a plausible accelerator. Causal masking and chunking are supported by the
literature and are unlikely to be responsible.
