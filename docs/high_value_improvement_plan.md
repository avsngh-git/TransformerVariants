# High-Value Scientific and Recruiter Improvement Plan

**Status:** implementation started 2026-07-18
**Canonical primary-study manifest:**
[`configs/experiment/main_500m_5seed.yaml`](../configs/experiment/main_500m_5seed.yaml)

## Outcome

The completed 1B-token/three-seed experiment remains a historical study. It is not
silently overwritten. The new primary study will retrain all ten recipes with:

- 7,629 steps, or exactly 499,974,144 tokens per run;
- five seeds: 42, 137, 2024, 31415, and 271828;
- independent logs and checkpoints for every run;
- async atomic checkpointing, SHA-256 verification, a three-checkpoint ring, health
  monitoring, and automatic resume from the latest verified checkpoint;
- a parameter-matched sparse FFN design for all three MoE placements;
- one canonical YAML manifest from which commands and provenance are generated.

This creates 50 runs and processes 24,998,707,200 training tokens. The completed
study processed approximately 29.49B tokens, so nominal token-level training compute
falls by approximately 15.2%, not 50%. Increasing from three to five seeds offsets
most of the per-run saving.

Five seeds reduce the standard error by a factor of
\(\sqrt{3/5}=0.775\), about 22.5%, if the underlying variance is unchanged. This
improves uncertainty estimates but does not guarantee statistical significance.
Halving the training horizon may also change effect sizes and variance. The old and
new studies must therefore be described as different training regimes.

## Decisions and scientific boundaries

### 1. MoE active-parameter matching

The dense main-scale SwiGLU hidden width is 1,408. Each sparse expert now uses width
704. With top-2 routing, the two selected experts collectively activate the same FFN
weights as one dense SwiGLU FFN. The router adds only 32,768 active parameters over
the whole eight-layer full-MoE model.

| Recipe | Active parameters | Stored parameters | Dense-active difference |
|---|---:|---:|---:|
| Modern | 51,430,400 | 51,430,400 | baseline |
| Full MoE | 51,463,168 | 103,367,680 | +0.064% |
| Interleaved MoE | 51,446,784 | 77,399,040 | +0.032% |
| Deep MoE | 51,446,784 | 77,399,040 | +0.032% |

The comparison will still report stored parameters because sparse conditional
capacity is part of the MoE mechanism. “Parameter matched” means active-per-token,
not equal checkpoint size.

### 2. Vanilla versus Modern remains a recipe comparison

The primary purpose of V0 versus V1 is to test whether the bundle of widely adopted
modern components improves a conventional decoder recipe under a shared training
protocol. It is not presented as a causal attribution study.

The actual V0 experiment uses GELU, not ReLU. The canonical configuration, registry,
CLI default, documentation, and new manifest now agree on this.

The case study will explain expected component-level effects from the original
literature, then clearly distinguish those external findings from what this bundled
experiment establishes:

- [Transformer](https://arxiv.org/abs/1706.03762): causal multi-head self-attention
  and position-wise FFNs;
- [GPT-2](https://arxiv.org/abs/1901.02860): the GELU-based decoder baseline;
- [RMSNorm](https://arxiv.org/abs/1910.07467): removal of mean centering with lower
  normalization overhead;
- [RoFormer/RoPE](https://arxiv.org/abs/2104.09864): relative position information
  through rotations of queries and keys;
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): gated FFNs,
  including SwiGLU;
- [FlashAttention](https://arxiv.org/abs/2205.14135): exact attention with reduced
  high-bandwidth-memory traffic.

### 3. Secondary surgical ablations

Attribution is a secondary study and will not expand the 50-run primary matrix. Once
the primary run is stable, create four Modern-backbone counterfactual recipes:

1. Modern with learned absolute position embeddings instead of RoPE.
2. Modern with LayerNorm instead of RMSNorm.
3. Modern with a parameter-matched dense GELU FFN instead of SwiGLU.
4. Identical Modern weights benchmarked through SDPA and FlashAttention where the
   kernels implement the same attention operator; no retraining is needed for this
   kernel-only comparison.

The three trainable counterfactuals use 3,750 steps (245.76M tokens) and the original
three seeds. They are compared with the 3,750-step validation measurements from the
five-seed Modern runs. Concretely, the primary metrics logs preserve the fresh step-3,750
validation measurement even though the bounded recovery ring may later evict that
checkpoint. An ambiguous effect can be promoted to all five seeds. The new
counterfactual runs add 2.212B tokens; the four-recipe comparison contains 2.949B
tokens when the reused Modern reference histories are counted.

The registry/config implementations and the isolated nine-run launcher are versioned
in [`configs/experiment/surgical_ablations.yaml`](../configs/experiment/surgical_ablations.yaml).
They are deliberately absent from the primary manifest.

The ablation report must say that marginal effects can depend on the chosen backbone
and interaction order. It must not add the individual effects and claim they explain
the entire V0-to-V1 difference.

### 4. Long-context evaluation

MQAR and paired-tail perplexity remain diagnostics. They are supplemented with:

- zero-shot passkey retrieval;
- zero-shot needle-in-a-haystack fact retrieval;
- exact top-1 and top-5 accuracy by context length and retrieval distance;
- expected answer-token probability and negative log-likelihood, which remain useful
  when exact accuracy has a floor effect;
- standard RoPE base 10,000 versus a no-training larger-base 100,000 extrapolation
  baseline.

The evaluator is [`scripts/evaluate_retrieval.py`](../scripts/evaluate_retrieval.py).
Multi-query associative recall after task-specific training is excluded because it
would violate the no-retraining requirement. Long-context claims will use the term
“extrapolation stability” unless retrieval accuracy demonstrates effective access to
distant information. RULER-style controlled retrieval motivates this separation; see
[RULER](https://arxiv.org/abs/2404.06654).

### 5. Serving and throughput protocol

The serving benchmark now separates prefill, steady-state cached decode, cached
end-to-end generation, and uncached generation. It records raw timing samples,
mean/p50/p95 latency, throughput, peak allocated GPU memory, and cache bytes.

The default matrix is:

- prompt lengths: 64, 512, 1,024, and 4,096;
- batch sizes: 1, 4, and 8;
- 128 generated tokens;
- 10 warm-ups and 30 measured repetitions.

FlashAttention-backed variants now build a reusable prompt cache. GQA retains fewer
KV heads, SWA applies its local window during cached generation, and causal linear
attention exposes a fixed-size recurrent numerator/denominator state. SWA currently
uses a pre-sized physical cache even though attention is window bounded; a circular
physical cache remains a possible memory optimization and must not be claimed yet.

The protocol follows the intended efficiency motivations of
[GQA](https://arxiv.org/abs/2305.13245),
[Mistral/SWA](https://arxiv.org/abs/2310.06825), and
[Linear Transformers](https://arxiv.org/abs/2006.16236).

### 6. Independent histories

All ten recipes are rerun. This is necessary if fixed-wall-clock, fixed-FLOP, and
learning-curve uncertainty are to be primary evidence for every recipe. Rerunning
only the affected files would leave the report as a mixture of training horizons,
seed counts, code revisions, and checkpoint policies.

The new report will reject duplicated histories. Fixed-data loss remains the primary
quality endpoint; wall-clock and model-estimated FLOP views are secondary efficiency
endpoints.

### 7. Statistical reporting

For every primary quality endpoint, report:

- all five seed values;
- mean, sample standard deviation, and a 95% confidence interval;
- paired seed-level differences for prespecified comparisons;
- effect size and uncertainty, not only rank;
- “inconclusive at this sample size” when intervals overlap materially.

The prespecified analysis is executable via
[`scripts/analyze_primary_statistics.py`](../scripts/analyze_primary_statistics.py).
It uses Student-t intervals (rather than a large-sample 1.96 shortcut) and matches
paired differences by the literal seed encoded in each checkpoint path.

A new untouched test shard must be evaluated once after model/evaluation decisions
are frozen. Validation remains the development set and cannot be relabeled as test
data.

### 8. MoE interpretability

After the new checkpoints finish, run
[`scripts/evaluate_moe_routing.py`](../scripts/evaluate_moe_routing.py) before the
case-study rewrite. It records:

- expert utilization by layer;
- router entropy relative to \(\log(E)\);
- position-bucket/expert affinity;
- expert-pair co-selection;
- pairwise cross-seed routing stability;
- the fact that this implementation has no capacity cutoff and therefore drops no
  selected routes.

Because expert IDs are permutation-symmetric across independent seeds, stability
first learns a one-to-one expert-label alignment on half of the matched validation
tokens and scores agreement on the held-out half. Raw expert IDs are never compared
directly. The recruiter story must connect MoE quality to measured routing behavior without
claiming that descriptive routing statistics causally explain the loss difference.
The architecture is motivated by
[Mixtral of Experts](https://arxiv.org/abs/2401.04088).

### 9. Reproducibility and configuration

The canonical manifest defines data, variants, seeds, optimizer settings, token
accounting, checkpoint policy, analysis endpoint, and output paths. The launcher
generates a resolved manifest containing the source-manifest hash, git SHA, dataset
manifest hash, platform, Python version, exact commands, and run status.

The exact prepared bytes are pinned by a tracked manifest and per-shard SHA-256
inventory. The original preprocessing run did not capture its upstream Hugging Face
revision, so that field remains explicitly unknown rather than being reconstructed
from a later cache state. Before final publication, create an untouched test split
and publish at least one representative checkpoint with a model card and data card.

### 10. Fault-tolerant primary training

Every new run uses the hardened checkpoint path. Completion requires a verified final
checkpoint. An interrupted run resumes only from the newest checkpoint whose file
hash matches the persisted ring metadata.

The trainer writes and verifies a step-zero bootstrap checkpoint before the first
optimizer update, requires ten finite observations before z-score anomaly decisions,
and stops after more than three consecutive rollback attempts. NaN/Inf checks remain
active immediately. A rollback restores loader/RNG/optimizer state, records a
structured recovery event, and retries the restored step; it is not counted as a
completed update. Logs, evaluations, and checkpoint intervals use one-based completed
optimizer-step numbers, so a record labeled step 3,750 represents exactly 3,750
updates and 245,760,000 tokens.

In addition to the existing injection tests, preserve one recruiter-facing recovery
demonstration: interrupt a debug run, corrupt its latest checkpoint, recover the
preceding verified checkpoint, and compare resumed versus uninterrupted loss. This is
systems evidence; it does not require rerunning the 50-run matrix.

The executable demonstration is
[`scripts/demonstrate_fault_recovery.py`](../scripts/demonstrate_fault_recovery.py);
its versioned result belongs at `reports/fault_recovery_demo.json`.

## Execution order and acceptance criteria

### Phase A — preflight and truth pass

- [x] Correct V0 GELU documentation and defaults.
- [x] Match MoE active parameters within 1% of Modern.
- [x] Add the canonical 500M/five-seed manifest and launcher.
- [x] Add a reproducible all-variant fault-tolerance smoke manifest.
- [x] Synchronize the experiment contract and shared defaults.
- [x] Pin all 101 prepared dataset shards with tracked SHA-256 values and explicitly
  record that the original upstream revision was not captured.
- [x] Run a one-step smoke test for all ten recipes with fault tolerance enabled;
  every run produced a SHA-256-verified final checkpoint on the L4.
- [x] Confirm at least 300 GB free checkpoint storage before launch (968 GB free at
  the 2026-07-18 preflight).
- [x] Demonstrate checksum-detected corruption, rollback, deterministic replay, and
  exact agreement with uninterrupted training.
- [x] Correct the launch-preflight health-monitor false positive, add a verified
  step-zero bootstrap, and make interval labels count completed updates.
- [x] Canonicalize relative checkpoint directories before async writes and ring
  registration, preventing the verifier from prefixing a project-relative path
  twice; cover the exact launch-path shape with a regression test.

### Phase B — primary retraining

- [x] Launch the manifest in named tmux session `transformer_500m_5seed` from
  commit `aa49e72bddd2403ecef54e896fb6b9b95663b7ce` (2026-07-18); the real matrix
  passed verified step-zero bootstrap and health-monitor warm-up.
- [ ] Produce 50 independent training histories.
- [ ] Produce 50 verified final checkpoints at step 7,629.
- [ ] Record any recovery event instead of hiding it.
- [ ] Stop and diagnose repeated non-finite loss, checksum, or health-monitor errors.

### Phase C — evaluation

- [x] Implement realistic serving matrix and recurrent/cache paths.
- [x] Implement passkey, needle, distance, and RoPE-base supplements.
- [x] Implement MoE routing evaluation.
- [x] Implement prespecified Student-t intervals and literal-seed paired differences.
- [ ] Evaluate fresh validation loss on all 50 final checkpoints.
- [ ] Build the untouched test split and run the frozen final evaluation.
- [ ] Run serving on one declared representative checkpoint per recipe.
- [ ] Run long-context and retrieval evaluation on all five seeds.
- [ ] Run MoE routing evaluation before interpreting MoE results.

### Phase D — secondary ablations

- [x] Implement three config-driven Modern counterfactuals and an isolated launcher.
- [ ] Run 3,750 steps over seeds 42, 137, and 2024.
- [ ] Promote only ambiguous, decision-relevant effects to five seeds.
- [ ] Benchmark SDPA versus FlashAttention with identical weights.

### Phase E — publication and recruiter presentation

- [ ] Replace the historical primary tables with the new report only after all 50
  runs and frozen evaluation complete.
- [ ] Keep the historical study in an explicitly labeled archive section.
- [ ] Export the new JSON/PNG asset bundle to the separate Jekyll repository.
- [ ] Add CI, a reproducible miniature experiment, model/data cards, and an artifact
  release link.

## Recruiter-facing narrative

The case study should lead with the experimental system rather than a universal
architecture winner:

> Built a fault-tolerant, single-GPU Transformer research platform and used it to run
> a 50-run, five-seed comparison of ten approximately 50M-active-parameter recipes on
> 25B processed tokens, with controlled data, active-parameter-matched MoE, realistic
> cache-aware serving benchmarks, and distance-controlled long-context diagnostics.

The page should present four acts:

1. **Question and controls.** Explain recipe-level comparison, fixed data/optimizer,
   five seeds, exact token budget, and active versus stored parameters.
2. **Quality and uncertainty.** Show fixed-data results with seed points and intervals;
   distinguish MoE conditional capacity from dense computation.
3. **Mechanisms and systems.** Contrast attention, position encoding, FFNs, caching,
   routing, and realized throughput. Use literature for expected component effects and
   this project for measured recipe-level outcomes.
4. **Failures and corrections.** Show the V5 causal-mask/stability correction, the
   discovered duplicate histories, the original MoE mismatch, and verified recovery.

Every figure needs a nearby statement of:

- what was measured;
- what the figure supports;
- what it does not establish;
- sample size and uncertainty unit;
- whether the evidence is primary, diagnostic, incomplete, or unsupported.

The strongest final claims should be limited to:

1. Modern recipe versus GELU Vanilla under this exact controlled regime.
2. Active-parameter-matched dense versus MoE quality and routing behavior.
3. Extrapolation stability versus actual distant retrieval.
4. The gap between theoretical attention efficiency and cache/kernel-aware realized
   serving performance.

Do not call these “1B-parameter models.” Use “approximately 50M-active-parameter
models trained for 500M tokens per seed.”
