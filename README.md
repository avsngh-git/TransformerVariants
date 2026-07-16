# Transformer Variant Lab

A controlled, single-GPU study of decoder-only Transformer architectures on an
NVIDIA L4-24Q. The repository contains the full path from streamed dataset preparation
through training, checkpoint recovery, multi-seed evaluation, diagnostic probes,
inference benchmarking, and a self-contained offline HTML report.

The objective is not to train a frontier model. It is to answer smaller, more useful
systems-and-architecture questions under a fixed hardware and training budget:

- Which architectural changes improve validation loss without changing the data or
  optimization protocol?
- Which variants trade quality for training FLOPs, wall-clock time, memory, or serving
  capability?
- How do local attention, grouped-query attention, causal linear attention, and sparse
  MoE routing behave at roughly 50M active parameters?
- Can those comparisons be made reproducibly without hiding failed parity checks,
  unsupported cache paths, or incomplete seed histories?

## At a glance

| Item | Current project state |
|------|-----------------------|
| Hardware target | One NVIDIA L4-24Q, 24 GB |
| Training objective | Causal next-token prediction |
| Main context length | 1,024 tokens |
| Long-context diagnostics | 1,024 / 2,048 / 4,096 tokens |
| Implemented recipes | 10 |
| Main experiment | 30 runs: 10 recipes × 3 seeds |
| Main token budget | Approximately 1B tokens per run |
| Primary parameter range | 48.3M–68.8M active parameters |
| Test suite | 713 passing tests |
| Report surface | One offline, serverless HTML file |

Open the current report directly in a browser:

- [Static HTML dashboard](reports/1B_comparison/index.html)
- [Markdown evaluation summary](reports/1B_comparison/summary.md)
- [Detailed scientific and implementation notes](docs/project_notes.tex)
- [Experiment contract](reports/experiment_contract.md)

## Model families

All recipes are decoder-only language models. The modern-family recipes retain
RMSNorm and SwiGLU unless the row states otherwise.

| ID | Registry name | Architecture | Main question |
|----|---------------|--------------|---------------|
| V0 | `vanilla` | Learned positions, LayerNorm, standard causal MHA, dense FFN | What does a conventional baseline achieve? |
| V1 | `modern` | RoPE, RMSNorm, SwiGLU, memory-efficient full attention | What is gained by a modernized dense baseline? |
| V2 | `alibi` | ALiBi positional bias with full attention | Does position bias improve extrapolation without learned positions? |
| V3 | `gqa` | Grouped-query attention with two KV heads at main scale | How much KV projection capacity can be removed? |
| V4a | `swa` | Sliding-window attention in every layer | What is the quality cost of a strictly local receptive field? |
| V4b | `swa_interleaved` | Alternating full-attention and sliding-window layers | Can periodic global mixing recover local-attention quality? |
| V5 | `linear` | Strictly causal ELU+1 prefix-state attention with RoPE | Can linear-time attention remain stable and competitive? |
| V6a | `moe` | Eight experts, top-2 routing in every FFN layer | What does maximum sparse capacity add? |
| V6b | `moe_interleaved` | MoE in alternating FFN layers | Is partial sparse capacity a better trade-off? |
| V6c | `moe_deep` | Dense first half, MoE second half | Does placing experts in deeper layers help? |

The registry in [`src/models/registry.py`](src/models/registry.py) is the canonical
construction interface. A recipe describes its model class, attention module,
normalization, position encoding, FFN type, precision requirement, and optional
per-layer configuration.

## Current results

The table below is the final fixed-data comparison. Each value is a fresh held-out
checkpoint evaluation reported as mean ± sample standard deviation across seeds
42, 137, and 2024. Lower is better.

| Variant | Validation loss |
|---------|----------------:|
| MoE | **3.4292 ± 0.0013** |
| MoE-interleaved | 3.4826 ± 0.0042 |
| MoE-deep | 3.4846 ± 0.0010 |
| Modern | **3.6176 ± 0.0034** |
| Vanilla | 3.6984 ± 0.0035 |
| Causal linear | 3.8344 ± 0.0022 |
| ALiBi | 3.9025 ± 0.0023 |
| SWA-interleaved | 3.9333 ± 0.0105 |
| SWA | 3.9427 ± 0.0043 |
| GQA | 3.9565 ± 0.0034 |

The important interpretation is not simply “MoE wins”:

- `modern` is the strongest dense, parameter-matched baseline in this run.
- Full MoE obtains the lowest loss, but it violates the declared ±5% active-parameter
  tolerance and stores substantially more total capacity. It is therefore a useful
  sparse-capacity result, not a fair drop-in winner over the dense recipes.
- The corrected V5 causal-linear checkpoints train stably and load cleanly, but their
  quality and long-context degradation remain behind the strongest dense and sparse
  recipes.
- The FLOP-versus-loss Pareto set contains `moe` and `vanilla`; consult the dashboard
  before interpreting any single axis as an overall ranking.

### Parameter accounting

Sparse models need two parameter counts. **Active parameters** count the router and
the experts selected for a token; **total parameters** count every stored expert.

| Recipe | Active parameters | Total parameters |
|--------|------------------:|-----------------:|
| Vanilla, Modern, ALiBi, SWA, SWA-interleaved, Linear | 51,430,400 | 51,430,400 |
| GQA | 48,284,672 | 48,284,672 |
| MoE-interleaved, MoE-deep | 60,097,536 | 112,002,048 |
| MoE | 68,764,672 | 172,573,696 |

### What is statistically complete

The report deliberately distinguishes primary results from diagnostics:

- Fixed-data checkpoint quality uses fresh validation over all three independently
  trained weight states and reports sample standard deviations.
- Probe aggregates retain per-seed data and elementwise uncertainty for MQAR, stable
  rank, CKA, and attention entropy.
- Nine historical `metrics.jsonl` seed triplets were duplicated. Fixed-wall-clock and
  pre-endpoint fixed-FLOP values derived from those histories are labeled
  **incomplete/non-statistical** and do not display artificial `±0.0000` error bars.
- Generation, KV-cache, long-context, and target-free prefill measurements use one
  representative checkpoint per recipe. They are capability diagnostics, not
  seed-aggregated quality claims.
- Unsupported cache or context-extension paths remain visible as `unsupported`; they
  are never converted to zero or silently emulated.

These limitations are persisted in `metadata.json` and rendered in the dashboard’s
provenance section.

## Experimental protocol

The main comparison controls the following variables:

1. Identical tokenized shards and data order.
2. Identical token budget and causal-language-modeling objective.
3. Identical AdamW hyperparameters unless an exception is documented.
4. Identical effective batch size and precision policy.
5. Identical validation code and comparison budgets.
6. Three seeds for primary checkpoint-quality results.
7. Active and total parameter accounting at the report boundary.

The three comparison views answer different questions:

- **Fixed data:** which model reaches the lowest loss after seeing the same number of
  tokens?
- **Fixed wall-clock:** which model reaches the lowest loss within the same elapsed
  time?
- **Fixed FLOPs:** which model reaches the lowest loss under the same estimated compute
  budget?

See [the experiment contract](reports/experiment_contract.md) for the formal rules and
[the project notes](docs/project_notes.tex) for derivations, literature context, causal
masking details, V5 stability corrections, probe definitions, and result caveats.

## Hardware and scale tiers

| Scale | Layers | Width | Heads | Native context | Intended use |
|-------|-------:|------:|------:|---------------:|--------------|
| `debug` | 4 | 256 | 4 | 512 | Fast correctness and integration checks |
| `main` | 8 | 512 | 8 | 1,024 | Primary controlled experiments |
| `stretch` | 12 | 768 | 12 | 1,024 | Near-memory-limit exploration |

The project targets bf16 on an NVIDIA L4. Some implementations can run in float32
for testing, while FlashAttention-backed recipes require a compatible CUDA/PyTorch
environment for the intended training path.

## Repository layout

```text
TransformerVariants/
├── configs/
│   ├── data/                 # Dataset preparation configs
│   ├── model/                # Per-recipe model configs
│   ├── train/                # Debug and L4 training configs
│   └── project_defaults.yaml
├── dashboard/                # Legacy Streamlit surface; not the supported report
├── docs/                     # Phase docs, status, PRDs, and project_notes.tex
├── reports/
│   ├── 1B_comparison/        # Final HTML, plots, metadata, CSV, and JSON
│   └── experiment_contract.md
├── scripts/                  # Data, training, evaluation, benchmark, report CLIs
├── src/
│   ├── data/                 # Streaming preparation and memory-mapped loaders
│   ├── evaluation/           # Metrics, probes, FLOPs, comparisons, plots
│   ├── models/               # Attention, FFN, Transformer, and MoE implementations
│   ├── training/             # Trainer, logging, health checks, checkpoints
│   ├── utils/                # Configuration, seeding, parameter helpers
│   └── viz/                  # Self-contained HTML dashboard generator
├── tests/                    # Unit, property, integration, and failure tests
├── checkpoints/              # Local model checkpoints; generally not portable
├── data/                     # Prepared token shards; git-ignored
└── runs/                     # Per-run logs and configs; git-ignored
```

## Installation

Python 3.10 or newer is required. A CUDA-enabled PyTorch installation is recommended
for training; CPU execution is suitable for many tests and small debug checks.

```bash
git clone <repository-url>
cd TransformerVariants

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[data,viz,dev]"
```

FlashAttention is optional because its wheel/build requirements depend on the local
CUDA and PyTorch versions:

```bash
pip install -e ".[flash]"
```

If FlashAttention installation fails, confirm that the CUDA toolkit, compiler, and
installed PyTorch CUDA version are compatible before changing project code.

## Quick start: debug workflow

### 1. Prepare a small dataset

Dataset preparation downloads from Hugging Face, tokenizes with GPT-2 BPE, filters
documents, and writes binary train/validation shards.

```bash
python scripts/prepare_data.py \
  --config configs/data/debug.yaml \
  --output-dir data/processed/wikitext-debug
```

### 2. Train a small model

```bash
python scripts/train.py \
  --variant modern \
  --scale debug \
  --data_dir data/processed/wikitext-debug \
  --max_steps 200 \
  --eval_interval 50 \
  --checkpoint_interval 100 \
  --checkpoint_dir checkpoints/modern_debug_s42 \
  --seed 42
```

Useful training options include:

- `--variant`: any key in the model-family table.
- `--scale`: `debug`, `main`, or `stretch`.
- `--dtype`: `bfloat16`, `float16`, or `float32`.
- `--compile`: enable `torch.compile` for compatible dense recipes.
- `--resume PATH`: restore model, optimizer, scheduler, scaler, RNG, and progress state.
- `--checkpoint_dir PATH`: choose a stable output directory for scripted sweeps.

### 3. Evaluate checkpoints

The evaluator loads logs and weights, computes fresh validation metrics and probes,
aggregates seeds, generates PNGs and raw data, writes `summary.md`, and builds
`index.html`.

```bash
python scripts/evaluate.py \
  --checkpoints checkpoints/modern_debug_s42 \
  --output reports/debug_comparison \
  --data_dir data/processed/wikitext-debug
```

The checkpoint path matches the explicit output directory in the training command above.

## Reproducing the main experiment

The full sweep is expensive: 30 main-scale runs and roughly 1B tokens per run. On the
target L4 it is a multi-day experiment, not a quick-start command.

```bash
# Prepare the 1B-token FineWeb-Edu shards expected by the sweep scripts.
bash scripts/prepare_1B_data.sh

# Train all registered recipes for seeds 42, 137, and 2024.
bash scripts/train_all_1B.sh 2>&1 | tee training_all_1B.log
```

The finalized V5 results use the numerically corrected causal-linear checkpoints:

```text
checkpoints/linear_main_1B_fixed_s42
checkpoints/linear_main_1B_fixed_s137
checkpoints/linear_main_1B_fixed_s2024
```

The older unsuffixed V5 checkpoint directories contain the superseded Linformer-style
`E`/`F` projection tensors and must not be mixed into the final report. The correction
script is [`scripts/retrain_v5_fixed_1B.sh`](scripts/retrain_v5_fixed_1B.sh).

To rebuild the final multi-seed evaluation, pass the nine standard recipe triplets and
the three fixed V5 checkpoints:

```bash
python scripts/evaluate.py \
  --checkpoints \
    checkpoints/{alibi,gqa,modern,moe_deep,moe_interleaved,moe,swa_interleaved,swa,vanilla}_main_1B_s*/ \
    checkpoints/linear_main_1B_fixed_s*/ \
  --output reports/1B_comparison \
  --data_dir data/processed/fineweb-1B
```

Brace and wildcard expansion in the command above requires a compatible shell such as
Bash.

## Inference and long-context benchmark

The benchmark keeps uncached generation, cached generation, persistent KV-cache
storage, validation loss, and target-free prefill throughput separate. A cached result
is emitted only when the model returns a genuinely reusable prompt cache.

```bash
python scripts/benchmark_inference.py \
  --checkpoints \
    checkpoints/{alibi,gqa,modern,moe_deep,moe_interleaved,moe,swa_interleaved,swa,vanilla}_main_1B_s42 \
    checkpoints/linear_main_1B_fixed_s42 \
  --output reports/1B_comparison/raw/benchmarks.json \
  --data-dir data/processed/fineweb-1B \
  --prompt-length 64 \
  --new-tokens 8 \
  --repeats 2 \
  --warmups 1 \
  --context-lengths 1024 2048 4096
```

These measurements intentionally use one representative checkpoint per recipe. Repeat
the benchmark across checkpoints and validation windows before making statistical
serving claims.

## Static HTML dashboard

Streamlit is no longer the supported reporting surface. The report generator produces
one portable HTML file containing its CSS, JavaScript, JSON data, metadata, benchmark
statuses, and PNG plots. It requires no Python server, CDN, network connection, or
tracking runtime.

```bash
python scripts/build_dashboard.py --report reports/1B_comparison
```

Then open [`reports/1B_comparison/index.html`](reports/1B_comparison/index.html)
directly in a browser. Rebuild it after changing `raw/metrics.json`,
`raw/benchmarks.json`, metadata, or plot files.

## Evaluation outputs

An evaluation report contains:

```text
reports/<experiment>/
├── index.html               # Self-contained dashboard
├── summary.md               # Human-readable comparison tables
├── metadata.json            # Environment, checkpoints, warnings, provenance
├── plots/                   # Publication-oriented PNG figures
└── raw/
    ├── metrics.csv          # Flat checkpoint metrics
    ├── metrics.json         # Versioned aggregate + per-seed schema
    └── benchmarks.json      # Serving and long-context diagnostics
```

The JSON report retains individual checkpoint provenance, named estimates
(`mean`, `std`, `n`), active and total parameter counts, per-seed probes, aggregate
probe uncertainty, Pareto membership, and explicit unsupported statuses.

## Testing and quality gates

Run the full suite:

```bash
pytest -q
```

Run lint on the actively maintained evaluation and reporting stack:

```bash
ruff check \
  src/evaluation/pipeline.py \
  src/evaluation/comparison.py \
  src/evaluation/benchmarks.py \
  src/evaluation/visualizations.py \
  src/viz/html_dashboard.py \
  scripts/evaluate.py \
  scripts/benchmark_inference.py \
  scripts/build_dashboard.py
```
Repository-wide Ruff currently also scans the legacy Streamlit surface and older test
modules that retain known lint debt, so it is not yet used as a clean global gate.


The suite covers, among other things:

- causal masking and future-token independence;
- linear-attention numerical properties;
- FlashAttention, ALiBi, GQA, SWA, and MoE integration;
- checkpoint atomicity, integrity, resume, rollback, and fault injection;
- parameter and FLOP accounting;
- multi-seed aggregation and duplicate-history handling;
- probes, inference benchmarks, report generation, and offline HTML integrity.

## Project status

| Phase | Goal | Status |
|------:|------|--------|
| 00–03 | Contract, repository, data, vanilla model | Complete |
| 04–07 | Training system and architecture variants | Complete |
| 08–10 | Evaluation, static reporting, large-scale data | Complete |
| 11 | Fault-tolerant training and recovery hardening | Partial |
| 12 | Main controlled benchmark | Complete |
| 13 | Final report, packaging, and demo assets | In progress |

The remaining work is primarily packaging and publication work. Fault-tolerance code
and tests exist, but Phase 11 remains marked partial until every planned recovery and
operational scenario in the experiment contract is closed.

## Adding a new recipe

1. Implement or reuse an attention/FFN/model component under `src/models/`.
2. Add a `VariantSpec` to `src/models/registry.py`.
3. Use `config_overrides` for recipe-wide settings and
   `per_layer_config_builder` for heterogeneous layers.
4. Add causality, shape, numerical, parameter-count, training, and checkpoint-loading
   tests.
5. Train the same seeds and token budget before adding the recipe to formal tables.
6. Document parameter-parity exceptions and unsupported serving behavior explicitly.

## Design principles

- **Controlled comparisons over leaderboard claims.** Every reported advantage should
  identify its data, compute, parameter, and seed boundary.
- **Unsupported is a result.** Missing cache or extrapolation support remains visible.
- **No fabricated uncertainty.** Duplicated histories never become zero-width error
  bars.
- **One model registry.** Construction knowledge stays centralized and testable.
- **Raw data and presentation stay separate.** Versioned JSON is the machine interface;
  HTML and Markdown are generated views.
- **Single-GPU realism.** Memory headroom, resumability, and wall-clock cost are part of
  the experiment rather than afterthoughts.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
