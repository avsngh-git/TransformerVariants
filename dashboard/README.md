# Transformer Variants Dashboard

An interactive Streamlit multi-page dashboard for comparing 7 decoder-only Transformer architecture variants. Visualizes pre-computed evaluation metrics as interactive Plotly charts with a dark-mode theme and colorblind-safe palette.

## What the Dashboard Shows

- **Overview** — Summary table of all variants with validation loss, perplexity, throughput, memory, and parameter parity badge
- **Learning Curves** — Validation loss over training (tokens seen, wall-clock time, or cumulative FLOPs) with optional seed envelope
- **Comparison Axes** — Bar charts comparing variants under fixed-data, fixed-wallclock, and fixed-FLOPs constraints with Pareto-front highlighting
- **Probes** — Diagnostic probe results: MQAR accuracy, stable rank, CKA similarity, and attention entropy
- **Efficiency** — Roofline diagram, FLOP breakdown, and MFU comparison
- **Per-Position Loss** — Per-position cross-entropy loss curves with ICL power-law fit overlays and decay exponent comparison table

## Prerequisites

- Python 3.11+
- Install dependencies:

```bash
pip install -r dashboard/requirements.txt
```

For development/testing, also install:

```bash
pip install -r dashboard/tests/requirements-test.txt
```

## Running Locally

From the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Configuring the Report Directory

The dashboard reads pre-computed evaluation results from a report directory. You can configure this in two ways:

1. **Environment variable** (takes precedence):

```bash
export REPORT_DIR=/path/to/your/reports
streamlit run dashboard/app.py
```

2. **Sidebar input** — If `REPORT_DIR` is not set, the dashboard displays a text input in the sidebar defaulting to `reports/` relative to the repository root.

The report directory should contain `raw/metrics.json` produced by the evaluation pipeline.

## Running Tests

```bash
cd dashboard && python -m pytest tests/ -v
```

## Deploying on Streamlit Cloud

1. Push your repository to GitHub (ensure the `dashboard/` directory and report data are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repository
3. Set the entry point to `dashboard/app.py`
4. Set the dependencies file to `dashboard/requirements.txt`
5. (Optional) Add `REPORT_DIR` as a secret in the Streamlit Cloud settings if your report data is at a non-default path
6. Deploy — the app will be available at a public URL
