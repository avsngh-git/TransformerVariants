# Legacy Streamlit Dashboard

This directory contains the superseded Streamlit/Plotly dashboard for historical
reference. It is no longer the supported report interface.

The current dashboard is one self-contained HTML file generated from evaluation
artifacts:

```bash
python scripts/build_dashboard.py --report reports/1B_comparison
```

Open `reports/1B_comparison/index.html` directly. It has no server, external
assets, tracking, or internet dependency. The evaluation pipeline also rebuilds
the file automatically.

Inputs:

- `raw/metrics.json`: seed-aware metrics, comparisons, and probes
- `raw/benchmarks.json`: optional inference/cache/long-context results
- `metadata.json`: environment provenance
- `plots/*.png`: embedded publication figures
