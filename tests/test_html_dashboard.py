"""Public-contract tests for the self-contained HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from src.viz.html_dashboard import build_dashboard


def test_build_dashboard_writes_one_offline_html_file(tmp_path: Path) -> None:
    """A report directory becomes a directly openable dashboard with embedded assets."""
    report_dir = tmp_path / "report"
    (report_dir / "raw").mkdir(parents=True)
    (report_dir / "plots").mkdir()
    metrics = {
        "schema_version": 2,
        "variants": {
            "vanilla": [{"checkpoint_dir": "vanilla_s42", "val_loss": 3.6}],
            "modern": [{"checkpoint_dir": "modern_s42", "val_loss": 3.4}],
        },
        "aggregated": {
            "vanilla": {"val_loss": {"mean": 3.6, "std": 0.1}},
            "modern": {"val_loss": {"mean": 3.4, "std": 0.05}},
        },
        "comparison": {
            "fixed_data": {
                "vanilla": {"mean": 3.6, "std": 0.1, "n": 3},
                "modern": {"mean": 3.4, "std": 0.05, "n": 3},
            },
            "fixed_wallclock": {},
            "fixed_flops": {},
            "pareto_front": ["modern"],
            "parameter_counts": {"vanilla": 51_000_000, "modern": 59_000_000},
            "parameter_parity_valid": False,
            "total_parameter_counts": {"vanilla": 51_000_000, "modern": 59_000_000},
        },
        "probes": {"aggregated": {}, "per_seed": {}},
    }
    (report_dir / "raw" / "metrics.json").write_text(json.dumps(metrics))
    (report_dir / "metadata.json").write_text(
        json.dumps({"timestamp": "2026-07-15T12:00:00+00:00", "hardware": "NVIDIA L4"})
    )
    (report_dir / "raw" / "benchmarks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "limitations": ["Representative checkpoint only."],
                "variants": {
                    "vanilla": {
                        "generation": {
                            "uncached": {"status": "ok", "tokens_per_second": 12.3},
                            "cached": {"status": "unsupported"},
                            "kv_cache": {"status": "unsupported"},
                        },
                        "long_context": {"1024": {"status": "ok", "perplexity": 42.0}},
                    }
                },
            }
        )
    )
    (report_dir / "plots" / "learning_curves_tokens.png").write_bytes(
        b"\x89PNG\r\n\x1a\nfixture"
    )

    output = build_dashboard(report_dir)

    assert output == report_dir / "index.html"
    html = output.read_text()
    assert "Transformer Variant Lab" in html
    assert 'id="overview"' in html
    assert 'id="comparisons"' in html
    assert 'id="probes"' in html
    assert 'id="artifacts"' in html
    assert "data:image/png;base64," in html
    assert '"schema_version": 2' in html
    assert "streamlit" not in html.lower()
    assert 'id="benchmarks"' in html
    assert "<script src=" not in html.lower()
    assert "https://" not in html.lower()
    assert "Active params" in html
    assert "Total params" in html
    assert "Representative checkpoint only." in html
    assert "12.3" in html
