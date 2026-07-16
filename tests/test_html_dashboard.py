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
            "vanilla": {
                "val_loss": {"mean": 3.6, "std": 0.1},
                "icl_alpha": {"mean": 0.5},
                "step_flops": {"mean": 100.0},
            },
            "modern": {
                "val_loss": {"mean": 3.4, "std": 0.05},
                "icl_alpha": {"mean": 0.6},
                "step_flops": {"mean": 120.0},
            },
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
        "probes": {
            "aggregated": {
                "modern": {
                    "mqar": {"accuracy": 0.5},
                    "stable_rank": {"mean": 9.0, "per_layer": [8.0, 10.0]},
                    "cka": {
                        "adjacent_curve": [0.8, 0.9],
                        "full_matrix": [[1.0, 0.2], [0.2, 1.0]],
                    },
                    "n": 3,
                }
            },
            "per_seed": {},
        },
    }
    (report_dir / "raw" / "metrics.json").write_text(json.dumps(metrics))
    (report_dir / "metadata.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-15T12:00:00+00:00",
                "hardware": "NVIDIA L4",
                "software_versions": {"python": "3.11"},
                "evaluated_checkpoints": ["modern_s42"],
                "warnings": ["Historical wall-clock uncertainty is incomplete."],
            }
        )
    )
    (report_dir / "raw" / "benchmarks.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "settings": {
                    "context_lengths": [1024, 2048, 4096],
                    "long_context_windows_per_checkpoint": 8,
                    "long_context_tail_tokens": 256,
                },
                "long_context_method": {
                    "target_alignment": "the same final tail tokens are scored",
                    "uncertainty_unit": "sample standard deviation across checkpoint seeds",
                },
                "limitations": ["Representative checkpoint only."],
                "variants": {
                    "vanilla": {
                        "generation": {
                            "uncached": {"status": "ok", "tokens_per_second": 12.3},
                            "cached": {"status": "unsupported"},
                            "kv_cache": {"status": "unsupported"},
                        },
                        "long_context": {
                            "1024": {
                                "status": "ok",
                                "val_loss": {"mean": 3.7, "std": 0.1, "n": 3},
                                "perplexity": {"mean": 40.5, "std": 4.0, "n": 3},
                                "perplexity_ratio": {"mean": 1.0, "std": 0.0, "n": 3},
                                "prefill_tokens_per_second": {
                                    "mean": 50000,
                                    "std": 1000,
                                    "n": 3,
                                },
                            },
                            "4096": {"status": "unsupported"},
                        },
                    },
                    "modern": {
                        "generation": {
                            "uncached": {"status": "ok", "tokens_per_second": 10.0},
                            "cached": {"status": "ok", "tokens_per_second": 15.0},
                            "kv_cache": {"status": "ok", "bytes": 4194304},
                        },
                        "long_context": {
                            "1024": {
                                "status": "ok",
                                "val_loss": {"mean": 3.5, "std": 0.1, "n": 3},
                                "perplexity": {"mean": 33.2, "std": 3.0, "n": 3},
                                "perplexity_ratio": {"mean": 1.0, "std": 0.0, "n": 3},
                                "prefill_tokens_per_second": {
                                    "mean": 45000,
                                    "std": 900,
                                    "n": 3,
                                },
                            },
                            "2048": {
                                "status": "ok",
                                "val_loss": {"mean": 3.8, "std": 0.1, "n": 3},
                                "perplexity": {"mean": 44.7, "std": 4.0, "n": 3},
                                "perplexity_ratio": {"mean": 1.35, "std": 0.05, "n": 3},
                                "prefill_tokens_per_second": {
                                    "mean": 42000,
                                    "std": 800,
                                    "n": 3,
                                },
                            },
                            "4096": {
                                "status": "ok",
                                "val_loss": {"mean": 4.1, "std": 0.2, "n": 3},
                                "perplexity": {"mean": 60.3, "std": 8.0, "n": 3},
                                "perplexity_ratio": {"mean": 1.82, "std": 0.1, "n": 3},
                                "prefill_tokens_per_second": {
                                    "mean": 39000,
                                    "std": 700,
                                    "n": 3,
                                },
                            },
                        },
                    },
                },
            }
        )
    )
    (report_dir / "plots" / "learning_curves_tokens.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (report_dir / "plots" / "stable_rank.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (report_dir / "plots" / "cka_adjacent.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (report_dir / "plots" / "cka_heatmap_modern.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    for plot_name in (
        "flop_breakdown",
        "pareto_flops_val_loss",
        "pareto_peak_memory_val_loss",
        "pareto_wallclock_val_loss",
        "roofline",
    ):
        (report_dir / "plots" / f"{plot_name}.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")

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

    assert 'id="longContextChart"' in html
    assert 'id="longContextSummary"' in html
    assert 'id="longContextRankings"' in html
    assert 'id="longContextChartSummary"' in html
    assert 'id="longContextRankingsSummary"' in html
    assert 'id="axisSummary"' in html
    assert 'class="plot-stack"' in html
    assert html.count('class="plot-panel"') == 9
    assert "Tracks validation loss against the number of training tokens" in html
    assert "Shows the effective dimensionality of each layer" in html
    assert "Representation similarity (CKA)" in html
    assert "The adjacent-layer curve highlights local transitions" in html
    assert "Training dynamics" in html
    assert "modern leads fixed-data quality" in html
    assert "modern averages 0.850 adjacent-layer CKA" in html
    assert html.count("The reported Pareto front contains modern.") == 2
    assert "wall-clock and memory plots mark their own axis-specific fronts" in html
    assert "the roofline relates arithmetic intensity to achieved throughput" in html
    assert "modern&#x27;s first-to-last-layer CKA is 0.200" in html
    assert 'id="provenanceCards"' in html
    assert "Evaluation hardware" in html
    assert "NVIDIA L4" in html
    assert "<summary>Software versions (1)</summary>" in html
    assert "<li><b>python</b>: 3.11</li>" in html
    assert "<summary>Evaluated checkpoints (1)</summary>" in html
    assert "<li>modern_s42</li>" in html
    assert "<summary>Methodology warnings (1)</summary>" in html
    assert "Historical wall-clock uncertainty is incomplete." in html
    assert "JSON.stringify({...meta" not in html
    embedded_report = html.split(
        '<script type="application/json" id="report-data">',
        maxsplit=1,
    )[1].split("</script>", maxsplit=1)[0]
    dashboard_data = json.loads(embedded_report)
    rankings = dashboard_data["benchmarks"]["long_context_rankings"]
    assert [entry["variant"] for entry in rankings["quality"]] == ["modern"]
    assert [entry["variant"] for entry in rankings["retention"]] == ["modern"]
    assert [entry["variant"] for entry in rankings["throughput"]] == ["modern"]
    assert "Paired tail-token extrapolation" in html
    assert "12.3" in html
