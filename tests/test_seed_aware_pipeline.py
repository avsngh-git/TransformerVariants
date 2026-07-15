"""Public-contract tests for seed-aware evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.comparison import VariantData, aggregate_pareto_variants
from src.evaluation.pipeline import EvaluationPipeline, ProbeResults
from src.evaluation.probes import CKAResult, MQARResult, StableRankResult
from src.models.config import ModelConfig


def _write_run(checkpoint_dir: Path, config: ModelConfig, final_loss: float) -> None:
    checkpoint_dir.mkdir(parents=True)
    run_config = {
        "model": {name: value for name, value in vars(config).items() if value is not None}
    }
    (checkpoint_dir / "run_config.json").write_text(json.dumps(run_config))
    entries = [
        {
            "type": "eval",
            "step": 1,
            "val_loss": final_loss + 1.0,
            "tokens_processed": 100,
            "elapsed_seconds": 1.0,
        },
        {
            "type": "eval",
            "step": 2,
            "val_loss": final_loss,
            "tokens_processed": 200,
            "elapsed_seconds": 2.0,
        },
    ]
    (checkpoint_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries)
    )


def test_pipeline_reports_seed_estimates_on_every_comparison_axis(tmp_path: Path) -> None:
    """The report contract exposes means and sample stds, never an arbitrary seed."""
    config = ModelConfig(
        n_layer=1,
        d_model=32,
        n_head=4,
        seq_len=16,
        vocab_size=128,
        variant="vanilla",
    )
    checkpoints: list[Path] = []
    for seed, loss in zip((42, 137, 2024), (3.0, 3.2, 2.8), strict=True):
        checkpoint_dir = tmp_path / f"vanilla_main_s{seed}"
        _write_run(checkpoint_dir, config, loss)
        checkpoints.append(checkpoint_dir)

    report_dir = tmp_path / "report"
    result = EvaluationPipeline(device="cpu").run(checkpoints, report_dir)

    assert result.comparison.fixed_data["vanilla"] == pytest.approx((3.0, 0.2))
    assert result.comparison.fixed_wallclock["vanilla"][1.0] == pytest.approx((3.0, 0.2))
    assert result.comparison.fixed_flops["vanilla"] == pytest.approx((3.0, 0.2))

    summary = (report_dir / "summary.md").read_text()
    assert "3.0000 ± 0.2000" in summary
    assert result.comparison.parameter_counts["vanilla"] > 0
    assert result.comparison.total_parameter_counts == result.comparison.parameter_counts

    raw = json.loads((report_dir / "raw" / "metrics.json").read_text())
    assert raw["schema_version"] == 2
    assert raw["probes"] == {"aggregated": {}, "per_seed": {}}
    assert raw["comparison"]["fixed_data"]["vanilla"] == {
        "mean": pytest.approx(3.0),
        "std": pytest.approx(0.2),
        "n": 3,
    }
    assert raw["comparison"]["total_parameter_counts"] == (raw["comparison"]["parameter_counts"])
    dashboard = report_dir / "index.html"
    assert dashboard.is_file()
    assert "<script src=" not in dashboard.read_text().lower()


def test_probe_results_retain_each_seed_and_publish_an_aggregate() -> None:
    """Probe collection keeps provenance while aggregate plots use every seed."""
    probes = ProbeResults()
    for checkpoint, accuracy, stable_rank in (
        ("linear_main_s42", 0.1, 4.0),
        ("linear_main_s137", 0.2, 5.0),
        ("linear_main_s2024", 0.3, 6.0),
    ):
        probes.record(
            variant="linear",
            checkpoint_dir=checkpoint,
            mqar=MQARResult(accuracy=accuracy, accuracy_by_distance={8: accuracy}),
            stable_rank=StableRankResult(
                per_layer=np.array([stable_rank, stable_rank + 1.0]),
                mean=stable_rank + 0.5,
                std=0.5,
            ),
            cka=CKAResult(
                adjacent_curve=np.array([accuracy]),
                full_matrix=np.array([[1.0, accuracy], [accuracy, 1.0]]),
            ),
            attention_entropy=None,
        )

    assert [seed.checkpoint_dir for seed in probes.per_seed["linear"]] == [
        "linear_main_s42",
        "linear_main_s137",
        "linear_main_s2024",
    ]
    assert probes.mqar["linear"].accuracy == pytest.approx(0.2)
    assert probes.stable_rank["linear"].per_layer == pytest.approx([5.0, 6.0])
    assert probes.cka["linear"].adjacent_curve == pytest.approx([0.2])

    payload = probes.to_dict()
    assert len(payload["per_seed"]["linear"]) == 3
    assert payload["per_seed"]["linear"][0]["checkpoint_dir"] == "linear_main_s42"
    assert payload["aggregated"]["linear"]["mqar"]["accuracy"] == pytest.approx(0.2)
    assert payload["aggregated"]["linear"]["stable_rank"]["per_layer"] == pytest.approx([5.0, 6.0])


def test_pareto_points_average_loss_time_and_memory_across_seeds() -> None:
    """Every Pareto axis uses a variant mean rather than seed zero."""
    config = ModelConfig(n_layer=1, d_model=32, n_head=4, vocab_size=128)
    seeds = [
        VariantData(
            name="vanilla",
            checkpoint_dir=Path(f"vanilla_s{seed}"),
            log_entries=[
                {
                    "val_loss": loss,
                    "elapsed_time": elapsed,
                    "peak_memory_mb": memory,
                }
            ],
            config=config,
        )
        for seed, loss, elapsed, memory in (
            (42, 3.0, 10.0, 100.0),
            (137, 3.3, 13.0, 130.0),
            (2024, 2.7, 7.0, 70.0),
        )
    ]

    point = aggregate_pareto_variants({"vanilla": seeds})[0]

    assert point.metrics is not None
    assert point.metrics.val_loss == pytest.approx(3.0)
    assert point.log_entries == [
        {"elapsed_time": pytest.approx(10.0), "peak_memory_mb": pytest.approx(100.0)}
    ]
