"""Launch a canonical serial training matrix from a versioned YAML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.models.registry import SCALES, VARIANTS
from src.training.checkpoint import AtomicCheckpointWriter


def token_accounting(manifest: dict) -> dict[str, int]:
    """Return exact per-step, per-run, and matrix token counts."""
    training = manifest["training"]
    tokens_per_step = (
        int(training["sequence_length"])
        * int(training["micro_batch_size"])
        * int(training["grad_accum_steps"])
    )
    tokens_per_run = tokens_per_step * int(training["max_steps"])
    total_runs = len(manifest["variants"]) * len(manifest["seeds"])
    return {
        "tokens_per_step": tokens_per_step,
        "tokens_per_run": tokens_per_run,
        "total_runs": total_runs,
        "total_tokens": tokens_per_run * total_runs,
    }


def _validate_manifest(manifest: dict) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "data_dir",
        "scale",
        "variants",
        "seeds",
        "training",
        "fault_tolerance",
        "variant_overrides",
        "output",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Manifest is missing required keys: {', '.join(missing)}")
    unknown_variants = sorted(set(manifest["variants"]) - VARIANTS.keys())
    if unknown_variants:
        raise ValueError(f"Unknown variants: {', '.join(unknown_variants)}")
    if len(set(manifest["variants"])) != len(manifest["variants"]):
        raise ValueError("variants must be unique")
    if len(set(manifest["seeds"])) != len(manifest["seeds"]):
        raise ValueError("seeds must be unique")
    if not manifest["variants"] or not manifest["seeds"]:
        raise ValueError("variants and seeds must not be empty")
    scale = manifest["scale"]
    if scale not in SCALES:
        raise ValueError(f"Unknown scale: {scale}")

    training_required = {
        "max_steps",
        "target_tokens_per_run",
        "sequence_length",
        "micro_batch_size",
        "grad_accum_steps",
        "max_lr",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "dtype",
        "eval_interval",
        "eval_steps",
        "checkpoint_interval",
        "log_interval",
    }
    missing_training = sorted(training_required - manifest["training"].keys())
    if missing_training:
        raise ValueError(
            f"Manifest training section is missing: {', '.join(missing_training)}"
        )
    declared_length = int(manifest["training"]["sequence_length"])
    registry_length = int(SCALES[scale]["seq_len"])
    if declared_length != registry_length:
        raise ValueError(
            f"training.sequence_length={declared_length} does not match "
            f"registry scale {scale!r} seq_len={registry_length}"
        )
    if manifest["training"]["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("training.dtype must be bfloat16, float16, or float32")

    output_required = {"run_template", "checkpoint_template", "resolved_manifest"}
    missing_output = sorted(output_required - manifest["output"].keys())
    if missing_output:
        raise ValueError(f"Manifest output section is missing: {', '.join(missing_output)}")
    for field in ("run_template", "checkpoint_template"):
        template = str(manifest["output"][field])
        if "{variant}" not in template or "{seed}" not in template:
            raise ValueError(f"output.{field} must contain {{variant}} and {{seed}}")

    if manifest["fault_tolerance"].get("enabled", False):
        if int(manifest["fault_tolerance"].get("checkpoint_ring_size", 0)) < 2:
            raise ValueError("A fault-tolerant run requires checkpoint_ring_size >= 2")
        if not manifest["fault_tolerance"].get("resume_latest_verified", False):
            raise ValueError("A fault-tolerant run must resume the latest verified checkpoint")

    accounting = token_accounting(manifest)
    target = int(manifest["training"]["target_tokens_per_run"])
    if accounting["tokens_per_run"] > target:
        raise ValueError(
            f"Configured run uses {accounting['tokens_per_run']:,} tokens, which exceeds "
            f"target_tokens_per_run={target:,}"
        )
    shortfall = target - accounting["tokens_per_run"]
    if shortfall >= accounting["tokens_per_step"]:
        raise ValueError(
            "max_steps leaves at least one full step unused relative to target_tokens_per_run"
        )


def load_manifest(path: str | Path) -> dict:
    """Load and validate a canonical experiment manifest."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    manifest = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(manifest, dict):
        raise ValueError("Experiment manifest must contain a JSON object")
    _validate_manifest(manifest)
    return manifest


def checkpoint_dir(manifest: dict, variant: str, seed: int) -> Path:
    template = manifest["output"]["checkpoint_template"]
    return Path(template.format(variant=variant, seed=seed, scale=manifest["scale"]))


def run_dir(manifest: dict, variant: str, seed: int) -> Path:
    template = manifest["output"]["run_template"]
    return Path(template.format(variant=variant, seed=seed, scale=manifest["scale"]))


def final_checkpoint(manifest: dict, variant: str, seed: int) -> Path:
    step = int(manifest["training"]["max_steps"])
    return checkpoint_dir(manifest, variant, seed) / f"checkpoint_step_{step:06d}.pt"


def build_training_command(
    manifest: dict,
    *,
    variant: str,
    seed: int,
    python_executable: str = sys.executable,
    resume: bool = False,
) -> list[str]:
    """Build one deterministic training command from the manifest."""
    training = manifest["training"]
    fault_tolerance = manifest["fault_tolerance"]
    override = manifest.get("variant_overrides", {}).get(variant, {})
    command = [
        python_executable,
        "scripts/train.py",
        "--variant",
        variant,
        "--scale",
        str(manifest["scale"]),
        "--seed",
        str(seed),
        "--data_dir",
        str(manifest["data_dir"]),
        "--max_steps",
        str(training["max_steps"]),
        "--max_lr",
        str(training["max_lr"]),
        "--min_lr",
        str(training["min_lr"]),
        "--weight_decay",
        str(training["weight_decay"]),
        "--beta1",
        str(training["beta1"]),
        "--beta2",
        str(training["beta2"]),
        "--warmup_steps",
        str(training["warmup_steps"]),
        "--micro_batch_size",
        str(training["micro_batch_size"]),
        "--grad_accum_steps",
        str(training["grad_accum_steps"]),
        "--grad_clip",
        str(training["grad_clip"]),
        "--eval_interval",
        str(training["eval_interval"]),
        "--eval_steps",
        str(training["eval_steps"]),
        "--checkpoint_interval",
        str(training["checkpoint_interval"]),
        "--log_interval",
        str(training["log_interval"]),
        "--checkpoint_dir",
        str(checkpoint_dir(manifest, variant, seed)),
        "--run-dir",
        str(run_dir(manifest, variant, seed)),
        "--dtype",
        str(training["dtype"]),
    ]
    if override.get("activation"):
        command.extend(["--activation", str(override["activation"])])
    if override.get("compile", False):
        command.append("--compile")
    if fault_tolerance.get("enabled", False):
        command.extend(
            [
                "--fault-tolerant",
                "--checkpoint-ring-size",
                str(fault_tolerance["checkpoint_ring_size"]),
            ]
        )
    if resume:
        command.extend(["--resume", "latest"])
    return command


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_resolved_manifest(manifest: dict, source_path: Path, runs: list[dict]) -> Path:
    """Snapshot code/data provenance and exact commands before training."""
    resolved = deepcopy(manifest)
    dataset_manifest = Path(manifest.get("dataset_manifest", ""))
    dataset_inventory = Path(manifest.get("dataset_inventory", ""))
    resolved["resolved"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256(source_path),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "dataset_inventory_sha256": _sha256(dataset_inventory),
        "token_accounting": token_accounting(manifest),
        "runs": runs,
    }
    output = Path(manifest["output"]["resolved_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    return output


def _has_verified_resume(checkpoint_path: Path) -> bool:
    ring_path = checkpoint_path / "checkpoint_ring.json"
    if not ring_path.exists():
        return False
    try:
        entries = json.loads(ring_path.read_text(encoding="utf-8")).get("entries", [])
    except (json.JSONDecodeError, OSError):
        return False
    return any(
        AtomicCheckpointWriter.verify_trusted(checkpoint_path / entry.get("path", ""))
        for entry in reversed(entries)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/experiment/main_500m_5seed.yaml",
        help="Canonical YAML experiment manifest (JSON accepted for compatibility)",
    )
    parser.add_argument("--variant", action="append", help="Run only selected variant(s)")
    parser.add_argument("--seed", action="append", type=int, help="Run only selected seed(s)")
    parser.add_argument("--max-runs", type=int, help="Stop after this many non-completed runs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print without training"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.manifest)
    manifest = load_manifest(source_path)
    variants = [v for v in manifest["variants"] if not args.variant or v in args.variant]
    seeds = [s for s in manifest["seeds"] if not args.seed or s in args.seed]
    if args.variant and set(args.variant) - set(manifest["variants"]):
        raise SystemExit("A requested variant is not part of the manifest")
    if args.seed and set(args.seed) - set(manifest["seeds"]):
        raise SystemExit("A requested seed is not part of the manifest")
    if not Path(manifest["data_dir"]).is_dir() and not args.dry_run:
        raise SystemExit(f"Data directory does not exist: {manifest['data_dir']}")

    accounting = token_accounting(manifest)
    print(f"Experiment: {manifest['experiment_id']}", flush=True)
    print(
        f"Matrix: {len(variants)} variants x {len(seeds)} seeds; "
        f"{accounting['tokens_per_run']:,} tokens/run",
        flush=True,
    )

    run_records: list[dict] = []
    pending: list[tuple[str, int, list[str], dict]] = []
    for variant in variants:
        for seed in seeds:
            final = final_checkpoint(manifest, variant, seed)
            completed = final.exists() and AtomicCheckpointWriter.verify_trusted(final)
            resume = not completed and _has_verified_resume(checkpoint_dir(manifest, variant, seed))
            command = build_training_command(
                manifest,
                variant=variant,
                seed=seed,
                resume=resume,
            )
            record = {
                "variant": variant,
                "seed": seed,
                "checkpoint_dir": str(checkpoint_dir(manifest, variant, seed)),
                "run_dir": str(run_dir(manifest, variant, seed)),
                "status": "completed" if completed else "resume_pending" if resume else "pending",
                "command": command,
            }
            if resume:
                record["recovery_event"] = "resume_from_latest_verified_checkpoint"
            run_records.append(record)
            if not completed:
                pending.append((variant, seed, command, record))

    resolved_path = write_resolved_manifest(manifest, source_path, run_records)
    print(f"Resolved manifest: {resolved_path}", flush=True)
    for variant, seed, command, _record in pending:
        print(f"PENDING {variant} seed={seed}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        print(f"Dry run complete: {len(pending)} run(s) pending", flush=True)
        return

    launch_count = 0
    total_pending = len(pending)
    for index, (variant, seed, command, record) in enumerate(pending, start=1):
        if args.max_runs is not None and launch_count >= args.max_runs:
            break
        launch_count += 1
        print(f"START [{index}/{total_pending}] {variant} seed={seed}", flush=True)
        record["status"] = "running"
        record["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_resolved_manifest(manifest, source_path, run_records)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["return_code"] = exc.returncode
            record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_resolved_manifest(manifest, source_path, run_records)
            raise
        final = final_checkpoint(manifest, variant, seed)
        if not final.exists() or not AtomicCheckpointWriter.verify_trusted(final):
            record["status"] = "failed_verification"
            record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_resolved_manifest(manifest, source_path, run_records)
            raise RuntimeError(f"Run returned without a verified final checkpoint: {final}")
        record["status"] = "completed"
        record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["final_checkpoint"] = str(final)
        write_resolved_manifest(manifest, source_path, run_records)
        print(f"DONE  [{index}/{total_pending}] {variant} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
