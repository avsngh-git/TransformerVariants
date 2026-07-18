"""Create a tracked SHA-256 inventory for an already prepared shard dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(data_dir: Path, source_revision: str | None = None) -> dict:
    """Return provenance and byte hashes for every shard named by the manifest."""
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_names = [
        name
        for split in ("train", "val", "test")
        for name in manifest.get(split, {}).get("shards", [])
    ]
    if not shard_names:
        raise ValueError(f"No shards are declared in {manifest_path}")

    shards = []
    for name in shard_names:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest shard does not exist: {path}")
        shards.append(
            {
                "filename": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(data_dir),
        "source": manifest.get("source"),
        "source_revision": source_revision,
        "source_revision_status": (
            "captured" if source_revision else "not captured during original preprocessing"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "tokenizer": manifest.get("tokenizer"),
        "vocab_size": manifest.get("vocab_size"),
        "dtype": manifest.get("dtype"),
        "split_method": manifest.get("split_method"),
        "splits": {
            split: {
                "num_shards": manifest.get(split, {}).get("num_shards", 0),
                "total_tokens": manifest.get(split, {}).get("total_tokens", 0),
            }
            for split in ("train", "val", "test")
            if split in manifest
        },
        "shards": shards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        help="Exact upstream revision, only when it was recorded during preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory(args.data_dir, args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(inventory['shards'])} shard hashes to {args.output}")


if __name__ == "__main__":
    main()
