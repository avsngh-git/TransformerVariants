"""Tests for byte-level prepared-dataset provenance."""

from __future__ import annotations

import hashlib
import json

from scripts.hash_dataset import build_inventory


def test_build_inventory_hashes_every_manifest_shard(tmp_path) -> None:
    train_bytes = b"train shard"
    val_bytes = b"validation shard"
    (tmp_path / "train.bin").write_bytes(train_bytes)
    (tmp_path / "val.bin").write_bytes(val_bytes)
    manifest = {
        "source": "example/source",
        "tokenizer": "gpt2",
        "vocab_size": 50257,
        "dtype": "uint16",
        "train": {"num_shards": 1, "total_tokens": 5, "shards": ["train.bin"]},
        "val": {"num_shards": 1, "total_tokens": 6, "shards": ["val.bin"]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    inventory = build_inventory(tmp_path)

    assert inventory["source_revision"] is None
    assert [entry["filename"] for entry in inventory["shards"]] == [
        "train.bin",
        "val.bin",
    ]
    assert inventory["shards"][0]["sha256"] == hashlib.sha256(train_bytes).hexdigest()
    assert inventory["shards"][1]["sha256"] == hashlib.sha256(val_bytes).hexdigest()
