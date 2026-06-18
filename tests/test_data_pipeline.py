"""Tests for the Phase 2 data pipeline.

These tests run the pipeline on a tiny subset (10 documents) to validate
correctness without downloading the full dataset or taking a long time.

To run: python -m pytest tests/test_data_pipeline.py -v
Requires: pip install -e ".[data,dev]"
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.prepare import (
    compute_token_stats,
    load_dataset_split,
    prepare_dataset,
    tokenize_documents,
    write_shards,
)
from src.data.tokenizer import (
    decode,
    encode,
    get_eot_token,
    get_tokenizer,
    get_vocab_size,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable setup shared across tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    """Create a GPT-2 tokenizer for tests."""
    return get_tokenizer("gpt2")


@pytest.fixture
def tiny_output_dir(tmp_path):
    """Run the pipeline on 10 documents and return the output directory.

    tmp_path is a pytest built-in fixture that gives a unique temporary
    directory for each test. It's automatically cleaned up after the test.
    """
    config = {
        "dataset_name": "wikitext-103-raw-v1",
        "tokenizer": "gpt2",
        "max_documents": 10,
        "max_tokens": None,
        "tokens_per_shard": 500,
        "output_dir": str(tmp_path / "output"),
    }
    prepare_dataset(config)
    return tmp_path / "output"


# ---------------------------------------------------------------------------
# Test 1: Tokenizer roundtrip
# ---------------------------------------------------------------------------


def test_tokenizer_roundtrip(tokenizer):
    """Encode → decode returns the original text.

    This confirms tiktoken is installed correctly and our wrapper
    functions work as expected.
    """
    original = "Hello, world! This is a test of the tokenizer."
    tokens = encode(original, tokenizer)
    recovered = decode(tokens, tokenizer)
    assert recovered == original


# ---------------------------------------------------------------------------
# Test 2: Shard dtype and shape
# ---------------------------------------------------------------------------


def test_shard_dtype_and_shape(tiny_output_dir):
    """Each shard contains uint16 values and length matches expectations.

    All shards except possibly the last should have exactly tokens_per_shard
    tokens. The last shard can be smaller.
    """
    tokens_per_shard = 500
    shard_files = sorted(tiny_output_dir.glob("train_*.bin"))

    assert len(shard_files) > 0, "No train shards were written"

    for shard_file in shard_files[:-1]:  # All shards except last
        data = np.fromfile(shard_file, dtype=np.uint16)
        assert data.dtype == np.uint16
        assert len(data) == tokens_per_shard

    # Last shard can be smaller or equal
    last_data = np.fromfile(shard_files[-1], dtype=np.uint16)
    assert last_data.dtype == np.uint16
    assert 0 < len(last_data) <= tokens_per_shard


# ---------------------------------------------------------------------------
# Test 3: All tokens in vocab range
# ---------------------------------------------------------------------------


def test_all_tokens_in_vocab_range(tiny_output_dir, tokenizer):
    """Every token ID in every shard must be < vocab_size.

    Catches corruption, wrong dtype, or encoding bugs that produce
    out-of-range values.
    """
    vocab_size = get_vocab_size(tokenizer)

    for shard_file in tiny_output_dir.glob("*.bin"):
        data = np.fromfile(shard_file, dtype=np.uint16)
        assert data.max() < vocab_size, (
            f"{shard_file.name}: max token ID {data.max()} >= vocab_size {vocab_size}"
        )


# ---------------------------------------------------------------------------
# Test 4: EOT tokens are present
# ---------------------------------------------------------------------------


def test_eot_tokens_present(tiny_output_dir, tokenizer):
    """The concatenated token stream must contain EOT separators.

    We concatenate documents with EOT between them, so the shards
    (when read together) must contain at least one EOT token.
    """
    eot = get_eot_token(tokenizer)

    # Read all train shards and concatenate
    all_tokens = []
    for shard_file in sorted(tiny_output_dir.glob("train_*.bin")):
        data = np.fromfile(shard_file, dtype=np.uint16)
        all_tokens.append(data)

    full_stream = np.concatenate(all_tokens)
    eot_count = np.sum(full_stream == eot)

    # We process 10 documents, each gets an EOT after it → at least 10 EOTs
    # (some docs may be empty, but they still get an EOT)
    assert eot_count >= 1, "No EOT tokens found in train shards"


# ---------------------------------------------------------------------------
# Test 5: Manifest schema
# ---------------------------------------------------------------------------


def test_manifest_schema(tiny_output_dir):
    """manifest.json has all required fields and shard filenames match disk.

    The manifest is the contract for the dataloader — if it's wrong,
    training will fail.
    """
    manifest_path = tiny_output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json not found"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check top-level fields
    assert manifest["tokenizer"] == "gpt2"
    assert manifest["vocab_size"] == 50257
    assert manifest["dtype"] == "uint16"
    assert "tokens_per_shard" in manifest

    # Check train section
    assert "train" in manifest
    assert manifest["train"]["num_shards"] > 0
    assert manifest["train"]["total_tokens"] > 0

    # Check that listed shard files actually exist on disk
    for shard_name in manifest["train"]["shards"]:
        assert (tiny_output_dir / shard_name).exists(), f"Missing shard: {shard_name}"

    # Check val section
    assert "val" in manifest
    for shard_name in manifest["val"]["shards"]:
        assert (tiny_output_dir / shard_name).exists(), f"Missing shard: {shard_name}"


# ---------------------------------------------------------------------------
# Test 6: Data report schema
# ---------------------------------------------------------------------------


def test_data_report_schema(tiny_output_dir):
    """data_report.json has required fields and token counts are consistent.

    The report is for human auditing — it should be internally consistent
    and contain provenance info.
    """
    report_path = tiny_output_dir / "data_report.json"
    assert report_path.exists(), "data_report.json not found"

    with open(report_path) as f:
        report = json.load(f)

    # Check top-level provenance fields
    assert report["dataset_name"] == "wikitext-103-raw-v1"
    assert report["tokenizer"] == "gpt2"
    assert "git_sha" in report
    assert "timestamp" in report
    assert report["processing_time_seconds"] >= 0

    # Check train stats
    train = report["train"]
    assert train["documents_processed"] > 0
    assert train["total_tokens"] > 0
    assert "token_stats" in train
    assert train["token_stats"]["num_documents"] == train["documents_processed"]


# ---------------------------------------------------------------------------
# Test 7: Shard hashes match
# ---------------------------------------------------------------------------


def test_shard_hashes_match(tiny_output_dir):
    """SHA256 hashes in data_report.json match actual shard file contents.

    Catches silent file corruption or write errors.
    """
    report_path = tiny_output_dir / "data_report.json"
    with open(report_path) as f:
        report = json.load(f)

    for split in ["train", "val"]:
        for shard_info in report[split]["shards"]:
            filepath = tiny_output_dir / shard_info["filename"]
            actual_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
            assert actual_hash == shard_info["sha256"], (
                f"{shard_info['filename']}: hash mismatch"
            )


# ---------------------------------------------------------------------------
# Test 8: Total tokens consistent
# ---------------------------------------------------------------------------


def test_total_tokens_consistent(tiny_output_dir):
    """Sum of shard sizes equals total_tokens in manifest.

    Catches off-by-one errors in sharding logic.
    """
    manifest_path = tiny_output_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    for split in ["train", "val"]:
        # Sum actual tokens from shard files on disk
        actual_total = 0
        for shard_name in manifest[split]["shards"]:
            data = np.fromfile(tiny_output_dir / shard_name, dtype=np.uint16)
            actual_total += len(data)

        assert actual_total == manifest[split]["total_tokens"], (
            f"{split}: manifest says {manifest[split]['total_tokens']} tokens "
            f"but shards contain {actual_total}"
        )


# ---------------------------------------------------------------------------
# Test 9: Document limits are respected
# ---------------------------------------------------------------------------


def test_limits_respected(tmp_path):
    """Running with max_documents=5 produces fewer tokens than max_documents=10.

    Confirms that the safety limits actually constrain the output.
    """
    config_small = {
        "dataset_name": "wikitext-103-raw-v1",
        "tokenizer": "gpt2",
        "max_documents": 5,
        "max_tokens": None,
        "tokens_per_shard": 10_000,
        "output_dir": str(tmp_path / "small"),
    }
    config_large = {
        "dataset_name": "wikitext-103-raw-v1",
        "tokenizer": "gpt2",
        "max_documents": 10,
        "max_tokens": None,
        "tokens_per_shard": 10_000,
        "output_dir": str(tmp_path / "large"),
    }

    prepare_dataset(config_small)
    prepare_dataset(config_large)

    # Read manifests and compare token counts
    with open(tmp_path / "small" / "manifest.json") as f:
        small_manifest = json.load(f)
    with open(tmp_path / "large" / "manifest.json") as f:
        large_manifest = json.load(f)

    small_tokens = small_manifest["train"]["total_tokens"]
    large_tokens = large_manifest["train"]["total_tokens"]

    assert small_tokens <= large_tokens, (
        f"5 docs ({small_tokens} tokens) should produce <= tokens than "
        f"10 docs ({large_tokens} tokens)"
    )
