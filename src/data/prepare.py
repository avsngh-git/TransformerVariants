"""Data preparation pipeline.

Downloads a HuggingFace dataset, tokenizes it into a flat token stream
with EOT separators, writes binary shards, and produces manifest + report files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from src.data.tokenizer import encode, get_eot_token, get_tokenizer, get_vocab_size


def load_dataset_split(
    dataset_name: str,
    split: str,
    max_documents: int | None = None,
) -> list[str]:
    """Download/load a HuggingFace dataset split and return documents as strings.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "wikitext-103-raw-v1").
                      We load from the "wikitext" parent with this as the config name.
        split: Which split to load ("train", "validation", or "test").
        max_documents: If set, only return the first N documents. Respects the
                       safety limit from CLAUDE.md to avoid downloading huge datasets.

    Returns:
        A list of document strings. Empty documents are included (they'll be
        handled during tokenization via concatenation with EOT).
    """
    ds = load_dataset("Salesforce/wikitext", dataset_name, split=split)
    documents = ds["text"]

    if max_documents is not None:
        documents = documents[:max_documents]

    return documents


def compute_token_stats(doc_token_lengths: list[int]) -> dict[str, float]:
    """Compute summary statistics for per-document token counts.

    Args:
        doc_token_lengths: A list where each element is the number of tokens
                          in one document (before concatenation).

    Returns:
        A dict with keys: mean, median, min, max, std, total, num_documents,
        num_empty (documents with 0 tokens).
    """
    lengths = np.array(doc_token_lengths)
    return {
        "mean": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
        "median": float(np.median(lengths)) if len(lengths) > 0 else 0.0,
        "min": int(np.min(lengths)) if len(lengths) > 0 else 0,
        "max": int(np.max(lengths)) if len(lengths) > 0 else 0,
        "std": float(np.std(lengths)) if len(lengths) > 0 else 0.0,
        "total": int(np.sum(lengths)),
        "num_documents": len(lengths),
        "num_empty": int(np.sum(lengths == 0)),
    }


def get_git_sha() -> str:
    """Get the current git commit SHA for provenance tracking.
    
    Why needed?
    To ensure reproducibility and track exactly which version of the code was used
    to generate the dataset.

    Returns:
        The short git SHA (7 characters), or "unknown" if not in a git repo
        or git is not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def tokenize_documents(
    documents: list[str],
    tokenizer_name: str = "gpt2",
) -> tuple[np.ndarray, list[int]]:
    """Tokenize all documents and concatenate into a single flat token stream.

    Each document is tokenized independently, then all tokens are joined into
    one continuous array with an EOT (end-of-text) token inserted between
    every pair of documents.

    The result looks like:
        [doc1_tokens...] [EOT] [doc2_tokens...] [EOT] [doc3_tokens...] [EOT]

    Note: A trailing EOT is added after the last document as well.

    Args:
        documents: List of raw text strings (one per document).
        tokenizer_name: Which tokenizer to use (default "gpt2").

    Returns:
        A tuple of:
        - token_stream: A 1-D numpy array of dtype uint16 containing all
          token IDs in concatenated order.
        - doc_token_lengths: A list of ints, one per document, recording
          how many tokens that document produced (excluding the EOT separator).
          Used for computing statistics in the data report.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    eot = get_eot_token(tokenizer)

    all_tokens: list[int] = []
    doc_token_lengths: list[int] = []

    for doc in documents:
        # Tokenize this document
        tokens = encode(doc, tokenizer)
        doc_token_lengths.append(len(tokens))

        # Append tokens + EOT separator
        all_tokens.extend(tokens) #extend instead of append because 
        #we want to add each element of tokens to all_tokens, not the list as an element.
        all_tokens.append(eot)
    #numpy array .append() is slow (copies the whole array each time), while Python list .extend() is amortized O(1).
    
    # Convert to numpy uint16 array
    token_stream = np.array(all_tokens, dtype=np.uint16)

    return token_stream, doc_token_lengths


def write_shards(
    token_stream: np.ndarray,
    output_dir: Path,
    split_name: str,
    tokens_per_shard: int,
) -> list[dict[str, Any]]:
    """Slice a token stream into fixed-size binary shards and write to disk.

    Each shard is a raw binary file of uint16 values. The last shard may be
    smaller than tokens_per_shard if the total token count isn't evenly divisible.

    Files are named: {split_name}_{shard_index:06d}.bin
    For example: train_000000.bin, train_000001.bin, val_000000.bin

    Args:
        token_stream: A 1-D numpy uint16 array of all tokens for this split.
        output_dir: Directory to write shard files into. Created if it doesn't exist.
        split_name: Prefix for filenames ("train" or "val").
        tokens_per_shard: How many tokens per shard file.

    Returns:
        A list of dicts, one per shard, each containing:
        - "filename": the shard's filename (not the full path)
        - "num_tokens": how many tokens in this shard
        - "sha256": hex digest of the file's contents
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = len(token_stream)
    shard_info: list[dict[str, Any]] = []

    # Calculate how many shards we need (ceiling division)
    num_shards = (total_tokens + tokens_per_shard - 1) // tokens_per_shard

    for i in range(num_shards):
        # Slice out this shard's chunk of tokens
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, total_tokens)
        shard_data = token_stream[start:end]

        # Build filename: train_000000.bin, train_000001.bin, etc.
        filename = f"{split_name}_{i:06d}.bin"
        filepath = output_dir / filename

        # Write raw bytes to disk
        shard_data.tofile(filepath)

        # Compute SHA256 hash for integrity verification
        file_hash = hashlib.sha256(shard_data.tobytes()).hexdigest()

        shard_info.append({
            "filename": filename,
            "num_tokens": len(shard_data),
            "sha256": file_hash,
        })

    return shard_info


def write_manifest(
    output_dir: Path,
    train_shards: list[dict[str, Any]],
    val_shards: list[dict[str, Any]],
    tokenizer_name: str,
    vocab_size: int,
    tokens_per_shard: int,
) -> None:
    """Write manifest.json — the dataloader contract file.

    This file contains everything the training code needs to load the dataset:
    filenames, dtype, token counts, and vocab size. It does NOT contain
    provenance info (that's in data_report.json).

    Args:
        output_dir: Directory to write manifest.json into.
        train_shards: Shard info list from write_shards() for the train split.
        val_shards: Shard info list from write_shards() for the val split.
        tokenizer_name: Name of the tokenizer used (e.g. "gpt2").
        vocab_size: Total vocabulary size (e.g. 50257).
        tokens_per_shard: Max tokens per shard file.
    """
    total_train_tokens = sum(s["num_tokens"] for s in train_shards)
    total_val_tokens = sum(s["num_tokens"] for s in val_shards)

    manifest = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "tokens_per_shard": tokens_per_shard,
        "train": {
            "num_shards": len(train_shards),
            "total_tokens": total_train_tokens,
            "shards": [s["filename"] for s in train_shards],
        },
        "val": {
            "num_shards": len(val_shards),
            "total_tokens": total_val_tokens,
            "shards": [s["filename"] for s in val_shards],
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def write_data_report(
    output_dir: Path,
    dataset_name: str,
    train_shards: list[dict[str, Any]],
    val_shards: list[dict[str, Any]],
    train_doc_lengths: list[int],
    val_doc_lengths: list[int],
    tokenizer_name: str,
    vocab_size: int,
    tokens_per_shard: int,
    processing_time_seconds: float,
    max_documents: int | None,
) -> None:
    """Write data_report.json — the rich provenance file.

    This file records how the dataset was created: source, stats, hashes,
    git version, and timing. It's for humans debugging or auditing, not
    for the training dataloader.

    Args:
        output_dir: Directory to write data_report.json into.
        dataset_name: Source dataset name (e.g. "wikitext-103-raw-v1").
        train_shards: Shard info from write_shards() for train (includes hashes).
        val_shards: Shard info from write_shards() for val (includes hashes).
        train_doc_lengths: Per-document token counts for train split.
        val_doc_lengths: Per-document token counts for val split.
        tokenizer_name: Tokenizer used.
        vocab_size: Vocabulary size.
        tokens_per_shard: Max tokens per shard.
        processing_time_seconds: How long the entire pipeline took.
        max_documents: Document limit that was applied (None if no limit).
    """
    from datetime import datetime, timezone

    report = {
        "dataset_name": dataset_name,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "tokens_per_shard": tokens_per_shard,
        "max_documents_limit": max_documents,
        "git_sha": get_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(processing_time_seconds, 2),
        "train": {
            "documents_processed": len(train_doc_lengths),
            "total_tokens": sum(s["num_tokens"] for s in train_shards),
            "num_shards": len(train_shards),
            "token_stats": compute_token_stats(train_doc_lengths),
            "shards": train_shards,  # includes filenames + sha256 hashes
        },
        "val": {
            "documents_processed": len(val_doc_lengths),
            "total_tokens": sum(s["num_tokens"] for s in val_shards),
            "num_shards": len(val_shards),
            "token_stats": compute_token_stats(val_doc_lengths),
            "shards": val_shards,
        },
    }

    report_path = output_dir / "data_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def prepare_dataset(config: dict[str, Any]) -> Path:
    """Top-level orchestrator: run the full data preparation pipeline.

    This function ties everything together in the correct order:
    1. Load train and val splits from HuggingFace
    2. Tokenize both into flat token streams
    3. Write binary shards for each split
    4. Write manifest.json (dataloader contract)
    5. Write data_report.json (provenance)

    Args:
        config: A dictionary with the data config. Expected keys:
            - dataset_name (str): e.g. "wikitext-103-raw-v1"
            - max_documents (int | None): document limit per split
            - max_tokens (int | None): token limit per split (applied after tokenization)
            - tokens_per_shard (int): tokens per binary shard file
            - tokenizer (str): tokenizer name, defaults to "gpt2"
            - output_dir (str | None): where to write output, defaults to
              "data/processed/{dataset_name}"

    Returns:
        The Path to the output directory containing shards + metadata.
    """
    start_time = time.time()

    # Extract config values with defaults
    dataset_name = config["dataset_name"]
    max_documents = config.get("max_documents")
    max_tokens = config.get("max_tokens")
    tokens_per_shard = config["tokens_per_shard"]
    tokenizer_name = config.get("tokenizer", "gpt2")
    output_dir = Path(config.get("output_dir", f"data/processed/{dataset_name}"))

    # Get vocab size for manifest
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = get_vocab_size(tokenizer)

    print(f"Preparing dataset: {dataset_name}")
    print(f"Tokenizer: {tokenizer_name} (vocab_size={vocab_size})")
    print(f"Output: {output_dir}")

    # --- Process train split ---
    print("\n--- Train split ---")
    train_docs = load_dataset_split(dataset_name, "train", max_documents)
    print(f"Loaded {len(train_docs)} documents")

    train_stream, train_doc_lengths = tokenize_documents(train_docs, tokenizer_name)
    print(f"Tokenized: {len(train_stream)} total tokens")

    # Apply max_tokens limit if set
    if max_tokens is not None and len(train_stream) > max_tokens:
        train_stream = train_stream[:max_tokens]
        print(f"Trimmed to {max_tokens} tokens (max_tokens limit)")

    train_shards = write_shards(train_stream, output_dir, "train", tokens_per_shard)
    print(f"Wrote {len(train_shards)} train shards")

    # --- Process val split ---
    print("\n--- Validation split ---")
    val_docs = load_dataset_split(dataset_name, "validation", max_documents)
    print(f"Loaded {len(val_docs)} documents")

    val_stream, val_doc_lengths = tokenize_documents(val_docs, tokenizer_name)
    print(f"Tokenized: {len(val_stream)} total tokens")

    if max_tokens is not None and len(val_stream) > max_tokens:
        val_stream = val_stream[:max_tokens]
        print(f"Trimmed to {max_tokens} tokens (max_tokens limit)")

    val_shards = write_shards(val_stream, output_dir, "val", tokens_per_shard)
    print(f"Wrote {len(val_shards)} val shards")

    # --- Write metadata ---
    write_manifest(
        output_dir, train_shards, val_shards,
        tokenizer_name, vocab_size, tokens_per_shard,
    )
    print("\nWrote manifest.json")

    processing_time = time.time() - start_time

    write_data_report(
        output_dir=output_dir,
        dataset_name=dataset_name,
        train_shards=train_shards,
        val_shards=val_shards,
        train_doc_lengths=train_doc_lengths,
        val_doc_lengths=val_doc_lengths,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        tokens_per_shard=tokens_per_shard,
        processing_time_seconds=processing_time,
        max_documents=max_documents,
    )
    print(f"Wrote data_report.json")
    print(f"\nDone! ({processing_time:.1f}s)")

    return output_dir
