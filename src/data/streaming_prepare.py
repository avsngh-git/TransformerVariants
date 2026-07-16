"""Streaming data preparation pipeline for FineWeb-Edu.

Ingests documents from HuggingFace streaming iterators, tokenizes them,
filters by length, routes deterministically to train/val splits, and
flushes to uint16 binary shards compatible with ShardedDataLoader.

This module provides:
    - PipelineConfig: validated configuration dataclass
    - FilterStats: document filter outcome counters
    - ResumeState: checkpoint state for pipeline resumption
    - PipelineResult: summary of a completed pipeline run
    - StreamingPipeline: main orchestrator class
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset

from src.data.tokenizer import encode, get_eot_token, get_tokenizer


@dataclass
class FilterStats:
    """Tracks document filter outcomes.

    All counters start at zero and are incremented as documents pass
    through the DocumentFilter.
    """

    documents_processed: int = 0
    documents_accepted: int = 0
    documents_filtered_short: int = 0
    documents_filtered_long: int = 0


@dataclass
class ResumeState:
    """State loaded from progress.json for pipeline resumption.

    All fields are required — a valid progress.json must contain
    each of these values.
    """

    documents_consumed: int
    train_shards_written: int
    val_shards_written: int
    train_tokens_written: int
    val_tokens_written: int


@dataclass
class PipelineConfig:
    """Configuration for the streaming pipeline.

    Validated on construction via __post_init__. Invalid configurations
    raise ValueError with a descriptive message.
    """

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    split: str = "train"
    max_tokens: int | None = 1_000_000_000
    min_doc_tokens: int = 50
    max_doc_tokens: int = 10_000
    tokens_per_shard: int = 10_000_000
    tokenizer_name: str = "gpt2"
    output_dir: Path = field(default_factory=lambda: Path("data/processed/fineweb-edu"))
    resume: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.min_doc_tokens < 1:
            raise ValueError(f"min_doc_tokens must be >= 1, got {self.min_doc_tokens}")
        if self.max_doc_tokens < self.min_doc_tokens:
            raise ValueError(
                f"max_doc_tokens ({self.max_doc_tokens}) must be >= "
                f"min_doc_tokens ({self.min_doc_tokens})"
            )
        if self.tokens_per_shard < 1000:
            raise ValueError(f"tokens_per_shard must be >= 1000, got {self.tokens_per_shard}")
        # Ensure output_dir is a Path instance
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""

    output_dir: Path
    train_shards: int
    val_shards: int
    train_tokens: int
    val_tokens: int
    filter_stats: FilterStats
    documents_consumed: int
    processing_time_seconds: float


class DocumentFilter:
    """Lightweight document length filter.

    Accepts or rejects documents based on token count thresholds.
    Documents with fewer than min_tokens are rejected (filtered_short).
    Documents with more than max_tokens are accepted but tracked as
    filtered_long (they will be truncated by the caller).
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 10000) -> None:
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._stats = FilterStats()

    def should_accept(self, token_count: int) -> bool:
        """Return True if document passes length filter.

        A document is accepted if token_count >= min_tokens.
        Documents exceeding max_tokens are still accepted (they get
        truncated), but are tracked in documents_filtered_long.
        """
        self._stats.documents_processed += 1

        if token_count < self._min_tokens:
            self._stats.documents_filtered_short += 1
            return False

        # Document is accepted
        self._stats.documents_accepted += 1
        if token_count > self._max_tokens:
            self._stats.documents_filtered_long += 1

        return True

    def truncate(self, tokens: list[int], token_count: int) -> list[int]:
        """Truncate tokens to max_tokens if needed.

        Returns the token list sliced to max_tokens length if token_count
        exceeds max_tokens, otherwise returns tokens unchanged.
        """
        if token_count > self._max_tokens:
            return tokens[: self._max_tokens]
        return tokens

    @property
    def stats(self) -> FilterStats:
        """Return current filter statistics."""
        return self._stats


class HashRouter:
    """SHA-256 hash-based deterministic train/val router.

    Routes documents to 'train' or 'val' splits based on the SHA-256
    hash of the first 256 bytes of the document text. This ensures
    deterministic, reproducible routing regardless of processing order.
    """

    def __init__(self, val_threshold: int = 1) -> None:
        """Initialize the router.

        Args:
            val_threshold: Hash bucket values below this threshold route
                to 'val'. Default 1 means ~1% of documents go to val.
        """
        self.val_threshold = val_threshold

    def route(self, text: str) -> str:
        """Return 'train' or 'val' based on SHA-256 of first 256 bytes.

        Algorithm:
            1. Take first 256 characters, encode to UTF-8, take first 256 bytes
            2. Compute SHA-256 hash
            3. Convert first 8 hex chars to integer, mod 100
            4. If result < val_threshold → 'val', else → 'train'

        Args:
            text: The document text to route.

        Returns:
            'val' if the hash bucket is below val_threshold, else 'train'.
        """
        # Take first 256 bytes for hashing (sufficient for uniqueness)
        prefix = text[:256].encode("utf-8")[:256]
        # SHA-256 hash
        hash_hex = hashlib.sha256(prefix).hexdigest()
        # Convert first 8 hex chars to integer, mod 100
        bucket = int(hash_hex[:8], 16) % 100
        if bucket < self.val_threshold:
            return "val"
        else:
            return "train"


class ShardBuffer:
    """Token buffer that flushes to uint16 binary shards.

    Accumulates tokens in an internal list and flushes to disk as a
    numpy uint16 binary file when the buffer reaches tokens_per_shard.
    Supports starting from a specific shard index for pipeline resumption.
    """

    def __init__(
        self,
        output_dir: Path,
        split_name: str,
        tokens_per_shard: int,
        start_shard_idx: int = 0,
    ) -> None:
        """Initialize the shard buffer.

        Args:
            output_dir: Directory where shard files are written.
            split_name: Split name prefix for filenames (e.g., 'train', 'val').
            tokens_per_shard: Number of tokens per shard file (flush threshold).
            start_shard_idx: Starting shard index (for resumption support).
        """
        self.output_dir = Path(output_dir)
        self.split_name = split_name
        self.tokens_per_shard = tokens_per_shard
        self._shard_idx = start_shard_idx
        self._buffer: list[int] = []
        self._total_tokens_written = 0

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, tokens: list[int]) -> list[str]:
        """Append tokens to buffer. Returns list of flushed shard filenames.

        Extends the internal buffer with the given tokens. If the buffer
        reaches or exceeds tokens_per_shard, flushes complete shards to disk.

        Args:
            tokens: Token IDs to append to the buffer.

        Returns:
            List of filenames that were flushed during this call.
        """
        flushed_files: list[str] = []
        self._buffer.extend(tokens)
        while len(self._buffer) >= self.tokens_per_shard:
            shard_data = self._buffer[: self.tokens_per_shard]
            self._buffer = self._buffer[self.tokens_per_shard :]
            filename = f"{self.split_name}_{self._shard_idx:06d}.bin"
            filepath = self.output_dir / filename
            np.array(shard_data, dtype=np.uint16).tofile(filepath)
            self._shard_idx += 1
            self._total_tokens_written += len(shard_data)
            flushed_files.append(filename)
        return flushed_files

    def flush(self) -> str | None:
        """Force-flush remaining buffer. Returns filename or None if empty.

        Writes whatever tokens remain in the buffer as a final shard,
        which may be smaller than tokens_per_shard.

        Returns:
            The filename of the flushed shard, or None if the buffer was empty.
        """
        if not self._buffer:
            return None
        filename = f"{self.split_name}_{self._shard_idx:06d}.bin"
        filepath = self.output_dir / filename
        np.array(self._buffer, dtype=np.uint16).tofile(filepath)
        self._total_tokens_written += len(self._buffer)
        self._shard_idx += 1
        self._buffer = []
        return filename

    @property
    def total_tokens_written(self) -> int:
        """Total number of tokens flushed to disk across all shards."""
        return self._total_tokens_written

    @property
    def shards_written(self) -> int:
        """Number of shards flushed to disk."""
        return self._shard_idx

    @property
    def buffer_size(self) -> int:
        """Current number of tokens in the buffer (not yet flushed)."""
        return len(self._buffer)


# =============================================================================
# Stub functions for manifest and progress (filled in by parallel tasks 6.2/6.3)
# =============================================================================


def write_manifest(
    config: PipelineConfig,
    train_buffer: ShardBuffer,
    val_buffer: ShardBuffer,
    doc_filter: DocumentFilter,
    documents_consumed: int,
) -> None:
    """Write manifest.json with standard fields and extended metadata.

    Produces a manifest compatible with ShardedDataLoader (which reads
    tokenizer, vocab_size, dtype, tokens_per_shard, and per-split shard
    lists) and adds provenance metadata that the loader ignores.

    Args:
        config: Pipeline configuration (dataset name, tokenizer, output dir, etc.).
        train_buffer: The train split ShardBuffer (used for shard count and token totals).
        val_buffer: The val split ShardBuffer (used for shard count and token totals).
        doc_filter: The DocumentFilter (used for filter_stats metadata).
        documents_consumed: Total documents consumed from the iterator.
    """
    from src.data.tokenizer import get_tokenizer, get_vocab_size

    tokenizer = get_tokenizer(config.tokenizer_name)
    vocab_size = get_vocab_size(tokenizer)

    # Build shard filename lists from buffer state
    train_shards = [f"train_{i:06d}.bin" for i in range(train_buffer.shards_written)]
    val_shards = [f"val_{i:06d}.bin" for i in range(val_buffer.shards_written)]

    # Standard fields (required by ShardedDataLoader)
    manifest = {
        "tokenizer": config.tokenizer_name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "tokens_per_shard": config.tokens_per_shard,
        "train": {
            "num_shards": train_buffer.shards_written,
            "total_tokens": train_buffer.total_tokens_written,
            "shards": train_shards,
        },
        "val": {
            "num_shards": val_buffer.shards_written,
            "total_tokens": val_buffer.total_tokens_written,
            "shards": val_shards,
        },
        # Extended metadata (ShardedDataLoader ignores extra keys)
        "source": f"{config.dataset_name}/{config.dataset_config}",
        "pipeline_version": "2.0",
        "filter_stats": {
            "documents_processed": doc_filter.stats.documents_processed,
            "documents_accepted": doc_filter.stats.documents_accepted,
            "documents_filtered_short": doc_filter.stats.documents_filtered_short,
            "documents_filtered_long": doc_filter.stats.documents_filtered_long,
        },
        "streaming_progress": {
            "documents_consumed": documents_consumed,
            "completed": True,
        },
        "split_method": "sha256_first_256_bytes_mod100_lt1_val",
    }

    manifest_path = config.output_dir / "manifest.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def write_progress(
    output_dir: Path,
    documents_consumed: int,
    train_buffer: ShardBuffer,
    val_buffer: ShardBuffer,
    *,
    completed: bool = False,
) -> None:
    """Write progress.json with current pipeline state.

    Written after each shard flush to enable resumption.
    Uses atomic write (write to temp file, then rename) to avoid
    corrupted state on crash.
    """
    progress = {
        "documents_consumed": documents_consumed,
        "train_shards_written": train_buffer.shards_written,
        "val_shards_written": val_buffer.shards_written,
        "train_tokens_written": train_buffer.total_tokens_written,
        "val_tokens_written": val_buffer.total_tokens_written,
        "completed": completed,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "progress.json"
    tmp = output_path / "progress.json.tmp"

    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(target)


def load_resume_state(output_dir: Path) -> ResumeState | None:
    """Load resume state from progress.json.

    Returns ResumeState if progress.json exists and is valid.
    Returns None if progress.json does not exist.
    Raises ValueError if progress.json exists but is malformed or missing required fields.
    """
    progress_path = Path(output_dir) / "progress.json"

    if not progress_path.exists():
        return None

    try:
        data = json.loads(progress_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"progress.json is malformed: {e}") from e

    required_fields = [
        "documents_consumed",
        "train_shards_written",
        "val_shards_written",
        "train_tokens_written",
        "val_tokens_written",
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"progress.json is missing required fields: {', '.join(missing)}")

    # Validate that values are non-negative integers
    for field_name in required_fields:
        value = data[field_name]
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"progress.json field '{field_name}' must be a non-negative integer, got {value!r}"
            )

    return ResumeState(
        documents_consumed=data["documents_consumed"],
        train_shards_written=data["train_shards_written"],
        val_shards_written=data["val_shards_written"],
        train_tokens_written=data["train_tokens_written"],
        val_tokens_written=data["val_tokens_written"],
    )


# =============================================================================
# StreamingPipeline orchestrator
# =============================================================================


class StreamingPipeline:
    """Main orchestrator for the streaming data preparation pipeline.

    Ties together document ingestion, filtering, tokenization, routing,
    buffering, and shard output into a single-pass streaming process.
    Supports resumption from the last checkpoint via progress.json.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline with the given configuration.

        Args:
            config: Validated pipeline configuration.
        """
        self.config = config

    def run(self) -> PipelineResult:
        """Execute the streaming pipeline.

        Main orchestration loop:
        1. Initialize tokenizer, filter, router, and shard buffers
        2. Load HuggingFace dataset with streaming=True
        3. Handle resume: load progress.json, skip consumed documents
        4. Iterate documents: filter → tokenize → route → buffer + EOT
        5. Check max_tokens limit after each accepted document
        6. Write progress.json after each shard flush
        7. Final flush + manifest writing on completion

        Returns:
            PipelineResult summarizing the completed run.
        """
        start_time = time.time()

        # Initialize components
        tokenizer = get_tokenizer(self.config.tokenizer_name)
        eot_token = get_eot_token(tokenizer)
        doc_filter = DocumentFilter(self.config.min_doc_tokens, self.config.max_doc_tokens)
        router = HashRouter(val_threshold=1)

        # Handle resume state
        resume_state = load_resume_state(self.config.output_dir) if self.config.resume else None

        train_buffer = ShardBuffer(
            self.config.output_dir,
            "train",
            self.config.tokens_per_shard,
            start_shard_idx=resume_state.train_shards_written if resume_state else 0,
        )
        val_buffer = ShardBuffer(
            self.config.output_dir,
            "val",
            self.config.tokens_per_shard,
            start_shard_idx=resume_state.val_shards_written if resume_state else 0,
        )

        # Open streaming dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
            streaming=True,
        )
        iterator = iter(dataset)

        # Skip already-consumed documents on resume
        if resume_state:
            for _ in range(resume_state.documents_consumed):
                next(iterator)

        docs_consumed = resume_state.documents_consumed if resume_state else 0

        # Main processing loop
        for doc in iterator:
            text = doc["text"]
            tokens = encode(text, tokenizer)
            token_count = len(tokens)

            if not doc_filter.should_accept(token_count):
                docs_consumed += 1
                continue

            tokens = doc_filter.truncate(tokens, token_count)
            split = router.route(text)
            buffer = train_buffer if split == "train" else val_buffer

            # Append tokens + EOT separator
            flushed = buffer.append(tokens + [eot_token])
            docs_consumed += 1

            # Write progress after shard flushes
            if flushed:
                write_progress(self.config.output_dir, docs_consumed, train_buffer, val_buffer)

            # Check max_tokens limit
            if self.config.max_tokens:
                total = (
                    train_buffer.total_tokens_written
                    + val_buffer.total_tokens_written
                    + train_buffer.buffer_size
                    + val_buffer.buffer_size
                )
                if total >= self.config.max_tokens:
                    break

        # Final flush
        train_buffer.flush()
        val_buffer.flush()

        # Write manifest and final progress
        write_manifest(self.config, train_buffer, val_buffer, doc_filter, docs_consumed)
        write_progress(
            self.config.output_dir, docs_consumed, train_buffer, val_buffer, completed=True
        )

        processing_time = time.time() - start_time

        return PipelineResult(
            output_dir=self.config.output_dir,
            train_shards=train_buffer.shards_written,
            val_shards=val_buffer.shards_written,
            train_tokens=train_buffer.total_tokens_written,
            val_tokens=val_buffer.total_tokens_written,
            filter_stats=doc_filter.stats,
            documents_consumed=docs_consumed,
            processing_time_seconds=processing_time,
        )
