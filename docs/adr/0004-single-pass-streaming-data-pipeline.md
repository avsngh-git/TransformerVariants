# ADR 0004: Single-Pass Streaming Data Pipeline

## Status

Accepted (2026-07-04)

## Context

Phase 10 introduces large-scale data preparation (FineWeb-Edu, up to 10B tokens).
The existing Phase 2 pipeline loads the entire dataset into memory via
`load_dataset()`, tokenizes all documents, then writes shards. This works for
WikiText-103 (~500MB raw) but cannot handle multi-GB corpora on a machine with
limited RAM.

Two architectural approaches were considered:

1. **Single-pass streaming** — consume documents one at a time from an HF streaming
   iterator, tokenize immediately, buffer tokens, and flush to shards on threshold.
   Memory is constant. Resumption via document-count checkpointing.

2. **Two-stage with intermediate storage** — Stage 1 downloads/filters to JSONL on
   disk. Stage 2 tokenizes JSONL to shards. Allows restarting Stage 2 without
   re-downloading. Supports offline deduplication between stages.

## Decision

We chose single-pass streaming with a resumption checkpoint.

## Consequences

**Benefits:**
- Constant memory (~200-500MB) regardless of corpus size
- Simpler code — one script, one pass, no intermediate storage
- HuggingFace streaming handles download/retry/pagination internally
- Resumption via `progress.json` (skip N documents on restart)
- The "constant memory at arbitrary scale" property is the primary portfolio signal

**Tradeoffs:**
- Cannot add offline deduplication (would need two-stage)
- Resumption requires re-streaming and skipping documents (slower than seeking into JSONL)
- If HF streaming changes iteration order between runs, resumed output may differ

**Mitigations:**
- FineWeb-Edu is pre-deduplicated upstream — no dedup needed
- HF datasets streaming uses fixed shard ordering — safe for resumption
- For corpora that need dedup, the two-stage approach would be a separate script (not needed here)
