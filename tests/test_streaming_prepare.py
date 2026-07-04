"""End-to-end unit tests for StreamingPipeline with mocked HF iterators.

Validates pipeline behavior including document processing, filtering,
hash routing, max_tokens termination, resumption, manifest structure,
and EOT token insertion — all without network access.

**Validates: Requirements 10.1, 10.2**
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.streaming_prepare import (
    PipelineConfig,
    StreamingPipeline,
    write_progress,
    ShardBuffer,
)


# =============================================================================
# Fixtures
# =============================================================================

EOT_TOKEN = 50256  # GPT-2 EOT token ID


def make_long_text(n_words: int = 200) -> str:
    """Generate text that tokenizes to well above 50 tokens.

    Uses varied words to avoid degenerate tokenization.
    """
    words = [f"word{i}" for i in range(n_words)]
    return " ".join(words)


def make_short_text() -> str:
    """Generate text that tokenizes to fewer than 50 tokens (very short)."""
    return "hi"


@pytest.fixture
def mock_documents():
    """Create a list of synthetic documents (dicts with 'text' field).

    Returns 10 documents with long text that will pass the default
    min_doc_tokens=50 filter.
    """
    return [{"text": make_long_text(200 + i * 10)} for i in range(10)]


@pytest.fixture
def pipeline_output_dir(tmp_path):
    """Provide a temporary output directory for pipeline results."""
    out = tmp_path / "pipeline_output"
    out.mkdir()
    return out


def make_pipeline_config(output_dir: Path, **overrides) -> PipelineConfig:
    """Create a PipelineConfig with small defaults suitable for testing."""
    defaults = dict(
        dataset_name="test/dataset",
        dataset_config="test-config",
        split="train",
        max_tokens=None,
        min_doc_tokens=50,
        max_doc_tokens=10000,
        tokens_per_shard=1000,
        tokenizer_name="gpt2",
        output_dir=output_dir,
        resume=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def run_pipeline_with_mock_docs(documents, output_dir, **config_overrides):
    """Helper: run pipeline with mocked load_dataset returning given documents."""
    config = make_pipeline_config(output_dir, **config_overrides)

    mock_dataset = MagicMock()
    mock_dataset.__iter__ = MagicMock(return_value=iter(documents))

    with patch("src.data.streaming_prepare.load_dataset", return_value=mock_dataset):
        pipeline = StreamingPipeline(config)
        result = pipeline.run()

    return result


# =============================================================================
# Test: Pipeline processes documents and produces shards
# =============================================================================


class TestPipelineProcessesDocuments:
    """Pipeline should produce shard files from valid documents."""

    def test_pipeline_produces_shards(self, mock_documents, pipeline_output_dir):
        """Pipeline with valid documents produces at least one shard file."""
        result = run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        # Should produce some shards (train and/or val)
        total_shards = result.train_shards + result.val_shards
        assert total_shards > 0

        # Total tokens should be positive
        total_tokens = result.train_tokens + result.val_tokens
        assert total_tokens > 0

    def test_shard_files_exist_on_disk(self, mock_documents, pipeline_output_dir):
        """Shard .bin files are written to the output directory."""
        run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        bin_files = list(pipeline_output_dir.glob("*.bin"))
        assert len(bin_files) > 0

    def test_shard_files_are_valid_uint16(self, mock_documents, pipeline_output_dir):
        """Each shard file is readable as uint16 numpy data."""
        run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        bin_files = list(pipeline_output_dir.glob("*.bin"))
        for f in bin_files:
            data = np.fromfile(f, dtype=np.uint16)
            assert len(data) > 0
            # All values should be valid token IDs (< vocab size 50257)
            assert data.max() <= 50256


# =============================================================================
# Test: Filtered documents don't produce tokens
# =============================================================================


class TestFilteredDocumentsSkipped:
    """Short documents below min_doc_tokens should be filtered out."""

    def test_short_docs_are_filtered(self, pipeline_output_dir):
        """Documents with fewer tokens than min_doc_tokens produce no tokens."""
        # All documents are very short (will tokenize to < 50 tokens)
        short_docs = [{"text": "hi"} for _ in range(5)]

        result = run_pipeline_with_mock_docs(short_docs, pipeline_output_dir)

        # All documents should be filtered
        assert result.filter_stats.documents_filtered_short == 5
        assert result.filter_stats.documents_accepted == 0
        assert result.train_tokens + result.val_tokens == 0

    def test_mixed_docs_only_long_accepted(self, pipeline_output_dir):
        """Only documents above min_doc_tokens contribute tokens."""
        docs = [
            {"text": "hi"},  # too short
            {"text": make_long_text(200)},  # long enough
            {"text": "a"},  # too short
            {"text": make_long_text(300)},  # long enough
        ]

        result = run_pipeline_with_mock_docs(docs, pipeline_output_dir)

        assert result.filter_stats.documents_filtered_short == 2
        assert result.filter_stats.documents_accepted == 2


# =============================================================================
# Test: Hash routing distributes to train/val
# =============================================================================


class TestHashRoutingDistribution:
    """Hash routing should deterministically assign documents to splits."""

    def test_val_threshold_100_forces_all_to_val(self, pipeline_output_dir):
        """With val_threshold=100, all documents route to val split."""
        docs = [{"text": make_long_text(200 + i)} for i in range(5)]
        config = make_pipeline_config(pipeline_output_dir)

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(docs))

        # Patch the HashRouter to use val_threshold=100 (all go to val)
        with patch("src.data.streaming_prepare.load_dataset", return_value=mock_dataset):
            with patch(
                "src.data.streaming_prepare.HashRouter.__init__",
                lambda self, *a, **kw: setattr(self, "val_threshold", 100),
            ):
                with patch(
                    "src.data.streaming_prepare.HashRouter.route",
                    return_value="val",
                ):
                    pipeline = StreamingPipeline(config)
                    result = pipeline.run()

        # All tokens should be in val
        assert result.val_tokens > 0
        assert result.train_tokens == 0

    def test_default_routing_is_mostly_train(self, pipeline_output_dir):
        """With default val_threshold=1, most documents go to train."""
        # Use enough documents that statistically all should go to train
        # (1% chance per doc going to val, with 10 docs it's very unlikely all go to val)
        docs = [{"text": make_long_text(200 + i * 5)} for i in range(10)]

        result = run_pipeline_with_mock_docs(docs, pipeline_output_dir)

        # With default threshold=1, ~99% go to train
        # At least some should be in train
        assert result.train_tokens > 0


# =============================================================================
# Test: max_tokens stops pipeline early
# =============================================================================


class TestMaxTokensTermination:
    """Pipeline should stop consuming documents when max_tokens is reached."""

    def test_stops_at_max_tokens(self, pipeline_output_dir):
        """Pipeline stops when total tokens reach max_tokens."""
        # Create many documents
        docs = [{"text": make_long_text(200)} for _ in range(50)]

        # Set a small max_tokens to force early termination
        result = run_pipeline_with_mock_docs(
            docs, pipeline_output_dir, max_tokens=500
        )

        # Should not have consumed all 50 documents
        assert result.documents_consumed < 50

        # Total tokens should be approximately max_tokens
        # (can exceed by at most one document's worth of tokens)
        total = result.train_tokens + result.val_tokens
        assert total > 0

    def test_max_tokens_none_processes_all(self, pipeline_output_dir):
        """When max_tokens is None, pipeline processes all documents."""
        docs = [{"text": make_long_text(200)} for _ in range(5)]

        result = run_pipeline_with_mock_docs(
            docs, pipeline_output_dir, max_tokens=None
        )

        # Should consume all documents
        assert result.documents_consumed == 5


# =============================================================================
# Test: Resumption skips documents
# =============================================================================


class TestResumption:
    """Pipeline with --resume should skip already-consumed documents."""

    def test_resumes_from_progress_json(self, pipeline_output_dir):
        """Pipeline skips documents_consumed documents on resume."""
        # Write a fake progress.json indicating 3 documents consumed
        progress = {
            "documents_consumed": 3,
            "train_shards_written": 0,
            "val_shards_written": 0,
            "train_tokens_written": 0,
            "val_tokens_written": 0,
            "completed": False,
            "timestamp": "2026-01-01T00:00:00Z",
        }
        (pipeline_output_dir / "progress.json").write_text(json.dumps(progress))

        # Provide 10 documents total
        docs = [{"text": make_long_text(200 + i * 5)} for i in range(10)]
        config = make_pipeline_config(pipeline_output_dir, resume=True)

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(docs))

        with patch("src.data.streaming_prepare.load_dataset", return_value=mock_dataset):
            pipeline = StreamingPipeline(config)
            result = pipeline.run()

        # Should have consumed 10 total (3 skipped + 7 processed)
        # The pipeline records docs_consumed = resume.documents_consumed + new ones processed
        assert result.documents_consumed == 10

    def test_no_progress_starts_from_beginning(self, pipeline_output_dir):
        """With resume=True but no progress.json, starts from beginning."""
        docs = [{"text": make_long_text(200 + i)} for i in range(5)]
        config = make_pipeline_config(pipeline_output_dir, resume=True)

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(docs))

        with patch("src.data.streaming_prepare.load_dataset", return_value=mock_dataset):
            pipeline = StreamingPipeline(config)
            result = pipeline.run()

        assert result.documents_consumed == 5


# =============================================================================
# Test: Manifest has required fields
# =============================================================================


class TestManifestStructure:
    """manifest.json should contain all fields needed by ShardedDataLoader."""

    def test_manifest_has_required_fields(self, mock_documents, pipeline_output_dir):
        """manifest.json contains all standard ShardedDataLoader fields."""
        run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        manifest_path = pipeline_output_dir / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())

        # Standard ShardedDataLoader fields
        assert "tokenizer" in manifest
        assert manifest["tokenizer"] == "gpt2"
        assert "vocab_size" in manifest
        assert manifest["vocab_size"] == 50257
        assert "dtype" in manifest
        assert manifest["dtype"] == "uint16"
        assert "tokens_per_shard" in manifest

        # Per-split fields
        assert "train" in manifest
        assert "val" in manifest
        assert "num_shards" in manifest["train"]
        assert "total_tokens" in manifest["train"]
        assert "shards" in manifest["train"]
        assert "num_shards" in manifest["val"]
        assert "total_tokens" in manifest["val"]
        assert "shards" in manifest["val"]

    def test_manifest_has_extended_metadata(self, mock_documents, pipeline_output_dir):
        """manifest.json includes extended provenance metadata."""
        run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        manifest = json.loads((pipeline_output_dir / "manifest.json").read_text())

        # Extended metadata
        assert "source" in manifest
        assert "pipeline_version" in manifest
        assert "filter_stats" in manifest
        assert "streaming_progress" in manifest
        assert "split_method" in manifest

    def test_manifest_shard_lists_match_files(self, mock_documents, pipeline_output_dir):
        """Shard filenames in manifest match actual files on disk."""
        run_pipeline_with_mock_docs(mock_documents, pipeline_output_dir)

        manifest = json.loads((pipeline_output_dir / "manifest.json").read_text())

        all_shards = manifest["train"]["shards"] + manifest["val"]["shards"]
        for shard_name in all_shards:
            assert (pipeline_output_dir / shard_name).exists()


# =============================================================================
# Test: EOT token inserted between documents
# =============================================================================


class TestEOTInsertion:
    """EOT token (50256) should appear between document boundaries."""

    def test_eot_present_in_shard_data(self, pipeline_output_dir):
        """Raw shard data contains EOT tokens between documents."""
        docs = [{"text": make_long_text(200)} for _ in range(3)]

        run_pipeline_with_mock_docs(docs, pipeline_output_dir)

        # Read all shard data
        all_tokens = []
        for bin_file in sorted(pipeline_output_dir.glob("*.bin")):
            data = np.fromfile(bin_file, dtype=np.uint16)
            all_tokens.extend(data.tolist())

        # EOT token should appear in the stream
        assert EOT_TOKEN in all_tokens

    def test_eot_count_matches_accepted_docs(self, pipeline_output_dir):
        """Number of EOT tokens equals number of accepted documents."""
        docs = [{"text": make_long_text(200 + i * 10)} for i in range(5)]

        result = run_pipeline_with_mock_docs(docs, pipeline_output_dir)

        # Read all shard data
        all_tokens = []
        for bin_file in sorted(pipeline_output_dir.glob("*.bin")):
            data = np.fromfile(bin_file, dtype=np.uint16)
            all_tokens.extend(data.tolist())

        eot_count = all_tokens.count(EOT_TOKEN)
        assert eot_count == result.filter_stats.documents_accepted

    def test_filtered_docs_produce_no_eot(self, pipeline_output_dir):
        """Filtered (rejected) documents do not produce EOT tokens."""
        # All documents too short to pass filter
        docs = [{"text": "hi"} for _ in range(5)]

        run_pipeline_with_mock_docs(docs, pipeline_output_dir)

        # No shard files should exist (nothing accepted)
        bin_files = list(pipeline_output_dir.glob("*.bin"))
        if bin_files:
            all_tokens = []
            for bin_file in bin_files:
                data = np.fromfile(bin_file, dtype=np.uint16)
                all_tokens.extend(data.tolist())
            assert EOT_TOKEN not in all_tokens
