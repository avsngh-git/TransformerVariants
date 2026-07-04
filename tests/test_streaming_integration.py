"""Integration test for streaming pipeline with real HuggingFace data.

Marked with @pytest.mark.network — skipped in normal CI runs.
Run with: pytest -m network tests/test_streaming_integration.py
"""

import json
import pytest
from pathlib import Path

from src.data.streaming_prepare import PipelineConfig, StreamingPipeline
from src.data.dataloader import ShardedDataLoader


@pytest.mark.network
class TestStreamingIntegration:
    """End-to-end test with real HuggingFace streaming."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "integration_output"

    def test_full_pipeline_with_real_data(self, output_dir):
        """Stream 50 real docs from FineWeb-Edu, produce shards, load with ShardedDataLoader."""
        config = PipelineConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-10BT",
            split="train",
            max_tokens=50_000,  # Small limit for test speed
            min_doc_tokens=50,
            max_doc_tokens=10000,
            tokens_per_shard=10_000,
            output_dir=output_dir,
        )

        pipeline = StreamingPipeline(config)
        result = pipeline.run()

        # Verify pipeline produced output
        assert result.train_tokens + result.val_tokens > 0
        assert result.documents_consumed > 0

        # Verify manifest exists and has required fields
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["tokenizer"] == "gpt2"
        assert manifest["vocab_size"] == 50257
        assert manifest["dtype"] == "uint16"
        assert "train" in manifest
        assert "val" in manifest
        assert "source" in manifest
        assert "filter_stats" in manifest

        # Verify ShardedDataLoader can load the output
        if manifest["train"]["num_shards"] > 0:
            loader = ShardedDataLoader(
                data_dir=output_dir,
                batch_size=2,
                seq_len=128,
                split="train",
            )
            x, y = loader.next_batch()
            assert x.shape == (2, 128)
            assert y.shape == (2, 128)
