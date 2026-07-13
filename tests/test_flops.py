"""Unit tests for the FLOPs computation module."""

import pytest

from src.evaluation.flops import FLOPBreakdown, MFUResult, compute_mfu, compute_step_flops
from src.models.config import ModelConfig


class TestComputeStepFlops:
    """Tests for compute_step_flops against hand-calculated values."""

    @pytest.fixture
    def debug_config(self) -> ModelConfig:
        """Debug-scale config: n_layer=2, d_model=64, seq_len=128, standard FFN."""
        return ModelConfig(
            n_layer=2,
            d_model=64,
            n_head=4,
            seq_len=128,
            ffn_multiplier=4,  # d_ff = 256
            attention_type="full",
            ffn_type="standard",
        )

    def test_total_flops_matches_hand_calculation(self, debug_config):
        """Total FLOPs for debug config should be 88,080,384 within 1% tolerance."""
        result = compute_step_flops(debug_config)
        expected_total = 88_080_384
        assert abs(result.total - expected_total) / expected_total < 0.01

    def test_qkv_proj_flops(self, debug_config):
        """QKV projections: 3 × 2 × 128 × 64 × 64 × 2 layers × 3 multiplier = 9,437,184."""
        result = compute_step_flops(debug_config)
        # Per layer: 3 * 2 * 128 * 64 * 64 = 3,145,728
        # × 2 layers = 6,291,456
        # × 3 training multiplier = 18,874,368
        expected_qkv = 3 * 2 * 128 * 64 * 64 * 2 * 3
        assert abs(result.qkv_proj - expected_qkv) / expected_qkv < 0.01

    def test_attention_score_flops_full(self, debug_config):
        """Attention score (full): 2 × 4 × 128 × 128 × 16 × 2 layers × 3 multiplier."""
        result = compute_step_flops(debug_config)
        # Per layer: 2 * 4 * 128 * 128 * 16 = 2,097,152
        # × 2 layers = 4,194,304
        # × 3 training multiplier = 12,582,912
        expected_attn = 2 * 4 * 128 * 128 * 16 * 2 * 3
        assert abs(result.attention_score - expected_attn) / expected_attn < 0.01

    def test_attention_output_flops(self, debug_config):
        """Attention output: 2 × 128 × 64 × 64 × 2 layers × 3 multiplier."""
        result = compute_step_flops(debug_config)
        # Per layer: 2 * 128 * 64 * 64 = 1,048,576
        # × 2 layers = 2,097,152
        # × 3 training multiplier = 6,291,456
        expected_output = 2 * 128 * 64 * 64 * 2 * 3
        assert abs(result.attention_output - expected_output) / expected_output < 0.01

    def test_ffn_flops_standard(self, debug_config):
        """FFN (standard): 2 × 2 × 128 × 64 × 256 × 2 layers × 3 multiplier."""
        result = compute_step_flops(debug_config)
        # Per layer: 2 * 2 * 128 * 64 * 256 = 8,388,608
        # × 2 layers = 16,777,216
        # × 3 training multiplier = 50,331,648
        expected_ffn = 2 * 2 * 128 * 64 * 256 * 2 * 3
        assert abs(result.ffn - expected_ffn) / expected_ffn < 0.01

    def test_returns_flop_breakdown_dataclass(self, debug_config):
        """compute_step_flops should return a FLOPBreakdown dataclass."""
        result = compute_step_flops(debug_config)
        assert isinstance(result, FLOPBreakdown)
        assert result.total == (
            result.qkv_proj + result.attention_score + result.attention_output + result.ffn
        )

    def test_swa_fewer_attention_flops_than_full(self):
        """SWA should use fewer attention FLOPs than full attention."""
        v1_config = ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            ffn_multiplier=4,
            attention_type="full",
            ffn_type="standard",
        )
        v4_config = ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            ffn_multiplier=4,
            attention_type="sliding_window",
            window_size=128,
            ffn_type="standard",
        )
        v1_result = compute_step_flops(v1_config)
        v4_result = compute_step_flops(v4_config)
        assert v4_result.attention_score < v1_result.attention_score

    def test_causal_linear_fewer_attention_flops_than_full(self):
        """Causal linear attention should return fewer attention FLOPs than full attention."""
        full_config = ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            ffn_multiplier=4,
            attention_type="full",
            ffn_type="standard",
        )
        linear_config = ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            seq_len=512,
            ffn_multiplier=4,
            attention_type="linear",
            ffn_type="standard",
        )
        full_result = compute_step_flops(full_config)
        linear_result = compute_step_flops(linear_config)
        assert linear_result.attention_score < full_result.attention_score

        d_head = linear_config.d_head
        expected = 4 * linear_config.n_head * linear_config.seq_len * d_head**2
        expected += 2 * linear_config.n_head * linear_config.seq_len * d_head
        assert linear_result.attention_score == expected * linear_config.n_layer * 3


class TestComputeMFU:
    """Tests for compute_mfu function."""

    def test_mfu_in_valid_range(self):
        """MFU should be in [0, 1] for realistic inputs."""
        # Realistic scenario: 88M FLOPs, 0.01s step time, L4 peak
        step_flops = 88_080_384
        step_time = 0.01  # 10ms
        result = compute_mfu(step_flops, step_time)
        assert 0.0 <= result.mfu <= 1.0

    def test_mfu_returns_mfu_result(self):
        """compute_mfu should return an MFUResult dataclass."""
        result = compute_mfu(step_flops=1_000_000_000, step_time_seconds=0.1)
        assert isinstance(result, MFUResult)
        assert result.peak_tflops == 242.0
        assert result.achieved_tflops > 0
        assert result.mfu == result.achieved_tflops / result.peak_tflops

    def test_mfu_achieved_tflops_calculation(self):
        """achieved_tflops = step_flops / (step_time_seconds × 1e12)."""
        step_flops = 242_000_000_000_000  # 242 TFLOPS worth
        step_time = 1.0  # 1 second
        result = compute_mfu(step_flops, step_time)
        assert abs(result.achieved_tflops - 242.0) < 1e-6
        assert abs(result.mfu - 1.0) < 1e-6

    def test_mfu_raises_on_zero_step_time(self):
        """compute_mfu should raise ValueError for step_time_seconds = 0."""
        with pytest.raises(ValueError):
            compute_mfu(step_flops=1_000_000, step_time_seconds=0.0)

    def test_mfu_raises_on_negative_step_time(self):
        """compute_mfu should raise ValueError for step_time_seconds < 0."""
        with pytest.raises(ValueError):
            compute_mfu(step_flops=1_000_000, step_time_seconds=-0.5)

    def test_mfu_custom_peak_tflops(self):
        """MFU should use the provided peak_tflops."""
        step_flops = 100_000_000_000_000  # 100 TFLOPS worth
        step_time = 1.0
        result = compute_mfu(step_flops, step_time, peak_tflops=100.0)
        assert abs(result.mfu - 1.0) < 1e-6
        assert result.peak_tflops == 100.0
