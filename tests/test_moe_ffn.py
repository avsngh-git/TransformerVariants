"""Unit tests for MoEFeedForward module — Task 2.1.

Covers:
  1. Construction with valid config
  2. Forward pass produces correct output shape
  3. Aux loss is computed in training mode
  4. Aux loss is NOT computed in eval mode
  5. get_aux_loss returns and clears
  6. Routing data is captured when enabled
  7. Invalid config raises ValueError
"""

import pytest
import torch

from src.models.config import ModelConfig
from src.models.moe_ffn import MoEFeedForward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moe_config(**overrides) -> ModelConfig:
    """Create a minimal MoE-enabled ModelConfig for testing."""
    defaults = dict(
        n_layer=4,
        d_model=64,
        n_head=4,
        vocab_size=100,
        seq_len=32,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        num_experts=4,
        moe_top_k=2,
        aux_loss_alpha=0.01,
        z_loss_beta=0.001,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Construction with valid config
# ---------------------------------------------------------------------------

class TestConstruction:
    """Verify MoEFeedForward constructs correctly with valid configurations."""

    def test_basic_construction(self):
        """MoEFeedForward with 4 experts, top-2 constructs without error."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        assert moe.num_experts == 4
        assert moe.top_k == 2
        assert len(moe.experts) == 4

    def test_router_shape(self):
        """Router linear layer has shape (d_model, num_experts) with no bias."""
        config = _moe_config(d_model=64, num_experts=8)
        moe = MoEFeedForward(config)
        assert moe.router.weight.shape == (8, 64)
        assert moe.router.bias is None

    def test_expert_count_matches_config(self):
        """Number of expert modules matches num_experts."""
        config = _moe_config(num_experts=6)
        moe = MoEFeedForward(config)
        assert len(moe.experts) == 6

    def test_top_k_equals_num_experts(self):
        """Construction works when moe_top_k == num_experts (full routing)."""
        config = _moe_config(num_experts=4, moe_top_k=4)
        moe = MoEFeedForward(config)
        assert moe.top_k == 4

    def test_record_routing_defaults_false(self):
        """record_routing defaults to False."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        assert moe.record_routing is False

    def test_routing_buffer_starts_empty(self):
        """_routing_buffer starts as empty list."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        assert moe._routing_buffer == []

    def test_aux_loss_starts_none(self):
        """_aux_loss starts as None."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        assert moe._aux_loss is None


# ---------------------------------------------------------------------------
# 2. Forward pass produces correct output shape
# ---------------------------------------------------------------------------

class TestForwardShape:
    """Verify forward pass input/output shape contract."""

    def test_basic_forward_shape(self):
        """(B=2, T=8, D=64) → output shape (2, 8, 64)."""
        config = _moe_config(d_model=64)
        moe = MoEFeedForward(config)
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = moe(x)
        assert out.shape == (2, 8, 64)

    def test_single_token_forward(self):
        """(B=1, T=1, D=64) → output shape (1, 1, 64)."""
        config = _moe_config(d_model=64)
        moe = MoEFeedForward(config)
        moe.eval()
        x = torch.randn(1, 1, 64)
        with torch.no_grad():
            out = moe(x)
        assert out.shape == (1, 1, 64)

    def test_larger_batch_forward(self):
        """(B=4, T=16, D=64) → output shape (4, 16, 64)."""
        config = _moe_config(d_model=64)
        moe = MoEFeedForward(config)
        moe.eval()
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            out = moe(x)
        assert out.shape == (4, 16, 64)


# ---------------------------------------------------------------------------
# 3. Aux loss is computed in training mode
# ---------------------------------------------------------------------------

class TestAuxLossTrainingMode:
    """Verify auxiliary loss is computed during training forward pass."""

    def test_aux_loss_stored_after_forward(self):
        """After forward in training mode, _aux_loss is not None."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        assert moe._aux_loss is not None

    def test_aux_loss_is_scalar(self):
        """Stored aux loss is a scalar tensor."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        assert moe._aux_loss.dim() == 0

    def test_aux_loss_is_finite(self):
        """Stored aux loss is finite."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        assert torch.isfinite(moe._aux_loss)

    def test_aux_loss_requires_grad(self):
        """Stored aux loss participates in autograd (requires_grad=True)."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        assert moe._aux_loss.requires_grad

    def test_aux_loss_zero_when_coefficients_zero(self):
        """Aux loss is zero when both alpha and beta are 0."""
        config = _moe_config(aux_loss_alpha=0.0, z_loss_beta=0.0)
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        assert moe._aux_loss.item() == 0.0


# ---------------------------------------------------------------------------
# 4. Aux loss is NOT computed in eval mode
# ---------------------------------------------------------------------------

class TestAuxLossEvalMode:
    """Verify auxiliary loss is NOT computed during eval forward pass."""

    def test_aux_loss_none_in_eval(self):
        """After forward in eval mode, _aux_loss remains None."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
        assert moe._aux_loss is None

    def test_aux_loss_not_overwritten_in_eval(self):
        """Eval forward does not overwrite previously cleared aux loss."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        # First do a training forward to set aux loss
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        # Clear it
        moe.get_aux_loss()
        assert moe._aux_loss is None
        # Eval forward should leave it None
        moe.eval()
        with torch.no_grad():
            _ = moe(x)
        assert moe._aux_loss is None


# ---------------------------------------------------------------------------
# 5. get_aux_loss returns and clears
# ---------------------------------------------------------------------------

class TestGetAuxLoss:
    """Verify get_aux_loss() retrieval and clearing behavior."""

    def test_returns_stored_loss(self):
        """get_aux_loss() returns the stored aux loss tensor."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        loss = moe.get_aux_loss()
        assert loss is not None
        assert loss.dim() == 0

    def test_clears_after_retrieval(self):
        """After get_aux_loss(), _aux_loss is set to None."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        _ = moe.get_aux_loss()
        assert moe._aux_loss is None

    def test_second_call_returns_zero(self):
        """Second call to get_aux_loss() without forward returns zero tensor."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        _ = moe.get_aux_loss()  # first call
        zero_loss = moe.get_aux_loss()  # second call
        assert zero_loss.item() == 0.0

    def test_returns_zero_when_never_forwarded(self):
        """get_aux_loss() returns zero if no forward pass has occurred."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        loss = moe.get_aux_loss()
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 6. Routing data is captured when enabled
# ---------------------------------------------------------------------------

class TestRoutingDataCapture:
    """Verify routing data capture with record_routing flag."""

    def test_routing_captured_when_enabled(self):
        """With record_routing=True, _routing_buffer has entries after forward."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.record_routing = True
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
        assert len(moe._routing_buffer) == 1

    def test_routing_not_captured_when_disabled(self):
        """With record_routing=False, _routing_buffer remains empty."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.record_routing = False
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
        assert len(moe._routing_buffer) == 0

    def test_routing_data_shapes(self):
        """Captured routing data has correct shapes: (B, T, top_k)."""
        config = _moe_config(num_experts=4, moe_top_k=2)
        moe = MoEFeedForward(config)
        moe.record_routing = True
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
        indices, weights = moe._routing_buffer[0]
        assert indices.shape == (2, 8, 2)
        assert weights.shape == (2, 8, 2)

    def test_routing_data_detached(self):
        """Captured routing tensors are detached (no grad)."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.record_routing = True
        moe.train()
        x = torch.randn(2, 8, 64)
        _ = moe(x)
        indices, weights = moe._routing_buffer[0]
        assert not indices.requires_grad
        assert not weights.requires_grad

    def test_get_routing_data_returns_and_clears(self):
        """get_routing_data() returns buffer and clears it."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.record_routing = True
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
            _ = moe(x)
        data = moe.get_routing_data()
        assert len(data) == 2
        # Buffer should be empty now
        assert len(moe._routing_buffer) == 0
        # Second call returns empty
        data2 = moe.get_routing_data()
        assert len(data2) == 0

    def test_multiple_forwards_accumulate(self):
        """Multiple forward passes accumulate routing data in buffer."""
        config = _moe_config()
        moe = MoEFeedForward(config)
        moe.record_routing = True
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            _ = moe(x)
            _ = moe(x)
            _ = moe(x)
        assert len(moe._routing_buffer) == 3


# ---------------------------------------------------------------------------
# 7. Invalid config raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidConfig:
    """Verify invalid configurations raise ValueError."""

    def test_top_k_zero_raises(self):
        """moe_top_k=0 raises ValueError (caught at config or module level)."""
        with pytest.raises(ValueError, match="moe_top_k must be between 1 and"):
            _moe_config(moe_top_k=0)

    def test_top_k_exceeds_num_experts_raises(self):
        """moe_top_k > num_experts raises ValueError (caught at config or module level)."""
        with pytest.raises(ValueError, match="moe_top_k must be between 1 and"):
            _moe_config(num_experts=4, moe_top_k=5)

    def test_top_k_negative_raises(self):
        """moe_top_k=-1 raises ValueError (caught at config or module level)."""
        with pytest.raises(ValueError, match="moe_top_k must be between 1 and"):
            _moe_config(moe_top_k=-1)

    def test_config_level_validation_num_experts_less_than_2(self):
        """num_experts=1 raises ValueError at ModelConfig level."""
        with pytest.raises(ValueError, match="num_experts must be >= 2"):
            ModelConfig(
                n_layer=4,
                d_model=64,
                n_head=4,
                vocab_size=100,
                seq_len=32,
                num_experts=1,
            )

    def test_config_level_validation_moe_top_k_exceeds(self):
        """moe_top_k > num_experts raises ValueError at ModelConfig level."""
        with pytest.raises(ValueError, match="moe_top_k must be between 1 and"):
            ModelConfig(
                n_layer=4,
                d_model=64,
                n_head=4,
                vocab_size=100,
                seq_len=32,
                num_experts=4,
                moe_top_k=5,
            )
