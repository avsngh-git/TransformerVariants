"""Integration tests for MoE routing data capture, generation support, and trainer integration.

Task 9.1: Verifies end-to-end behavior of:
  1. Routing data capture (record_routing flag, get_routing_data, shape checks)
  2. Generation support (autoregressive generation with KV-cache, no aux loss in eval)
  3. Trainer integration (mini training step, aux_loss flows gradients to router)
  4. Dense model compatibility (get_aux_loss returns 0, get_routing_data returns empty)

Requirements traced: 6.1, 6.2, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import pytest
import torch

from src.models.config import ModelConfig
from src.models.modern_transformer import ModernTransformer
from src.models.modern_attention import ModernAttention
from src.models.generate import generate
from src.models.moe_ffn import MoEFeedForward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moe_config(**overrides) -> ModelConfig:
    """Create a minimal MoE-enabled config for fast test execution."""
    defaults = dict(
        n_layer=2,
        d_model=64,
        n_head=2,
        vocab_size=100,
        seq_len=32,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        variant="moe",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_backend="sdpa",
        num_experts=4,
        moe_top_k=2,
        aux_loss_alpha=0.01,
        z_loss_beta=0.001,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _dense_config(**overrides) -> ModelConfig:
    """Create a minimal dense (modern) config for compatibility testing."""
    defaults = dict(
        n_layer=2,
        d_model=64,
        n_head=2,
        vocab_size=100,
        seq_len=32,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        variant="modern",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_backend="sdpa",
        num_experts=None,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _build_moe_model(**config_overrides) -> ModernTransformer:
    """Build a small MoE model for testing."""
    config = _moe_config(**config_overrides)
    model = ModernTransformer(config, attention_class=ModernAttention)
    return model


def _build_dense_model(**config_overrides) -> ModernTransformer:
    """Build a small dense model for testing."""
    config = _dense_config(**config_overrides)
    model = ModernTransformer(config, attention_class=ModernAttention)
    return model


def _set_record_routing(model: ModernTransformer, enabled: bool) -> None:
    """Set record_routing flag on all MoE layers."""
    for block in model.blocks:
        if hasattr(block.ffn, 'record_routing'):
            block.ffn.record_routing = enabled


# ---------------------------------------------------------------------------
# 1. Routing Data Capture
# ---------------------------------------------------------------------------

class TestRoutingDataCapture:
    """Verify routing data capture through the full model pipeline."""

    def test_routing_data_returns_nonempty_dict(self):
        """With record_routing=True, get_routing_data() returns non-empty dict."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_routing_data_keys_are_layer_indices(self):
        """Keys of routing data dict are integer layer indices."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        for key in data.keys():
            assert isinstance(key, int)
            assert 0 <= key < model.config.n_layer

    def test_routing_data_correct_shapes(self):
        """Values have correct shapes: (batch, seq_len, top_k)."""
        batch_size = 2
        seq_len = 8
        top_k = 2

        model = _build_moe_model(moe_top_k=top_k)
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (batch_size, seq_len))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        for layer_idx, entries in data.items():
            assert len(entries) == 1  # one forward pass
            indices, weights = entries[0]
            assert indices.shape == (batch_size, seq_len, top_k)
            assert weights.shape == (batch_size, seq_len, top_k)

    def test_second_call_returns_empty_dict(self):
        """After get_routing_data(), a second call returns empty dict (buffers cleared)."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        # First call: non-empty
        data1 = model.get_routing_data()
        assert len(data1) > 0

        # Second call: empty (buffers cleared)
        data2 = model.get_routing_data()
        assert len(data2) == 0

    def test_record_routing_false_returns_empty_dict(self):
        """With record_routing=False, get_routing_data() returns empty dict."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, False)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        assert len(data) == 0

    def test_multiple_forwards_accumulate_in_buffer(self):
        """Multiple forward passes accumulate entries in the routing buffer."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)
            model(x)
            model(x)

        data = model.get_routing_data()
        for layer_idx, entries in data.items():
            assert len(entries) == 3  # three forward passes

    def test_expert_indices_are_valid(self):
        """Expert indices are within [0, num_experts)."""
        num_experts = 4
        model = _build_moe_model(num_experts=num_experts)
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        for layer_idx, entries in data.items():
            indices, _ = entries[0]
            assert (indices >= 0).all()
            assert (indices < num_experts).all()

    def test_expert_weights_sum_to_one(self):
        """Expert weights per token sum to approximately 1.0 (renormalized)."""
        model = _build_moe_model()
        model.eval()
        _set_record_routing(model, True)

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        for layer_idx, entries in data.items():
            _, weights = entries[0]
            weight_sums = weights.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Generation Support
# ---------------------------------------------------------------------------

class TestGenerationSupport:
    """Verify MoE model supports autoregressive generation with KV-cache."""

    def test_generation_produces_valid_tokens(self):
        """Generated tokens are valid indices in [0, vocab_size)."""
        vocab_size = 100
        model = _build_moe_model(vocab_size=vocab_size)
        model.eval()

        prompt = torch.randint(0, vocab_size, (1, 3))
        output = generate(model, prompt, max_new_tokens=10, temperature=0.0)

        # Output should include prompt + generated tokens
        assert output.shape == (1, 13)
        # All tokens should be valid
        assert (output >= 0).all()
        assert (output < vocab_size).all()

    def test_generation_no_errors_with_cache(self):
        """Generation with KV-cache completes without errors."""
        vocab_size = 100
        model = _build_moe_model(vocab_size=vocab_size)
        model.eval()

        prompt = torch.randint(0, vocab_size, (1, 4))
        # This should run without any exceptions
        output = generate(model, prompt, max_new_tokens=15, temperature=1.0, use_cache=True)
        assert output.shape == (1, 19)

    def test_generation_no_errors_without_cache(self):
        """Generation without KV-cache also works correctly."""
        vocab_size = 100
        model = _build_moe_model(vocab_size=vocab_size)
        model.eval()

        prompt = torch.randint(0, vocab_size, (1, 4))
        output = generate(model, prompt, max_new_tokens=10, temperature=0.0, use_cache=False)
        assert output.shape == (1, 14)
        assert (output >= 0).all()
        assert (output < vocab_size).all()

    def test_no_aux_loss_during_generation(self):
        """Aux loss is NOT accumulated during generation (eval mode)."""
        model = _build_moe_model()
        model.eval()

        prompt = torch.randint(0, 100, (1, 4))
        generate(model, prompt, max_new_tokens=5, temperature=0.0)

        # After generation in eval mode, aux loss should be zero
        aux_loss = model.get_aux_loss()
        assert aux_loss.item() == 0.0

    def test_generation_greedy_is_deterministic(self):
        """Greedy generation (temperature=0) is deterministic for MoE models."""
        model = _build_moe_model()
        model.eval()

        prompt = torch.randint(0, 100, (1, 5))
        out1 = generate(model, prompt, max_new_tokens=8, temperature=0.0)
        out2 = generate(model, prompt, max_new_tokens=8, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_cached_matches_uncached_greedy(self):
        """Cached and uncached greedy generation produce identical results."""
        model = _build_moe_model()
        model.eval()

        prompt = torch.randint(0, 100, (1, 5))
        out_cached = generate(model, prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
        out_uncached = generate(model, prompt, max_new_tokens=8, temperature=0.0, use_cache=False)
        assert torch.equal(out_cached, out_uncached)


# ---------------------------------------------------------------------------
# 3. Trainer Integration (Mini Training Step)
# ---------------------------------------------------------------------------

class TestTrainerIntegration:
    """Verify MoE model works in a training step with aux loss and gradients."""

    def test_loss_backward_works(self):
        """loss.backward() succeeds — gradients flow through MoE routing."""
        model = _build_moe_model()
        model.train()

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        logits, ce_loss, _ = model(x, targets=targets)
        aux_loss = model.get_aux_loss()
        total_loss = ce_loss + aux_loss

        total_loss.backward()

        # Verify at least some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_aux_loss_is_nonzero(self):
        """Aux loss is non-zero for MoE model in training mode."""
        model = _build_moe_model()
        model.train()

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        _, _, _ = model(x, targets=targets)
        aux_loss = model.get_aux_loss()

        # With non-zero alpha and beta, aux_loss should be > 0
        assert aux_loss.item() > 0.0

    def test_router_parameters_receive_gradients(self):
        """Router parameters receive gradients via aux loss backpropagation."""
        model = _build_moe_model()
        model.train()

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        _, ce_loss, _ = model(x, targets=targets)
        aux_loss = model.get_aux_loss()
        total_loss = ce_loss + aux_loss
        total_loss.backward()

        # Check that router weights in each MoE layer have gradients
        for block in model.blocks:
            if isinstance(block.ffn, MoEFeedForward):
                assert block.ffn.router.weight.grad is not None
                assert block.ffn.router.weight.grad.abs().sum() > 0

    def test_expert_parameters_receive_gradients(self):
        """Expert FFN parameters also receive gradients."""
        model = _build_moe_model()
        model.train()

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        _, ce_loss, _ = model(x, targets=targets)
        aux_loss = model.get_aux_loss()
        total_loss = ce_loss + aux_loss
        total_loss.backward()

        # Check that at least some expert parameters have gradients
        for block in model.blocks:
            if isinstance(block.ffn, MoEFeedForward):
                expert_has_grad = False
                for expert in block.ffn.experts:
                    for p in expert.parameters():
                        if p.grad is not None and p.grad.abs().sum() > 0:
                            expert_has_grad = True
                            break
                    if expert_has_grad:
                        break
                assert expert_has_grad

    def test_gradient_accumulation_pattern(self):
        """Simulate gradient accumulation: multiple forward/backward steps."""
        model = _build_moe_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        # Simulate 2 micro-batches of gradient accumulation
        for _ in range(2):
            x = torch.randint(0, 100, (2, 16))
            targets = torch.randint(0, 100, (2, 16))

            _, ce_loss, _ = model(x, targets=targets)
            aux_loss = model.get_aux_loss()
            total_loss = (ce_loss + aux_loss) / 2  # scale by grad_accum_steps
            total_loss.backward()

        # Optimizer step should succeed
        optimizer.step()
        optimizer.zero_grad()


# ---------------------------------------------------------------------------
# 4. Dense Model Compatibility
# ---------------------------------------------------------------------------

class TestDenseModelCompatibility:
    """Verify dense (non-MoE) model compatibility with MoE API methods."""

    def test_get_aux_loss_returns_zero(self):
        """Dense model's get_aux_loss() returns 0.0."""
        model = _build_dense_model()
        model.train()

        x = torch.randint(0, 100, (2, 8))
        _, _, _ = model(x)

        aux_loss = model.get_aux_loss()
        assert aux_loss.item() == 0.0

    def test_get_routing_data_returns_empty_dict(self):
        """Dense model's get_routing_data() returns empty dict."""
        model = _build_dense_model()
        model.eval()

        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            model(x)

        data = model.get_routing_data()
        assert data == {}

    def test_dense_model_no_moe_layers(self):
        """Dense model has no MoEFeedForward instances in its blocks."""
        model = _build_dense_model()
        for block in model.blocks:
            assert not isinstance(block.ffn, MoEFeedForward)

    def test_dense_aux_loss_does_not_affect_training(self):
        """Adding get_aux_loss() to dense training step has no effect (returns 0)."""
        model = _build_dense_model()
        model.train()

        x = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))

        _, ce_loss, _ = model(x, targets=targets)
        aux_loss = model.get_aux_loss()

        # aux_loss should be exactly 0.0 — doesn't change total loss
        assert aux_loss.item() == 0.0

        total_loss = ce_loss + aux_loss
        total_loss.backward()

        # Training should still work fine
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
