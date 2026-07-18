"""Tests for FlashAttention torch.compile compatibility.

Verifies that a model using FlashAttention:
1. Runs 3+ forward+backward steps under torch.compile(fullgraph=True) without graph breaks
2. Produces the same loss/gradients as eager mode (within tolerance)

Requirements satisfied: 8.1, 8.2
"""

import pytest
import torch

from src.models.config import ModelConfig
from src.models.modern_transformer import ModernTransformer

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _try_import_flash_attn():
    """Check if flash_attn is importable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# Also skip if flash_attn is not installed
pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not _try_import_flash_attn() if torch.cuda.is_available() else True,
        reason="flash_attn not installed",
    ),
]


@pytest.fixture
def small_config():
    """Small model config for fast testing."""
    return ModelConfig(
        n_layer=2,
        d_model=64,
        n_head=4,
        vocab_size=256,
        seq_len=64,
        ffn_multiplier=4,
        dropout=0.0,
        bias=False,
        tie_embeddings=True,
        variant="modern",
        norm_type="rmsnorm",
        position_encoding="rope",
        ffn_type="swiglu",
        attention_type="flash_sdpa",
        attention_backend="flash_attn",
    )


@pytest.fixture
def flash_model(small_config):
    """Create a ModernTransformer with FlashAttention on CUDA in bfloat16."""
    from src.models.flash_attention import FlashAttention

    model = ModernTransformer(small_config, attention_class=FlashAttention)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    return model


class TestTorchCompileCompatibility:
    """Verify torch.compile works with FlashAttention without graph breaks."""

    def test_compiled_model_runs_multiple_training_steps(self, flash_model, small_config):
        """A compiled model with FlashAttention can run 3+ forward+backward steps.

        Validates: Requirements 8.1
        """
        compiled_model = torch.compile(flash_model, fullgraph=True)
        optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-4)

        batch_size = 2
        seq_len = 32

        # Run 4 training steps to verify no graph breaks
        for step in range(4):
            idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")
            targets = torch.randint(
                0, small_config.vocab_size, (batch_size, seq_len), device="cuda"
            )

            optimizer.zero_grad()
            logits, loss, _ = compiled_model(idx, targets)

            assert loss is not None, f"Loss is None at step {step}"
            assert loss.dim() == 0, f"Loss is not scalar at step {step}"
            assert loss.item() > 0, f"Loss should be positive at step {step}"

            loss.backward()
            optimizer.step()

    def test_compiled_gradients_match_eager(self, small_config):
        """The compiled model produces the same loss/gradients as eager mode.

        Validates: Requirements 8.2
        """
        from src.models.flash_attention import FlashAttention

        # Create two identical models in bfloat16
        eager_model = (
            ModernTransformer(small_config, attention_class=FlashAttention)
            .cuda()
            .to(torch.bfloat16)
        )
        compiled_model = (
            ModernTransformer(small_config, attention_class=FlashAttention)
            .cuda()
            .to(torch.bfloat16)
        )

        # Copy weights from eager to compiled so they are identical
        compiled_model.load_state_dict(eager_model.state_dict())

        # Compile the second model
        compiled_model = torch.compile(compiled_model, fullgraph=True)

        batch_size = 2
        seq_len = 32

        # Use a fixed seed for reproducible input
        torch.manual_seed(42)
        idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")
        targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")

        # Run eager forward + backward
        eager_model.train()
        logits_eager, loss_eager, _ = eager_model(idx, targets)
        loss_eager.backward()

        # Run compiled forward + backward
        compiled_model.train()
        logits_compiled, loss_compiled, _ = compiled_model(idx, targets)
        loss_compiled.backward()

        # Compare losses — allow bf16 tolerance since compile may reorder ops
        assert torch.allclose(loss_eager, loss_compiled, atol=1e-2, rtol=1e-2), (
            f"Loss mismatch: eager={loss_eager.item():.6f}, compiled={loss_compiled.item():.6f}"
        )

        # Compare logits — bf16 precision with compiled reordering
        assert torch.allclose(logits_eager, logits_compiled, atol=1e-2, rtol=1e-2), (
            f"Logits differ between eager and compiled modes. "
            f"Max diff: {(logits_eager - logits_compiled).abs().max().item():.6e}"
        )

        # Compare gradients for all parameters
        eager_params = dict(eager_model.named_parameters())
        # compiled_model._orig_mod gives us the underlying module with gradients
        compiled_orig = (
            compiled_model._orig_mod
            if hasattr(compiled_model, "_orig_mod")
            else compiled_model
        )
        compiled_params = dict(compiled_orig.named_parameters())

        for name in eager_params:
            eager_grad = eager_params[name].grad
            compiled_grad = compiled_params[name].grad

            if eager_grad is None and compiled_grad is None:
                continue

            assert eager_grad is not None and compiled_grad is not None, (
                f"Gradient mismatch for {name}: one is None"
            )
            assert torch.allclose(eager_grad, compiled_grad, atol=1e-2, rtol=1e-2), (
                f"Gradient mismatch for parameter '{name}': "
                f"max diff = {(eager_grad - compiled_grad).abs().max().item():.6e}"
            )


def test_eager_prefill_and_cached_decode_match_full_sequence(flash_model, small_config):
    """The eager KV-cache path must preserve full-sequence attention logits."""
    torch.manual_seed(91)
    prompt = torch.randint(0, small_config.vocab_size, (2, 16), device="cuda")
    next_token = torch.randint(0, small_config.vocab_size, (2, 1), device="cuda")

    with torch.no_grad():
        flash_model.train()
        reference, _, _ = flash_model(torch.cat((prompt, next_token), dim=1))

        flash_model.eval()
        _, _, cache = flash_model(prompt)
        cached_logits, _, updated_cache = flash_model(next_token, kv_cache=cache)

    assert all(layer_cache is not None for layer_cache in cache)
    assert all(layer_cache[2].tolist() == [17, 17] for layer_cache in updated_cache)
    torch.testing.assert_close(
        cached_logits[:, -1].float(),
        reference[:, -1].float(),
        atol=2e-2,
        rtol=2e-2,
    )
