"""Unit tests for the SWA Interleaved variant — Tasks 5.1 and 5.9.

Covers:
  5.1 - Pattern verification at each scale (Requirements 2.3, 2.4, 2.5)
      - VariantSpec fields validation (Requirements 3.1, 3.2)
      - Per-layer config validation (wrong length, None fallback)
      - Forward pass shape / loss / backward (Requirements 8.2)
  5.9 - Backward compatibility for existing variants (Requirements 6.1–6.5)
      - Parameter count parity (Requirements 5.1, 5.4)
"""

import pytest
import torch

from src.models.registry import build, VARIANTS, SCALES
from src.models.config import ModelConfig
from src.models.flash_attention import FlashAttention
from src.models.modern_transformer import ModernTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 5.1  Pattern verification at each scale
# Requirements: 2.3, 2.4, 2.5
# ---------------------------------------------------------------------------

class TestInterleavedWindowPattern:
    """Verify the alternating window pattern at each scale."""

    def test_debug_scale_pattern(self):
        """build('swa_interleaved', 'debug') → window pattern [None, 128, None, 128].

        Validates: Requirements 2.3
        """
        model, config = build("swa_interleaved", "debug", dtype="float32")
        expected = [None, 128, None, 128]
        actual = [block.attn._window_size for block in model.blocks]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_main_scale_pattern(self):
        """build('swa_interleaved', 'main') → window pattern [None, 256, ...].

        Validates: Requirements 2.4
        """
        model, config = build("swa_interleaved", "main", dtype="float32")
        expected = [None, 256, None, 256, None, 256, None, 256]
        actual = [block.attn._window_size for block in model.blocks]
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_stretch_scale_pattern(self):
        """build('swa_interleaved', 'stretch') → window pattern [None, 256] * 6.

        Validates: Requirements 2.5
        """
        model, config = build("swa_interleaved", "stretch", dtype="float32")
        expected = [None, 256, None, 256, None, 256, None, 256, None, 256, None, 256]
        actual = [block.attn._window_size for block in model.blocks]
        assert actual == expected, f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 5.1  VariantSpec fields
# Requirements: 3.1, 3.2
# ---------------------------------------------------------------------------

class TestVariantSpecFields:
    """Verify the swa_interleaved VariantSpec has correct field values."""

    def test_variant_field(self):
        """VARIANTS['swa_interleaved'].variant == 'swa_interleaved'.

        Validates: Requirements 3.1
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.variant == "swa_interleaved"

    def test_attention_type(self):
        """VARIANTS['swa_interleaved'].attention_type == 'sliding_window'.

        Validates: Requirements 3.1
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.attention_type == "sliding_window"

    def test_attention_class(self):
        """VARIANTS['swa_interleaved'].attention_class == FlashAttention.

        Validates: Requirements 3.1
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.attention_class is FlashAttention

    def test_requires_bf16(self):
        """VARIANTS['swa_interleaved'].requires_bf16 == True.

        Validates: Requirements 3.2
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.requires_bf16 is True

    def test_model_class(self):
        """VARIANTS['swa_interleaved'].model_class == ModernTransformer.

        Validates: Requirements 3.1
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.model_class is ModernTransformer

    def test_norm_type(self):
        """VARIANTS['swa_interleaved'].norm_type == 'rmsnorm'.

        Validates: Requirements 3.2
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.norm_type == "rmsnorm"

    def test_position_encoding(self):
        """VARIANTS['swa_interleaved'].position_encoding == 'rope'.

        Validates: Requirements 3.2
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.position_encoding == "rope"

    def test_ffn_type(self):
        """VARIANTS['swa_interleaved'].ffn_type == 'swiglu'.

        Validates: Requirements 3.2
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.ffn_type == "swiglu"

    def test_default_activation(self):
        """VARIANTS['swa_interleaved'].default_activation == 'swiglu'.

        Validates: Requirements 3.2
        """
        spec = VARIANTS["swa_interleaved"]
        assert spec.default_activation == "swiglu"


# ---------------------------------------------------------------------------
# 5.1  Per-layer config validation
# ---------------------------------------------------------------------------

class TestPerLayerConfigValidation:
    """Verify per_layer_configs validation and fallback behavior."""

    def test_wrong_length_raises_value_error(self):
        """per_layer_configs with wrong length (too short) must raise ValueError."""
        config = ModelConfig(
            n_layer=4,
            d_model=64,
            n_head=4,
            vocab_size=100,
            seq_len=64,
            ffn_multiplier=4,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
        )
        # Create a list of 3 configs (but n_layer is 4)
        wrong_length_configs = [config, config, config]
        with pytest.raises(ValueError, match="per_layer_configs length"):
            ModernTransformer(config, per_layer_configs=wrong_length_configs)

    def test_wrong_length_too_long_raises_value_error(self):
        """per_layer_configs longer than n_layer must raise ValueError."""
        config = ModelConfig(
            n_layer=4,
            d_model=64,
            n_head=4,
            vocab_size=100,
            seq_len=64,
            ffn_multiplier=4,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
        )
        too_long_configs = [config] * 6
        with pytest.raises(ValueError, match="per_layer_configs length"):
            ModernTransformer(config, per_layer_configs=too_long_configs)

    def test_none_per_layer_configs_uniform_window_size(self):
        """per_layer_configs=None → all blocks have same window_size as config."""
        if not _flash_attn_available():
            pytest.skip("flash_attn not installed — cannot instantiate FlashAttention")

        config = ModelConfig(
            n_layer=4,
            d_model=64,
            n_head=4,
            vocab_size=100,
            seq_len=64,
            ffn_multiplier=4,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
            window_size=16,
            attention_backend="flash_attn",
        )
        model = ModernTransformer(config, attention_class=FlashAttention, per_layer_configs=None)
        for block in model.blocks:
            assert block.attn._window_size == 16


# ---------------------------------------------------------------------------
# 5.1  Forward pass shape
# Requirements: 8.2
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_flash_attn_available() and torch.cuda.is_available()),
    reason="flash_attn and CUDA required for SWA interleaved forward pass",
)
class TestForwardPass:
    """Verify forward pass shape, loss, and backward for swa_interleaved at debug scale."""

    @pytest.fixture(scope="class")
    @classmethod
    def model_and_config(cls):
        """Build the swa_interleaved debug model on CUDA."""
        model, config = build("swa_interleaved", "debug", dtype="bfloat16")
        model = model.cuda()
        return model, config

    def test_output_shape(self, model_and_config):
        """Forward with (B=2, T=64) → logits shape (2, 64, 50257).

        Validates: Requirements 8.2
        """
        model, config = model_and_config
        model.eval()
        B, T = 2, 64
        idx = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        with torch.no_grad():
            logits, loss, _ = model(idx)

        assert logits.shape == (B, T, 50257), f"Expected (2, 64, 50257), got {logits.shape}"

    def test_loss_is_finite(self, model_and_config):
        """Forward with targets → loss is finite.

        Validates: Requirements 8.2
        """
        model, config = model_and_config
        model.train()
        B, T = 2, 64
        idx = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        targets = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        logits, loss, _ = model(idx, targets)

        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_backward_pass(self, model_and_config):
        """loss.backward() doesn't raise.

        Validates: Requirements 8.2
        """
        model, config = model_and_config
        model.train()
        # Zero grads to get clean state
        model.zero_grad()
        B, T = 2, 64
        idx = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        targets = torch.randint(0, config.vocab_size, (B, T), device="cuda")
        logits, loss, _ = model(idx, targets)

        # Should not raise
        loss.backward()

        # Verify at least some gradients were computed
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Expected at least one parameter to have non-zero gradients"


# ---------------------------------------------------------------------------
# 5.9  Backward compatibility tests
# Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5
# ---------------------------------------------------------------------------

class TestSWAVariantUnchanged:
    """Verify that the 'swa' variant is unchanged after per_layer_configs extension.

    Requirements: 6.1, 6.2
    """

    def test_swa_debug_window_size_uniform(self):
        """SWA at debug scale: config.window_size == 128 (uniform, not interleaved).

        Validates: Requirements 6.1, 6.2
        """
        _, config = build("swa", "debug", dtype="float32")
        assert config.window_size == 128

    def test_swa_debug_all_blocks_same_window_size(self):
        """SWA at debug scale: ALL blocks have window_size == 128 (not alternating).

        Validates: Requirements 6.1, 6.2
        """
        model, config = build("swa", "debug", dtype="float32")
        for i, block in enumerate(model.blocks):
            assert hasattr(block.attn, "_window_size"), (
                f"Block {i} attention must have _window_size attribute"
            )
            assert block.attn._window_size == 128, (
                f"Block {i}: expected _window_size=128, got {block.attn._window_size}"
            )


class TestModernVariantUnchanged:
    """Verify that the 'modern' variant is unchanged after per_layer_configs extension.

    Requirements: 6.3
    """

    def test_modern_debug_window_size_none(self):
        """Modern at debug scale: config.window_size is None.

        Validates: Requirements 6.3
        """
        _, config = build("modern", "debug", dtype="float32")
        assert config.window_size is None

    def test_modern_debug_all_blocks_window_size_none(self):
        """Modern at debug scale: ALL blocks have window_size == None (if they have _window_size).

        Validates: Requirements 6.3
        """
        model, _ = build("modern", "debug", dtype="float32")
        for i, block in enumerate(model.blocks):
            if hasattr(block.attn, "_window_size"):
                assert block.attn._window_size is None, (
                    f"Block {i}: expected _window_size=None, got {block.attn._window_size}"
                )


class TestPerLayerConfigsNoneIdenticalBehavior:
    """Verify per_layer_configs=None produces identical behavior to standard path.

    Requirements: 6.1, 6.5
    """

    def test_modern_no_per_layer_configs(self):
        """Modern variant built via standard path has no per_layer_configs (all blocks same config).

        Validates: Requirements 6.1, 6.5
        """
        model, config = build("modern", "debug", dtype="float32")

        # All blocks should use the same config values — verify d_model consistency
        for i, block in enumerate(model.blocks):
            assert block.ln1.weight.shape[0] == config.d_model, (
                f"Block {i}: ln1 dimension mismatch"
            )
            assert block.attn.out_proj.weight.shape[0] == config.d_model, (
                f"Block {i}: attn output dimension mismatch"
            )

    def test_explicit_none_matches_default(self):
        """Explicitly passing per_layer_configs=None matches default construction.

        Validates: Requirements 6.1, 6.5
        """
        from src.models.modern_attention import ModernAttention

        config = ModelConfig(
            n_layer=4,
            d_model=256,
            n_head=4,
            vocab_size=50257,
            seq_len=512,
            ffn_multiplier=4,
            dropout=0.0,
            bias=False,
            tie_embeddings=True,
        )

        # Build with explicit None
        torch.manual_seed(42)
        model_explicit = ModernTransformer(config, attention_class=ModernAttention, per_layer_configs=None)

        # Build without the kwarg (default)
        torch.manual_seed(42)
        model_default = ModernTransformer(config, attention_class=ModernAttention)

        # Verify identical parameters
        for (name1, p1), (name2, p2) in zip(
            model_explicit.named_parameters(), model_default.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.equal(p1, p2), f"Parameter {name1} values differ"


class TestParameterCountParity:
    """Verify swa_interleaved has identical parameter count to swa.

    Requirements: 5.1, 5.4
    """

    def test_debug_scale_same_total_params(self):
        """SWA and SWA_Interleaved at debug scale have identical total parameter count.

        Validates: Requirements 5.1
        """
        swa_model, _ = build("swa", "debug", dtype="float32")
        interleaved_model, _ = build("swa_interleaved", "debug", dtype="float32")

        swa_params = sum(p.numel() for p in swa_model.parameters())
        interleaved_params = sum(p.numel() for p in interleaved_model.parameters())

        assert swa_params == interleaved_params, (
            f"Parameter count mismatch: swa={swa_params}, "
            f"swa_interleaved={interleaved_params}"
        )

    def test_debug_scale_same_param_names_and_shapes(self):
        """SWA and SWA_Interleaved at debug scale have identical parameter name and shape sets.

        Validates: Requirements 5.4
        """
        swa_model, _ = build("swa", "debug", dtype="float32")
        interleaved_model, _ = build("swa_interleaved", "debug", dtype="float32")

        swa_param_set = {(name, tuple(p.shape)) for name, p in swa_model.named_parameters()}
        interleaved_param_set = {(name, tuple(p.shape)) for name, p in interleaved_model.named_parameters()}

        assert swa_param_set == interleaved_param_set, (
            f"Parameter name/shape mismatch.\n"
            f"In SWA but not interleaved: {swa_param_set - interleaved_param_set}\n"
            f"In interleaved but not SWA: {interleaved_param_set - swa_param_set}"
        )
