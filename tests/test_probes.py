"""Unit tests for the MQAR probe and CKA in src/evaluation/probes.py."""

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.probes import (
    CKAResult,
    MQARResult,
    _generate_mqar_sequences,
    compute_cka,
    run_mqar_probe,
)
from src.models.config import ModelConfig
from src.models.vanilla_transformer import VanillaTransformer


class DebugModel(nn.Module):
    """A trivial model that always predicts a fixed token — used to test probe wiring."""

    def __init__(self, vocab_size: int = 50257):
        super().__init__()
        self.vocab_size = vocab_size
        # A dummy parameter so the model is recognized as a module
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        # Return uniform logits — random predictions
        logits = torch.zeros(B, T, self.vocab_size, device=idx.device)
        return logits, None, []


class CopyModel(nn.Module):
    """A model that 'cheats' by copying the input token as its prediction.

    At each position, it puts high logit on the input token at that position.
    This simulates a model that echoes its input, which is useful for testing
    because MQAR places the key at the query position — so if the model predicts
    the value at the next-token position, we can test with a model that always
    predicts the key itself (which won't match the expected value).
    """

    def __init__(self, vocab_size: int = 50257):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.vocab_size, device=idx.device)
        # Put high logit on the input token at each position
        for b in range(B):
            for t in range(T):
                logits[b, t, idx[b, t]] = 100.0
        return logits, None, []


class BatchTrackingModel(DebugModel):
    """Debug model that records MQAR inference micro-batch sizes."""

    def __init__(self, vocab_size: int = 50257):
        super().__init__(vocab_size=vocab_size)
        self.batch_sizes = []

    def forward(self, idx, targets=None, kv_cache=None):
        self.batch_sizes.append(idx.shape[0])
        return super().forward(idx, targets=targets, kv_cache=kv_cache)


class TestMQARSequenceGeneration:
    """Tests for _generate_mqar_sequences."""

    def test_output_shape(self):
        """Tokens tensor has correct shape."""
        seq_len = 64
        n_sequences = 8
        n_associations = 4
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=n_sequences,
            n_associations=n_associations,
            seq_len=seq_len,
            vocab_size=1000,
            device="cpu",
        )
        assert tokens.shape == (n_sequences, seq_len)

    def test_positions_within_bounds(self):
        """All key and query positions are within [0, seq_len)."""
        seq_len = 64
        n_sequences = 16
        n_associations = 4
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=n_sequences,
            n_associations=n_associations,
            seq_len=seq_len,
            vocab_size=1000,
            device="cpu",
        )
        for seq_idx in range(n_sequences):
            for kp in key_pos[seq_idx]:
                assert 0 <= kp < seq_len
            for qp in query_pos[seq_idx]:
                assert 0 <= qp < seq_len

    def test_queries_after_keys(self):
        """Query positions appear after all key-value pair positions."""
        seq_len = 128
        n_sequences = 16
        n_associations = 8
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=n_sequences,
            n_associations=n_associations,
            seq_len=seq_len,
            vocab_size=1000,
            device="cpu",
        )
        for seq_idx in range(n_sequences):
            max_key_val_pos = max(key_pos[seq_idx]) + 1  # value is right after key
            min_query_pos = min(query_pos[seq_idx])
            assert min_query_pos > max_key_val_pos

    def test_correct_number_of_associations(self):
        """Each sequence has exactly n_associations keys, queries, and expected values."""
        n_associations = 6
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=4,
            n_associations=n_associations,
            seq_len=64,
            vocab_size=1000,
            device="cpu",
        )
        for seq_idx in range(4):
            assert len(key_pos[seq_idx]) == n_associations
            assert len(query_pos[seq_idx]) == n_associations
            assert len(expected[seq_idx]) == n_associations

    def test_key_value_consistency(self):
        """The value token after each key matches the expected value for that query."""
        n_associations = 4
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=8,
            n_associations=n_associations,
            seq_len=64,
            vocab_size=1000,
            device="cpu",
        )
        for seq_idx in range(8):
            for i in range(n_associations):
                # Value is at key_pos + 1
                val_pos = key_pos[seq_idx][i] + 1
                actual_val = tokens[seq_idx, val_pos].item()
                assert actual_val == expected[seq_idx][i]

    def test_exact_sequence_count(self):
        """Generates exactly n_sequences sequences (requirement 5.6)."""
        n_sequences = 256
        tokens, key_pos, query_pos, expected = _generate_mqar_sequences(
            n_sequences=n_sequences,
            n_associations=8,
            seq_len=128,
            vocab_size=1000,
            device="cpu",
        )
        assert tokens.shape[0] == n_sequences
        assert len(key_pos) == n_sequences


class TestRunMQARProbe:
    """Tests for run_mqar_probe end-to-end."""

    def test_returns_mqar_result(self):
        """run_mqar_probe returns an MQARResult dataclass."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        result = run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=8, device="cpu"
        )
        assert isinstance(result, MQARResult)

    def test_accuracy_in_valid_range(self):
        """Overall accuracy is between 0 and 1."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        result = run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=8, device="cpu"
        )
        assert 0.0 <= result.accuracy <= 1.0

    def test_accuracy_by_distance_keys_positive(self):
        """All distances in accuracy_by_distance are positive (query after key)."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        result = run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=16, device="cpu"
        )
        for dist in result.accuracy_by_distance:
            assert dist > 0

    def test_accuracy_by_distance_values_valid(self):
        """Per-distance accuracy values are in [0, 1]."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        result = run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=16, device="cpu"
        )
        for dist, acc in result.accuracy_by_distance.items():
            assert 0.0 <= acc <= 1.0

    def test_random_model_low_accuracy(self):
        """A model with uniform logits should have near-zero accuracy on a large vocab."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        result = run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=32, device="cpu"
        )
        # With vocab_size=1000, random chance is ~0.1%, expect very low accuracy
        assert result.accuracy < 0.1

    def test_no_grad_context(self):
        """Probe runs in inference mode (no gradients tracked)."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = DebugModel(vocab_size=1000)
        # If gradients were tracked, this would increase memory; we verify no grad
        # by checking that model parameters don't have .grad set after probe
        run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=8, device="cpu"
        )
        for param in model.parameters():
            assert param.grad is None

    def test_default_micro_batch_bounds_full_vocab_logits(self):
        """MQAR must not materialize logits for more than eight sequences at once."""
        config = ModelConfig(n_layer=2, d_model=64, seq_len=128, vocab_size=1000)
        model = BatchTrackingModel(vocab_size=1000)

        run_mqar_probe(
            model, config, vocab_size=1000, n_associations=4, n_sequences=17, device="cpu"
        )

        assert model.batch_sizes == [8, 8, 1]


class _FakeLoader:
    """Minimal loader with next_batch() for CKA testing."""

    def __init__(self, batch_size: int, seq_len: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def next_batch(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        return x, y


class TestComputeCKA:
    """Tests for compute_cka function.

    Validates: Requirements 7.3, 7.4, 7.5
    """

    def _make_model_and_loader(self, n_layer=3, d_model=64, seq_len=32, vocab_size=100):
        config = ModelConfig(
            n_layer=n_layer,
            d_model=d_model,
            n_head=4,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dropout=0.0,
        )
        model = VanillaTransformer(config)
        model.eval()
        loader = _FakeLoader(batch_size=2, seq_len=seq_len, vocab_size=vocab_size)
        return model, loader, config

    def test_returns_cka_result(self):
        """compute_cka returns a CKAResult dataclass."""
        model, loader, config = self._make_model_and_loader()
        result = compute_cka(model, loader, n_batches=3, device="cpu")
        assert isinstance(result, CKAResult)

    def test_full_matrix_shape(self):
        """Requirement 7.2: full_matrix has shape (n_layer, n_layer)."""
        n_layer = 3
        model, loader, config = self._make_model_and_loader(n_layer=n_layer)
        result = compute_cka(model, loader, n_batches=3, device="cpu")
        assert result.full_matrix.shape == (n_layer, n_layer)

    def test_adjacent_curve_shape(self):
        """Requirement 7.2: adjacent_curve has shape (n_layer - 1,)."""
        n_layer = 4
        model, loader, config = self._make_model_and_loader(n_layer=n_layer)
        result = compute_cka(model, loader, n_batches=3, device="cpu")
        assert result.adjacent_curve.shape == (n_layer - 1,)

    def test_diagonal_equals_one(self):
        """Requirement 7.3: Diagonal entries of full_matrix are all 1.0."""
        model, loader, config = self._make_model_and_loader(n_layer=4)
        result = compute_cka(model, loader, n_batches=5, device="cpu")
        for i in range(config.n_layer):
            assert abs(result.full_matrix[i, i] - 1.0) < 1e-6, (
                f"Diagonal entry [{i},{i}] = {result.full_matrix[i, i]}, expected 1.0"
            )

    def test_symmetry(self):
        """Requirement 7.4: full_matrix is symmetric within floating-point tolerance."""
        model, loader, config = self._make_model_and_loader(n_layer=4)
        result = compute_cka(model, loader, n_batches=5, device="cpu")
        np.testing.assert_allclose(
            result.full_matrix,
            result.full_matrix.T,
            atol=1e-6,
            err_msg="CKA matrix is not symmetric",
        )

    def test_values_in_zero_one(self):
        """Requirement 7.5: All CKA values are in [0.0, 1.0]."""
        model, loader, config = self._make_model_and_loader(n_layer=4)
        result = compute_cka(model, loader, n_batches=5, device="cpu")
        assert np.all(result.full_matrix >= 0.0), (
            f"Found CKA values < 0: {result.full_matrix.min()}"
        )
        assert np.all(result.full_matrix <= 1.0), (
            f"Found CKA values > 1: {result.full_matrix.max()}"
        )

    def test_adjacent_curve_values_in_zero_one(self):
        """Requirement 7.5: Adjacent curve values are also in [0.0, 1.0]."""
        model, loader, config = self._make_model_and_loader(n_layer=4)
        result = compute_cka(model, loader, n_batches=5, device="cpu")
        assert np.all(result.adjacent_curve >= 0.0)
        assert np.all(result.adjacent_curve <= 1.0)

    def test_adjacent_curve_matches_full_matrix(self):
        """adjacent_curve[i] should equal full_matrix[i, i+1]."""
        n_layer = 4
        model, loader, config = self._make_model_and_loader(n_layer=n_layer)
        result = compute_cka(model, loader, n_batches=5, device="cpu")
        for i in range(n_layer - 1):
            np.testing.assert_allclose(
                result.adjacent_curve[i],
                result.full_matrix[i, i + 1],
                atol=1e-10,
                err_msg=f"adjacent_curve[{i}] != full_matrix[{i},{i+1}]",
            )
