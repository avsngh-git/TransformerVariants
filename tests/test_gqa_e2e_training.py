"""End-to-end training smoke test for GQA attention variant.

Validates:
- Task 5.1: End-to-end training works (loss decreases, no errors)
- Task 5.2: torch.compile compatibility (fullgraph=True, no graph breaks)
"""

import pytest
import torch

from src.models.registry import build


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash_attn")
class TestGQAEndToEndTraining:
    """Task 5.1: Verify end-to-end training works."""

    def test_training_loss_decreases(self):
        """Run training steps and verify the loss decreases."""
        model, config = build("gqa", "debug")
        model = model.cuda().to(torch.bfloat16)
        model.train()

        # Use a higher LR to ensure measurable loss decrease within few steps in bf16
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Use fixed random data so each step sees the same batch (memorization test)
        torch.manual_seed(42)
        idx = torch.randint(0, config.vocab_size, (4, 64), device="cuda")
        targets = torch.randint(0, config.vocab_size, (4, 64), device="cuda")

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            logits, loss, _ = model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify no NaN losses
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), (
            f"NaN loss detected during training: {losses}"
        )

        # Verify loss is finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
            f"Non-finite loss detected: {losses}"
        )

        # Verify loss decreased (final < initial)
        # With fixed data and 10 steps at lr=1e-3, the model should memorize
        assert losses[-1] < losses[0], (
            f"Training loss did not decrease: went from {losses[0]:.4f} to {losses[-1]:.4f}. "
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash_attn")
class TestGQATorchCompile:
    """Task 5.2: Verify torch.compile compatibility."""

    def test_compile_fullgraph_training(self):
        """Wrap GQA model with torch.compile(fullgraph=True) and run training steps."""
        model, config = build("gqa", "debug")
        model = model.cuda().to(torch.bfloat16)
        model.train()

        compiled_model = torch.compile(model, fullgraph=True)
        optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-4)

        losses = []
        for step in range(3):
            idx = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
            targets = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

            optimizer.zero_grad()
            logits, loss, _ = compiled_model(idx, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify no NaN losses under compilation
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), (
            f"NaN loss under torch.compile: {losses}"
        )

        # Verify all steps completed without graph breaks (implicit — fullgraph=True
        # raises an error if there are any graph breaks)
        assert len(losses) == 3, "Not all training steps completed"

    def test_compiled_vs_eager_output_similarity(self):
        """Compare compiled model output to eager mode for correctness."""
        torch.manual_seed(42)

        model, config = build("gqa", "debug")
        model = model.cuda().to(torch.bfloat16)
        model.eval()

        # Get eager output
        idx = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
        with torch.no_grad():
            eager_logits, _, _ = model(idx)

        # Get compiled output (same model, same weights)
        compiled_model = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled_logits, _, _ = compiled_model(idx)

        # Outputs should be very close (torch.compile may reorder ops causing
        # small numerical differences in bf16 due to kernel fusion)
        max_diff = (eager_logits - compiled_logits).abs().max().item()
        assert torch.allclose(eager_logits, compiled_logits, atol=0.05, rtol=0.01), (
            f"Compiled output diverges too much from eager. "
            f"Max diff: {max_diff}"
        )
