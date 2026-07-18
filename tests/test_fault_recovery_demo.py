"""End-to-end recruiter-facing fault-recovery demonstration test."""

from scripts.demonstrate_fault_recovery import run_demo


def test_corruption_rollback_resume_matches_uninterrupted_training(tmp_path) -> None:
    result = run_demo(tmp_path)

    assert result["corruption_detected"] is True
    assert result["recovered_step"] == 1
    assert result["equivalent"] is True
    assert result["absolute_loss_difference"] < 1e-8
    assert result["parameter_max_absolute_difference"] < 1e-8
