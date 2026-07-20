"""Tests for the HealthMonitor anomaly detection system.

Tests cover:
- NaN loss triggers ROLLBACK
- Inf grad_norm triggers ROLLBACK
- Baseline warmup period (< 10 data points) returns CONTINUE
- Z-score spike returns SKIP_STEP
- 3 consecutive skips escalate to ROLLBACK
- Normal step resets consecutive skip counter
- reset() clears window and counter
- Zero std doesn't cause division by zero
"""

import pytest

from src.training.health_monitor import Action, HealthMonitor


@pytest.fixture
def monitor():
    """Create a HealthMonitor with default settings."""
    return HealthMonitor()


@pytest.fixture
def warmed_monitor():
    """Create a HealthMonitor with a populated window (past warmup).

    Uses slightly varied values so that standard deviation is non-zero,
    enabling z-score computation to detect spikes.
    """
    m = HealthMonitor(window_size=100, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
    # Feed 10 stable steps with small variance to get non-zero std
    for i in range(10):
        m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)
    return m


class TestActionEnum:
    """Tests for the Action enum values."""

    def test_action_values(self):
        """Action enum should have the expected string values."""
        assert Action.CONTINUE.value == "continue"
        assert Action.SKIP_STEP.value == "skip_step"
        assert Action.ROLLBACK.value == "rollback"

    def test_action_members(self):
        """Action enum should have exactly three members."""
        assert len(Action) == 3


class TestHealthMonitorNaNInf:
    """Tests for NaN/Inf detection → ROLLBACK."""

    def test_nan_loss_triggers_rollback(self, monitor):
        """NaN loss should immediately return ROLLBACK."""
        result = monitor.check(step=0, loss=float('nan'), grad_norm=1.0)
        assert result == Action.ROLLBACK

    def test_inf_loss_triggers_rollback(self, monitor):
        """Inf loss should immediately return ROLLBACK."""
        result = monitor.check(step=0, loss=float('inf'), grad_norm=1.0)
        assert result == Action.ROLLBACK

    def test_neg_inf_loss_triggers_rollback(self, monitor):
        """Negative Inf loss should immediately return ROLLBACK."""
        result = monitor.check(step=0, loss=float('-inf'), grad_norm=1.0)
        assert result == Action.ROLLBACK

    def test_nan_grad_norm_triggers_rollback(self, monitor):
        """NaN grad_norm should immediately return ROLLBACK."""
        result = monitor.check(step=0, loss=1.0, grad_norm=float('nan'))
        assert result == Action.ROLLBACK

    def test_inf_grad_norm_triggers_rollback(self, monitor):
        """Inf grad_norm should immediately return ROLLBACK."""
        result = monitor.check(step=0, loss=1.0, grad_norm=float('inf'))
        assert result == Action.ROLLBACK

    def test_nan_detected_regardless_of_window_state(self, warmed_monitor):
        """NaN should trigger ROLLBACK even with a populated window."""
        result = warmed_monitor.check(step=100, loss=float('nan'), grad_norm=0.5)
        assert result == Action.ROLLBACK

    def test_both_nan_triggers_rollback(self, monitor):
        """Both loss and grad_norm being NaN should return ROLLBACK."""
        result = monitor.check(step=0, loss=float('nan'), grad_norm=float('nan'))
        assert result == Action.ROLLBACK


class TestHealthMonitorWarmup:
    """Tests for warmup handling (< min_samples data points → CONTINUE)."""

    def test_first_step_returns_continue(self, monitor):
        """The very first step should return CONTINUE (warmup)."""
        result = monitor.check(step=0, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE

    def test_second_step_returns_continue(self, monitor):
        """The second step should also return CONTINUE (still warmup, need 2 before computing)."""
        monitor.check(step=0, loss=1.0, grad_norm=0.5)
        result = monitor.check(step=1, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE

    def test_third_step_remains_in_baseline_warmup(self, monitor):
        """Two observations are not a reliable anomaly baseline."""
        monitor.check(step=0, loss=1.0, grad_norm=0.5)
        monitor.check(step=1, loss=1.0, grad_norm=0.5)
        # Third step with normal values should still CONTINUE
        result = monitor.check(step=2, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE

    def test_healthy_main_scale_warmup_trajectory_does_not_trigger_recovery(self):
        """Normal early optimization drift must build a baseline, not look anomalous."""
        monitor = HealthMonitor()
        healthy_metrics = [
            (10.9537, 7.70),
            (10.9391, 7.84),
            (10.8874, 7.78),
            (10.8332, 7.05),
            (10.7474, 7.08),
            (10.6551, 6.25),
            (10.5720, 5.44),
            (10.5057, 4.74),
            (10.4246, 4.25),
            (10.3712, 3.82),
        ]

        actions = [
            monitor.check(step, loss=loss, grad_norm=grad_norm)
            for step, (loss, grad_norm) in enumerate(healthy_metrics)
        ]

        assert actions == [Action.CONTINUE] * len(healthy_metrics)

    def test_healthy_gqa_warmup_does_not_treat_falling_gradients_as_anomalies(self):
        """A fast but finite optimization improvement must not request rollback."""
        monitor = HealthMonitor()
        # Deterministic replay of main-scale GQA seed 42 from the failed matrix run.
        healthy_metrics = [
            (10.9540, 6.56),
            (10.9521, 6.75),
            (10.9484, 7.09),
            (10.9473, 6.50),
            (10.9316, 6.88),
            (10.9163, 6.62),
            (10.9097, 6.62),
            (10.8626, 6.31),
            (10.8269, 6.53),
            (10.7771, 6.69),
            (10.7349, 6.56),
            (10.6789, 6.59),
            (10.6263, 6.62),
            (10.5665, 6.28),
            (10.4907, 5.91),
            (10.4131, 4.97),
            (10.3308, 4.44),
            (10.2885, 4.03),
            (10.2529, 3.47),
            (10.2029, 3.20),
        ]

        actions = [
            monitor.check(step, loss=loss, grad_norm=grad_norm)
            for step, (loss, grad_norm) in enumerate(healthy_metrics)
        ]

        assert actions == [Action.CONTINUE] * len(healthy_metrics)


class TestHealthMonitorZScore:
    """Tests for z-score spike detection → SKIP_STEP."""

    def test_loss_spike_returns_skip_step(self, warmed_monitor):
        """A large loss spike should return SKIP_STEP."""
        # With window of loss=1.0, a spike of 1000.0 should exceed z-threshold
        result = warmed_monitor.check(step=100, loss=1000.0, grad_norm=0.5)
        assert result == Action.SKIP_STEP

    def test_grad_norm_spike_returns_skip_step(self, warmed_monitor):
        """A large grad_norm spike should return SKIP_STEP."""
        # With window of grad_norm=0.5, a spike of 500.0 should exceed z-threshold
        result = warmed_monitor.check(step=100, loss=1.0, grad_norm=500.0)
        assert result == Action.SKIP_STEP

    def test_both_spike_returns_skip_step(self, warmed_monitor):
        """Spikes in both loss and grad_norm should return SKIP_STEP."""
        result = warmed_monitor.check(step=100, loss=1000.0, grad_norm=500.0)
        assert result == Action.SKIP_STEP

    def test_normal_value_after_warmup_returns_continue(self, warmed_monitor):
        """A normal value after warmup should return CONTINUE."""
        result = warmed_monitor.check(step=100, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE

    def test_spike_does_not_pollute_window(self, warmed_monitor):
        """Anomalous values should NOT be added to the rolling window."""
        # Trigger a skip
        warmed_monitor.check(step=100, loss=1000.0, grad_norm=0.5)
        # Next normal step should still be fine (window unaffected)
        result = warmed_monitor.check(step=101, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE


class TestHealthMonitorConsecutiveSkips:
    """Tests for consecutive skip tracking and escalation."""

    def test_max_consecutive_skips_escalates_to_rollback(self):
        """After max_consecutive_skips, should escalate to ROLLBACK."""
        m = HealthMonitor(max_consecutive_skips=3, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        # Fill warmup with varied values (non-zero std)
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        # Trigger 3 consecutive skips → 3rd should be ROLLBACK
        assert m.check(10, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(11, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(12, loss=1000.0, grad_norm=0.5) == Action.ROLLBACK

    def test_consecutive_counter_resets_after_rollback(self):
        """After escalation to ROLLBACK, the counter should be reset."""
        m = HealthMonitor(max_consecutive_skips=3, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        # Trigger escalation
        m.check(10, loss=1000.0, grad_norm=0.5)  # SKIP_STEP (skip count = 1)
        m.check(11, loss=1000.0, grad_norm=0.5)  # SKIP_STEP (skip count = 2)
        m.check(12, loss=1000.0, grad_norm=0.5)  # ROLLBACK (resets to 0)

        # After rollback, next spike starts fresh
        assert m.check(13, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP

    def test_normal_step_resets_consecutive_skip_counter(self):
        """A normal step should reset the consecutive skip counter."""
        m = HealthMonitor(max_consecutive_skips=3, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        # Two skips
        assert m.check(10, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(11, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP

        # Normal step resets counter
        assert m.check(12, loss=1.05, grad_norm=0.55) == Action.CONTINUE

        # Now spikes start fresh — need 3 more to escalate
        assert m.check(13, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(14, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(15, loss=1000.0, grad_norm=0.5) == Action.ROLLBACK

    def test_max_consecutive_skips_configurable(self):
        """max_consecutive_skips parameter should control escalation threshold."""
        m = HealthMonitor(max_consecutive_skips=2, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        assert m.check(10, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        # With max_consecutive_skips=2, second skip escalates
        assert m.check(11, loss=1000.0, grad_norm=0.5) == Action.ROLLBACK


class TestHealthMonitorReset:
    """Tests for the reset() method."""

    def test_reset_clears_window(self, warmed_monitor):
        """reset() should clear the rolling window."""
        warmed_monitor.reset()
        # After reset, first step should be warmup again
        result = warmed_monitor.check(step=0, loss=1.0, grad_norm=0.5)
        assert result == Action.CONTINUE

    def test_reset_clears_consecutive_skip_counter(self):
        """reset() should clear the consecutive skip counter."""
        m = HealthMonitor(max_consecutive_skips=3, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        # Build up 2 consecutive skips
        m.check(10, loss=1000.0, grad_norm=0.5)
        m.check(11, loss=1000.0, grad_norm=0.5)

        # Reset
        m.reset()

        # After reset, refill window and verify counter is at 0
        for i in range(10):
            m.check(i, loss=1.0 + i * 0.01, grad_norm=0.5 + i * 0.01)

        # Should need full 3 skips again before rollback
        assert m.check(10, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(11, loss=1000.0, grad_norm=0.5) == Action.SKIP_STEP
        assert m.check(12, loss=1000.0, grad_norm=0.5) == Action.ROLLBACK

    def test_reset_returns_none(self, warmed_monitor):
        """reset() should return None."""
        result = warmed_monitor.reset()
        assert result is None


class TestHealthMonitorZeroStd:
    """Tests for zero standard deviation handling."""

    def test_constant_values_no_division_error(self):
        """Constant loss values (std=0) should not cause ZeroDivisionError."""
        m = HealthMonitor(window_size=10, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        # Fill window with identical values
        for i in range(5):
            result = m.check(i, loss=1.0, grad_norm=1.0)
            assert result == Action.CONTINUE

        # Another identical value should still CONTINUE (z-score = 0 when std = 0)
        result = m.check(5, loss=1.0, grad_norm=1.0)
        assert result == Action.CONTINUE

    def test_constant_loss_varying_grad_no_error(self):
        """Constant loss (std=0) with varying grad_norm should work correctly."""
        m = HealthMonitor(window_size=10, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        for i in range(5):
            m.check(i, loss=1.0, grad_norm=0.5 + i * 0.01)

        # Loss z=0 (std=0), grad should be within threshold
        result = m.check(5, loss=1.0, grad_norm=0.55)
        assert result == Action.CONTINUE

    def test_zero_std_with_different_value_still_passes(self):
        """When std=0, a different value should yield z=0 (no anomaly flagged)."""
        m = HealthMonitor(window_size=10, loss_z_threshold=5.0, grad_norm_z_threshold=5.0)
        # Fill with constant values
        for i in range(5):
            m.check(i, loss=1.0, grad_norm=1.0)

        # A different value when std=0: z-score formula gives 0 (not infinity)
        result = m.check(5, loss=2.0, grad_norm=1.0)
        assert result == Action.CONTINUE


class TestHealthMonitorInit:
    """Tests for __init__ configuration."""

    def test_default_parameters(self):
        """Default parameters should match spec."""
        m = HealthMonitor()
        assert m.window_size == 100
        assert m.grad_norm_z_threshold == 5.0
        assert m.loss_z_threshold == 5.0
        assert m.max_consecutive_skips == 3
        assert m.min_samples == 10

    def test_custom_parameters(self):
        """Custom parameters should be stored correctly."""
        m = HealthMonitor(
            window_size=50,
            grad_norm_z_threshold=3.0,
            loss_z_threshold=4.0,
            max_consecutive_skips=5,
            min_samples=20,
        )
        assert m.window_size == 50
        assert m.grad_norm_z_threshold == 3.0
        assert m.loss_z_threshold == 4.0
        assert m.max_consecutive_skips == 5
        assert m.min_samples == 20

    def test_window_uses_deque_with_maxlen(self):
        """Internal windows should be bounded deques."""
        m = HealthMonitor(window_size=10)
        assert m._loss_window.maxlen == 10
        assert m._grad_norm_window.maxlen == 10
