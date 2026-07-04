"""Health monitor with z-score anomaly detection for training metrics.

Provides rolling-window statistical analysis of loss and gradient norm values
to detect numerical instabilities and recommend recovery actions.
"""

import math
import statistics
from collections import deque
from enum import Enum


class Action(Enum):
    """Recovery action recommended by the health monitor."""

    CONTINUE = "continue"
    SKIP_STEP = "skip_step"
    ROLLBACK = "rollback"


class HealthMonitor:
    """Anomaly detector for training metrics using rolling z-score analysis.

    Maintains a rolling window of recent loss and gradient norm values,
    computes z-scores for incoming values, and recommends recovery actions
    when anomalies are detected.

    Parameters
    ----------
    window_size : int
        Number of recent steps to keep in the rolling window (default 100).
    grad_norm_z_threshold : float
        Z-score threshold above which a gradient norm is considered anomalous (default 5.0).
    loss_z_threshold : float
        Z-score threshold above which a loss value is considered anomalous (default 5.0).
    max_consecutive_skips : int
        Number of consecutive SKIP_STEP actions before escalating to ROLLBACK (default 3).
    """

    def __init__(
        self,
        window_size: int = 100,
        grad_norm_z_threshold: float = 5.0,
        loss_z_threshold: float = 5.0,
        max_consecutive_skips: int = 3,
    ):
        self.window_size = window_size
        self.grad_norm_z_threshold = grad_norm_z_threshold
        self.loss_z_threshold = loss_z_threshold
        self.max_consecutive_skips = max_consecutive_skips

        self._loss_window: deque[float] = deque(maxlen=window_size)
        self._grad_norm_window: deque[float] = deque(maxlen=window_size)
        self._consecutive_skips: int = 0

    def check(self, step: int, loss: float, grad_norm: float) -> Action:
        """Analyze a training step's metrics and return a recovery action.

        Parameters
        ----------
        step : int
            Current training step number.
        loss : float
            Loss value for this step (may be NaN/Inf).
        grad_norm : float
            Gradient norm for this step (may be NaN/Inf).

        Returns
        -------
        Action
            CONTINUE if metrics are normal, SKIP_STEP if anomalous,
            or ROLLBACK if NaN/Inf detected or too many consecutive skips.
        """
        # NaN/Inf check — always rollback
        if math.isnan(loss) or math.isinf(loss) or math.isnan(grad_norm) or math.isinf(grad_norm):
            return Action.ROLLBACK

        # Insufficient data — can't compute z-score yet
        if len(self._loss_window) < 2:
            self._loss_window.append(loss)
            self._grad_norm_window.append(grad_norm)
            return Action.CONTINUE

        # Compute z-scores (handle zero std gracefully)
        loss_mean = statistics.mean(self._loss_window)
        loss_std = statistics.pstdev(self._loss_window)
        grad_mean = statistics.mean(self._grad_norm_window)
        grad_std = statistics.pstdev(self._grad_norm_window)

        loss_z = abs(loss - loss_mean) / loss_std if loss_std > 0 else 0.0
        grad_z = abs(grad_norm - grad_mean) / grad_std if grad_std > 0 else 0.0

        is_anomaly = (loss_z > self.loss_z_threshold) or (grad_z > self.grad_norm_z_threshold)

        if is_anomaly:
            self._consecutive_skips += 1
            if self._consecutive_skips >= self.max_consecutive_skips:
                self._consecutive_skips = 0
                return Action.ROLLBACK
            return Action.SKIP_STEP

        # Normal step — update window, reset skip counter
        self._consecutive_skips = 0
        self._loss_window.append(loss)
        self._grad_norm_window.append(grad_norm)
        return Action.CONTINUE

    def reset(self) -> None:
        """Clear the rolling window and reset the consecutive skip counter."""
        self._loss_window.clear()
        self._grad_norm_window.clear()
        self._consecutive_skips = 0
