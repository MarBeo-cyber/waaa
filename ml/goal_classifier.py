"""
WAAA ML — Goal selection.

Goal switching is a **deterministic rule** over the node's perceptual
state — see ``_rule_based_goal``. It is not a learned model, and this
module does not claim to be one.

Why there is no classifier here any more
----------------------------------------
Earlier versions ran a RandomForest alongside the rule and handed over to
it after 30 samples. The training labels for that forest were produced by
``_rule_based_goal`` itself, so the forest could only ever learn to
imitate the rule it was said to replace — including the very
``coherence < 0.35`` comparison it was advertised as replacing. Training
a model on its own baseline's output teaches it nothing the baseline did
not already encode, so the forest has been removed rather than dressed up.

Training it on something better would need labels derived from *outcomes*
— whether the goal chosen at time t actually restored coherence. The node
records no such outcome signal today, so there is nothing honest to train
on. Adding one is the natural next step, and it is the reason the
temporal features below are kept.

Temporal feature engineering
----------------------------
``_extract_features`` computes sliding-window statistics that a
single-reading comparison cannot see: coherence trend (slope), prediction
error volatility, window means and peaks, time in the current goal, and
time-of-day terms. These are computed on every prediction and published
through ``status["last_features"]`` for analysis and for future
outcome-labelled training. They do **not** feed the goal decision today.
"""

import logging
import os
import pickle
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("waaa.ml.goal_classifier")

# Goal labels
GOALS = [
    "monitor_anomalies",
    "restore_perceptual_capacity",
    "execute_recovery",
]

# Sliding window for temporal features
WINDOW_SIZE = 8

FEATURE_NAMES = [
    "coherence", "pred_error", "frame_quality", "signal_mag",
    "aptc_theta", "coh_slope", "err_slope", "coh_mean",
    "coh_std", "err_mean", "err_vol", "qual_mean",
    "sig_mean", "sig_max", "goal_dur",
    "goal_monitor", "goal_restore", "goal_recovery",
    "hour_sin", "hour_cos",
]


def _slope(values: list) -> float:
    """Linear regression slope over a list of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    y = np.array(values, dtype=np.float32)
    if np.std(x) < 1e-9:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


class GoalClassifier:
    """
    Deterministic goal selector with temporal feature engineering.

    ``predict()`` applies ``_rule_based_goal`` to the current reading and
    reports ``method="rule"``. The temporal features are computed on the
    same call and kept in ``last_features`` for inspection; they are not
    part of the decision. See the module docstring for why.
    """

    def __init__(self, model_path: Optional[str] = None):
        # Sliding window buffers for temporal features
        self.coherence_window      = deque(maxlen=WINDOW_SIZE)
        self.pred_error_window     = deque(maxlen=WINDOW_SIZE)
        self.frame_quality_window  = deque(maxlen=WINDOW_SIZE)
        self.signal_window         = deque(maxlen=WINDOW_SIZE)

        self.last_features: Optional[np.ndarray] = None
        self.current_goal     = "monitor_anomalies"
        self.goal_start_time  = time.time()
        self.total_predictions = 0

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"[GoalSelector] State loaded from {model_path}")
        else:
            logger.info("[GoalSelector] Initialised — deterministic rule")

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_features(self,
                           coherence: float,
                           prediction_error: float,
                           frame_quality: float,
                           signal_magnitude: float,
                           aptc_theta: float) -> np.ndarray:
        """
        Build the temporal feature vector for the current reading.

        Combines instantaneous metrics with sliding-window statistics.
        Published through ``status["last_features"]``; not used by the
        goal decision (see module docstring).
        """
        # Update sliding windows
        self.coherence_window.append(coherence)
        self.pred_error_window.append(prediction_error)
        self.frame_quality_window.append(frame_quality)
        self.signal_window.append(signal_magnitude)

        cw = list(self.coherence_window)
        ew = list(self.pred_error_window)
        qw = list(self.frame_quality_window)
        sw = list(self.signal_window)

        # Temporal features
        coherence_slope    = _slope(cw)
        pred_error_slope   = _slope(ew)
        coherence_mean     = float(np.mean(cw))
        coherence_std      = float(np.std(cw)) if len(cw) > 1 else 0.0
        pred_error_mean    = float(np.mean(ew))
        pred_error_volatility = float(np.std(ew)) if len(ew) > 1 else 0.0
        frame_quality_mean = float(np.mean(qw))
        signal_mean        = float(np.mean(sw))
        signal_max         = float(np.max(sw))

        # Duration in current goal (normalised)
        goal_duration = min(1.0, (time.time() - self.goal_start_time) / 60.0)

        # Goal encoding (one-hot)
        goal_monitor  = float(self.current_goal == "monitor_anomalies")
        goal_restore  = float(self.current_goal == "restore_perceptual_capacity")
        goal_recovery = float(self.current_goal == "execute_recovery")

        # Time-of-day (proxy for seasonality)
        hour_sin = float(np.sin(2 * np.pi * time.localtime().tm_hour / 24))
        hour_cos = float(np.cos(2 * np.pi * time.localtime().tm_hour / 24))

        features = np.array([
            coherence,              # instantaneous
            prediction_error,       # instantaneous
            frame_quality,          # instantaneous
            signal_magnitude,       # instantaneous
            aptc_theta,             # current threshold
            coherence_slope,        # trend
            pred_error_slope,       # trend
            coherence_mean,         # window mean
            coherence_std,          # window volatility
            pred_error_mean,        # window mean
            pred_error_volatility,  # window volatility
            frame_quality_mean,     # window mean
            signal_mean,            # window mean
            signal_max,             # window max (peak anomaly)
            goal_duration,          # time in current goal
            goal_monitor,           # current goal encoding
            goal_restore,
            goal_recovery,
            hour_sin,               # seasonality
            hour_cos,
        ], dtype=np.float32)

        return features

    # ------------------------------------------------------------------ #
    # The goal decision                                                   #
    # ------------------------------------------------------------------ #

    def _rule_based_goal(self,
                          coherence: float,
                          prediction_error: float,
                          frame_quality: float) -> str:
        """
        The goal decision, in full. Deterministic, ordered by severity.
        """
        if coherence < 0.10 or prediction_error > 0.85:
            return "execute_recovery"
        elif coherence < 0.20 or frame_quality < 0.25:
            return "restore_perceptual_capacity"
        elif coherence < 0.35 or prediction_error > 0.60:
            return "restore_perceptual_capacity"
        else:
            return "monitor_anomalies"

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(self,
                coherence: float,
                prediction_error: float,
                frame_quality: float,
                signal_magnitude: float,
                aptc_theta: float) -> tuple[str, str, float]:
        """
        Select the goal for the current state.

        Returns ``(goal, method, confidence)``. ``method`` is always
        "rule". ``confidence`` is always 1.0: the decision is a
        deterministic function of the inputs, so there is no uncertainty
        estimate to report — the field is kept because callers unpack it.
        """
        self.total_predictions += 1
        self.last_features = self._extract_features(
            coherence, prediction_error, frame_quality,
            signal_magnitude, aptc_theta
        )

        goal = self._rule_based_goal(
            coherence, prediction_error, frame_quality
        )
        logger.debug(f"[GoalSelector] rule→{goal} (coh={coherence:.2f})")
        return goal, "rule", 1.0

    def update_current_goal(self, goal: str):
        """Called by the node when a goal switch occurs."""
        if goal != self.current_goal:
            self.current_goal = goal
            self.goal_start_time = time.time()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Persist the little state there is: which goal is active, since
        when, and how many decisions have been made. No model is stored —
        the decision rule is code, not data."""
        state = {
            "current_goal": self.current_goal,
            "goal_start_time": self.goal_start_time,
            "total_predictions": self.total_predictions,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[GoalSelector] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.current_goal      = state.get("current_goal", "monitor_anomalies")
        self.goal_start_time   = state.get("goal_start_time", time.time())
        self.total_predictions = state.get("total_predictions", 0)

    @property
    def status(self) -> dict:
        return {
            "method": "rule",
            "model": None,
            "total_predictions": self.total_predictions,
            "current_goal": self.current_goal,
            "goal_duration_s": round(time.time() - self.goal_start_time, 1),
            "last_features": (
                {name: round(float(v), 4)
                 for name, v in zip(FEATURE_NAMES, self.last_features,
                                    strict=True)}
                if self.last_features is not None else {}
            ),
        }
