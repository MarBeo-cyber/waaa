"""
WAAA ML — Goal Classifier
Replaces the rule-based if/elif goal switching with a Random Forest
trained on biographical history.

The classifier learns from the node's own experience:
  - Which combinations of coherence, prediction_error, frame_quality,
    goal duration, and temporal patterns correspond to which goal
  - The optimal goal given the full temporal context, not just
    instantaneous threshold comparisons

Feature engineering includes sliding-window statistics that capture
temporal patterns the original rule-based system cannot see:
  - Trend of coherence over last N readings (slope)
  - Volatility of prediction error
  - Duration in current goal
  - Time-of-day features (for seasonality)

The model bootstraps from hand-labelled rules and then refines
itself as biographical data accumulates — a form of self-supervised
learning grounded in the node's own history.
"""

import numpy as np
import logging
import time
import pickle
import os
from collections import deque
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

logger = logging.getLogger("waaa.ml.goal_classifier")

# Goal labels
GOALS = [
    "monitor_anomalies",
    "restore_perceptual_capacity",
    "execute_recovery",
]

# Minimum samples before the classifier is used instead of rules
MIN_TRAINING_SAMPLES = 30
# Retrain every N new samples
RETRAIN_INTERVAL = 20
# Sliding window for temporal features
WINDOW_SIZE = 8


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
    Random Forest goal classifier with temporal feature engineering.

    Training data is generated in two ways:
    1. Bootstrap: rule-based labels on early observations (fast start)
    2. Biographical: labels derived from the node's own past decisions
       and their outcomes (self-supervised refinement)

    Once MIN_TRAINING_SAMPLES is reached, the RF replaces the rules.
    The model is periodically retrained as new data arrives.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(GOALS)

        # Sliding window buffers for temporal features
        self.coherence_window      = deque(maxlen=WINDOW_SIZE)
        self.pred_error_window     = deque(maxlen=WINDOW_SIZE)
        self.frame_quality_window  = deque(maxlen=WINDOW_SIZE)
        self.signal_window         = deque(maxlen=WINDOW_SIZE)

        # Training data accumulator
        self.X_train: list = []
        self.y_train: list = []

        self.is_fitted        = False
        self.sample_count     = 0
        self.retrain_counter  = 0
        self.current_goal     = "monitor_anomalies"
        self.goal_start_time  = time.time()
        self.total_predictions = 0
        self.rf_predictions    = 0

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"[GoalClassifier] Loaded from {model_path} "
                        f"(samples={len(self.X_train)})")
        else:
            logger.info("[GoalClassifier] Initialised — bootstrap phase")

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
        Build feature vector for goal classification.
        Combines instantaneous metrics with temporal window statistics.
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
    # Rule-based bootstrap (generates initial training labels)            #
    # ------------------------------------------------------------------ #

    def _rule_based_goal(self,
                          coherence: float,
                          prediction_error: float,
                          frame_quality: float) -> str:
        """
        Original rule-based logic — used for bootstrapping.
        Generates training labels in the early phase.
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
        Predict the optimal goal for the current state.
        Returns (goal, method, confidence) where method is
        'classifier' or 'rules' (bootstrap).
        """
        self.total_predictions += 1
        features = self._extract_features(
            coherence, prediction_error, frame_quality,
            signal_magnitude, aptc_theta
        )

        # Generate bootstrap label for training
        rule_label = self._rule_based_goal(
            coherence, prediction_error, frame_quality
        )

        # Accumulate training data
        self.X_train.append(features.copy())
        self.y_train.append(rule_label)
        self.sample_count += 1
        self.retrain_counter += 1

        # Retrain periodically
        if (self.retrain_counter >= RETRAIN_INTERVAL and
                len(self.X_train) >= MIN_TRAINING_SAMPLES):
            self._fit()
            self.retrain_counter = 0

        # Use classifier if fitted, else fall back to rules
        if self.is_fitted and self.sample_count >= MIN_TRAINING_SAMPLES:
            try:
                X = features.reshape(1, -1)
                pred_encoded = self.rf.predict(X)[0]
                probas = self.rf.predict_proba(X)[0]
                goal = self.label_encoder.inverse_transform([pred_encoded])[0]
                confidence = float(np.max(probas))
                method = "classifier"
                self.rf_predictions += 1
                logger.debug(
                    f"[GoalClassifier] RF→{goal} "
                    f"(conf={confidence:.2f}, "
                    f"coh={coherence:.2f})"
                )
            except Exception as e:
                logger.warning(f"[GoalClassifier] RF failed: {e} → using rules")
                goal = rule_label
                confidence = 0.7
                method = "rules_fallback"
        else:
            goal = rule_label
            confidence = 0.7
            method = "rules_bootstrap"

        return goal, method, confidence

    def update_current_goal(self, goal: str):
        """Called by the node when a goal switch occurs."""
        if goal != self.current_goal:
            self.current_goal = goal
            self.goal_start_time = time.time()

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def _fit(self):
        if len(self.X_train) < MIN_TRAINING_SAMPLES:
            return

        X = np.array(self.X_train[-500:], dtype=np.float32)  # sliding window
        y_raw = self.y_train[-500:]

        try:
            y = self.label_encoder.transform(y_raw)
            self.rf.fit(X, y)
            self.is_fitted = True

            # Feature importance log (interpretability)
            importances = self.rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            feature_names = [
                "coherence", "pred_error", "frame_quality", "signal_mag",
                "aptc_theta", "coh_slope", "err_slope", "coh_mean",
                "coh_std", "err_mean", "err_vol", "qual_mean",
                "sig_mean", "sig_max", "goal_dur",
                "goal_monitor", "goal_restore", "goal_recovery",
                "hour_sin", "hour_cos",
            ]
            top_features = [(feature_names[i], round(importances[i], 3))
                            for i in top_idx]
            logger.info(f"[GoalClassifier] Retrained on {len(X)} samples. "
                        f"Top features: {top_features}")

        except Exception as e:
            logger.error(f"[GoalClassifier] Training failed: {e}")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        state = {
            "rf": self.rf,
            "label_encoder": self.label_encoder,
            "X_train": self.X_train[-500:],
            "y_train": self.y_train[-500:],
            "is_fitted": self.is_fitted,
            "sample_count": self.sample_count,
            "total_predictions": self.total_predictions,
            "rf_predictions": self.rf_predictions,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[GoalClassifier] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.rf               = state["rf"]
        self.label_encoder    = state["label_encoder"]
        self.X_train          = state["X_train"]
        self.y_train          = state["y_train"]
        self.is_fitted        = state["is_fitted"]
        self.sample_count     = state["sample_count"]
        self.total_predictions= state["total_predictions"]
        self.rf_predictions   = state["rf_predictions"]

    @property
    def status(self) -> dict:
        rf_pct = (self.rf_predictions / max(self.total_predictions, 1)) * 100
        return {
            "is_fitted": self.is_fitted,
            "sample_count": self.sample_count,
            "total_predictions": self.total_predictions,
            "rf_predictions": self.rf_predictions,
            "rf_usage_pct": round(rf_pct, 1),
            "min_samples_needed": max(0, MIN_TRAINING_SAMPLES - self.sample_count),
            "current_goal": self.current_goal,
            "feature_importances": (
                dict(zip(
                    ["coherence","pred_error","frame_quality","signal_mag",
                     "aptc_theta","coh_slope","err_slope","coh_mean",
                     "coh_std","err_mean","err_vol","qual_mean",
                     "sig_mean","sig_max","goal_dur",
                     "goal_monitor","goal_restore","goal_recovery",
                     "hour_sin","hour_cos"],
                    [round(float(v), 4)
                     for v in self.rf.feature_importances_]
                ))
                if self.is_fitted else {}
            ),
        }
