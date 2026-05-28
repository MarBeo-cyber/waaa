"""
WAAA ML — Anomaly Detection for Recovery Level Assessment
Replaces the rule-based recovery threshold logic with an
Isolation Forest that learns the distribution of healthy node states.

The Isolation Forest learns what "normal" looks like for this specific
node — not a generic threshold, but the actual manifold of healthy
operation. Recovery level is then determined by how far the current
state lies from that manifold.

Anomaly score → Recovery level mapping:
  score < 0.3  → NONE   (healthy)
  score < 0.5  → L0     (mild — in-operation self-repair)
  score < 0.7  → L1     (moderate — rollback to snapshot)
  score < 0.85 → L2     (severe — rollback to intermediate)
  score ≥ 0.85 → L3     (critical — full reset)
"""

import numpy as np
import logging
import pickle
import os
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("waaa.ml.recovery_detector")

CALIBRATION_SAMPLES = 25
RETRAIN_INTERVAL    = 40


class RecoveryLevelDetector:
    """
    Isolation Forest anomaly detector for recovery level assessment.

    The detector learns the multivariate distribution of the node's
    healthy operating state. It monitors:
      - coherence trajectory
      - prediction error trajectory
      - frame quality
      - APTC theta stability
      - blind streak
      - goal duration

    Once fitted, it produces a continuous anomaly score in [0,1]
    that maps to discrete recovery levels.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,   # expect ~10% anomalous states
            random_state=42,
            warm_start=False,
        )

        self.is_fitted       = False
        self.sample_count    = 0
        self.retrain_counter = 0
        self.buffer: list    = []
        self.score_history: list = []

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"[RecoveryDetector] Loaded from {model_path}")
        else:
            logger.info("[RecoveryDetector] Initialised — calibrating healthy baseline")

    def _extract_features(self,
                           coherence: float,
                           prediction_error: float,
                           frame_quality: float,
                           aptc_theta: float,
                           blind_streak: int,
                           goal_duration_s: float) -> np.ndarray:
        return np.array([
            coherence,
            prediction_error,
            frame_quality,
            1.0 - coherence,               # degradation proxy
            prediction_error * (1 - coherence),  # joint stress indicator
            aptc_theta,
            min(blind_streak / 5.0, 1.0),  # normalised blind streak
            min(goal_duration_s / 60.0, 1.0),  # normalised goal duration
        ], dtype=np.float32)

    def assess(self,
               coherence: float,
               prediction_error: float,
               frame_quality: float,
               aptc_theta: float,
               blind_streak: int = 0,
               goal_duration_s: float = 0.0) -> tuple[int, float]:
        """
        Assess the required recovery level.
        Returns (recovery_level: int, anomaly_score: float).
        recovery_level: -1=healthy, 0=L0, 1=L1, 2=L2, 3=L3
        """
        features = self._extract_features(
            coherence, prediction_error, frame_quality,
            aptc_theta, blind_streak, goal_duration_s
        )

        # Only train the healthy-state manifold on genuinely healthy samples
        # (coherence > 0.5 and prediction_error < 0.5 and frame_quality > 0.4)
        # This prevents the IsoForest from learning anomalous states as "normal"
        is_healthy_sample = (
            coherence > 0.50 and
            prediction_error < 0.50 and
            frame_quality > 0.40
        )
        if is_healthy_sample:
            self.buffer.append(features)

        self.sample_count += 1
        self.retrain_counter += 1

        if len(self.buffer) > 300:
            self.buffer.pop(0)

        # Retrain periodically
        if (self.retrain_counter >= RETRAIN_INTERVAL and
                len(self.buffer) >= CALIBRATION_SAMPLES):
            self._fit()
            self.retrain_counter = 0

        # During calibration: use simple rule-based fallback
        if not self.is_fitted or len(self.buffer) < CALIBRATION_SAMPLES:
            return self._rule_fallback(coherence, prediction_error, frame_quality)

        # Isolation Forest scoring
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            # score_samples returns negative values: more negative = more anomalous
            raw_score = float(self.iso_forest.score_samples(X)[0])
            # Normalise to [0,1]: typical range is roughly [-0.5, 0.5]
            anomaly_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"[RecoveryDetector] Scoring failed: {e}")
            return self._rule_fallback(coherence, prediction_error, frame_quality)

        self.score_history.append(anomaly_score)
        if len(self.score_history) > 100:
            self.score_history.pop(0)

        # Map score to recovery level
        level = self._score_to_level(anomaly_score)

        logger.debug(f"[RecoveryDetector] score={anomaly_score:.3f} → L{level}")
        return level, anomaly_score

    def _score_to_level(self, score: float) -> int:
        if score < 0.30:
            return -1  # healthy
        elif score < 0.50:
            return 0   # L0
        elif score < 0.70:
            return 1   # L1
        elif score < 0.85:
            return 2   # L2
        else:
            return 3   # L3

    def _rule_fallback(self, coherence: float,
                        prediction_error: float,
                        frame_quality: float) -> tuple[int, float]:
        """Rule-based fallback during calibration phase."""
        if coherence < 0.10 or prediction_error > 0.85:
            return 3, 0.90
        elif coherence < 0.20 or prediction_error > 0.70:
            return 2, 0.75
        elif coherence < 0.35 or prediction_error > 0.55:
            return 1, 0.55
        elif coherence < 0.50 or frame_quality < 0.35:
            return 0, 0.35
        else:
            return -1, 0.10

    def _fit(self):
        if len(self.buffer) < CALIBRATION_SAMPLES:
            return
        X = np.array(self.buffer, dtype=np.float32)
        try:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.iso_forest.fit(X_scaled)
            self.is_fitted = True
            logger.info(f"[RecoveryDetector] Trained on {len(X)} samples")
        except Exception as e:
            logger.error(f"[RecoveryDetector] Training failed: {e}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "iso_forest": self.iso_forest,
                "is_fitted": self.is_fitted,
                "sample_count": self.sample_count,
                "score_history": self.score_history,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler       = state["scaler"]
        self.iso_forest   = state["iso_forest"]
        self.is_fitted    = state["is_fitted"]
        self.sample_count = state["sample_count"]
        self.score_history= state["score_history"]

    @property
    def status(self) -> dict:
        return {
            "is_fitted": self.is_fitted,
            "sample_count": self.sample_count,
            "calibration_needed": max(0, CALIBRATION_SAMPLES - len(self.buffer)),
            "mean_anomaly_score": round(float(np.mean(self.score_history)), 4)
                                  if self.score_history else None,
            "recent_max_score": round(float(np.max(self.score_history[-10:])), 4)
                                if len(self.score_history) >= 10 else None,
        }
