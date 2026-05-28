"""
WAAA ML — Autoencoder Scene Model
Replaces the rule-based SceneModel with a genuine anomaly detection model.

Architecture:
  Input  : flattened frame features (luminance, noise, motion + spatial stats)
  Encoder: MLP compression to latent space
  Decoder: MLP reconstruction
  Score  : MSE reconstruction error = anomaly score

The autoencoder learns the distribution of NORMAL frames during a
calibration phase. Any deviation from the learned manifold produces
a reconstruction error proportional to the anomaly's rarity —
not to its distance from a hand-coded average.

This is a genuine learned model: it does not know in advance what
"normal" looks like — it discovers it from data.
"""

import numpy as np
import logging
import time
import pickle
import os
from typing import Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

logger = logging.getLogger("waaa.ml.autoencoder")

# Feature vector dimension
FEATURE_DIM = 12


def extract_features(frame: np.ndarray,
                     luminance: float,
                     noise: float,
                     motion: float) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a raw frame + scalar metrics.

    Features:
      [0]   mean luminance (scalar)
      [1]   luminance std (spatial variance across frame)
      [2]   noise estimate (scalar)
      [3]   motion score (scalar)
      [4]   frame mean pixel value (normalised)
      [5]   frame std pixel value (normalised)
      [6]   frame min pixel value (normalised)
      [7]   frame max pixel value (normalised)
      [8]   luminance gradient (row-wise mean absolute difference)
      [9]   noise*luminance interaction
      [10]  frame entropy proxy (std of pixel histogram)
      [11]  edge density proxy (mean absolute laplacian approximation)
    """
    if frame is not None and frame.size > 0:
        frame_f = frame.astype(np.float32) / 255.0
        mean_px   = float(np.mean(frame_f))
        std_px    = float(np.std(frame_f))
        min_px    = float(np.min(frame_f))
        max_px    = float(np.max(frame_f))

        # Row-wise gradient (proxy for spatial luminance variation)
        if frame_f.ndim == 2 and frame_f.shape[0] > 1:
            grad = float(np.mean(np.abs(np.diff(frame_f, axis=0))))
        else:
            grad = 0.0

        # Entropy proxy: std of histogram bins
        hist, _ = np.histogram(frame_f.ravel(), bins=16, range=(0, 1))
        entropy_proxy = float(np.std(hist.astype(np.float32)))

        # Edge density: mean absolute difference between adjacent pixels
        if frame_f.ndim == 2 and frame_f.shape[1] > 1:
            edge_density = float(np.mean(np.abs(np.diff(frame_f, axis=1))))
        else:
            edge_density = 0.0

        lum_std = float(np.std(frame_f))
    else:
        mean_px = luminance
        std_px = noise
        min_px = max(0.0, luminance - noise)
        max_px = min(1.0, luminance + noise)
        grad = 0.0
        entropy_proxy = 0.0
        edge_density = 0.0
        lum_std = noise

    features = np.array([
        luminance,
        lum_std,
        noise,
        motion,
        mean_px,
        std_px,
        min_px,
        max_px,
        grad,
        noise * luminance,       # interaction term
        entropy_proxy / 100.0,   # normalise histogram std
        edge_density,
    ], dtype=np.float32)

    return features


class AutoencoderSceneModel:
    """
    MLP Autoencoder for frame-level anomaly detection.

    Lifecycle:
      1. CALIBRATING: accumulates normal frames, no predictions
      2. TRAINING:    fits the autoencoder on collected normal frames
      3. ACTIVE:      computes genuine reconstruction error per frame
      4. UPDATING:    periodically retrains on expanding history

    The reconstruction error is the anomaly score:
      low error  → frame matches learned normal distribution
      high error → genuine anomaly, unseen during calibration
    """

    # Calibration phase: collect this many normal frames before first fit
    CALIBRATION_FRAMES = 20
    # Retrain every N frames once active
    RETRAIN_INTERVAL   = 50
    # Max frames kept in training buffer (sliding window)
    MAX_BUFFER_SIZE    = 200
    # Anomaly threshold: frames with error > mean + k*std are anomalous
    ANOMALY_K          = 2.0

    def __init__(self, model_path: Optional[str] = None):
        self.scaler      = StandardScaler()
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(8, 4, 8),   # encoder: 12→8→4, decoder: 4→8→12
            activation="tanh",
            solver="adam",
            max_iter=500,
            random_state=42,
            warm_start=True,                # incremental retraining
            learning_rate_init=0.005,
        )

        self.phase            = "calibrating"
        self.frame_buffer     : list[np.ndarray] = []   # raw feature vectors
        self.frame_count      : int = 0
        self.retrain_counter  : int = 0
        self.error_history    : list[float] = []
        self.is_fitted        : bool = False

        # Dynamic anomaly threshold (learned from error distribution)
        self._error_mean      : float = 0.0
        self._error_std       : float = 0.1
        self._anomaly_threshold: float = 0.5   # initial conservative value

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"[Autoencoder] Loaded from {model_path}")
        else:
            logger.info("[Autoencoder] Initialised — calibration phase")

    # ------------------------------------------------------------------ #
    # Core update and prediction                                           #
    # ------------------------------------------------------------------ #

    def update(self,
               frame: np.ndarray,
               luminance: float,
               noise: float,
               motion: float) -> tuple[float, float, float]:
        """
        Process one frame. Returns (prediction_error, coherence, anomaly_score).

        During calibration: returns conservative defaults.
        Once active: returns genuine reconstruction error.
        """
        self.frame_count += 1
        features = extract_features(frame, luminance, noise, motion)

        # Accumulate in buffer
        self.frame_buffer.append(features)
        if len(self.frame_buffer) > self.MAX_BUFFER_SIZE:
            self.frame_buffer.pop(0)

        # Phase transitions
        if self.phase == "calibrating":
            if len(self.frame_buffer) >= self.CALIBRATION_FRAMES:
                self._fit()
                self.phase = "active"
                logger.info(f"[Autoencoder] Calibration complete "
                            f"({self.CALIBRATION_FRAMES} frames). Now active.")
            # Return conservative defaults during calibration
            calibration_progress = len(self.frame_buffer) / self.CALIBRATION_FRAMES
            return 0.15, calibration_progress * 0.6, 0.1

        # Active phase: compute reconstruction error
        error = self._reconstruction_error(features)
        self.error_history.append(error)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        # Update dynamic threshold
        if len(self.error_history) >= 10:
            self._error_mean = float(np.mean(self.error_history))
            self._error_std  = float(np.std(self.error_history)) + 1e-6
            self._anomaly_threshold = (
                self._error_mean + self.ANOMALY_K * self._error_std
            )

        # Normalise error to [0,1] using dynamic threshold
        normalised_error = min(1.0, error / max(self._anomaly_threshold, 1e-6))

        # Coherence: inverse of normalised error, weighted by fit quality
        coherence = max(0.0, 1.0 - normalised_error)

        # Anomaly score: how far above threshold
        anomaly_score = max(0.0, (error - self._error_mean) /
                           max(self._error_std * 3, 1e-6))
        anomaly_score = min(1.0, anomaly_score)

        # Periodic retraining (incremental — warm_start=True)
        self.retrain_counter += 1
        if self.retrain_counter >= self.RETRAIN_INTERVAL:
            self._fit(incremental=True)
            self.retrain_counter = 0

        logger.debug(f"[Autoencoder] error={error:.4f} "
                     f"threshold={self._anomaly_threshold:.4f} "
                     f"coherence={coherence:.3f} "
                     f"anomaly={anomaly_score:.3f}")

        return normalised_error, coherence, anomaly_score

    def _reconstruction_error(self, features: np.ndarray) -> float:
        """Compute MSE between input and reconstructed output."""
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            X_reconstructed = self.autoencoder.predict(X)
            mse = float(np.mean((X - X_reconstructed) ** 2))
            return mse
        except (NotFittedError, Exception) as e:
            logger.warning(f"[Autoencoder] Reconstruction error: {e}")
            return 0.1

    def _fit(self, incremental: bool = False):
        """Fit or refit the autoencoder on the current frame buffer."""
        if len(self.frame_buffer) < 5:
            return

        X = np.array(self.frame_buffer, dtype=np.float32)

        try:
            if not self.is_fitted or not incremental:
                self.scaler.fit(X)

            X_scaled = self.scaler.transform(X)
            self.autoencoder.fit(X_scaled, X_scaled)
            self.is_fitted = True

            # Compute baseline error distribution on training data
            X_reconstructed = self.autoencoder.predict(X_scaled)
            errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            self._error_mean = float(np.mean(errors))
            self._error_std  = float(np.std(errors)) + 1e-6
            self._anomaly_threshold = (
                self._error_mean + self.ANOMALY_K * self._error_std
            )

            logger.info(f"[Autoencoder] {'Retrained' if incremental else 'Trained'} "
                        f"on {len(X)} frames. "
                        f"Baseline error: {self._error_mean:.4f} "
                        f"± {self._error_std:.4f} "
                        f"threshold: {self._anomaly_threshold:.4f}")

        except Exception as e:
            logger.error(f"[Autoencoder] Training failed: {e}")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        state = {
            "scaler": self.scaler,
            "autoencoder": self.autoencoder,
            "phase": self.phase,
            "error_history": self.error_history,
            "error_mean": self._error_mean,
            "error_std": self._error_std,
            "anomaly_threshold": self._anomaly_threshold,
            "frame_count": self.frame_count,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[Autoencoder] Saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler              = state["scaler"]
        self.autoencoder         = state["autoencoder"]
        self.phase               = state["phase"]
        self.error_history       = state["error_history"]
        self._error_mean         = state["error_mean"]
        self._error_std          = state["error_std"]
        self._anomaly_threshold  = state["anomaly_threshold"]
        self.frame_count         = state["frame_count"]
        self.is_fitted           = state["is_fitted"]

    @property
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "frame_count": self.frame_count,
            "is_fitted": self.is_fitted,
            "error_mean": round(self._error_mean, 5),
            "error_std": round(self._error_std, 5),
            "anomaly_threshold": round(self._anomaly_threshold, 5),
            "buffer_size": len(self.frame_buffer),
            "error_history_len": len(self.error_history),
        }
