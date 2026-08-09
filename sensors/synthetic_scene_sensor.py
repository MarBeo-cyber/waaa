"""
WAAA ML — Synthetic scene sensor with Autoencoder Scene Model.

**No camera is read anywhere in this module.** Frames are generated
in-process by ``_synthesise_frame``: a constant luminance plane plus
Gaussian noise, with the luminance and noise levels chosen by the current
scene state. There is no cv2 import in this repository and no hardware
capture path. The sensor exists so the cognitive loop and the models can
be exercised deterministically; swapping in a real capture device means
replacing ``_synthesise_frame``.

What *is* real here: the frame statistics (luminance, noise level, frame
quality) are measured from the generated frame, and the autoencoder that
scores it is a genuine model fitted on those frames.
"""

import numpy as np
import logging
import time
import random
from dataclasses import dataclass
from typing import Optional

from ml.autoencoder_scene import AutoencoderSceneModel

logger = logging.getLogger("waaa.ml.sensor")


@dataclass
class SensorReading:
    timestamp: float
    signal_magnitude: float
    coherence: float
    prediction_error: float
    frame_quality: float
    luminance: float
    noise_level: float
    motion_score: float
    anomaly_score: float = 0.0
    raw_frame: Optional[np.ndarray] = None
    # "active"      — coherence/prediction_error/anomaly_score come from the
    #                 fitted autoencoder
    # "warming_up"  — the autoencoder has no fitted network yet, so those
    #                 three fields are direct frame statistics instead
    scene_model_status: str = "active"

    @property
    def is_perceptually_degraded(self) -> bool:
        return self.frame_quality < 0.35 or self.noise_level > 0.60

    @property
    def summary(self) -> str:
        return (f"mag={self.signal_magnitude:.2f} "
                f"coh={self.coherence:.2f} "
                f"err={self.prediction_error:.2f} "
                f"qual={self.frame_quality:.2f} "
                f"anomaly={self.anomaly_score:.2f} "
                f"[{self.scene_model_status}]")


class SyntheticSceneSensor:
    """
    Synthetic scene sensor backed by an MLP Autoencoder.

    The autoencoder learns the distribution of frames it sees during the
    first CALIBRATION_FRAMES cycles. After that, the reconstruction error
    replaces hand-coded prediction error formulas. Until then the sensor
    reports directly measured frame statistics and marks the reading
    ``scene_model_status="warming_up"``.

    Scene states: NORMAL | DIM | NOISY | RECOVERED
    """

    SCENE_STATES = ["NORMAL", "DIM", "NOISY", "RECOVERED"]

    def __init__(self, node_id: str,
                 width: int = 64, height: int = 48,
                 model_path: Optional[str] = None):
        self.node_id = node_id
        self.width = width
        self.height = height
        self.current_scene_state = "NORMAL"
        self._frame_count = 0
        self._noise_filter = 0.5

        # ML scene model
        self.scene_model = AutoencoderSceneModel(model_path=model_path)

        logger.info(f"[SyntheticSensor:{node_id}] Initialised ({width}x{height}) "
                    f"with AutoencoderSceneModel — frames are synthetic, "
                    f"no camera is read")

    def set_scene_state(self, state: str):
        if state not in self.SCENE_STATES:
            raise ValueError(f"Unknown state: {state}. Valid: {self.SCENE_STATES}")
        prev = self.current_scene_state
        self.current_scene_state = state
        logger.info(f"[MLSensor:{self.node_id}] Scene: {prev} → {state}")

    def _synthesise_frame(self) -> tuple:
        state = self.current_scene_state
        rng = random.Random(self._frame_count)

        if state == "NORMAL":
            luminance  = 0.75 + rng.uniform(-0.05, 0.05)
            noise_base = 0.08 + rng.uniform(0, 0.04)
            motion     = rng.uniform(0, 0.15)
        elif state == "DIM":
            luminance  = 0.35 + rng.uniform(-0.08, 0.08)
            noise_base = 0.25 + rng.uniform(0, 0.10)
            motion     = rng.uniform(0, 0.10)
        elif state == "NOISY":
            luminance  = 0.20 + rng.uniform(-0.10, 0.10)
            noise_base = 0.65 + rng.uniform(0, 0.20)
            motion     = rng.uniform(0, 0.08)
        elif state == "RECOVERED":
            luminance  = 0.70 + rng.uniform(-0.05, 0.05)
            noise_base = 0.12 + rng.uniform(0, 0.05)
            motion     = rng.uniform(0, 0.12)
        else:
            luminance, noise_base, motion = 0.5, 0.1, 0.0

        effective_noise = max(0.0, noise_base - self._noise_filter * 0.2)

        base  = np.full((self.height, self.width),
                        luminance * 255, dtype=np.float32)
        noise = np.random.normal(0, effective_noise * 80,
                                 (self.height, self.width)).astype(np.float32)
        frame = np.clip(base + noise, 0, 255).astype(np.uint8)
        return frame, luminance, effective_noise, motion

    def read(self) -> SensorReading:
        self._frame_count += 1
        frame, luminance, noise, motion = self._synthesise_frame()

        # Frame quality: independent of scene model (absolute metric)
        frame_quality = max(0.0, min(1.0,
            0.5 * luminance + 0.5 * (1.0 - noise)
        ))

        # ML: autoencoder-based prediction error, coherence, anomaly score.
        # update() returns None while the autoencoder is still collecting
        # calibration frames.
        assessment = self.scene_model.update(frame, luminance, noise, motion)
        if assessment is None:
            # No trained model yet, so there is no reconstruction error to
            # report. Fall back to statistics measured directly from this
            # frame and label the reading, so nothing downstream can mistake
            # these for autoencoder output.
            scene_model_status = "warming_up"
            prediction_error = min(1.0, noise)
            coherence        = max(0.0, 1.0 - noise)
            anomaly_score    = min(1.0, max(0.0, (0.5 - frame_quality) * 2.0))
        else:
            scene_model_status = "active"
            prediction_error, coherence, anomaly_score = assessment

        # Signal magnitude combines autoencoder anomaly score with
        # absolute frame quality degradation
        quality_anomaly = max(0.0, 0.5 - frame_quality)  # absolute degradation
        signal_magnitude = min(1.0,
            0.6 * anomaly_score + 0.4 * quality_anomaly
        )

        reading = SensorReading(
            timestamp=time.time(),
            signal_magnitude=round(signal_magnitude, 4),
            coherence=round(coherence, 4),
            prediction_error=round(prediction_error, 4),
            frame_quality=round(frame_quality, 4),
            luminance=round(luminance, 4),
            noise_level=round(noise, 4),
            motion_score=round(motion, 4),
            anomaly_score=round(anomaly_score, 4),
            raw_frame=frame,
            scene_model_status=scene_model_status,
        )

        logger.debug(f"[SyntheticSensor:{self.node_id}] {reading.summary}")
        return reading

    def adjust_noise_filter(self, value: float):
        self._noise_filter = max(0.0, min(1.0, value))

    def save_model(self, path: str):
        self.scene_model.save(path)

    @property
    def status(self) -> dict:
        return {
            "scene_state": self.current_scene_state,
            "frame_count": self._frame_count,
            "frames_are_synthetic": True,
            "autoencoder": self.scene_model.status,
        }
