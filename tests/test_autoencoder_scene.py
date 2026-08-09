"""Regression tests for the autoencoder's warm-up behaviour.

The bug these guard against: for the first 20 frames update() returned the
hard-coded constants 0.15 (prediction error) and 0.1 (anomaly score). The
demo's first phase is 15 cycles, so it printed those constants formatted
as measurements for its entire duration. _reconstruction_error also caught
``(NotFittedError, Exception)`` — i.e. everything — and returned a
plausible-looking 0.1 for any failure whatsoever.
"""

import inspect

import numpy as np
import pytest

from ml.autoencoder_scene import AutoencoderSceneModel, extract_features


def _frame(luminance: float = 0.75, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((16, 16), luminance * 255, dtype=np.float32)
    return np.clip(base + rng.normal(0, 4, (16, 16)), 0, 255).astype(np.uint8)


def test_update_reports_nothing_while_warming_up():
    model = AutoencoderSceneModel()
    for i in range(model.CALIBRATION_FRAMES - 1):
        assert model.update(_frame(seed=i), 0.75, 0.08, 0.05) is None
        assert model.is_warming_up is True


def test_no_hard_coded_constants_during_calibration():
    """The old code returned 0.15/0.1 for every warm-up frame."""
    model = AutoencoderSceneModel()
    results = [model.update(_frame(seed=i), 0.75, 0.08, 0.05)
               for i in range(model.CALIBRATION_FRAMES - 1)]
    assert all(r is None for r in results)
    assert 0.15 not in [r for r in results if r is not None]


def test_model_becomes_active_and_reports_real_numbers():
    model = AutoencoderSceneModel()
    result = None
    for i in range(model.CALIBRATION_FRAMES):
        result = model.update(_frame(seed=i), 0.75, 0.08, 0.05)

    assert result is not None, "must report on the frame that completes calibration"
    prediction_error, coherence, anomaly = result
    assert model.is_fitted is True
    assert model.is_warming_up is False
    assert 0.0 <= coherence <= 1.0
    assert 0.0 <= anomaly <= 1.0
    assert prediction_error >= 0.0


def test_reconstruction_error_varies_with_the_input():
    """A real measurement moves when the input moves; a constant does not."""
    model = AutoencoderSceneModel()
    for i in range(model.CALIBRATION_FRAMES):
        model.update(_frame(luminance=0.75, seed=i), 0.75, 0.08, 0.05)

    normal = model.update(_frame(luminance=0.75, seed=99), 0.75, 0.08, 0.05)
    odd = model.update(_frame(luminance=0.10, seed=99), 0.10, 0.90, 0.05)
    assert normal[0] != odd[0]


def test_reconstruction_error_returns_none_when_unfitted():
    model = AutoencoderSceneModel()
    features = extract_features(_frame(), 0.75, 0.08, 0.05)
    assert model._reconstruction_error(features) is None


def test_only_notfitted_is_caught():
    """Any other failure must propagate instead of becoming a 0.1."""
    source = inspect.getsource(AutoencoderSceneModel._reconstruction_error)
    assert "except NotFittedError" in source
    assert "Exception" not in source.split("except")[-1]

    model = AutoencoderSceneModel()
    for i in range(model.CALIBRATION_FRAMES):
        model.update(_frame(seed=i), 0.75, 0.08, 0.05)

    def boom(_):
        raise ValueError("scaler exploded")

    model.scaler.transform = boom
    with pytest.raises(ValueError):
        model._reconstruction_error(extract_features(_frame(), 0.75, 0.08, 0.05))


def test_status_exposes_the_warming_up_state():
    model = AutoencoderSceneModel()
    model.update(_frame(), 0.75, 0.08, 0.05)
    status = model.status
    assert status["is_warming_up"] is True
    assert status["calibration_frames_remaining"] == model.CALIBRATION_FRAMES - 1


def test_extract_features_dimension():
    from ml.autoencoder_scene import FEATURE_DIM
    assert extract_features(_frame(), 0.75, 0.08, 0.05).shape == (FEATURE_DIM,)


def test_save_and_load_round_trip(tmp_path):
    model = AutoencoderSceneModel()
    for i in range(model.CALIBRATION_FRAMES):
        model.update(_frame(seed=i), 0.75, 0.08, 0.05)
    path = str(tmp_path / "ae.pkl")
    model.save(path)

    restored = AutoencoderSceneModel(model_path=path)
    assert restored.is_fitted is True
    assert restored.phase == "active"
    with np.errstate(all="ignore"):
        assert restored.update(_frame(seed=5), 0.75, 0.08, 0.05) is not None


def test_calibration_countdown_is_zero_once_fitted(tmp_path):
    """A model restored from disk is fitted even with an empty frame buffer."""
    model = AutoencoderSceneModel()
    for i in range(model.CALIBRATION_FRAMES):
        model.update(_frame(seed=i), 0.75, 0.08, 0.05)
    path = str(tmp_path / "ae.pkl")
    model.save(path)

    restored = AutoencoderSceneModel(model_path=path)
    assert restored.frame_buffer == []
    assert restored.status["is_warming_up"] is False
    assert restored.status["calibration_frames_remaining"] == 0
