"""Regression tests for RecoveryLevelDetector score normalisation.

The bug these guard against: the anomaly score was computed as
``0.5 - iso_forest.score_samples(X)``. score_samples returns *negative*
values (about -0.4 to -0.7), so the expression was always >= 0.9, which
maps to L3 "critical — full reset" for every input the detector ever saw,
healthy or degraded.
"""

import numpy as np

from ml.recovery_detector import CALIBRATION_SAMPLES, RecoveryLevelDetector

# Range of the healthy states the detector below is trained on.
COHERENCE_RANGE = (0.60, 0.95)
PRED_ERROR_RANGE = (0.00, 0.40)
QUALITY_RANGE = (0.50, 0.90)

# Centre of that range: the most typical healthy state there is.
HEALTHY = dict(coherence=0.775, prediction_error=0.20, frame_quality=0.70,
               aptc_theta=0.35, blind_streak=0, goal_duration_s=5.0)
DEGRADED = dict(coherence=0.15, prediction_error=0.78, frame_quality=0.22,
                aptc_theta=0.35, blind_streak=3, goal_duration_s=30.0)


def _fitted_detector(seed: int = 0) -> RecoveryLevelDetector:
    """A detector trained on a spread of healthy states."""
    rng = np.random.default_rng(seed)
    det = RecoveryLevelDetector()
    while not det.is_fitted:
        det.assess(
            coherence=float(rng.uniform(*COHERENCE_RANGE)),
            prediction_error=float(rng.uniform(*PRED_ERROR_RANGE)),
            frame_quality=float(rng.uniform(*QUALITY_RANGE)),
            aptc_theta=0.35, blind_streak=0, goal_duration_s=5.0,
        )
    assert len(det.buffer) >= CALIBRATION_SAMPLES
    return det


def test_healthy_and_degraded_get_different_levels():
    """The whole point of the detector: it must tell the two apart."""
    det = _fitted_detector()
    healthy_level, healthy_score = det.assess(**HEALTHY)
    degraded_level, degraded_score = det.assess(**DEGRADED)

    assert healthy_level != degraded_level
    assert healthy_level == -1, "a state drawn from the training distribution is healthy"
    assert degraded_level >= 1, "a clearly degraded state needs more than L0"
    assert healthy_score < degraded_score


def test_healthy_state_is_not_reported_as_critical():
    """Direct guard on the old behaviour: everything scored >= 0.9 -> L3."""
    det = _fitted_detector()
    level, score = det.assess(**HEALTHY)
    assert score < 0.30, f"healthy state scored {score:.3f}"
    assert level < 3


def test_severity_is_graded_not_binary():
    det = _fitted_detector()
    mild, _ = det.assess(coherence=0.55, prediction_error=0.45,
                         frame_quality=0.45, aptc_theta=0.35,
                         blind_streak=1, goal_duration_s=5.0)
    severe, _ = det.assess(coherence=0.02, prediction_error=0.98,
                           frame_quality=0.05, aptc_theta=0.95,
                           blind_streak=5, goal_duration_s=60.0)
    assert mild < severe


def test_scores_stay_in_unit_interval():
    det = _fitted_detector()
    for kwargs in (HEALTHY, DEGRADED):
        _, score = det.assess(**kwargs)
        assert 0.0 <= score <= 1.0


def test_decision_scale_is_learned_at_fit_time():
    det = _fitted_detector()
    assert det.decision_scale > 0.0
    # and it survives a save/load round trip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl") as fh:
        det.save(fh.name)
        restored = RecoveryLevelDetector()
        restored.load(fh.name)
    assert restored.decision_scale == det.decision_scale


def test_rule_fallback_before_the_forest_is_fitted():
    det = RecoveryLevelDetector()
    level, score = det.assess(**HEALTHY)
    assert det.is_fitted is False
    assert level == -1 and 0.0 <= score <= 1.0
