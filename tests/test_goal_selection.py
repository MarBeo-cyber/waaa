"""Tests for goal selection.

Goal selection is a deterministic rule. It used to be a RandomForest
trained on labels produced by that same rule, which meant the "learned"
model could only reproduce its own baseline — including the
``coherence < 0.35`` comparison it was documented as replacing. These
tests pin the rule's behaviour and check that no self-trained model has
crept back in.
"""

import numpy as np

from ml.goal_classifier import FEATURE_NAMES, GOALS, GoalClassifier


def _goal(coherence, prediction_error=0.2, frame_quality=0.9):
    goal, method, confidence = GoalClassifier().predict(
        coherence=coherence, prediction_error=prediction_error,
        frame_quality=frame_quality, signal_magnitude=0.4, aptc_theta=0.35,
    )
    assert method == "rule"
    assert confidence == 1.0
    return goal


def test_nominal_state_monitors():
    assert _goal(0.90) == "monitor_anomalies"


def test_low_coherence_restores_perception():
    assert _goal(0.30) == "restore_perceptual_capacity"


def test_very_low_coherence_executes_recovery():
    assert _goal(0.05) == "execute_recovery"


def test_high_prediction_error_executes_recovery():
    assert _goal(0.90, prediction_error=0.95) == "execute_recovery"


def test_poor_frame_quality_restores_perception():
    assert _goal(0.90, frame_quality=0.10) == "restore_perceptual_capacity"


def test_every_outcome_is_a_declared_goal():
    for coherence in np.linspace(0.0, 1.0, 21):
        assert _goal(float(coherence)) in GOALS


def test_selection_is_deterministic():
    """Same input, same output — every time, from a fresh selector."""
    assert len({_goal(0.30) for _ in range(20)}) == 1


def test_no_self_trained_model_remains():
    selector = GoalClassifier()
    for attr in ("rf", "label_encoder", "X_train", "y_train", "is_fitted"):
        assert not hasattr(selector, attr), (
            f"{attr} is back: a model trained on its own rule's labels "
            f"learns nothing the rule did not already encode"
        )
    assert selector.status["model"] is None
    assert selector.status["method"] == "rule"


def test_temporal_features_are_computed_and_published():
    """The windowed feature engineering is kept and exported for analysis."""
    selector = GoalClassifier()
    for coherence in (0.9, 0.8, 0.6, 0.4, 0.2):
        selector.predict(coherence=coherence, prediction_error=0.3,
                         frame_quality=0.7, signal_magnitude=0.4,
                         aptc_theta=0.35)

    features = selector.status["last_features"]
    assert set(features) == set(FEATURE_NAMES)
    assert features["coh_slope"] < 0, "a falling coherence must show a falling slope"
    assert features["coh_mean"] == 0.58
    assert selector.last_features.shape == (len(FEATURE_NAMES),)


def test_goal_duration_resets_on_switch():
    selector = GoalClassifier()
    selector.update_current_goal("execute_recovery")
    assert selector.current_goal == "execute_recovery"
    assert selector.status["goal_duration_s"] < 1.0


def test_save_and_load_round_trip(tmp_path):
    selector = GoalClassifier()
    selector.predict(0.5, 0.3, 0.7, 0.4, 0.35)
    selector.update_current_goal("restore_perceptual_capacity")
    path = str(tmp_path / "goal.pkl")
    selector.save(path)

    restored = GoalClassifier(model_path=path)
    assert restored.current_goal == "restore_perceptual_capacity"
    assert restored.total_predictions == 1
