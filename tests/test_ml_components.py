"""
WAAA — Basic test suite.
Tests that each ML component initialises and produces outputs of correct type/shape.
No training required: tests run in bootstrap/fallback mode.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


class TestAutoencoderSceneModel:
    def test_import(self):
        from ml.autoencoder_scene import AutoencoderSceneModel
        model = AutoencoderSceneModel(input_dim=16)
        assert model is not None

    def test_predict_returns_float(self):
        from ml.autoencoder_scene import AutoencoderSceneModel
        model = AutoencoderSceneModel(input_dim=4)
        features = [0.5, 0.3, 0.1, 0.8]
        error = model.predict_anomaly_score(features)
        assert isinstance(error, float)
        assert 0.0 <= error <= 1.0 or error >= 0.0  # reconstruction error is non-negative


class TestRLAPTCAgent:
    def test_import(self):
        from ml.rl_aptc import RLAPTCAgent
        agent = RLAPTCAgent()
        assert agent is not None

    def test_choose_action_returns_float(self):
        from ml.rl_aptc import RLAPTCAgent
        agent = RLAPTCAgent()
        state = (0.5, 0.3, 0.7, 0.2)
        action = agent.choose_action(state)
        assert isinstance(action, float)

    def test_update_does_not_crash(self):
        from ml.rl_aptc import RLAPTCAgent
        agent = RLAPTCAgent()
        state = (0.5, 0.3, 0.7, 0.2)
        action = agent.choose_action(state)
        agent.update(state, action, reward=1.0, next_state=(0.5, 0.3, 0.7, 0.2))


class TestGoalClassifier:
    def test_import(self):
        from ml.goal_classifier import GoalClassifier
        clf = GoalClassifier()
        assert clf is not None

    def test_predict_returns_string(self):
        from ml.goal_classifier import GoalClassifier
        clf = GoalClassifier()
        features = [0.5, 0.3, 0.7, 0.2, 0.6]
        goal = clf.predict(features)
        assert isinstance(goal, str)
        assert len(goal) > 0


class TestRecoveryLevelDetector:
    def test_import(self):
        from ml.recovery_detector import RecoveryLevelDetector
        det = RecoveryLevelDetector()
        assert det is not None

    def test_assess_returns_int(self):
        from ml.recovery_detector import RecoveryLevelDetector
        det = RecoveryLevelDetector()
        features = [0.8, 0.1, 0.9, 0.05]
        level = det.assess(features)
        assert isinstance(level, int)
        assert 0 <= level <= 3


class TestVectorBiography:
    def test_import(self):
        from ml.vector_biography import VectorBiography
        vb = VectorBiography(dim=8)
        assert vb is not None

    def test_add_and_search(self):
        from ml.vector_biography import VectorBiography
        vb = VectorBiography(dim=4)
        vb.add(vector=[0.1, 0.2, 0.3, 0.4], metadata={"event": "test"})
        results = vb.search(query=[0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) >= 1
