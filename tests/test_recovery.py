"""Tests for the L0–L3 recovery hierarchy."""

import time

from core.biography import Snapshot
from core.recovery import DEFAULT_NOISE_FILTER, RecoveryLevel, RecoveryManager
from ml.rl_aptc import RLAPTCAgent, RLAPTCConfig
from ml.vector_biography import VectorBiography

INITIAL_CONFIG = {
    "goal": "monitor_anomalies",
    "functional_dispositions": {"pertinence_weight": 0.8,
                                "continuity_priority": 0.9},
    "telos": "guarantee_operational_continuity",
}


def _agent(theta=0.35):
    agent = RLAPTCAgent(config=RLAPTCConfig(theta_0=theta,
                                            observation_interval_steps=4))
    for _ in range(40):
        agent.evaluate(0.9, coherence=0.8)
    return agent


def _biography_with_snapshot(tmp_path, theta=0.42):
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    bio.save_snapshot(Snapshot(
        snapshot_id="snap-1", timestamp=time.time(), threshold_theta=theta,
        calibration_history=[], bia_config={}, current_goal="monitor_anomalies",
        functional_dispositions={}, node_id="node-a",
    ))
    return bio


def test_level0_restores_operating_parameters():
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    result = mgr.execute_level0({"sensor_noise_filter": 0.05})
    assert result["sensor_noise_filter"] == DEFAULT_NOISE_FILTER
    assert mgr.recovery_history[-1].biographical_loss == "none"
    assert mgr.recovery_history[-1].level is RecoveryLevel.LEVEL0


def test_level1_restores_theta_and_keeps_what_was_learned(tmp_path):
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    bio = _biography_with_snapshot(tmp_path, theta=0.42)
    agent = _agent()
    steps_before = agent.state.total_steps
    assert steps_before > 0

    mgr.execute_level1(bio, agent)
    assert agent.theta == 0.42
    assert agent.state.total_steps == steps_before, "L1 must not lose experience"
    assert mgr.recovery_history[-1].biographical_loss == "none"


def test_level1_without_a_snapshot_fails_honestly(tmp_path):
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    result = mgr.execute_level1(bio, _agent())
    assert result == {"error": "no_snapshot_available"}
    assert mgr.recovery_history[-1].success is False


def test_level2_discards_calibration_history(tmp_path):
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    bio = _biography_with_snapshot(tmp_path, theta=0.42)
    agent = _agent()
    assert agent.state.calibration_log

    mgr.execute_level2(bio, agent)
    assert agent.theta == 0.42
    assert agent.state.calibration_log == []
    assert agent.state.total_steps == 0
    assert mgr.recovery_history[-1].biographical_loss == "partial"


def test_level3_resets_everything_learned():
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    agent = _agent(theta=0.35)
    assert agent.status["q_table_nonzero"] > 0

    result = mgr.execute_level3(agent)
    assert agent.theta == 0.35
    assert agent.status["q_table_nonzero"] == 0
    assert agent.state.cumulative_reward == 0.0
    assert result["functional_dispositions"] == INITIAL_CONFIG["functional_dispositions"]
    assert mgr.recovery_history[-1].biographical_loss == "significant"


def test_level3_returns_a_copy_of_the_dispositions():
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    result = mgr.execute_level3(_agent())
    result["functional_dispositions"]["pertinence_weight"] = 0.0
    assert INITIAL_CONFIG["functional_dispositions"]["pertinence_weight"] == 0.8


def test_history_and_status():
    mgr = RecoveryManager("node-a", INITIAL_CONFIG)
    mgr.execute_level0({"sensor_noise_filter": 0.1})
    mgr.execute_level3(_agent())

    assert mgr.status["total_recoveries"] == 2
    assert mgr.status["by_level"] == {"LEVEL0": 1, "LEVEL3": 1}
    assert mgr.status["last_recovery"]["level"] == "LEVEL3"
    for event in mgr.recovery_history:
        assert event.duration_seconds >= 0.0
        assert event.trigger == "node_request"


def test_recovery_levels_are_ordered():
    assert RecoveryLevel(0) is RecoveryLevel.LEVEL0
    assert RecoveryLevel.LEVEL0 < RecoveryLevel.LEVEL3
    assert [lvl.value for lvl in RecoveryLevel] == [0, 1, 2, 3]
