"""Regression tests for the RL-APTC observation interval and Q-learning.

The bug these guard against: an observation interval could only be closed
by the wall clock (12 seconds). _update_q returns early until
_prev_discrete_state is set, and that only happens inside
_select_and_apply_action, which only runs when an interval closes. A demo
that runs for ten seconds therefore finished with a Q-table that was 100%
zero and a threshold that had never moved, while narrating that the agent
was learning.
"""

import time

import numpy as np

from ml.rl_aptc import RATE_BINS, RLAPTCAgent, RLAPTCConfig, discretise_state


def _run(agent: RLAPTCAgent, n: int, signal: float = 0.9,
         coherence: float = 0.8) -> None:
    for _ in range(n):
        agent.evaluate(signal, coherence=coherence)


def test_q_table_learns_without_waiting_for_the_wall_clock():
    """N steps of a fast loop must produce Bellman updates, not zeros."""
    started = time.time()
    agent = RLAPTCAgent(config=RLAPTCConfig(observation_interval_steps=4))
    _run(agent, 40)
    elapsed = time.time() - started

    assert elapsed < 1.0, "the test must not depend on wall-clock time passing"
    assert agent.status["q_table_nonzero"] > 0
    assert agent.state.total_intervals >= 4
    assert agent.state.total_steps > 0


def test_threshold_moves():
    agent = RLAPTCAgent(config=RLAPTCConfig(theta_0=0.35,
                                            observation_interval_steps=4))
    _run(agent, 60)
    assert agent.theta != 0.35


def test_interval_closes_after_exactly_the_configured_step_count():
    agent = RLAPTCAgent(config=RLAPTCConfig(observation_interval_steps=5,
                                            observation_interval_s=1e9))
    _run(agent, 5)
    assert agent.state.total_intervals == 0, "closes on the 6th evaluation"
    _run(agent, 1)
    assert agent.state.total_intervals == 1
    assert agent.state.evaluations_this_interval == 1


def test_wall_clock_gate_still_closes_a_slow_interval():
    """A slow loop must not have to reach the step count to make progress."""
    agent = RLAPTCAgent(config=RLAPTCConfig(observation_interval_steps=10_000,
                                            observation_interval_s=0.05))
    agent.evaluate(0.9, coherence=0.8)
    time.sleep(0.06)
    agent.evaluate(0.9, coherence=0.8)
    assert agent.state.total_intervals == 1


def test_bellman_update_moves_q_towards_the_reward():
    """The update rule itself: Q += alpha * (r + gamma * maxQ' - Q)."""
    agent = RLAPTCAgent(config=RLAPTCConfig(learning_rate=0.5, discount=0.0))
    agent._prev_discrete_state = agent._get_discrete_state()
    agent._prev_action = 2
    before = float(agent.Q[agent._prev_discrete_state][2])
    agent._update_q(reward=1.0)
    after = float(agent.Q[agent._prev_discrete_state][2])
    assert after == before + 0.5 * (1.0 - before)


def test_rate_bins_cover_high_event_rates():
    """Rates above the top bin used to collapse into one indistinguishable
    state; 20 was the old ceiling and rates in the hundreds occur when the
    interval is driven by the wall clock in a fast loop."""
    assert RATE_BINS.max() >= 400
    high = discretise_state(0.35, 402.0, 0, 0, 0.5)
    mid = discretise_state(0.35, 40.0, 0, 0, 0.5)
    assert high[1] != mid[1]


def test_state_is_within_q_table_bounds_for_extreme_inputs():
    agent = RLAPTCAgent()
    for rate in (0.0, 1.0, 19.0, 402.0, 10_000.0):
        state = discretise_state(0.99, rate, 99, 99, 1.0)
        assert agent.Q[state].shape == (5,)


def test_save_and_load_round_trip(tmp_path):
    agent = RLAPTCAgent(config=RLAPTCConfig(observation_interval_steps=4))
    _run(agent, 40)
    path = str(tmp_path / "rl.pkl")
    agent.save(path)

    restored = RLAPTCAgent(model_path=path)
    assert np.array_equal(restored.Q, agent.Q)
    assert restored.theta == agent.theta
    assert restored.state.total_steps == agent.state.total_steps
