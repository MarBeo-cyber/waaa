"""
WAAA ML — Reinforcement Learning APTC
Replaces the rule-based APTC with a Q-learning agent.

The threshold calibration problem is framed as an MDP:
  State  : (current_theta, ema_event_rate, blind_streak, saturated_streak,
             recent_coherence, phase_encoded)
  Actions: DECREASE_LARGE | DECREASE_SMALL | HOLD | INCREASE_SMALL | INCREASE_LARGE
  Reward :
    +2.0  detected a real anomaly (signal above theta, coherence confirms)
    -1.0  false positive (signal above theta, coherence nominal)
    -2.0  blind interval (zero events detected)
    -0.5  saturated interval (too many events — possible noise)
    +0.1  productive interval (n_min ≤ events ≤ n_max)
    -0.3  theta at extreme boundary (min or max)

The agent uses a tabular Q-table with discretised state space,
making it lightweight and interpretable — no GPU required.
As the agent accumulates experience, it learns the optimal
sensitivity policy for the specific target entity it monitors.
"""

import numpy as np
import logging
import time
import pickle
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("waaa.ml.rl_aptc")

# Action space
ACTIONS = {
    0: -0.08,   # DECREASE_LARGE
    1: -0.03,   # DECREASE_SMALL
    2:  0.00,   # HOLD
    3: +0.03,   # INCREASE_SMALL
    4: +0.08,   # INCREASE_LARGE
}
N_ACTIONS = len(ACTIONS)

# State discretisation bins
THETA_BINS       = np.linspace(0.05, 0.95, 10)
# The event rate is an EMA of the events counted in one observation
# interval, so its range follows observation_interval_steps. The top bins
# cover long intervals: with the default 12-step interval the measured EMA
# stays under 12, but a server configured with a wall-clock interval and a
# fast tick loop reaches the hundreds, and every rate above the last bin
# used to collapse into a single indistinguishable state.
RATE_BINS        = np.array([0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400])
BLIND_BINS       = np.array([0, 1, 2, 3, 5])
SATURATED_BINS   = np.array([0, 1, 2, 3])
COHERENCE_BINS   = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def discretise_state(theta: float,
                     ema_rate: float,
                     blind_streak: int,
                     saturated_streak: int,
                     coherence: float) -> tuple:
    """Map continuous state to discrete Q-table index."""
    return (
        int(np.digitize(theta, THETA_BINS)),
        int(np.digitize(ema_rate, RATE_BINS)),
        int(np.digitize(blind_streak, BLIND_BINS)),
        int(np.digitize(saturated_streak, SATURATED_BINS)),
        int(np.digitize(coherence, COHERENCE_BINS)),
    )


STATE_DIMS = (
    len(THETA_BINS) + 1,
    len(RATE_BINS) + 1,
    len(BLIND_BINS) + 1,
    len(SATURATED_BINS) + 1,
    len(COHERENCE_BINS) + 1,
)


@dataclass
class RLAPTCConfig:
    theta_0: float = 0.35
    theta_min: float = 0.05
    theta_max: float = 0.95
    # An observation interval ends after observation_interval_steps
    # evaluations *or* observation_interval_s seconds, whichever comes
    # first. The step counter is what makes the agent learn in a short
    # run: with a wall-clock gate alone a demo that ticks for ten seconds
    # never closes an interval, never selects an action, and therefore
    # never performs a single Bellman update.
    observation_interval_steps: int = 12
    observation_interval_s: float = 12.0
    n_min: int = 1
    n_max: int = 10
    # RL hyperparameters
    learning_rate: float = 0.15     # alpha
    discount: float = 0.90          # gamma
    epsilon_start: float = 0.40     # exploration rate
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995    # per episode


@dataclass
class RLAPTCState:
    theta: float
    phase: str = "initialising"
    interval_start: float = field(default_factory=time.time)
    events_this_interval: int = 0
    evaluations_this_interval: int = 0
    total_intervals: int = 0
    blind_streak: int = 0
    saturated_streak: int = 0
    ema_event_rate: float = 0.0
    last_coherence: float = 0.5
    epsilon: float = 0.40
    total_steps: int = 0
    cumulative_reward: float = 0.0
    calibration_log: list = field(default_factory=list)


class RLAPTCAgent:
    """
    Tabular Q-learning agent for adaptive threshold calibration.

    The agent learns — through trial and error — the optimal policy
    for maintaining the minimum productive threshold given the
    specific event distribution of the monitored entity.

    Unlike the rule-based APTC, the RL agent:
    - Adapts its exploration/exploitation trade-off over time
    - Learns non-linear relationships between state and optimal action
    - Improves its policy continuously as it accumulates experience
    - Can be persisted and resumed across sessions
    """

    def __init__(self, config: Optional[RLAPTCConfig] = None,
                 model_path: Optional[str] = None):
        self.cfg   = config or RLAPTCConfig()
        self.state = RLAPTCState(
            theta=self.cfg.theta_0,
            epsilon=self.cfg.epsilon_start,
        )

        # Q-table: state_dims × n_actions, initialised optimistically
        self.Q = np.zeros(STATE_DIMS + (N_ACTIONS,), dtype=np.float32)
        self._prev_discrete_state: Optional[tuple] = None
        self._prev_action: Optional[int] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"[RL-APTC] Q-table loaded from {model_path} "
                        f"(steps={self.state.total_steps})")
        else:
            logger.info(f"[RL-APTC] Initialised. "
                        f"θ={self.state.theta:.3f} "
                        f"ε={self.state.epsilon:.3f}")

    # ------------------------------------------------------------------ #
    # Core interface (matches original APTC)                              #
    # ------------------------------------------------------------------ #

    def evaluate(self, signal_magnitude: float,
                 coherence: float = 0.5) -> bool:
        """
        Evaluate whether signal_magnitude crosses the current threshold.
        Returns True if classified as perturbation.
        Also tracks events and advances the interval clock.
        """
        self._maybe_advance_interval(coherence)
        self.state.evaluations_this_interval += 1
        self.state.last_coherence = coherence

        is_perturbation = signal_magnitude >= self.state.theta

        if is_perturbation:
            self.state.events_this_interval += 1

            # Immediate reward signal for real detection
            # (only if coherence confirms it's not noise)
            if coherence < 0.5:
                # Signal detected but coherence is low: likely noise
                self._update_q(reward=-0.5)
            else:
                self._update_q(reward=+1.0)

        return is_perturbation

    # ------------------------------------------------------------------ #
    # Interval management                                                  #
    # ------------------------------------------------------------------ #

    def _maybe_advance_interval(self, coherence: float):
        """Close the observation interval on whichever gate trips first.

        Step count and wall clock are both checked so the agent learns at
        the same rate whether it is driven by a fast demo loop or by a
        slow real-time deployment.
        """
        elapsed = time.time() - self.state.interval_start
        steps = self.state.evaluations_this_interval
        if (steps >= self.cfg.observation_interval_steps or
                elapsed >= self.cfg.observation_interval_s):
            self._close_interval(coherence)

    def _close_interval(self, coherence: float):
        n = self.state.events_this_interval
        self.state.total_intervals += 1

        # Update EMA of event rate
        alpha = 0.2
        self.state.ema_event_rate = (
            alpha * n + (1 - alpha) * self.state.ema_event_rate
        )

        # Compute reward for this interval
        if n == 0:
            reward = -2.0
            self.state.blind_streak += 1
            self.state.saturated_streak = 0
            self.state.phase = "exploring_blind"
            logger.warning(f"[RL-APTC] Blind interval "
                           f"(streak={self.state.blind_streak})")
        elif n > self.cfg.n_max:
            reward = -0.5
            self.state.saturated_streak += 1
            self.state.blind_streak = 0
            self.state.phase = "saturated"
        else:
            reward = +0.1
            self.state.blind_streak = 0
            self.state.saturated_streak = 0
            self.state.phase = "productive"

        # Penalise extreme theta values
        if self.state.theta <= self.cfg.theta_min + 0.02:
            reward -= 0.3
        elif self.state.theta >= self.cfg.theta_max - 0.02:
            reward -= 0.3

        # Update Q-table and select next action
        self._update_q(reward)
        self._select_and_apply_action(coherence)

        # Log calibration record
        record = {
            "interval": self.state.total_intervals,
            "timestamp": time.time(),
            "n_detected": n,
            "theta": round(self.state.theta, 4),
            "reward": round(reward, 3),
            "epsilon": round(self.state.epsilon, 3),
            "ema_rate": round(self.state.ema_event_rate, 3),
        }
        self.state.calibration_log.append(record)

        # Reset interval
        self.state.events_this_interval = 0
        self.state.evaluations_this_interval = 0
        self.state.interval_start = time.time()

        if self.state.total_intervals >= 3:
            self.state.phase = "learning"

    # ------------------------------------------------------------------ #
    # Q-learning core                                                      #
    # ------------------------------------------------------------------ #

    def _get_discrete_state(self, coherence: Optional[float] = None) -> tuple:
        coh = coherence if coherence is not None else self.state.last_coherence
        return discretise_state(
            self.state.theta,
            self.state.ema_event_rate,
            self.state.blind_streak,
            self.state.saturated_streak,
            coh,
        )

    def _select_and_apply_action(self, coherence: float):
        """ε-greedy action selection + apply to theta."""
        discrete_state = self._get_discrete_state(coherence)

        # ε-greedy exploration
        if np.random.random() < self.state.epsilon:
            action_idx = np.random.randint(N_ACTIONS)
        else:
            action_idx = int(np.argmax(self.Q[discrete_state]))

        # Apply action to theta
        delta = ACTIONS[action_idx]
        new_theta = float(np.clip(
            self.state.theta + delta,
            self.cfg.theta_min,
            self.cfg.theta_max,
        ))

        logger.debug(f"[RL-APTC] Action {action_idx} (Δ={delta:+.3f}): "
                     f"θ {self.state.theta:.3f}→{new_theta:.3f} "
                     f"ε={self.state.epsilon:.3f}")

        self.state.theta = new_theta
        self._prev_discrete_state = discrete_state
        self._prev_action = action_idx
        self.state.total_steps += 1

        # Decay epsilon
        self.state.epsilon = max(
            self.cfg.epsilon_min,
            self.state.epsilon * self.cfg.epsilon_decay,
        )

    def _update_q(self, reward: float):
        """Bellman update for the Q-table."""
        if self._prev_discrete_state is None or self._prev_action is None:
            return

        current_state = self._get_discrete_state()
        best_next_q = float(np.max(self.Q[current_state]))

        old_q = self.Q[self._prev_discrete_state][self._prev_action]
        new_q = old_q + self.cfg.learning_rate * (
            reward + self.cfg.discount * best_next_q - old_q
        )
        self.Q[self._prev_discrete_state][self._prev_action] = new_q

        self.state.cumulative_reward += reward

    # ------------------------------------------------------------------ #
    # State export / import (for snapshots and federation)                #
    # ------------------------------------------------------------------ #

    def export_state(self) -> dict:
        return {
            "theta": self.state.theta,
            "phase": self.state.phase,
            "ema_event_rate": self.state.ema_event_rate,
            "total_intervals": self.state.total_intervals,
            "blind_streak": self.state.blind_streak,
            "epsilon": self.state.epsilon,
            "total_steps": self.state.total_steps,
            "cumulative_reward": self.state.cumulative_reward,
            "calibration_log": self.state.calibration_log[-50:],
        }

    def import_state(self, state_dict: dict):
        self.state.theta              = state_dict.get("theta", self.cfg.theta_0)
        self.state.phase              = state_dict.get("phase", "initialising")
        self.state.ema_event_rate     = state_dict.get("ema_event_rate", 0.0)
        self.state.total_intervals    = state_dict.get("total_intervals", 0)
        self.state.blind_streak       = state_dict.get("blind_streak", 0)
        self.state.epsilon            = state_dict.get("epsilon", self.cfg.epsilon_start)
        self.state.total_steps        = state_dict.get("total_steps", 0)
        self.state.cumulative_reward  = state_dict.get("cumulative_reward", 0.0)
        self.state.calibration_log    = state_dict.get("calibration_log", [])
        logger.info(f"[RL-APTC] State imported: θ={self.state.theta:.3f} "
                    f"steps={self.state.total_steps}")

    def save(self, path: str):
        state = {
            "Q": self.Q,
            "agent_state": self.export_state(),
            "config": self.cfg,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"[RL-APTC] Q-table saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.Q   = state["Q"]
        self.cfg = state.get("config", self.cfg)
        self.import_state(state["agent_state"])

    def force_interval_close(self):
        self._close_interval(self.state.last_coherence)

    @property
    def theta(self) -> float:
        return self.state.theta

    @property
    def status(self) -> dict:
        return {
            "theta": round(self.state.theta, 4),
            "phase": self.state.phase,
            "epsilon": round(self.state.epsilon, 4),
            "ema_event_rate": round(self.state.ema_event_rate, 3),
            "events_this_interval": self.state.events_this_interval,
            "blind_streak": self.state.blind_streak,
            "total_intervals": self.state.total_intervals,
            "total_steps": self.state.total_steps,
            "cumulative_reward": round(self.state.cumulative_reward, 2),
            "q_table_nonzero": int(np.count_nonzero(self.Q)),
        }
