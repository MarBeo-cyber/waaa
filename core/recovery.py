"""
WAAA — Recovery hierarchy (L0–L3).

Four escalating repair actions, ordered by how much of the node's
accumulated state they discard:

  L0  in-operation self-repair   restore operating parameters to defaults
  L1  rollback to snapshot       restore θ from the last snapshot
  L2  rollback intermediate      restore θ, discard calibration history
  L3  reset invariant core       full reset of θ, policy and dispositions

Every level is executed against the objects the node already owns (the
biographical store and the APTC agent) through their public interfaces;
this module holds no learned state of its own. Biographical *events* are
never deleted by any level — what a higher level discards is calibration
and policy state, which is what "biographical loss" refers to below.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger("waaa.recovery")

# Default value of the sensor noise filter, restored by L0.
DEFAULT_NOISE_FILTER = 0.5


class RecoveryLevel(IntEnum):
    LEVEL0 = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3


@dataclass
class RecoveryEvent:
    timestamp: float
    level: RecoveryLevel
    trigger: str
    biographical_loss: str
    success: bool
    duration_seconds: float
    details: dict = field(default_factory=dict)


class RecoveryManager:
    """Executes the L0–L3 recovery actions and records what it did."""

    def __init__(self, node_id: str, initial_config: Optional[dict] = None):
        self.node_id = node_id
        self.initial_config = dict(initial_config or {})
        self.recovery_history: list[RecoveryEvent] = []

    # ------------------------------------------------------------------ #
    # Levels                                                              #
    # ------------------------------------------------------------------ #

    def execute_level0(self, operating_params: Optional[dict] = None,
                       trigger: str = "node_request") -> dict:
        """L0 — in-operation self-repair.

        Restores the operating parameters handed in by the caller to their
        defaults. Nothing learned and nothing recorded is discarded.
        Returns the repaired parameters for the caller to apply.
        """
        started = time.time()
        params = dict(operating_params or {})
        repaired = dict(params)
        if "sensor_noise_filter" in params:
            repaired["sensor_noise_filter"] = DEFAULT_NOISE_FILTER

        self._record(RecoveryLevel.LEVEL0, trigger, "none", True,
                     time.time() - started,
                     {"before": params, "after": repaired})
        logger.info("[Recovery:%s] L0 executed: %s", self.node_id, repaired)
        return repaired

    def execute_level1(self, biography, aptc, trigger: str = "node_request") -> dict:
        """L1 — rollback to the last snapshot.

        Restores θ from the most recent snapshot and keeps everything the
        agent has learned (Q-table, ε, step counters, calibration log).
        """
        started = time.time()
        snap = self._latest_snapshot(biography)
        if snap is None:
            return self._no_snapshot(RecoveryLevel.LEVEL1, trigger, started)

        state = aptc.export_state()
        state["theta"] = snap.threshold_theta
        aptc.import_state(state)

        details = {
            "snapshot_id": snap.snapshot_id,
            "restored_theta": snap.threshold_theta,
            "restored_goal": snap.current_goal,
        }
        self._record(RecoveryLevel.LEVEL1, trigger, "none", True,
                     time.time() - started, details)
        logger.warning("[Recovery:%s] L1 executed from %s",
                       self.node_id, snap.snapshot_id)
        return details

    def execute_level2(self, biography, aptc, trigger: str = "node_request") -> dict:
        """L2 — rollback to the invariant core plus partial history.

        Restores θ from the last snapshot (falling back to the configured
        θ₀ if there is none) and discards the calibration history and the
        exploration counters. The Q-table survives.
        """
        started = time.time()
        snap = self._latest_snapshot(biography)
        theta = snap.threshold_theta if snap is not None else aptc.theta

        # Passing only theta resets every other field to its default:
        # epsilon back to its start value, counters to zero, log emptied.
        aptc.import_state({"theta": theta})

        details = {
            "snapshot_id": snap.snapshot_id if snap is not None else None,
            "restored_theta": theta,
            "calibration_history_discarded": True,
        }
        self._record(RecoveryLevel.LEVEL2, trigger, "partial", True,
                     time.time() - started, details)
        logger.warning("[Recovery:%s] L2 executed (θ=%.3f)", self.node_id, theta)
        return details

    def execute_level3(self, aptc, trigger: str = "node_request") -> dict:
        """L3 — reset of the invariant core.

        Everything the agent learned is discarded: θ returns to θ₀, ε to
        its start value, the calibration log is emptied and the Q-table is
        zeroed. The functional dispositions from the node's initial
        configuration are returned to the caller. Only the node identity
        and the recorded biographical events survive.
        """
        started = time.time()
        aptc.import_state({})            # every field back to its default
        q_table = getattr(aptc, "Q", None)
        if q_table is not None:
            q_table.fill(0.0)

        dispositions = dict(
            self.initial_config.get("functional_dispositions", {})
        )
        details = {
            "reset_theta": aptc.theta,
            "q_table_cleared": q_table is not None,
            "functional_dispositions": dispositions,
            "goal": self.initial_config.get("goal"),
            "telos": self.initial_config.get("telos"),
        }
        self._record(RecoveryLevel.LEVEL3, trigger, "significant", True,
                     time.time() - started, details)
        logger.error("[Recovery:%s] L3 executed — invariant core reset",
                     self.node_id)
        return details

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _latest_snapshot(biography):
        loader = getattr(biography, "load_latest_snapshot", None)
        if loader is None:
            return None
        try:
            return loader()
        except Exception as exc:                       # pragma: no cover
            logger.warning("[Recovery] Snapshot load failed: %s", exc)
            return None

    def _no_snapshot(self, level: RecoveryLevel, trigger: str,
                     started: float) -> dict:
        logger.warning("[Recovery:%s] %s requested but no snapshot exists",
                       self.node_id, level.name)
        self._record(level, trigger, "none", False, time.time() - started,
                     {"error": "no_snapshot_available"})
        return {"error": "no_snapshot_available"}

    def _record(self, level: RecoveryLevel, trigger: str, loss: str,
                success: bool, duration: float, details: dict) -> None:
        self.recovery_history.append(RecoveryEvent(
            timestamp=time.time(),
            level=level,
            trigger=trigger,
            biographical_loss=loss,
            success=success,
            duration_seconds=round(duration, 6),
            details=details,
        ))

    @property
    def status(self) -> dict:
        counts: dict[str, int] = {}
        for ev in self.recovery_history:
            counts[ev.level.name] = counts.get(ev.level.name, 0) + 1
        last = self.recovery_history[-1] if self.recovery_history else None
        return {
            "node_id": self.node_id,
            "total_recoveries": len(self.recovery_history),
            "by_level": counts,
            "last_recovery": {
                "timestamp": last.timestamp,
                "level": last.level.name,
                "trigger": last.trigger,
                "biographical_loss": last.biographical_loss,
                "success": last.success,
            } if last else None,
        }
