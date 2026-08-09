"""
WAAA — Biographical record types.

Two plain data carriers used by the node and by the persistence layer
(``ml/vector_biography.py``):

  BiographicalEntry — one significant event in the node's history
  Snapshot          — the restorable configuration of a node at a point
                      in time (used by recovery levels L1 and L2)

These types carry data only. All storage, indexing and retrieval logic
lives in ``ml/vector_biography.py``.
"""

from dataclasses import dataclass, field


@dataclass
class BiographicalEntry:
    """A single significant event in the node's history.

    ``payload`` is free-form and is stored as JSON; the scalar fields are
    stored as columns so they can be queried and embedded directly.
    """

    timestamp: float
    event_type: str
    payload: dict = field(default_factory=dict)
    coherence_at_event: float = 0.0
    prediction_error_at_event: float = 0.0
    node_goal_at_event: str = ""
    impact_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
            "coherence_at_event": self.coherence_at_event,
            "prediction_error_at_event": self.prediction_error_at_event,
            "node_goal_at_event": self.node_goal_at_event,
            "impact_score": self.impact_score,
        }


@dataclass
class Snapshot:
    """A restorable configuration of the node at a point in time.

    A snapshot deliberately holds configuration, not history: restoring it
    resets the threshold, the goal and the functional dispositions, while
    the biographical events themselves stay in the event store.
    """

    snapshot_id: str
    timestamp: float
    threshold_theta: float
    calibration_history: list = field(default_factory=list)
    bia_config: dict = field(default_factory=dict)
    current_goal: str = ""
    functional_dispositions: dict = field(default_factory=dict)
    node_id: str = ""

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "threshold_theta": self.threshold_theta,
            "calibration_history": self.calibration_history,
            "bia_config": self.bia_config,
            "current_goal": self.current_goal,
            "functional_dispositions": self.functional_dispositions,
            "node_id": self.node_id,
        }
