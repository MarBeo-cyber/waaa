"""Tests for in-process biographical reconciliation between nodes."""

import time

from core.biography import BiographicalEntry
from federation.federation import Federation
from ml.vector_biography import VectorBiography


class FakeNode:
    """The two attributes Federation actually uses."""

    def __init__(self, node_id, db_path):
        self.node_id = node_id
        self.biography = VectorBiography(node_id, db_path)

    def record(self, content="event"):
        self.biography.record(BiographicalEntry(
            timestamp=time.time(),
            event_type="perturbation",
            payload={"content": content},
            coherence_at_event=0.5,
            prediction_error_at_event=0.5,
            node_goal_at_event="monitor_anomalies",
            impact_score=0.5,
        ))


def _pair(tmp_path):
    a = FakeNode("node-a", str(tmp_path / "a.db"))
    b = FakeNode("node-b", str(tmp_path / "b.db"))
    fed = Federation("fed-test", sync_interval_s=0.0)
    fed.register_node(a)
    fed.register_node(b)
    return fed, a, b


def test_events_reach_the_peer(tmp_path):
    fed, a, b = _pair(tmp_path)
    a.record("only a saw this")

    assert fed.sync() == 1
    payloads = [e["payload"] for e in b.biography.recent_events(10)]
    assert any("only a saw this" in p for p in payloads)


def test_repeated_syncs_do_not_duplicate_events(tmp_path):
    """A federated event must not be forwarded back to its origin, or the
    two nodes copy it to each other forever."""
    fed, a, b = _pair(tmp_path)
    a.record()
    fed.sync()

    for _ in range(5):
        fed.sync()

    assert len(a.biography.recent_events(50)) == 1
    assert len(b.biography.recent_events(50)) == 1


def test_only_new_events_are_moved(tmp_path):
    fed, a, b = _pair(tmp_path)
    a.record("first")
    assert fed.sync() == 1
    assert fed.sync() == 0

    a.record("second")
    assert fed.sync() == 1


def test_rate_limit_blocks_a_sync_within_the_interval(tmp_path):
    fed, a, b = _pair(tmp_path)
    fed.sync_interval_s = 60.0
    a.record()
    fed.sync()
    a.record()
    assert fed.sync() == 0, "second sync is inside the interval"
    assert fed.sync(force=True) == 1


def test_isolated_node_receives_nothing_until_restored(tmp_path):
    fed, a, b = _pair(tmp_path)
    fed.isolate_node("node-b")
    a.record("during the partition")
    fed.sync()
    assert b.biography.recent_events(10) == []
    assert fed.status["isolated"] == ["node-b"]

    result = fed.restore_node("node-b")
    assert result["events_reconciled"] == 1
    payloads = [e["payload"] for e in b.biography.recent_events(10)]
    assert any("during the partition" in p for p in payloads)


def test_single_node_federation_is_a_no_op(tmp_path):
    a = FakeNode("node-a", str(tmp_path / "a.db"))
    fed = Federation("fed-solo", sync_interval_s=0.0)
    fed.register_node(a)
    a.record()
    assert fed.sync() == 0


def test_status_reports_membership(tmp_path):
    fed, a, b = _pair(tmp_path)
    a.record()
    fed.sync()
    status = fed.status
    assert status["node_count"] == 2
    assert sorted(status["connected"]) == ["node-a", "node-b"]
    assert status["events_reconciled"] == 1
    assert status["sync_count"] == 1
