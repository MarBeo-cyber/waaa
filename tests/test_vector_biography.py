"""Tests for the vector-store biographical memory."""

import time

from core.biography import BiographicalEntry, Snapshot
from ml.vector_biography import EMBEDDING_DIM, VectorBiography, embed_event


def _entry(coherence, event_type="perturbation", goal="monitor_anomalies",
           impact=0.5):
    return BiographicalEntry(
        timestamp=time.time(),
        event_type=event_type,
        payload={"frame_quality": 0.7, "signal_magnitude": 0.4},
        coherence_at_event=coherence,
        prediction_error_at_event=1.0 - coherence,
        node_goal_at_event=goal,
        impact_score=impact,
    )


def test_embedding_dimension():
    vec = embed_event(0.5, 0.5, 0.5, 0.5, "monitor_anomalies",
                      "perturbation", 0.5, time.time())
    assert vec.shape == (EMBEDDING_DIM,)


def test_record_then_retrieve(tmp_path):
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    bio.record(_entry(0.2, goal="restore_perceptual_capacity"))
    bio.record(_entry(0.9))

    assert bio.status["index_size"] == 2
    assert len(bio.recent_events(10)) == 2


def test_find_similar_ranks_the_closer_event_first(tmp_path):
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    bio.record(_entry(0.95, goal="monitor_anomalies"))
    bio.record(_entry(0.10, goal="restore_perceptual_capacity"))

    results = bio.find_similar(
        coherence=0.12, prediction_error=0.88, frame_quality=0.3,
        signal_magnitude=0.4, goal="restore_perceptual_capacity", top_k=2,
    )
    assert len(results) == 2
    assert results[0].similarity >= results[1].similarity
    assert results[0].goal == "restore_perceptual_capacity"


def test_find_similar_on_empty_index(tmp_path):
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    assert bio.find_similar(0.5, 0.5, 0.5, 0.5, "monitor_anomalies") == []


def test_index_is_rebuilt_from_sqlite(tmp_path):
    db = str(tmp_path / "bio.db")
    first = VectorBiography("node-a", db)
    for coherence in (0.1, 0.5, 0.9):
        first.record(_entry(coherence))

    reopened = VectorBiography("node-a", db)
    assert reopened.status["index_size"] == 3
    assert reopened.vectors.shape == (3, EMBEDDING_DIM)


def test_snapshot_round_trip(tmp_path):
    bio = VectorBiography("node-a", str(tmp_path / "bio.db"))
    bio.save_snapshot(Snapshot(
        snapshot_id="snap-1",
        timestamp=time.time(),
        threshold_theta=0.42,
        calibration_history=[{"interval": 1}],
        bia_config={"entities": []},
        current_goal="monitor_anomalies",
        functional_dispositions={"pertinence_weight": 0.8},
        node_id="node-a",
    ))
    assert [s["snapshot_id"] for s in bio.list_snapshots()] == ["snap-1"]

    loaded = bio.load_latest_snapshot()
    assert loaded.threshold_theta == 0.42
    assert loaded.functional_dispositions == {"pertinence_weight": 0.8}


def test_export_since_and_import(tmp_path):
    src = VectorBiography("node-a", str(tmp_path / "a.db"))
    dst = VectorBiography("node-b", str(tmp_path / "b.db"))
    src.record(_entry(0.4))

    events = src.export_since(0.0)
    assert len(events) == 1
    assert dst.import_foreign_events(events, "node-a") == 1
    assert dst.recent_events(10)[0]["payload"].count("_federated_from") == 1
