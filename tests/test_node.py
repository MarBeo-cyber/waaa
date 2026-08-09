"""End-to-end tests for MLWaaaNode.

The node, main_ml.py and the REST API were unreachable: core/ml_node.py
imported core.biography, core.bia and core.recovery, and main_ml.py
imported federation.federation — none of which existed in the repository.
Every documented entry point died with ModuleNotFoundError. The import
test below is the regression guard for that.
"""

import pytest

from core.bia import BIA, TargetEntity
from core.ml_node import Goal, MLWaaaNode, NodeState


def _node(tmp_path, node_id="test-node"):
    return MLWaaaNode(
        node_id=node_id,
        db_path=str(tmp_path / "node.db"),
        bia_config={"entities": [
            {"entity_id": "room_001", "name": "Room", "priority": 1,
             "is_self": False},
            {"entity_id": "self", "name": "Node", "priority": 0,
             "is_self": True},
        ]},
        model_dir=str(tmp_path / "models"),
    )


def test_every_documented_entry_point_imports():
    import api.ml_rest_api
    import core.ml_node
    import federation.federation
    import main_ml

    assert callable(main_ml.main)
    assert callable(api.ml_rest_api.create_ml_app)
    assert callable(core.ml_node.MLWaaaNode)
    assert callable(federation.federation.Federation)


def test_single_tick(tmp_path):
    node = _node(tmp_path)
    result = node.tick()

    assert result["loop"] == 1
    assert result["goal"] in vars(Goal).values()
    assert result["state"] in vars(NodeState).values()
    assert result["sensor"]["scene_model_status"] == "warming_up"
    assert 0.0 <= result["sensor"]["coherence"] <= 1.0


def test_scene_model_becomes_active_and_is_labelled(tmp_path):
    node = _node(tmp_path)
    statuses = [node.tick()["sensor"]["scene_model_status"] for _ in range(25)]
    assert statuses[0] == "warming_up"
    assert statuses[-1] == "active"
    assert node.sensor.scene_model.is_fitted is True


def test_many_ticks_across_scenes(tmp_path):
    node = _node(tmp_path)
    for scene in ("NORMAL", "DIM", "NOISY", "RECOVERED"):
        node.set_scene(scene)
        for _ in range(12):
            result = node.tick()
            assert result["cycle_ms"] >= 0

    assert node.loop_count == 48
    assert node.status["architecture"] == "A — Full ML"
    assert node.biography.status["index_size"] > 0


def test_degraded_scene_drives_a_goal_switch(tmp_path):
    node = _node(tmp_path)
    node.set_scene("NORMAL")
    for _ in range(25):
        node.tick()
    node.set_scene("NOISY")
    for _ in range(15):
        node.tick()

    assert node.goal_switch_log, "a severe scene must move the node off monitoring"
    assert all(sw["method"] == "rule" for sw in node.goal_switch_log)


def test_sensor_frames_are_synthetic(tmp_path):
    node = _node(tmp_path)
    node.tick()
    assert node.sensor.status["frames_are_synthetic"] is True


def test_snapshot_is_persisted(tmp_path):
    node = _node(tmp_path)
    node.tick()
    snapshot_id = node.force_snapshot()
    assert snapshot_id != "none"
    assert node.biography.load_latest_snapshot().node_id == "test-node"


def test_models_are_saved_to_the_configured_dir(tmp_path):
    node = _node(tmp_path)
    for _ in range(22):
        node.tick()
    node._save_models()

    saved = {p.name for p in (tmp_path / "models").iterdir()}
    assert {"rl_aptc.pkl", "goal_classifier.pkl",
            "recovery_detector.pkl", "autoencoder.pkl"} <= saved


def test_bia_entities_are_registered(tmp_path):
    node = _node(tmp_path)
    assert node.bia.status["entity_count"] == 2
    assert [e.entity_id for e in node.bia.self_entities()] == ["self"]


def test_bia_config_round_trip():
    bia = BIA({"entities": [{"entity_id": "e1", "rto_seconds": 30.0}]})
    rebuilt = BIA(bia.export_config())
    assert rebuilt.get("e1").rto_seconds == 30.0


def test_entity_disruption_bookkeeping():
    entity = TargetEntity(entity_id="e1", mtpd_seconds=100.0)
    assert entity.is_disrupted is False
    assert entity.residual_mtpd() is None

    entity.mark_disrupted(when=0.0)
    first = entity.disruption_start
    entity.mark_disrupted(when=999.0)
    assert entity.disruption_start == first, "an open disruption is not restarted"

    entity.mark_restored()
    assert entity.is_disrupted is False


def test_rest_app_builds_and_serves_status(tmp_path):
    from api.ml_rest_api import create_ml_app
    from federation.federation import Federation

    node = _node(tmp_path)
    fed = Federation("fed-test")
    fed.register_node(node)
    app = create_ml_app(node, fed)

    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/status").status_code == 200
    assert client.get("/ml/status").status_code == 200
    assert client.get("/federation").json["node_count"] == 1

    assert client.get("/ml/similar").status_code == 400  # no readings yet
    assert client.post("/tick").status_code == 200
    assert client.get("/ml/similar").status_code == 200
    assert client.post("/scene/NOISY").json["scene_state"] == "NOISY"
    assert client.post("/scene/BOGUS").status_code == 400


@pytest.mark.parametrize("scene", ["NORMAL", "DIM", "NOISY", "RECOVERED"])
def test_all_scene_states_are_accepted(tmp_path, scene):
    node = _node(tmp_path)
    node.set_scene(scene)
    assert node.tick()["sensor"]["scene_state"] == scene
