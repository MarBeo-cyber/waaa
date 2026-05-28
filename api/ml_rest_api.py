"""
WAAA ML — REST API (Architecture A)
Extends the base REST API with ML-specific endpoints.

Additional endpoints:
  GET  /ml/status              — status of all ML models
  GET  /ml/similar             — semantic search in biographical memory
  GET  /ml/autoencoder         — autoencoder anomaly detector status
  GET  /ml/rl_aptc             — RL agent Q-table stats
  GET  /ml/goal_classifier     — RF classifier status + feature importances
  GET  /ml/recovery_detector   — Isolation Forest status
  POST /ml/save                — persist all ML models to disk
  POST /ml/reset/<component>   — reset a specific ML model
"""

from flask import Flask, jsonify, request
import logging

logger = logging.getLogger("waaa.ml.api")


def create_ml_app(node, federation=None):
    app = Flask(__name__)
    app.json.sort_keys = False

    # ── Base endpoints (identical to base API) ──────────────────────── #

    @app.route("/")
    def index():
        return jsonify({
            "name": "WAAA REST API — Architecture A (Full ML)",
            "node_id": node.node_id,
            "version": "2.0",
            "architecture": "A",
            "ml_components": [
                "AutoencoderSceneModel",
                "RLAPTCAgent",
                "GoalClassifier (RandomForest)",
                "RecoveryLevelDetector (IsolationForest)",
                "VectorBiography (cosine similarity)",
            ],
        })

    @app.route("/status")
    def status():
        return jsonify(node.status)

    @app.route("/tick", methods=["POST"])
    def tick():
        result = node.tick()
        if federation:
            federation.sync()
        return jsonify(result)

    @app.route("/tick/n/<int:n>", methods=["POST"])
    def tick_n(n):
        n = min(n, 300)
        results = []
        for _ in range(n):
            results.append(node.tick())
            if federation:
                federation.sync()
        goal_switches = [r for r in results if r.get("goal_switched")]
        return jsonify({
            "cycles_run": n,
            "goal_switches": len(goal_switches),
            "goal_switch_events": goal_switches,
            "final_state": results[-1] if results else None,
        })

    @app.route("/scene/<state>", methods=["POST"])
    def set_scene(state):
        valid = ["NORMAL", "DIM", "NOISY", "RECOVERED"]
        state = state.upper()
        if state not in valid:
            return jsonify({"error": f"Invalid state. Valid: {valid}"}), 400
        node.set_scene(state)
        return jsonify({"scene_state": state})

    @app.route("/biography")
    def biography():
        n = request.args.get("n", 20, type=int)
        return jsonify({
            "recent_events": node.biography.recent_events(n),
            "coherence_trend": node.biography.recent_coherence_trend(),
        })

    @app.route("/biography/snapshots")
    def snapshots():
        return jsonify(node.biography.list_snapshots())

    @app.route("/snapshot", methods=["POST"])
    def force_snapshot():
        snap_id = node.force_snapshot()
        return jsonify({"snapshot_id": snap_id})

    @app.route("/aptc")
    def aptc():
        return jsonify(node.aptc.status)

    @app.route("/bia")
    def bia():
        return jsonify({
            "status": node.bia.status,
            "entities": [e.to_dict() for e in node.bia.all_entities()],
        })

    @app.route("/recovery")
    def recovery():
        return jsonify({
            "status": node.recovery_manager.status,
            "history": [
                {
                    "timestamp": ev.timestamp,
                    "level": ev.level.name,
                    "trigger": ev.trigger,
                    "biographical_loss": ev.biographical_loss,
                    "success": ev.success,
                    "duration_s": ev.duration_seconds,
                }
                for ev in node.recovery_manager.recovery_history
            ],
        })

    @app.route("/goal_log")
    def goal_log():
        return jsonify({
            "total_switches": len(node.goal_switch_log),
            "switches": node.goal_switch_log,
        })

    # ── ML-specific endpoints ───────────────────────────────────────── #

    @app.route("/ml/status")
    def ml_status():
        return jsonify({
            "autoencoder": node.sensor.scene_model.status,
            "rl_aptc": node.aptc.status,
            "goal_classifier": node.goal_classifier.status,
            "recovery_detector": node.recovery_detector.status,
            "vector_biography": node.biography.status,
        })

    @app.route("/ml/similar")
    def similar():
        """
        Semantic search: find past events similar to the current state.
        Query params: coherence, prediction_error, frame_quality,
                      signal_magnitude, goal, top_k
        """
        if node.last_reading is None:
            return jsonify({"error": "No readings yet. Run /tick first."}), 400

        r = node.last_reading
        coherence        = request.args.get("coherence",        r.coherence,        type=float)
        prediction_error = request.args.get("prediction_error", r.prediction_error, type=float)
        frame_quality    = request.args.get("frame_quality",    r.frame_quality,    type=float)
        signal_magnitude = request.args.get("signal_magnitude", r.signal_magnitude, type=float)
        goal             = request.args.get("goal",             node.current_goal)
        top_k            = request.args.get("top_k",            5,                  type=int)

        similar_events = node.biography.find_similar(
            coherence=coherence,
            prediction_error=prediction_error,
            frame_quality=frame_quality,
            signal_magnitude=signal_magnitude,
            goal=goal,
            top_k=top_k,
        )

        return jsonify({
            "query": {
                "coherence": coherence,
                "prediction_error": prediction_error,
                "frame_quality": frame_quality,
                "signal_magnitude": signal_magnitude,
                "goal": goal,
            },
            "results": [
                {
                    "similarity": round(e.similarity, 4),
                    "event_type": e.event_type,
                    "goal": e.goal,
                    "coherence": e.coherence,
                    "prediction_error": e.prediction_error,
                    "impact_score": e.impact_score,
                    "timestamp": e.timestamp,
                }
                for e in similar_events
            ],
        })

    @app.route("/ml/autoencoder")
    def autoencoder():
        return jsonify(node.sensor.scene_model.status)

    @app.route("/ml/rl_aptc")
    def rl_aptc():
        s = node.aptc.status
        s["q_table_shape"] = list(node.aptc.Q.shape)
        return jsonify(s)

    @app.route("/ml/goal_classifier")
    def goal_classifier():
        return jsonify(node.goal_classifier.status)

    @app.route("/ml/recovery_detector")
    def recovery_detector():
        return jsonify(node.recovery_detector.status)

    @app.route("/ml/save", methods=["POST"])
    def save_models():
        node._save_models()
        return jsonify({
            "message": "All ML models saved",
            "model_dir": node.model_dir,
        })

    @app.route("/ml/reset/<component>", methods=["POST"])
    def reset_component(component):
        valid = ["autoencoder", "rl_aptc", "goal_classifier",
                 "recovery_detector", "vector_biography"]
        if component not in valid:
            return jsonify({"error": f"Valid components: {valid}"}), 400

        if component == "autoencoder":
            from ml.autoencoder_scene import AutoencoderSceneModel
            node.sensor.scene_model = AutoencoderSceneModel()
            msg = "Autoencoder reset — recalibrating from next frames"
        elif component == "rl_aptc":
            from ml.rl_aptc import RLAPTCAgent, RLAPTCConfig
            node.aptc = RLAPTCAgent(config=RLAPTCConfig())
            msg = "RL-APTC reset — Q-table cleared"
        elif component == "goal_classifier":
            from ml.goal_classifier import GoalClassifier
            node.goal_classifier = GoalClassifier()
            msg = "Goal classifier reset — bootstrap phase"
        elif component == "recovery_detector":
            from ml.recovery_detector import RecoveryLevelDetector
            node.recovery_detector = RecoveryLevelDetector()
            msg = "Recovery detector reset — recalibrating baseline"
        else:
            msg = "Vector biography index rebuilt"
            node.biography._rebuild_index()

        return jsonify({"message": msg, "component": component})

    # ── Federation endpoints ─────────────────────────────────────────── #

    if federation:
        @app.route("/federation")
        def fed_status():
            return jsonify(federation.status)

        @app.route("/federation/isolate/<node_id>", methods=["POST"])
        def isolate(node_id):
            if node_id not in federation.nodes:
                return jsonify({"error": f"Node '{node_id}' not found"}), 404
            federation.isolate_node(node_id)
            return jsonify({"message": f"Node '{node_id}' isolated"})

        @app.route("/federation/restore/<node_id>", methods=["POST"])
        def restore(node_id):
            if node_id not in federation.nodes:
                return jsonify({"error": f"Node '{node_id}' not found"}), 404
            federation.restore_node(node_id)
            return jsonify({"message": f"Node '{node_id}' restored. BRP executed."})

        @app.route("/federation/sync", methods=["POST"])
        def fed_sync():
            federation.last_sync = 0
            federation.sync()
            return jsonify({"message": "Federation sync executed"})

    return app
