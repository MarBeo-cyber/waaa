"""
WAAA Architecture A — Main Entry Point
Full ML implementation: Autoencoder + RL-APTC + RandomForest + IsolationForest + VectorBiography

Run modes:
  python main_ml.py demo    — four-phase webcam demo
  python main_ml.py server  — REST API on :5001
  python main_ml.py both    — server + demo simultaneously

The demo shows the ML system learning in real time:
  - Autoencoder calibrates on NORMAL frames, then detects NOISY as anomalous
  - RL agent explores threshold adjustments and learns the reward structure
  - Goal classifier bootstraps from rules, then switches to RF predictions
  - Isolation Forest builds the healthy-state manifold
  - Vector biography accumulates embeddings for semantic recall
"""

import sys
import time
import os
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("waaa.ml.main")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

import sys
sys.path.insert(0, os.path.dirname(__file__))

from core.ml_node import MLWaaaNode, Goal
from core.bia import BIA, TargetEntity
from federation.federation import Federation
from api.ml_rest_api import create_ml_app

DIVIDER = "─" * 72


def build_bia_config() -> dict:
    return {
        "entities": [
            {
                "entity_id": "room_001",
                "name": "Room under observation",
                "priority": 1,
                "rto_seconds": 30.0,
                "rpo_seconds": 10.0,
                "mtpd_seconds": 120.0,
                "tpp_seconds": 15.0,
                "eb_hours": 0.5,
                "dependencies": [],
                "is_self": False,
                "disruption_start": None,
            },
            {
                "entity_id": "waaa_node_self",
                "name": "WAAA Node — perceptual subsystem",
                "priority": 0,
                "rto_seconds": 10.0,
                "rpo_seconds": 5.0,
                "mtpd_seconds": 60.0,
                "tpp_seconds": 5.0,
                "eb_hours": 0.1,
                "dependencies": [],
                "is_self": True,
                "disruption_start": None,
            },
        ]
    }


def print_cycle(result: dict, verbose: bool = False):
    goal_marker = " *** GOAL SWITCH ***" if result.get("goal_switched") else ""
    print(f"\n[Cycle {result['loop']:04d}] "
          f"State={result['state']} "
          f"Goal={result['goal']}{goal_marker}")

    s = result["sensor"]
    print(f"  Scene    : {s['scene_state']}")
    print(f"  Signal   : {s['signal_magnitude']:.2f}  "
          f"Coherence: {s['coherence']:.2f}  "
          f"PredErr: {s['prediction_error']:.2f}  "
          f"Quality: {s['frame_quality']:.2f}")
    print(f"  Anomaly  : {s['anomaly_score']:.2f}  "
          f"Degraded: {s['perceptually_degraded']}")

    ae = s.get("autoencoder", {})
    print(f"  Autoenc  : phase={ae.get('phase','?')}  "
          f"fitted={ae.get('is_fitted','?')}  "
          f"threshold={ae.get('anomaly_threshold', 0):.4f}")

    aptc = result["aptc"]
    print(f"  RL-APTC  : θ={aptc['theta']:.3f}  "
          f"ε={aptc['epsilon']:.3f}  "
          f"steps={aptc['total_steps']}  "
          f"reward={aptc['cumulative_reward']:.1f}")

    gc = result.get("goal_classifier", {})
    print(f"  GoalCls  : fitted={gc.get('is_fitted','?')}  "
          f"samples={gc.get('sample_count','?')}  "
          f"rf_usage={gc.get('rf_usage_pct','?')}%")

    rd = result.get("recovery_detector", {})
    print(f"  RecovDet : fitted={rd.get('is_fitted','?')}  "
          f"samples={rd.get('sample_count','?')}")

    sa = result["self_assessment"]
    print(f"  Self     : verdict={sa['verdict']}  "
          f"anomaly={sa['anomaly_score']:.2f}  "
          f"rec_level=L{sa['recommended_recovery_level']}")

    action = result["action"]
    if action["type"] != "none":
        print(f"  Action   : {action['type']}")
        if verbose and "what_changed" in action.get("details", {}):
            wc = action["details"]["what_changed"]
            print(f"    what : {wc.get('what_to_observe', '-')}")
            print(f"    how  : {wc.get('how_to_observe', '-')}")
            print(f"    why  : {wc.get('why_to_observe', '-')}")

    if result.get("goal_switched"):
        print(f"\n  {'='*62}")
        print(f"  GOAL SWITCH: {result['previous_goal']}")
        print(f"            → {result['goal']}")
        gs = result
        print(f"  Method: {result.get('goal_classifier', {}).get('current_goal', '?')}")
        print(f"  {'='*62}")


def run_demo(node: MLWaaaNode, federation: Federation):
    print(f"\n{DIVIDER}")
    print("  WAAA Architecture A — Full ML Demo")
    print(f"  Node: {node.node_id}")
    print(f"  ML components:")
    print(f"    Autoencoder (MLP)  — scene anomaly detection")
    print(f"    RL Agent (Q-table) — adaptive threshold calibration")
    print(f"    Random Forest      — goal switching classifier")
    print(f"    Isolation Forest   — recovery level detector")
    print(f"    Vector Biography   — cosine similarity episodic memory")
    print(DIVIDER)

    phases = [
        {
            "name": "Phase 1 — NORMAL: Autoencoder calibrates on normal frames",
            "scene": "NORMAL",
            "cycles": 15,
            "narrative": (
                "Autoencoder learns the distribution of normal frames. "
                "RL agent begins exploring threshold adjustments. "
                "Goal classifier accumulates bootstrap training samples. "
                "Isolation Forest builds the healthy-state manifold."
            ),
        },
        {
            "name": "Phase 2 — DIM: First perturbation — models begin to differentiate",
            "scene": "DIM",
            "cycles": 10,
            "narrative": (
                "Luminance drops. Autoencoder reconstruction error rises "
                "as frames diverge from the learned normal distribution. "
                "RL agent receives negative reward for blind intervals. "
                "Goal classifier may switch to restore_perceptual_capacity."
            ),
        },
        {
            "name": "Phase 3 — NOISY: Severe degradation — all ML models respond",
            "scene": "NOISY",
            "cycles": 12,
            "narrative": (
                "High reconstruction error from autoencoder. "
                "Isolation Forest detects the state as anomalous. "
                "Goal classifier (now using RF predictions) switches goal. "
                "RL agent learns to lower threshold to maintain sensitivity."
            ),
        },
        {
            "name": "Phase 4 — RECOVERED: Models adapt back to nominal state",
            "scene": "RECOVERED",
            "cycles": 10,
            "narrative": (
                "Light restored. Autoencoder reconstruction error drops. "
                "Isolation Forest returns to healthy-state classification. "
                "Goal classifier returns to monitor_anomalies. "
                "Vector biography now contains semantically searchable history."
            ),
        },
    ]

    all_goal_switches = []

    for phase in phases:
        print(f"\n{DIVIDER}")
        print(f"  {phase['name']}")
        print(f"  {phase['narrative']}")
        print(DIVIDER)

        node.set_scene(phase["scene"])
        time.sleep(0.1)

        for i in range(phase["cycles"]):
            result = node.tick()
            federation.sync()
            verbose = result.get("goal_switched", False)
            print_cycle(result, verbose=verbose)

            if result.get("goal_switched"):
                all_goal_switches.append({
                    "phase": phase["name"],
                    "loop": result["loop"],
                    "from": result["previous_goal"],
                    "to": result["goal"],
                    "coherence": result["sensor"]["coherence"],
                    "anomaly_score": result["sensor"]["anomaly_score"],
                })

            time.sleep(0.2)

    # ── Summary ──────────────────────────────────────────────────────── #
    print(f"\n{DIVIDER}")
    print("  DEMO SUMMARY — Architecture A")
    print(DIVIDER)
    print(f"  Total cycles         : {node.loop_count}")
    print(f"  Goal switches        : {len(all_goal_switches)}")
    print(f"  Biographical events  : {node.biography.event_count_in_window(3600)}")
    print(f"  Vector index size    : {node.biography.status['index_size']}")
    print(f"  Snapshots saved      : {len(node.biography.list_snapshots())}")
    print(f"  Recovery events      : {len(node.recovery_manager.recovery_history)}")

    # ML model states
    print(f"\n  ML Model States:")
    ae = node.sensor.scene_model.status
    print(f"    Autoencoder       : phase={ae['phase']} "
          f"fitted={ae['is_fitted']} "
          f"threshold={ae['anomaly_threshold']:.4f}")

    aptc = node.aptc.status
    print(f"    RL-APTC           : θ={aptc['theta']:.3f} "
          f"ε={aptc['epsilon']:.3f} "
          f"steps={aptc['total_steps']} "
          f"Q_nonzero={aptc['q_table_nonzero']}")

    gc = node.goal_classifier.status
    print(f"    Goal Classifier   : fitted={gc['is_fitted']} "
          f"samples={gc['sample_count']} "
          f"rf_usage={gc['rf_usage_pct']}%")

    rd = node.recovery_detector.status
    print(f"    Recovery Detector : fitted={rd['is_fitted']} "
          f"samples={rd['sample_count']}")

    if all_goal_switches:
        print(f"\n  Goal Switch Log:")
        for gs in all_goal_switches:
            print(f"    [Cycle {gs['loop']:04d}] {gs['from']}")
            print(f"             → {gs['to']}")
            print(f"             (coherence={gs['coherence']:.2f} "
                  f"anomaly={gs['anomaly_score']:.2f})")

    # Semantic memory demo
    if node.biography.status["index_size"] > 0 and node.last_reading:
        print(f"\n  Semantic Memory Demo — find_similar():")
        r = node.last_reading
        similar = node.biography.find_similar(
            coherence=0.25,
            prediction_error=0.70,
            frame_quality=0.30,
            signal_magnitude=0.65,
            goal=Goal.RESTORE_PERCEPTION,
            top_k=3,
        )
        print(f"  Query: low coherence (0.25), high error (0.70), "
              f"goal=restore_perception")
        if similar:
            for e in similar:
                print(f"    sim={e.similarity:.3f} | "
                      f"type={e.event_type} | "
                      f"goal={e.goal} | "
                      f"impact={e.impact_score:.2f}")
        else:
            print("    (no similar events yet — needs more cycles)")

    print(f"\n  REST API: http://localhost:5001")
    print(f"  Try: curl http://localhost:5001/ml/status")
    print(f"       curl http://localhost:5001/ml/similar")
    print(f"       curl -X POST http://localhost:5001/scene/NOISY")
    print(f"       curl -X POST http://localhost:5001/tick/n/20")
    print(DIVIDER)

    # Save all models at end of demo
    node._save_models()
    print(f"  Models saved to: {node.model_dir}")


def build_system():
    for path in ["/tmp/waaa_ml.db", "/tmp/waaa_ml2.db"]:
        if os.path.exists(path):
            os.remove(path)

    model_dir = "/tmp/waaa_ml_models"
    os.makedirs(model_dir, exist_ok=True)

    node = MLWaaaNode(
        node_id="waaa-ml-webcam-01",
        db_path="/tmp/waaa_ml.db",
        bia_config=build_bia_config(),
        model_dir=model_dir,
    )

    node2 = MLWaaaNode(
        node_id="waaa-ml-webcam-02",
        db_path="/tmp/waaa_ml2.db",
        bia_config=build_bia_config(),
        model_dir=model_dir + "_02",
    )

    federation = Federation("fed-ml-01")
    federation.register_node(node)
    federation.register_node(node2)

    app = create_ml_app(node, federation)
    return node, node2, federation, app


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    node, node2, federation, app = build_system()

    if mode == "demo":
        run_demo(node, federation)

    elif mode == "server":
        print(f"[WAAA-ML] REST API on http://0.0.0.0:5001")
        print(f"[WAAA-ML] Node: {node.node_id} — Architecture A")
        app.run(host="0.0.0.0", port=5001, debug=False)

    elif mode == "both":
        t = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0", port=5001, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        t.start()
        time.sleep(0.5)
        print(f"[WAAA-ML] REST API running on http://localhost:5001")
        run_demo(node, federation)
        print("\n[WAAA-ML] Server running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[WAAA-ML] Shutting down.")
    else:
        print("Usage: python main_ml.py [demo|server|both]")
        sys.exit(1)


if __name__ == "__main__":
    main()
