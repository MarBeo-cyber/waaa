"""
WAAA Architecture A — Main Entry Point
Autoencoder + RL-APTC + rule-based goal selection + IsolationForest +
VectorBiography.

Run modes:
  python main_ml.py demo    — four-phase synthetic-scene demo
  python main_ml.py server  — REST API on :5001
  python main_ml.py both    — server + demo simultaneously

What the demo actually shows:
  - The autoencoder collects 20 frames, fits, and then reports genuine
    reconstruction error (before that it reports "warming up")
  - The RL agent closes an observation interval every 12 evaluations,
    updates its Q-table and moves θ
  - The goal selector applies its documented rule to every reading
  - The Isolation Forest builds a healthy-state manifold and flags
    states that fall off it
  - The vector biography accumulates embeddings for semantic recall

All sensor frames are synthesised in-process. No camera is read.
"""

import logging
import os
import sys
import tempfile
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("waaa.ml.main")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.ml_rest_api import create_ml_app  # noqa: E402
from core.ml_node import Goal, MLWaaaNode  # noqa: E402
from federation.federation import Federation  # noqa: E402

DIVIDER = "─" * 72

SYNTHETIC_DATA_BANNER = (
    "All sensor data is synthetic: frames are generated in-process by "
    "sensors/synthetic_scene_sensor.py. No camera is read."
)

# Where the trained models are written. Overridable so the Docker image
# and scripts/run_ml.sh can point at the mounted ./waaa_models volume.
# I percorsi di default vivono nella directory temporanea DEL SISTEMA.
# Scrivere "/tmp/..." a mano funziona su Linux e su Windows finisce in
# C:\tmp\, che spesso non esiste: il file non si apre e l'errore arriva
# lontano dalla causa. tempfile.gettempdir() risolve la cosa ovunque.
DB_PATH_1 = os.path.join(tempfile.gettempdir(), "waaa_ml.db")
DB_PATH_2 = os.path.join(tempfile.gettempdir(), "waaa_ml2.db")

DEFAULT_MODEL_DIR = os.environ.get(
    "WAAA_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "waaa_models"),
)


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
          f"threshold={ae.get('anomaly_threshold', 0):.4f}  "
          f"source={s.get('scene_model_status','?')}")

    aptc = result["aptc"]
    print(f"  RL-APTC  : θ={aptc['theta']:.3f}  "
          f"ε={aptc['epsilon']:.3f}  "
          f"steps={aptc['total_steps']}  "
          f"reward={aptc['cumulative_reward']:.1f}")

    gc = result.get("goal_classifier", {})
    print(f"  GoalSel  : method={gc.get('method','?')}  "
          f"decisions={gc.get('total_predictions','?')}")

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
        print(f"  Method: {result.get('goal_classifier', {}).get('method', '?')}")
        print(f"  {'='*62}")


def run_demo(node: MLWaaaNode, federation: Federation):
    print(f"\n{DIVIDER}")
    print("  WAAA Architecture A — Demo")
    print(f"  Node: {node.node_id}")
    print(f"  {SYNTHETIC_DATA_BANNER}")
    print("  Components:")
    print("    Autoencoder (MLP)  — scene anomaly detection (learned)")
    print("    RL Agent (Q-table) — adaptive threshold calibration (learned)")
    print("    Isolation Forest   — healthy-manifold detection (learned)")
    print("    Goal selector      — deterministic rule, not a model")
    print("    Vector Biography   — cosine similarity episodic memory")
    print(DIVIDER)

    phases = [
        {
            "name": "Phase 1 — NORMAL: Autoencoder calibrates on normal frames",
            "scene": "NORMAL",
            "cycles": 15,
            "narrative": (
                "Autoencoder collects its 20 calibration frames, then fits. "
                "Until it does, readings are marked warming_up and carry "
                "direct frame statistics, not model output. "
                "RL agent begins exploring threshold adjustments. "
                "Isolation Forest accumulates healthy samples."
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
                "The goal rule may switch to restore_perceptual_capacity."
            ),
        },
        {
            "name": "Phase 3 — NOISY: Severe degradation — all ML models respond",
            "scene": "NOISY",
            "cycles": 12,
            "narrative": (
                "High reconstruction error from autoencoder. "
                "Isolation Forest detects the state as off-manifold and "
                "the severity ladder picks the recovery level. "
                "RL agent keeps updating Q-values from interval rewards; "
                "which way θ moves in 47 cycles is not predetermined."
            ),
        },
        {
            "name": "Phase 4 — RECOVERED: Models adapt back to nominal state",
            "scene": "RECOVERED",
            "cycles": 10,
            "narrative": (
                "Light restored. Autoencoder reconstruction error drops. "
                "Isolation Forest returns to healthy-state classification. "
                "The goal rule returns to monitor_anomalies. "
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

        for _ in range(phase["cycles"]):
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

    # Model states
    print("\n  Model States:")
    ae = node.sensor.scene_model.status
    print(f"    Autoencoder       : phase={ae['phase']} "
          f"fitted={ae['is_fitted']} "
          f"threshold={ae['anomaly_threshold']:.4f}")

    aptc = node.aptc.status
    print(f"    RL-APTC           : θ={aptc['theta']:.3f} "
          f"ε={aptc['epsilon']:.3f} "
          f"steps={aptc['total_steps']} "
          f"intervals={aptc['total_intervals']} "
          f"ema_rate={aptc['ema_event_rate']:.2f} "
          f"Q_nonzero={aptc['q_table_nonzero']}")

    gc = node.goal_classifier.status
    print(f"    Goal selector     : method={gc['method']} "
          f"decisions={gc['total_predictions']} "
          f"(deterministic rule, no model)")

    rd = node.recovery_detector.status
    print(f"    Recovery Detector : fitted={rd['is_fitted']} "
          f"samples={rd['sample_count']}")

    print(f"\n  {SYNTHETIC_DATA_BANNER}")

    if all_goal_switches:
        print("\n  Goal Switch Log:")
        for gs in all_goal_switches:
            print(f"    [Cycle {gs['loop']:04d}] {gs['from']}")
            print(f"             → {gs['to']}")
            print(f"             (coherence={gs['coherence']:.2f} "
                  f"anomaly={gs['anomaly_score']:.2f})")

    # Semantic memory demo
    if node.biography.status["index_size"] > 0 and node.last_reading:
        print("\n  Semantic Memory Demo — find_similar():")
        similar = node.biography.find_similar(
            coherence=0.25,
            prediction_error=0.70,
            frame_quality=0.30,
            signal_magnitude=0.65,
            goal=Goal.RESTORE_PERCEPTION,
            top_k=3,
        )
        print("  Query: low coherence (0.25), high error (0.70), "
              "goal=restore_perception")
        if similar:
            for e in similar:
                print(f"    sim={e.similarity:.3f} | "
                      f"type={e.event_type} | "
                      f"goal={e.goal} | "
                      f"impact={e.impact_score:.2f}")
        else:
            print("    (no similar events yet — needs more cycles)")

    print("\n  REST API: http://localhost:5001")
    print("  Try: curl http://localhost:5001/ml/status")
    print("       curl http://localhost:5001/ml/similar")
    print("       curl -X POST http://localhost:5001/scene/NOISY")
    print("       curl -X POST http://localhost:5001/tick/n/20")
    print(DIVIDER)

    # Save all models at end of demo
    node._save_models()
    print(f"  Models saved to: {node.model_dir}")


def build_system(model_dir: str = DEFAULT_MODEL_DIR):
    for path in [DB_PATH_1, DB_PATH_2]:
        if os.path.exists(path):
            os.remove(path)

    os.makedirs(model_dir, exist_ok=True)

    node = MLWaaaNode(
        node_id="waaa-ml-webcam-01",
        db_path=DB_PATH_1,
        bia_config=build_bia_config(),
        model_dir=model_dir,
    )

    node2 = MLWaaaNode(
        node_id="waaa-ml-webcam-02",
        db_path=DB_PATH_2,
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
        print("[WAAA-ML] REST API on http://0.0.0.0:5001")
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
        print("[WAAA-ML] REST API running on http://localhost:5001")
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
