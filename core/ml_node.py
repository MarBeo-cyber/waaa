"""
WAAA ML — Node (Architecture A)
Integrates all five ML modules replacing rule-based components:

  SceneModel      → AutoencoderSceneModel     (ml/autoencoder_scene.py)
  APTC            → RLAPTCAgent               (ml/rl_aptc.py)
  Goal switching  → GoalClassifier            (ml/goal_classifier.py)
  Recovery assess → RecoveryLevelDetector     (ml/recovery_detector.py)
  Biography query → VectorBiography           (ml/vector_biography.py)

The cognitive loop structure is identical to the base node.
What changes is the intelligence behind each decision:
rules → learned models.
"""

import os
import tempfile
import time
import uuid
import logging
import threading
from typing import Optional

from core.biography import BiographicalEntry, Snapshot
from core.bia import BIA
from core.recovery import RecoveryManager, RecoveryLevel
from ml.rl_aptc import RLAPTCAgent, RLAPTCConfig
from ml.goal_classifier import GoalClassifier
from ml.recovery_detector import RecoveryLevelDetector
from ml.vector_biography import VectorBiography
from sensors.synthetic_scene_sensor import SensorReading, SyntheticSceneSensor

# I percorsi di default vivono nella directory temporanea DEL SISTEMA.
# Scrivere "/tmp/..." a mano funziona su Linux e su Windows finisce in
# C:\tmp\, che spesso non esiste: il file non si apre e l'errore arriva
# lontano dalla causa. tempfile.gettempdir() risolve la cosa ovunque.
DEFAULT_DB_PATH   = os.path.join(tempfile.gettempdir(), "waaa_ml.db")
DEFAULT_MODEL_DIR = os.path.join(tempfile.gettempdir(), "waaa_models")

logger = logging.getLogger("waaa.ml.node")


class Goal:
    MONITOR_ANOMALIES   = "monitor_anomalies"
    RESTORE_PERCEPTION  = "restore_perceptual_capacity"
    EXECUTE_RECOVERY    = "execute_recovery"


class NodeState:
    MONITORING          = "MONITORING"
    SELF_DIAGNOSING     = "SELF_DIAGNOSING"
    RESTORING_PERCEPTION= "RESTORING_PERCEPTION"
    RECOVERING          = "RECOVERING"
    ISOLATED            = "ISOLATED"


DEFAULT_DISPOSITIONS = {
    "pertinence_weight":       0.8,
    "non_contradiction_floor": 0.6,
    "proportionality_factor":  1.0,
    "continuity_priority":     0.9,
}

SNAPSHOT_INTERVAL_S = 60.0


class MLWaaaNode:
    """
    WAAA Node — Architecture A (Full ML).

    Identical cognitive loop to the base WaaaNode.
    All decision components replaced with learned models.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        db_path: str = DEFAULT_DB_PATH,
        bia_config: Optional[dict] = None,
        model_dir: str = DEFAULT_MODEL_DIR,
    ):
        self.node_id = node_id or f"waaa-ml-{uuid.uuid4().hex[:8]}"
        self.model_dir = model_dir

        import os
        os.makedirs(model_dir, exist_ok=True)

        # ── ML: Vector Biography (replaces Biography) ─────────────────
        self.biography = VectorBiography(self.node_id, db_path)

        # ── ML: RL-APTC (replaces APTC) ───────────────────────────────
        self.aptc = RLAPTCAgent(
            config=RLAPTCConfig(
                theta_0=0.35,
                observation_interval_s=12.0,
                n_min=1,
                epsilon_start=0.40,
                epsilon_decay=0.995,
            ),
            model_path=f"{model_dir}/rl_aptc.pkl",
        )

        # ── ML: Goal Classifier (replaces if/elif goal switching) ──────
        self.goal_classifier = GoalClassifier(
            model_path=f"{model_dir}/goal_classifier.pkl",
        )

        # ── ML: Recovery Detector (replaces threshold logic) ───────────
        self.recovery_detector = RecoveryLevelDetector(
            model_path=f"{model_dir}/recovery_detector.pkl",
        )

        # ── BIA (unchanged — human-defined value system) ───────────────
        self.bia = BIA(bia_config)

        # ── Recovery Manager (unchanged — structural) ──────────────────
        initial_config = {
            "goal": Goal.MONITOR_ANOMALIES,
            "functional_dispositions": DEFAULT_DISPOSITIONS.copy(),
            "telos": "guarantee_operational_continuity_of_target_entities",
        }
        self.recovery_manager = RecoveryManager(self.node_id, initial_config)

        # ── Sensor (synthetic frames) with Autoencoder ─────────────────
        self.sensor = SyntheticSceneSensor(
            self.node_id,
            model_path=f"{model_dir}/autoencoder.pkl",
        )

        # ── Runtime state ──────────────────────────────────────────────
        self.telos            = initial_config["telos"]
        self.state            = NodeState.MONITORING
        self.current_goal     = Goal.MONITOR_ANOMALIES
        self.functional_dispositions = DEFAULT_DISPOSITIONS.copy()
        self.last_reading: Optional[SensorReading] = None
        self.last_snapshot_time: float = 0.0
        self.goal_start_time: float = time.time()
        self.loop_count: int = 0
        self.goal_switch_log: list = []
        self._lock = threading.Lock()

        logger.info(f"[MLNode:{self.node_id}] Initialised — Architecture A (Full ML)")

    # ------------------------------------------------------------------ #
    # Main cognitive loop                                                  #
    # ------------------------------------------------------------------ #

    def tick(self) -> dict:
        with self._lock:
            self.loop_count += 1
            cycle_start = time.time()

            # 1. SENSE
            reading = self.sensor.read()
            self.last_reading = reading

            # 2. EVALUATE — RL-APTC (passes coherence for reward signal)
            is_external_perturbation = self.aptc.evaluate(
                reading.signal_magnitude,
                coherence=reading.coherence,
            )

            # 3. SELF-MONITOR — Recovery Detector (Isolation Forest)
            self_assessment = self._assess_own_state(reading)

            # 4. DECIDE GOAL — Goal Classifier (Random Forest)
            previous_goal = self.current_goal
            self._update_goal(reading, self_assessment)
            goal_switched = self.current_goal != previous_goal

            # 5. ACT
            action_taken = self._act(reading, self_assessment,
                                     is_external_perturbation)

            # 6. RECORD to Vector Biography
            self._record_if_significant(
                reading, is_external_perturbation,
                goal_switched, previous_goal, action_taken
            )

            # 7. SNAPSHOT
            self._maybe_snapshot()

            # 8. PERIODIC MODEL SAVE
            if self.loop_count % 50 == 0:
                self._save_models()

            return {
                "node_id": self.node_id,
                "loop": self.loop_count,
                "timestamp": cycle_start,
                "state": self.state,
                "goal": self.current_goal,
                "goal_switched": goal_switched,
                "previous_goal": previous_goal if goal_switched else None,
                "sensor": {
                    "scene_state": self.sensor.current_scene_state,
                    "signal_magnitude": reading.signal_magnitude,
                    "coherence": reading.coherence,
                    "prediction_error": reading.prediction_error,
                    "frame_quality": reading.frame_quality,
                    "luminance": reading.luminance,
                    "noise_level": reading.noise_level,
                    "anomaly_score": reading.anomaly_score,
                    "perceptually_degraded": reading.is_perceptually_degraded,
                    "scene_model_status": reading.scene_model_status,
                    "autoencoder": self.sensor.scene_model.status,
                },
                "aptc": self.aptc.status,
                "self_assessment": self_assessment,
                "external_perturbation": is_external_perturbation,
                "action": action_taken,
                "goal_classifier": self.goal_classifier.status,
                "recovery_detector": self.recovery_detector.status,
                "vector_biography": self.biography.status,
                "cycle_ms": round((time.time() - cycle_start) * 1000, 1),
            }

    # ------------------------------------------------------------------ #
    # Self-monitoring — Isolation Forest                                   #
    # ------------------------------------------------------------------ #

    def _assess_own_state(self, reading: SensorReading) -> dict:
        goal_duration_s = time.time() - self.goal_start_time

        level, anomaly_score = self.recovery_detector.assess(
            coherence=reading.coherence,
            prediction_error=reading.prediction_error,
            frame_quality=reading.frame_quality,
            aptc_theta=self.aptc.theta,
            blind_streak=self.aptc.state.blind_streak,
            goal_duration_s=goal_duration_s,
        )

        # Map level to verdict label
        verdict_map = {
            -1: "nominal",
            0:  "mild_degradation",
            1:  "moderate_degradation",
            2:  "severe_degradation",
            3:  "critical_degradation",
        }

        return {
            "verdict": verdict_map.get(level, "nominal"),
            "severity": round(anomaly_score, 3),
            "coherence": reading.coherence,
            "prediction_error": reading.prediction_error,
            "frame_quality": reading.frame_quality,
            "anomaly_score": round(anomaly_score, 3),
            "recommended_recovery_level": level,
            "perceptual_capacity_ok": level == -1,
            "detector_fitted": self.recovery_detector.is_fitted,
        }

    # ------------------------------------------------------------------ #
    # Goal switching — Random Forest Classifier                           #
    # ------------------------------------------------------------------ #

    def _update_goal(self, reading: SensorReading, self_assessment: dict):
        goal, method, confidence = self.goal_classifier.predict(
            coherence=reading.coherence,
            prediction_error=reading.prediction_error,
            frame_quality=reading.frame_quality,
            signal_magnitude=reading.signal_magnitude,
            aptc_theta=self.aptc.theta,
        )

        # Map goal to node state
        state_map = {
            Goal.MONITOR_ANOMALIES:  NodeState.MONITORING,
            Goal.RESTORE_PERCEPTION: NodeState.RESTORING_PERCEPTION,
            Goal.EXECUTE_RECOVERY:   NodeState.RECOVERING,
        }

        previous = self.current_goal
        if goal != self.current_goal:
            self.current_goal = goal
            self.state = state_map.get(goal, NodeState.MONITORING)
            self.goal_classifier.update_current_goal(goal)
            self.goal_start_time = time.time()

            switch = {
                "timestamp": time.time(),
                "from": previous,
                "to": goal,
                "method": method,
                "confidence": round(confidence, 3),
                "coherence": reading.coherence,
                "loop": self.loop_count,
            }
            self.goal_switch_log.append(switch)
            logger.warning(
                f"[MLNode:{self.node_id}] *** GOAL SWITCH *** "
                f"{previous} → {goal} "
                f"[{method} conf={confidence:.2f}]"
            )
        else:
            self.state = state_map.get(goal, NodeState.MONITORING)

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def _act(self, reading: SensorReading,
             self_assessment: dict,
             external_perturbation: bool) -> dict:

        if self.current_goal == Goal.MONITOR_ANOMALIES:
            if external_perturbation:
                # Semantic memory: find similar past perturbations
                similar = self.biography.find_similar(
                    coherence=reading.coherence,
                    prediction_error=reading.prediction_error,
                    frame_quality=reading.frame_quality,
                    signal_magnitude=reading.signal_magnitude,
                    goal=self.current_goal,
                    top_k=3,
                )
                return {
                    "type": "report_external_perturbation",
                    "details": {
                        "signal_magnitude": reading.signal_magnitude,
                        "anomaly_score": reading.anomaly_score,
                        "aptc_theta": self.aptc.theta,
                        "similar_past_events": [
                            {
                                "similarity": round(e.similarity, 3),
                                "event_type": e.event_type,
                                "goal": e.goal,
                                "coherence": e.coherence,
                                "impact": e.impact_score,
                            }
                            for e in similar
                        ],
                    }
                }
            return {"type": "none", "details": {}}

        elif self.current_goal == Goal.RESTORE_PERCEPTION:
            return self._restore_perception(reading)

        elif self.current_goal == Goal.EXECUTE_RECOVERY:
            return self._execute_recovery(self_assessment)

        return {"type": "none", "details": {}}

    def _restore_perception(self, reading: SensorReading) -> dict:
        adjustments = []

        # Relax noise filter
        old_filter = self.sensor._noise_filter
        new_filter = max(0.2, old_filter - 0.05)
        if new_filter != old_filter:
            self.sensor._noise_filter = new_filter
            adjustments.append(f"noise_filter: {old_filter:.2f}→{new_filter:.2f}")

        # Signal to autoencoder: low-light is now expected
        if reading.luminance < 0.35:
            # Add current frame to buffer to update expectations
            self.sensor.scene_model.frame_buffer.append(
                self.sensor.scene_model.frame_buffer[-1]
                if self.sensor.scene_model.frame_buffer else
                [reading.luminance] * 12
            )
            adjustments.append("updated_autoencoder_expectation")

        # Mark self as disrupted in BIA
        for entity in self.bia.all_entities():
            if entity.is_self and entity.disruption_start is None:
                entity.mark_disrupted()
                adjustments.append("self_marked_disrupted_in_bia")

        return {
            "type": "restore_perception",
            "details": {
                "adjustments": adjustments,
                "what_changed": {
                    "what_to_observe": "luminance_and_anomaly_score_primary",
                    "how_to_observe": "relaxed_noise_filter_autoencoder_recalibrating",
                    "why_to_observe": "restore_perceptual_capacity_before_monitoring",
                },
                "autoencoder_phase": self.sensor.scene_model.phase,
            }
        }

    def _execute_recovery(self, self_assessment: dict) -> dict:
        level_int = self_assessment["recommended_recovery_level"]
        level = RecoveryLevel(max(level_int, 0))

        if level == RecoveryLevel.LEVEL0:
            result = self.recovery_manager.execute_level0(
                {"sensor_noise_filter": self.sensor._noise_filter}
            )
            self.sensor._noise_filter = result.get(
                "sensor_noise_filter", self.sensor._noise_filter
            )
        elif level == RecoveryLevel.LEVEL1:
            result = self.recovery_manager.execute_level1(
                self.biography, self.aptc
            )
        elif level == RecoveryLevel.LEVEL2:
            result = self.recovery_manager.execute_level2(
                self.biography, self.aptc
            )
        else:
            result = self.recovery_manager.execute_level3(self.aptc)
            # Do not force goal here — let the classifier decide next cycle
            if result:
                self.functional_dispositions = result.get(
                    "functional_dispositions", DEFAULT_DISPOSITIONS.copy()
                )

        return {"type": f"recovery_level_{level.value}", "details": result or {}}

    # ------------------------------------------------------------------ #
    # Biography recording                                                  #
    # ------------------------------------------------------------------ #

    def _record_if_significant(self, reading, external_perturbation,
                                goal_switched, previous_goal, action):
        significant = (
            goal_switched or
            external_perturbation or
            reading.is_perceptually_degraded or
            reading.anomaly_score > 0.5 or
            action["type"].startswith("recovery")
        )
        if not significant:
            return

        impact = max(
            reading.anomaly_score,
            0.8 if goal_switched else 0.0,
            reading.signal_magnitude if external_perturbation else 0.0,
        )

        entry = BiographicalEntry(
            timestamp=time.time(),
            event_type=(
                "goal_switch" if goal_switched else
                "perturbation" if external_perturbation else
                "perceptual_degradation" if reading.is_perceptually_degraded else
                "recovery"
            ),
            payload={
                "scene_state": self.sensor.current_scene_state,
                "goal": self.current_goal,
                "previous_goal": previous_goal if goal_switched else None,
                "action": action,
                "external_perturbation": external_perturbation,
                "frame_quality": reading.frame_quality,
                "signal_magnitude": reading.signal_magnitude,
                "anomaly_score": reading.anomaly_score,
            },
            coherence_at_event=reading.coherence,
            prediction_error_at_event=reading.prediction_error,
            node_goal_at_event=self.current_goal,
            impact_score=round(impact, 3),
        )
        self.biography.record(entry)

    # ------------------------------------------------------------------ #
    # Snapshot                                                             #
    # ------------------------------------------------------------------ #

    def _maybe_snapshot(self):
        if time.time() - self.last_snapshot_time < SNAPSHOT_INTERVAL_S:
            return
        snap = Snapshot(
            snapshot_id=f"snap-ml-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            threshold_theta=self.aptc.theta,
            calibration_history=self.aptc.state.calibration_log[-20:],
            bia_config=self.bia.export_config(),
            current_goal=self.current_goal,
            functional_dispositions=self.functional_dispositions.copy(),
            node_id=self.node_id,
        )
        self.biography.save_snapshot(snap)
        self.last_snapshot_time = time.time()
        logger.info(f"[MLNode:{self.node_id}] Snapshot: {snap.snapshot_id}")

    def force_snapshot(self) -> str:
        self.last_snapshot_time = 0
        self._maybe_snapshot()
        snaps = self.biography.list_snapshots()
        return snaps[0]["snapshot_id"] if snaps else "none"

    # ------------------------------------------------------------------ #
    # Model persistence                                                    #
    # ------------------------------------------------------------------ #

    def _save_models(self):
        try:
            self.aptc.save(f"{self.model_dir}/rl_aptc.pkl")
            self.goal_classifier.save(f"{self.model_dir}/goal_classifier.pkl")
            self.recovery_detector.save(f"{self.model_dir}/recovery_detector.pkl")
            self.sensor.save_model(f"{self.model_dir}/autoencoder.pkl")
            logger.info(f"[MLNode:{self.node_id}] Models saved to {self.model_dir}")
        except Exception as e:
            logger.warning(f"[MLNode:{self.node_id}] Model save failed: {e}")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def set_scene(self, scene_state: str):
        self.sensor.set_scene_state(scene_state)

    @property
    def status(self) -> dict:
        return {
            "node_id": self.node_id,
            "architecture": "A — Full ML",
            "state": self.state,
            "goal": self.current_goal,
            "telos": self.telos,
            "loop_count": self.loop_count,
            "aptc": self.aptc.status,
            "bia": self.bia.status,
            "recovery": self.recovery_manager.status,
            "sensor": self.sensor.status,
            "goal_classifier": self.goal_classifier.status,
            "recovery_detector": self.recovery_detector.status,
            "vector_biography": self.biography.status,
            "goal_switches": len(self.goal_switch_log),
            "last_goal_switch": self.goal_switch_log[-1]
                                if self.goal_switch_log else None,
            "snapshots": self.biography.list_snapshots(),
        }
