"""
WAAA ML — Vector Store Biographical Memory
Replaces plain SQLite event queries with embedding-based
semantic similarity search.

Instead of querying by event type or timestamp, the node
retrieves "situations similar to now" — even if they were
never explicitly categorised.

Architecture:
  - Each biographical event is embedded as a fixed-size feature vector
  - Vectors are stored in a numpy matrix (FAISS-equivalent, pure numpy)
  - Retrieval uses cosine similarity
  - SQLite is retained as the persistent backing store
  - The vector index is rebuilt from SQLite on startup
"""

import numpy as np
import logging
import time
import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("waaa.ml.vector_store")

EMBEDDING_DIM = 14


def embed_event(coherence: float,
                prediction_error: float,
                frame_quality: float,
                signal_magnitude: float,
                goal: str,
                event_type: str,
                impact_score: float,
                timestamp: float) -> np.ndarray:
    """
    Embed a biographical event as a fixed-size feature vector.

    Dimensions:
      [0]  coherence
      [1]  prediction_error
      [2]  frame_quality
      [3]  signal_magnitude
      [4]  impact_score
      [5]  1 - coherence  (degradation proxy)
      [6]  coherence * frame_quality  (joint quality)
      [7]  goal_monitor (one-hot)
      [8]  goal_restore (one-hot)
      [9]  goal_recovery (one-hot)
      [10] event_perturbation (one-hot)
      [11] event_goal_switch (one-hot)
      [12] event_degradation (one-hot)
      [13] hour_of_day_sin (seasonality)
    """
    hour = time.localtime(timestamp).tm_hour
    return np.array([
        coherence,
        prediction_error,
        frame_quality,
        signal_magnitude,
        impact_score,
        1.0 - coherence,
        coherence * frame_quality,
        float(goal == "monitor_anomalies"),
        float(goal == "restore_perceptual_capacity"),
        float(goal == "execute_recovery"),
        float(event_type == "perturbation"),
        float(event_type == "goal_switch"),
        float(event_type == "perceptual_degradation"),
        float(np.sin(2 * np.pi * hour / 24)),
    ], dtype=np.float32)


@dataclass
class SimilarEvent:
    event_id: int
    similarity: float
    event_type: str
    goal: str
    coherence: float
    prediction_error: float
    impact_score: float
    timestamp: float
    payload: dict


class VectorBiography:
    """
    Vector store + SQLite biographical memory.

    Provides two access modes:
    1. Semantic: find_similar(current_state) → events with similar structure
    2. Temporal: recent_events(n) → latest N events (from SQLite)

    The semantic mode enables the node to recall past situations by
    structural similarity rather than category — a genuine episodic
    memory that goes beyond keyword search.
    """

    MAX_INDEX_SIZE = 500   # max vectors kept in memory

    def __init__(self, node_id: str, db_path: str):
        self.node_id = node_id
        self.db_path = db_path
        self._init_db()

        # In-memory vector index
        self.vectors: np.ndarray = np.zeros(
            (0, EMBEDDING_DIM), dtype=np.float32
        )
        self.index_ids: list[int] = []   # maps row → SQLite event id

        self._rebuild_index()
        logger.info(f"[VectorBiography:{node_id}] "
                    f"Initialised with {len(self.index_ids)} vectors")

    # ------------------------------------------------------------------ #
    # DB init                                                              #
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    coherence REAL,
                    prediction_error REAL,
                    goal TEXT,
                    impact_score REAL,
                    embedding BLOB
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    threshold_theta REAL,
                    calibration_history TEXT,
                    bia_config TEXT,
                    current_goal TEXT,
                    functional_dispositions TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_node_time
                    ON events(node_id, timestamp);
            """)

    # ------------------------------------------------------------------ #
    # Record                                                               #
    # ------------------------------------------------------------------ #

    def record(self, entry) -> int:
        """Record a BiographicalEntry and add its embedding to the index."""
        embedding = embed_event(
            coherence=entry.coherence_at_event,
            prediction_error=entry.prediction_error_at_event,
            frame_quality=entry.payload.get("frame_quality", 0.5),
            signal_magnitude=entry.payload.get("signal_magnitude", 0.0),
            goal=entry.node_goal_at_event,
            event_type=entry.event_type,
            impact_score=entry.impact_score,
            timestamp=entry.timestamp,
        )

        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO events
                  (node_id, timestamp, event_type, payload,
                   coherence, prediction_error, goal, impact_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.node_id,
                entry.timestamp,
                entry.event_type,
                json.dumps(entry.payload),
                entry.coherence_at_event,
                entry.prediction_error_at_event,
                entry.node_goal_at_event,
                entry.impact_score,
                embedding.tobytes(),
            ))
            event_id = cursor.lastrowid

        self._add_to_index(event_id, embedding)
        return event_id

    # ------------------------------------------------------------------ #
    # Semantic retrieval                                                   #
    # ------------------------------------------------------------------ #

    def find_similar(self,
                     coherence: float,
                     prediction_error: float,
                     frame_quality: float,
                     signal_magnitude: float,
                     goal: str,
                     top_k: int = 5) -> list[SimilarEvent]:
        """
        Retrieve the top_k most similar past events to the current state.
        Uses cosine similarity on the embedding vectors.
        """
        if len(self.index_ids) == 0:
            return []

        query_vec = embed_event(
            coherence=coherence,
            prediction_error=prediction_error,
            frame_quality=frame_quality,
            signal_magnitude=signal_magnitude,
            goal=goal,
            event_type="query",
            impact_score=0.0,
            timestamp=time.time(),
        )

        # Cosine similarity: dot product of unit vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9
        normalised = self.vectors / norms
        similarities = normalised @ query_norm   # shape: (N,)

        # Top-k indices
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            event_id = self.index_ids[idx]
            sim = float(similarities[idx])
            event = self._load_event(event_id)
            if event:
                event.similarity = sim
                results.append(event)

        logger.debug(f"[VectorBiography] find_similar: "
                     f"top similarity={results[0].similarity:.3f}" if results else
                     "[VectorBiography] find_similar: no results")
        return results

    def _load_event(self, event_id: int) -> Optional[SimilarEvent]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE id=?", (event_id,)
            ).fetchone()
        if not row:
            return None
        return SimilarEvent(
            event_id=row["id"],
            similarity=0.0,
            event_type=row["event_type"],
            goal=row["goal"] or "",
            coherence=row["coherence"] or 0.0,
            prediction_error=row["prediction_error"] or 0.0,
            impact_score=row["impact_score"] or 0.0,
            timestamp=row["timestamp"],
            payload=json.loads(row["payload"]),
        )

    # ------------------------------------------------------------------ #
    # Index management                                                     #
    # ------------------------------------------------------------------ #

    def _add_to_index(self, event_id: int, embedding: np.ndarray):
        self.vectors = np.vstack([self.vectors, embedding.reshape(1, -1)]) \
            if self.vectors.shape[0] > 0 else embedding.reshape(1, -1)
        self.index_ids.append(event_id)

        # Trim if too large
        if len(self.index_ids) > self.MAX_INDEX_SIZE:
            self.vectors = self.vectors[-self.MAX_INDEX_SIZE:]
            self.index_ids = self.index_ids[-self.MAX_INDEX_SIZE:]

    def _rebuild_index(self):
        """Rebuild in-memory vector index from SQLite on startup."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, embedding FROM events
                WHERE node_id=? AND embedding IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """, (self.node_id, self.MAX_INDEX_SIZE)).fetchall()

        if not rows:
            return

        vecs = []
        ids  = []
        for row in reversed(rows):
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.shape[0] == EMBEDDING_DIM:
                vecs.append(emb)
                ids.append(row["id"])

        if vecs:
            self.vectors   = np.array(vecs, dtype=np.float32)
            self.index_ids = ids
            logger.info(f"[VectorBiography] Index rebuilt: {len(ids)} vectors")

    # ------------------------------------------------------------------ #
    # Temporal retrieval (SQLite)                                          #
    # ------------------------------------------------------------------ #

    def recent_events(self, n: int = 20) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, timestamp, event_type, payload,
                       coherence, prediction_error, goal, impact_score
                FROM events WHERE node_id=?
                ORDER BY timestamp DESC LIMIT ?
            """, (self.node_id, n)).fetchall()
        return [dict(r) for r in rows]

    def event_count_in_window(self, seconds: float,
                               event_type: Optional[str] = None) -> int:
        since = time.time() - seconds
        with self._conn() as conn:
            if event_type:
                row = conn.execute("""
                    SELECT COUNT(*) as cnt FROM events
                    WHERE node_id=? AND timestamp>=? AND event_type=?
                """, (self.node_id, since, event_type)).fetchone()
            else:
                row = conn.execute("""
                    SELECT COUNT(*) as cnt FROM events
                    WHERE node_id=? AND timestamp>=?
                """, (self.node_id, since)).fetchone()
        return row["cnt"]

    def recent_coherence_trend(self, n: int = 10) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT timestamp, coherence FROM events
                WHERE node_id=? AND coherence IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            """, (self.node_id, n)).fetchall()
        return [{"timestamp": r["timestamp"], "coherence": r["coherence"]}
                for r in reversed(rows)]

    def export_since(self, since_timestamp: float) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, timestamp, event_type, payload,
                       coherence, prediction_error, goal, impact_score
                FROM events WHERE node_id=? AND timestamp>?
                ORDER BY timestamp ASC
            """, (self.node_id, since_timestamp)).fetchall()
        return [dict(r) for r in rows]

    def import_foreign_events(self, events: list, source_node_id: str) -> int:
        imported = 0
        with self._conn() as conn:
            for ev in events:
                payload = json.loads(ev.get("payload", "{}"))
                payload["_federated_from"] = source_node_id
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO events
                          (node_id, timestamp, event_type, payload,
                           coherence, prediction_error, goal, impact_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        self.node_id,
                        ev["timestamp"],
                        ev["event_type"],
                        json.dumps(payload),
                        ev.get("coherence"),
                        ev.get("prediction_error"),
                        ev.get("goal"),
                        ev.get("impact_score"),
                    ))
                    imported += 1
                except Exception as e:
                    logger.warning(f"[VectorBiography] Import conflict: {e}")
        logger.info(f"[VectorBiography:{self.node_id}] "
                    f"Imported {imported} events from {source_node_id}")
        return imported

    # ------------------------------------------------------------------ #
    # Snapshot support (delegated to SQLite)                              #
    # ------------------------------------------------------------------ #

    def save_snapshot(self, snap):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO snapshots
                  (snapshot_id, node_id, timestamp, threshold_theta,
                   calibration_history, bia_config, current_goal,
                   functional_dispositions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snap.snapshot_id, snap.node_id, snap.timestamp,
                snap.threshold_theta,
                json.dumps(snap.calibration_history),
                json.dumps(snap.bia_config),
                snap.current_goal,
                json.dumps(snap.functional_dispositions),
            ))

    def load_latest_snapshot(self):
        from core.biography import Snapshot
        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM snapshots WHERE node_id=?
                ORDER BY timestamp DESC LIMIT 1
            """, (self.node_id,)).fetchone()
        if not row:
            return None
        return Snapshot(
            snapshot_id=row["snapshot_id"],
            timestamp=row["timestamp"],
            threshold_theta=row["threshold_theta"],
            calibration_history=json.loads(row["calibration_history"]),
            bia_config=json.loads(row["bia_config"]),
            current_goal=row["current_goal"],
            functional_dispositions=json.loads(row["functional_dispositions"]),
            node_id=row["node_id"],
        )

    def list_snapshots(self) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT snapshot_id, timestamp, current_goal
                FROM snapshots WHERE node_id=?
                ORDER BY timestamp DESC
            """, (self.node_id,)).fetchall()
        return [dict(r) for r in rows]

    @property
    def status(self) -> dict:
        return {
            "index_size": len(self.index_ids),
            "embedding_dim": EMBEDDING_DIM,
            "total_events": self.event_count_in_window(86400 * 365),
        }
