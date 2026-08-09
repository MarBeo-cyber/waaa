"""
WAAA — Federation of nodes (in-process).

What this module does: it keeps a registry of nodes that live in the same
Python process and periodically reconciles their biographies — every node
imports the biographical events the others recorded while it was not
looking. This is the Biographical Reconciliation Procedure (BRP) used by
``main_ml.py`` and by the ``/federation`` REST endpoints.

What this module does NOT do: there is no network transport, no
discovery, no consensus, no conflict resolution beyond timestamp
watermarks, and no federated learning — models are not shared, only
recorded events are. Anything in the docs about federated cognition
beyond this is design, not code.
"""

import json
import logging
import time

logger = logging.getLogger("waaa.federation")

# Minimum wall-clock seconds between two automatic reconciliations.
SYNC_INTERVAL_S = 5.0

# Marker written by VectorBiography.import_foreign_events into the payload
# of an imported event. Events carrying it are never forwarded again,
# otherwise a pair of nodes would copy the same event back and forth.
FEDERATED_MARKER = "_federated_from"


class Federation:
    """In-process registry of WAAA nodes with biographical reconciliation."""

    def __init__(self, federation_id: str, sync_interval_s: float = SYNC_INTERVAL_S):
        self.federation_id = federation_id
        self.sync_interval_s = sync_interval_s
        self.nodes: dict = {}
        self.isolated: set[str] = set()
        self.last_sync: float = 0.0
        self.sync_count: int = 0
        self.events_reconciled: int = 0
        # Highest event timestamp already exported from each node.
        self._watermarks: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Membership                                                          #
    # ------------------------------------------------------------------ #

    def register_node(self, node) -> None:
        self.nodes[node.node_id] = node
        self._watermarks.setdefault(node.node_id, 0.0)
        logger.info("[Federation:%s] Registered node %s",
                    self.federation_id, node.node_id)

    def isolate_node(self, node_id: str) -> None:
        """Exclude a node from reconciliation (simulated partition)."""
        if node_id in self.nodes:
            self.isolated.add(node_id)
            logger.warning("[Federation:%s] Node %s isolated",
                           self.federation_id, node_id)

    def restore_node(self, node_id: str) -> dict:
        """Re-admit an isolated node and run BRP for it immediately."""
        self.isolated.discard(node_id)
        imported = self._reconcile(only=node_id)
        logger.warning("[Federation:%s] Node %s restored, BRP moved %d events",
                       self.federation_id, node_id, imported)
        return {"node_id": node_id, "events_reconciled": imported}

    def connected_nodes(self) -> list:
        return [n for nid, n in self.nodes.items() if nid not in self.isolated]

    # ------------------------------------------------------------------ #
    # Reconciliation                                                      #
    # ------------------------------------------------------------------ #

    def sync(self, force: bool = False) -> int:
        """Reconcile biographies if the sync interval has elapsed.

        Returns the number of events moved between nodes (0 when the call
        was rate-limited). ``self.last_sync = 0`` forces the next call.
        """
        now = time.time()
        if not force and (now - self.last_sync) < self.sync_interval_s:
            return 0
        self.last_sync = now
        self.sync_count += 1
        return self._reconcile()

    def _reconcile(self, only: str | None = None) -> int:
        """Move every not-yet-shared event from each node to its peers."""
        peers = self.connected_nodes()
        if len(peers) < 2:
            return 0

        moved = 0
        for source in peers:
            targets = [n for n in peers if n.node_id != source.node_id]
            if only is not None and source.node_id != only:
                targets = [n for n in targets if n.node_id == only]
            if not targets:
                continue

            events = self._exportable_events(source)
            if not events:
                continue
            for target in targets:
                moved += target.biography.import_foreign_events(
                    events, source.node_id
                )
            self._watermarks[source.node_id] = max(
                ev["timestamp"] for ev in events
            )

        self.events_reconciled += moved
        return moved

    def _exportable_events(self, source) -> list:
        """Own events of ``source`` newer than its watermark.

        Events previously imported from another node are skipped: they are
        already on their originating node, and forwarding them would make
        each reconciliation duplicate them.
        """
        since = self._watermarks.get(source.node_id, 0.0)
        own = []
        for ev in source.biography.export_since(since):
            try:
                payload = json.loads(ev.get("payload") or "{}")
            except (TypeError, ValueError):
                payload = {}
            if FEDERATED_MARKER not in payload:
                own.append(ev)
        return own

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> dict:
        return {
            "federation_id": self.federation_id,
            "node_count": len(self.nodes),
            "connected": [nid for nid in self.nodes if nid not in self.isolated],
            "isolated": sorted(self.isolated),
            "sync_interval_s": self.sync_interval_s,
            "sync_count": self.sync_count,
            "last_sync": self.last_sync,
            "events_reconciled": self.events_reconciled,
        }
