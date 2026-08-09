"""
WAAA — BIA (Business Impact Analysis) registry.

The BIA holds the entities the node is responsible for keeping running,
together with their continuity parameters (RTO, RPO, MTPD, TPP, EB).
Those parameters are *human-defined*: nothing in this module learns or
adjusts them.

Scope note — this module implements exactly what the node uses today:
a registry of target entities plus disruption bookkeeping. The triage
formalism sketched in ``docs/ARCHITECTURE.md`` (the CUF ordering over
residual MTPD / TPP / RTO) is a design note, not implemented here.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger("waaa.bia")


@dataclass
class TargetEntity:
    """An entity whose operational continuity the node is responsible for.

    Times are in seconds unless the field name says otherwise:
      rto_seconds   Recovery Time Objective
      rpo_seconds   Recovery Point Objective
      mtpd_seconds  Maximum Tolerable Period of Disruption
      tpp_seconds   Time to Point of Prejudice
      eb_hours      Effort Backlog accumulated by a disruption
    """

    entity_id: str
    name: str = ""
    priority: int = 1
    rto_seconds: float = 60.0
    rpo_seconds: float = 30.0
    mtpd_seconds: float = 300.0
    tpp_seconds: float = 60.0
    eb_hours: float = 0.0
    dependencies: list = field(default_factory=list)
    is_self: bool = False
    disruption_start: Optional[float] = None

    def mark_disrupted(self, when: Optional[float] = None) -> None:
        """Record the start of a disruption (no-op if one is already open)."""
        if self.disruption_start is None:
            self.disruption_start = when if when is not None else time.time()
            logger.warning("[BIA] Entity '%s' marked disrupted", self.entity_id)

    def mark_restored(self) -> None:
        """Close an open disruption."""
        if self.disruption_start is not None:
            logger.info("[BIA] Entity '%s' restored", self.entity_id)
            self.disruption_start = None

    @property
    def is_disrupted(self) -> bool:
        return self.disruption_start is not None

    def disruption_seconds(self, now: Optional[float] = None) -> float:
        """How long the current disruption has been open (0.0 if none)."""
        if self.disruption_start is None:
            return 0.0
        return (now if now is not None else time.time()) - self.disruption_start

    def residual_mtpd(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds left before MTPD is exceeded, or None if not disrupted."""
        if self.disruption_start is None:
            return None
        return self.mtpd_seconds - self.disruption_seconds(now)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_disrupted"] = self.is_disrupted
        return d


class BIA:
    """Registry of target entities and their continuity parameters."""

    def __init__(self, config: Optional[dict] = None):
        self._entities: dict[str, TargetEntity] = {}
        for spec in (config or {}).get("entities", []):
            self.register(TargetEntity(**spec))

    def register(self, entity: TargetEntity) -> TargetEntity:
        self._entities[entity.entity_id] = entity
        return entity

    def get(self, entity_id: str) -> Optional[TargetEntity]:
        return self._entities.get(entity_id)

    def all_entities(self) -> list[TargetEntity]:
        return list(self._entities.values())

    def self_entities(self) -> list[TargetEntity]:
        """Entities that represent the node itself (is_self=True)."""
        return [e for e in self._entities.values() if e.is_self]

    def disrupted_entities(self) -> list[TargetEntity]:
        return [e for e in self._entities.values() if e.is_disrupted]

    def export_config(self) -> dict:
        """Round-trippable config: ``BIA(bia.export_config())`` rebuilds it."""
        return {"entities": [asdict(e) for e in self._entities.values()]}

    @property
    def status(self) -> dict:
        return {
            "entity_count": len(self._entities),
            "disrupted_count": len(self.disrupted_entities()),
            "entities": sorted(self._entities),
        }
