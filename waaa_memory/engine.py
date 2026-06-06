from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4
from collections import Counter

@dataclass
class MemoryEvent:
    content: str
    event_type: str = "observation"
    importance: float = 0.5
    causal: bool = False
    preserve: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid4()))

    def high_value(self) -> bool:
        return (
            self.preserve or self.causal or self.importance >= 0.85
            or self.event_type in {"anomaly", "error", "recovery", "critical"}
        )

@dataclass
class MemoryMetrics:
    event_count: int
    max_events: int
    memory_pressure: float
    retrieval_noise: float
    high_value_count: int

@dataclass
class CompactionResult:
    executed: bool
    reason: str
    before_count: int
    after_count: int
    removed_count: int
    preserved_count: int

class InMemoryEventStore:
    def __init__(self):
        self.events = []

    def add(self, event: MemoryEvent):
        self.events.append(event)

    def all(self):
        return list(self.events)

    def replace_all(self, events):
        self.events = list(events)

def canonical(text):
    return " ".join(text.lower().strip().split())

class MemoryMetricsEngine:
    def __init__(self, max_events=80):
        self.max_events = max_events

    def memory_pressure(self, events):
        return min(1.0, len(events) / max(1, self.max_events))

    def retrieval_noise(self, events):
        if not events:
            return 0.0
        counts = Counter(canonical(e.content) for e in events)
        duplicates = sum(c - 1 for c in counts.values() if c > 1)
        low_value = sum(1 for e in events if not e.high_value() and e.importance < 0.35)
        return round(min(1.0, (0.65 * duplicates + 0.35 * low_value) / len(events)), 4)

    def compute(self, events):
        return MemoryMetrics(
            event_count=len(events),
            max_events=self.max_events,
            memory_pressure=round(self.memory_pressure(events), 4),
            retrieval_noise=self.retrieval_noise(events),
            high_value_count=sum(1 for e in events if e.high_value()),
        )

class ConservativePruner:
    def compact_to_target(self, events, target):
        if len(events) <= target:
            return events, []

        removable = [e for e in events if not e.high_value()]
        seen = set()

        def score(e):
            key = canonical(e.content)
            duplicate = key in seen
            seen.add(key)
            return (
                0 if duplicate else 1,     # duplicates first
                e.importance,             # low importance first
                -e.timestamp.timestamp(), # older first
            )

        ranked = sorted(removable, key=score)
        remove_needed = max(0, len(events) - target)
        to_remove = ranked[:remove_needed]
        remove_ids = {e.id for e in to_remove}
        remaining = [e for e in events if e.id not in remove_ids]
        return remaining, to_remove

class MemoryCompactionEngine:
    def __init__(
        self,
        max_events=80,
        pressure_threshold=0.60,
        prune_target=40,
        retrieval_noise_threshold=0.45,
        min_interval_seconds=300,
    ):
        self.metrics_engine = MemoryMetricsEngine(max_events)
        self.pruner = ConservativePruner()
        self.pressure_threshold = pressure_threshold
        self.prune_target = prune_target
        self.retrieval_noise_threshold = retrieval_noise_threshold
        self.min_interval = timedelta(seconds=min_interval_seconds)
        self.last_compaction_time = None

    def should_compact(self, events, now=None):
        now = now or datetime.utcnow()
        if self.last_compaction_time and now - self.last_compaction_time < self.min_interval:
            return False, "cooldown_active"
        m = self.metrics_engine.compute(events)
        if m.memory_pressure >= self.pressure_threshold:
            return True, "memory_pressure_threshold"
        if m.retrieval_noise >= self.retrieval_noise_threshold:
            return True, "retrieval_noise_threshold"
        return False, "below_threshold"

    def compact(self, store, now=None):
        now = now or datetime.utcnow()
        events = store.all()
        before = len(events)
        should, reason = self.should_compact(events, now)
        if not should:
            return CompactionResult(False, reason, before, before, 0, sum(e.high_value() for e in events))

        remaining, removed = self.pruner.compact_to_target(events, self.prune_target)
        store.replace_all(remaining)
        self.last_compaction_time = now
        return CompactionResult(
            True, reason, before, len(remaining), len(removed),
            sum(e.high_value() for e in remaining)
        )
