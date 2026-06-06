from datetime import datetime, timedelta
from waaa_memory.engine import MemoryEvent, InMemoryEventStore, MemoryCompactionEngine

store = InMemoryEventStore()

for i in range(65):
    store.add(MemoryEvent(
        content="room stable" if i % 3 == 0 else f"ordinary observation {i}",
        importance=0.2,
        timestamp=datetime.utcnow() - timedelta(minutes=i),
    ))

store.add(MemoryEvent("camera failure detected", event_type="anomaly", importance=0.95, preserve=True))
store.add(MemoryEvent("perception recovery successful", event_type="recovery", importance=0.9, causal=True))

engine = MemoryCompactionEngine(max_events=80, pressure_threshold=0.60, prune_target=40)

print("Before:", engine.metrics_engine.compute(store.all()))
print("Compaction:", engine.compact(store))
print("After:", engine.metrics_engine.compute(store.all()))
print("Second attempt:", engine.compact(store))
