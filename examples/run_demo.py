"""Memory pruning demo — runnable straight from a clone:

    python examples/run_demo.py
"""

import os
import sys
from datetime import datetime, timedelta

# Allow running from a plain clone, without an install or PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waaa_memory.engine import (  # noqa: E402
    InMemoryEventStore,
    MemoryCompactionEngine,
    MemoryEvent,
)

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
