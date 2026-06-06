from waaa_memory.engine import MemoryEvent, InMemoryEventStore, MemoryCompactionEngine, MemoryMetricsEngine

def test_memory_pressure():
    events = [MemoryEvent(str(i)) for i in range(40)]
    m = MemoryMetricsEngine(max_events=80).compute(events)
    assert m.memory_pressure == 0.5

def test_high_value_preserved():
    store = InMemoryEventStore()
    for _ in range(70):
        store.add(MemoryEvent("duplicate event", importance=0.1))
    critical = MemoryEvent("critical anomaly", event_type="anomaly", importance=1.0, preserve=True)
    store.add(critical)
    engine = MemoryCompactionEngine(max_events=80, pressure_threshold=0.6, prune_target=40, min_interval_seconds=0)
    result = engine.compact(store)
    assert result.executed is True
    assert critical.id in {e.id for e in store.all()}

def test_cooldown_prevents_loop():
    store = InMemoryEventStore()
    for i in range(70):
        store.add(MemoryEvent(f"event {i}", importance=0.1))
    engine = MemoryCompactionEngine(max_events=80, pressure_threshold=0.6, prune_target=40, min_interval_seconds=300)
    first = engine.compact(store)
    second = engine.compact(store)
    assert first.executed is True
    assert second.executed is False
    assert second.reason == "cooldown_active"

def test_retrieval_noise_detects_duplicates():
    events = [MemoryEvent("same") for _ in range(10)] + [MemoryEvent(f"unique {i}") for i in range(5)]
    assert MemoryMetricsEngine(max_events=80).retrieval_noise(events) > 0.3
