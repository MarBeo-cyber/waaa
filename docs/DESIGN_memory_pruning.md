# Design

Il WAAA mantiene memoria compatta e utile.

Metriche:
- memory_pressure = len(events) / MEMORY_MAX_EVENTS
- retrieval_noise = duplicati + eventi a basso valore

Eventi preservati:
- anomaly
- error
- recovery
- critical
- causal=True
- preserve=True
- importance >= 0.85

Anti-loop:
- compaction non ripetuta se cooldown attivo.
