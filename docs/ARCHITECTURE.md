# WAAA — Architecture Reference

## Federation — Known Limitations and Open Problems

### Empirically Observed Bottlenecks

Three-node federation testing on independent machines with local Ollama models (Llama 3 8B, 4-bit quantised) revealed:

**Latency:** Full LLM reasoning cycle = 2–8s per node. Synchronous BRP adds one round-trip per peer. Three nodes in peer-to-peer topology = 6–24s per federated decision cycle. This exceeds the RTO of many operational scenarios.

**Hardware floor:** 16 GB RAM per node minimum for reliable structured output. Below this, JSON parsing failures cascade through the reasoning loop.

**Quantisation threshold:** At Q4_0, models below ~7B parameters fail to reliably produce the 8-section JSON response format required by Architecture B. The fallback kicks in, degrading the federation to rule-based mode.

### Proposed Solutions

**Hub-and-spoke** — one LLM orchestrator node, N lightweight Architecture A peripheral nodes. Peripheral nodes handle continuous monitoring; the orchestrator handles federated decisions. Reduces LLM instances from N to 1.

**Async BRP** — replace synchronous reconciliation with an event queue (Redis Streams or NATS JetStream). Each node publishes biographical events with a vector clock timestamp. Reconciliation runs as a background process. The cognitive loop is never blocked by federation latency.

**Tiered models** — use Phi-3 mini (3.8B, ~2GB RAM) or Qwen 1.5B for synchronisation heartbeats and routine monitoring cycles. Reserve the larger model for decisions that trigger MTPD-proximity alerts or cross-perimeter coordination.

### Theoretical Status

These are implementation constraints, not theoretical failures. The emergent properties of federation — irreducible Φ, collective biographical memory, distributed CUF — are formally demonstrable in the current architecture. The gap between theoretical property and practical realisation is a hardware and latency gap that closes as local LLM inference improves.

---

## The Five Constitutive Dimensions

A WAAA node is defined by five properties that distinguish it from a standard reactive agent.

### 1. Partial Operational Closure

The node monitors not only the environment but also its own perceptual subsystem. When coherence falls below the APTC threshold, the node recognises that its perception is unreliable — and modifies what it observes, how it observes, and why it observes.

### 2. Minimal Biographical Continuity

Every significant event is recorded in a persistent SQLite database. On restart, the node loads its history and resumes from the last known state. The biography is not a log: it is the node's memory of having been in situations before.

### 3. Stable Internal Orientation

The node has a goal that persists across perturbations: `monitor_anomalies` → `restore_perceptual_capacity` → `monitor_anomalies`. The transition is deliberate, not a failure state.

### 4. Context Sensitivity as Affordance

Sensor parameters change with the goal. In `monitor_anomalies` mode, sensitivity is calibrated for anomaly detection. In `restore_perceptual_capacity` mode, it shifts to luminance and signal quality assessment.

### 5. Extended Reactive Self-Interrogation (Architecture B)

The LLM generates spontaneous questions about the node's own state. These questions are stored and fed back into the next cycle's prompt — a temporal feedback loop approximating autonomous inner dialogue.

---

## BIA and the CUF

**Priority order: residual(MTPD) > TPP > RTO**

The entity closest to its irreversibility threshold takes absolute priority. For biological entities, MTPD = death. For regulated entities, MTPD is externally imposed.

**MTPD formalisation for technological entities:**

```
MTPD = RTO + EB(RTO)
```

Where EB (Effort Backlog) is a monotonically increasing function of RTO. Beyond the reconciliation threshold, damage is structurally irreversible.

---

## Recovery Hierarchy

| Level | Name | Action | Biographical Loss |
|---|---|---|---|
| L0 | Self-repair | Reset APTC θ to θ₀ | None |
| L1 | Rollback snapshot | Restore from last stable snapshot | Minimal |
| L2 | Rollback intermediate | Restore invariant core + partial history | Moderate |
| L3 | Reset invariant core | Full reset; preserve only node identity | Significant |
