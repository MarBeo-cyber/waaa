# WAAA — Architecture Reference

Two kinds of statement appear in this document and they are kept apart
deliberately:

- **Implemented** — describes code in this repository. Every such statement
  can be checked by reading the file named next to it or by running the tests.
- **Design** — describes where the architecture is meant to go. Not code.

---

## The Five Constitutive Dimensions

A WAAA node is defined by five properties that distinguish it from a standard
reactive agent. These are the conceptual core of the project; four of the five
have a partial realisation in code.

### 1. Partial Operational Closure — *implemented*

The node monitors not only the environment but also its own perceptual
subsystem. When coherence falls below the thresholds in
`ml/goal_classifier.py`, the node recognises that its perception is unreliable
and switches goal, changing what it observes, how it observes, and why.

Note the distinction from the APTC threshold θ: θ is the sensitivity applied to
signal magnitude (`ml/rl_aptc.py`), and it is what the RL agent adjusts. The
goal switch is driven by coherence, which is a different quantity.

### 2. Minimal Biographical Continuity — *partly implemented*

Significant events and snapshots are written to SQLite
(`ml/vector_biography.py`), and the vector index is rebuilt from that database
on startup. Model state — the Q-table, θ, the autoencoder and its scaler —
reloads from `waaa_models/`.

What does **not** survive a restart: the node's current goal and node state.
They are re-derived from the first reading rather than restored from the last
snapshot. The biography is not a log — it is the node's record of having been
in situations before — but the node does not yet wake up inside the situation
it left.

### 3. Stable Internal Orientation — *implemented*

The node has a goal that persists across perturbations:
`monitor_anomalies` → `restore_perceptual_capacity` → `monitor_anomalies`.
The transition is a deliberate reorientation, not a failure state.

### 4. Context Sensitivity as Affordance — *implemented*

Sensor parameters change with the goal. In `restore_perceptual_capacity` the
node relaxes the sensor's noise filter and feeds the current frame back into
the scene model's expectations (`core/ml_node.py::_restore_perception`), so
low light becomes the new baseline rather than a permanent alarm.

### 5. Extended Reactive Self-Interrogation — *design only, Architecture B*

The intended mechanism: an LLM generates spontaneous questions about the node's
own state, stores them, and feeds them into the next cycle's prompt — a
temporal feedback loop approximating autonomous inner dialogue.

**There is no LLM in this repository** and no Architecture B implementation.
Nothing here calls a language model, local or remote.

---

## The Cognitive Loop — *implemented*

One `tick()` in `core/ml_node.py`:

1. **Sense** — read a synthetic frame; the autoencoder scores it, or reports
   that it is still warming up
2. **Evaluate** — RL-APTC compares signal magnitude against θ and accumulates
   interval statistics
3. **Self-monitor** — the recovery detector assesses the node's own state
4. **Decide goal** — the goal rule maps the reading to a goal
5. **Act** — report, restore perception, or execute recovery
6. **Record** — significant events are embedded and persisted
7. **Snapshot** — every 60 s
8. **Persist models** — every 50 cycles

---

## Recovery Hierarchy — *implemented*

`core/recovery.py`. What each level discards, exactly as coded:

| Level | Name | Action | Loss |
|---|---|---|---|
| L0 | Self-repair | Restore the operating parameters passed in (sensor noise filter → default) | None |
| L1 | Rollback snapshot | Restore θ from the last snapshot; keep the Q-table, ε and step counters | None |
| L2 | Rollback intermediate | Restore θ from the last snapshot; discard the calibration log and exploration counters; keep the Q-table | Partial |
| L3 | Reset invariant core | θ → θ₀, ε → start, calibration log emptied, Q-table zeroed, dispositions restored from the initial configuration | Significant |

Biographical *events* are never deleted by any level. "Biographical loss"
refers to calibration and policy state.

L1 fails honestly when no snapshot exists: it returns
`{"error": "no_snapshot_available"}` and records an unsuccessful recovery event
rather than silently doing nothing.

**Known coupling.** The goal rule switches to `execute_recovery` on
`coherence < 0.10 or prediction_error > 0.85`, which is the same condition the
severity ladder uses for L3. Recoveries triggered by the goal rule therefore
always execute L3, and L0–L2 are unreachable from that path. Decoupling the two
thresholds is open work.

---

## Recovery Assessment — *implemented*

`ml/recovery_detector.py` combines a learned component with an explicit one,
because the learned component cannot do the whole job:

- The **Isolation Forest** learns the node's own healthy operating manifold
  from states that passed a health filter, and decides on-manifold vs
  off-manifold. This is what a fixed threshold cannot do — a node whose normal
  coherence is 0.4 would be permanently "degraded" by a global rule.
- The **severity ladder** picks L0–L3 for states the forest rejected. The
  forest cannot do this: trained on a narrow healthy region, it isolates every
  clearly abnormal state at the same minimum depth, so mild, degraded and
  severe states receive an identical score.

The anomaly score is derived from `decision_function`, which is centred on the
forest's own inlier/outlier boundary, rescaled by the spread of decision values
measured on the training set. An earlier version used `0.5 - score_samples(X)`;
`score_samples` returns negative values, so that expression was always ≥ 0.9
and every state mapped to L3.

---

## Threshold Calibration — *implemented*

`ml/rl_aptc.py`. The calibration problem is framed as an MDP:

- **State**: discretised (θ, EMA event rate, blind streak, saturated streak,
  coherence)
- **Actions**: θ ± 0.08, θ ± 0.03, hold
- **Reward**: +0.1 productive interval, −2.0 blind interval, −0.5 saturated,
  −0.3 at a θ boundary, plus immediate ±1.0/−0.5 on detections

An observation interval closes after `observation_interval_steps` evaluations
**or** `observation_interval_s` seconds, whichever comes first. Both gates
matter: with only the wall clock, a fast demo loop never closes an interval,
never selects an action and therefore never performs a Bellman update.

---

## BIA and the CUF

**Implemented** (`core/bia.py`): a registry of target entities with their
continuity parameters — RTO, RPO, MTPD, TPP, EB — plus disruption bookkeeping
(`mark_disrupted`, `residual_mtpd`).

**Design, not implemented**: the triage ordering itself.

> Priority order: residual(MTPD) > TPP > RTO

The entity closest to its irreversibility threshold takes absolute priority.
For biological entities, MTPD = death. For regulated entities, MTPD is
externally imposed.

MTPD formalisation for technological entities:

```
MTPD = RTO + EB(RTO)
```

where EB (Effort Backlog) is a monotonically increasing function of RTO. Beyond
the reconciliation threshold, damage is structurally irreversible.

Nothing in the code orders entities by this rule today; `core/bia.py` stores
the parameters the rule would need.

---

## Federation — *in-process only*

**Implemented** (`federation/federation.py`): a registry of nodes in one Python
process, with periodic biographical reconciliation. Each node exports its own
events newer than a per-node watermark and the peers import them. Events that
arrived by reconciliation are never forwarded again, so a pair of nodes cannot
copy the same event back and forth. `isolate_node` simulates a partition;
`restore_node` runs reconciliation for the returning node.

**Not implemented**: network transport, discovery, consensus, conflict
resolution beyond timestamp watermarks, and federated learning. Models are
never shared — only recorded events are.

### Design — federated emergence

The architectural claim is that a federation has properties no node has:
biographical reconciliation builds a timeline neither node held; global triage
priority is computable only across nodes; and collective memory survives the
loss of any single node because it is physically distributed.

Earlier versions of this document reported figures from "three-node federation
testing on independent machines with local Ollama models" — latency of 2–8 s
per reasoning cycle, a 16 GB RAM floor, a Q4_0 quantisation threshold. Those
figures have been removed. There is no LLM code in this repository, no
multi-machine deployment, and no measurement to support them.

A related claim, that the integrated information Φ (Tononi, 2008) of a
federation exceeds the sum of the individual Φ values, is a theoretical
conjecture about the architecture. Nothing here computes Φ.

### Directions under investigation

| Approach | Description |
|---|---|
| Hub-and-spoke | Lightweight peripheral nodes, one orchestrator for federated decisions |
| Async BRP with vector clocks | Nodes publish events to a shared queue; reconciliation never blocks the cognitive loop |
| Real transport | Replace the in-process registry with a network layer, which is where consensus and conflict resolution become real problems |
