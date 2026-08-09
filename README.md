[![AURA Framework](https://img.shields.io/badge/AURA-Level%201%20%7C%20waaa-1F3864)](https://github.com/MarBeo-cyber/AURA)

# 🤖 WAAA — Weak Autopoietic Artificial Agent

> **An agent that monitors its own perceptual capacity, notices when that capacity degrades, and repairs itself — Architecture A**

[![CI](https://github.com/MarBeo-cyber/waaa/actions/workflows/ci.yml/badge.svg)](https://github.com/MarBeo-cyber/waaa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Status

Research prototype. Everything documented below runs, and the CI badge above
covers lint, the test suite, the memory-pruning example and a full demo run on
Python 3.10–3.12. To keep that true, here is what the code does *not* do:

- **All sensor input is synthetic.** Frames are generated in-process by
  `sensors/synthetic_scene_sensor.py` — a luminance plane plus Gaussian noise.
  There is no camera capture path and no `cv2` anywhere in the repository.
- **There is no LLM anywhere.** Architecture B, the reasoning loop and the
  self-interrogation described in the design notes are not implemented.
- **Federation is in-process only.** `federation/federation.py` reconciles the
  biographies of nodes living in the same Python process. No network, no
  discovery, no consensus, no federated learning.
- **Goal switching is a rule, not a model.** See the table below.
- **The demo is a demonstration, not a benchmark.** No accuracy, latency or
  hardware figure in this repository has been measured against a real
  deployment, so none is quoted.

The autopoietic framing — a system whose job is to maintain the conditions of
its own operation — is the point of the project. The code is an early,
partial realisation of it.

---

## What is WAAA?

WAAA is an agent built around a single idea: the thing most worth monitoring is
the monitor itself. A conventional sensor loop reports what it sees. WAAA also
asks whether what it sees can still be trusted, and when the answer is no, it
changes *what* it observes, *how* it observes, and *why* — then repairs itself
along a graded hierarchy that trades away as little of its own history as the
damage requires.

WAAA is the predecessor of [MAAA](https://github.com/MarBeo-cyber/MAAA) — the
Metacognitive Autopoietic Adaptive Agent.

### Component map — what each part actually is

| Component | What it is | Learned? |
|---|---|---|
| Scene anomaly detection | MLP autoencoder (`sklearn.MLPRegressor`); reconstruction error on 12 frame features | **Yes** — fitted on the first 20 frames, refitted every 50 |
| Threshold calibration | Tabular Q-learning agent over a discretised state, 5 actions on θ | **Yes** — Bellman updates from interval rewards |
| Recovery assessment | Isolation Forest over 8 state features, plus an explicit severity ladder | **Partly** — see below |
| Goal switching | Deterministic rule on coherence / prediction error / frame quality | **No** — see below |
| Episodic memory | Cosine similarity over 14-dim event embeddings, SQLite-backed | No (retrieval, not learning) |
| Memory pruning | Pressure + retrieval-noise metrics with a conservative pruner | No |

**Recovery assessment is a hybrid, on purpose.** The Isolation Forest learns
*this node's* healthy operating manifold, which a fixed threshold cannot: it
answers "is this state unlike how I normally am?". It cannot answer "how bad is
it" — a forest trained on a narrow healthy region isolates every clearly
abnormal state at the same minimum depth, giving mild and severe states an
identical score. So the forest decides on-manifold vs off-manifold, and an
explicit ladder in `ml/recovery_detector.py` picks L0–L3.

**Goal switching is honestly a rule.** It used to be a RandomForest. That
forest was trained on labels produced by the rule it was said to replace, so it
could only learn to imitate its own baseline — including the `coherence < 0.35`
comparison it was advertised as replacing. Training a model on its own
baseline's output teaches it nothing, so the forest was removed rather than
dressed up. Doing better needs labels derived from *outcomes* — whether the
goal chosen at time *t* actually restored coherence — and the node records no
such signal yet. The temporal feature engineering (coherence slope, prediction
error volatility, window statistics) is kept, computed every cycle and
published in `status["last_features"]`, because it is what an outcome-labelled
model would train on.

---

## Quick Start

```bash
git clone https://github.com/MarBeo-cyber/waaa.git
cd waaa

pip install -e .

# Four-phase demo: NORMAL → DIM → NOISY → RECOVERED, 47 cycles, ~11 s
python main_ml.py demo

# Memory pruning engine, standalone
python examples/run_demo.py

# REST API on :5001
python main_ml.py server
```

`bash scripts/run_ml.sh demo|server|both|test|docker` wraps the same commands.

### What the demo prints

Each cycle reports the sensor reading, which component produced it, the RL
agent's threshold and Q-table, the self-assessment and any goal switch. The
first ~20 cycles are marked `source=warming_up`: the autoencoder has no fitted
network yet, so the reading carries frame statistics measured directly rather
than model output. Once it fits, the marker becomes `source=active`.

Counts vary between runs — the frame noise is not seeded — so the demo reports
what happened rather than a fixed expected result.

---

## Model persistence

Models are written to `waaa_models/` in the repository root, overridable with
`WAAA_MODEL_DIR`. The Docker image sets it to `/app/waaa_models`, and
`scripts/run_ml.sh docker` mounts `./waaa_models` there, so a container run
picks up the previous run's models.

```
Warming up (no model) → Calibration (20–40 samples) → Active (learned model)
        ↑                                                      |
        └──────────── periodic refit on a sliding window ◄──────┘
```

---

## REST API

```bash
curl http://localhost:5001/ml/status                 # all model states
curl "http://localhost:5001/ml/similar?coherence=0.2&prediction_error=0.8"
curl -X POST http://localhost:5001/scene/NOISY       # force a degraded scene
curl -X POST http://localhost:5001/tick/n/20
curl http://localhost:5001/goal_log
curl -X POST http://localhost:5001/ml/save
curl -X POST http://localhost:5001/ml/reset/autoencoder
```

Full reference: [`api/ml_rest_api.py`](api/ml_rest_api.py).

---

## Project structure

```
waaa/
├── main_ml.py                        # Entry point: demo | server | both
├── core/
│   ├── ml_node.py                    # MLWaaaNode — the cognitive loop
│   ├── biography.py                  # BiographicalEntry, Snapshot
│   ├── bia.py                        # Target entities + continuity parameters
│   └── recovery.py                   # L0–L3 recovery hierarchy
├── ml/
│   ├── autoencoder_scene.py          # MLP autoencoder anomaly detector
│   ├── rl_aptc.py                    # Q-learning threshold calibration
│   ├── goal_classifier.py            # Goal rule + temporal features
│   ├── recovery_detector.py          # Isolation Forest + severity ladder
│   └── vector_biography.py           # Cosine-similarity episodic memory
├── sensors/
│   └── synthetic_scene_sensor.py     # Synthetic frames — no camera
├── federation/federation.py          # In-process biographical reconciliation
├── waaa_memory/engine.py             # Memory pruning engine
├── api/ml_rest_api.py                # REST API :5001
├── examples/run_demo.py              # Memory pruning example
└── tests/                            # pytest
```

---

## Known limitations

Observed by running the code, not estimated:

- **Every recovery the demo triggers is L3.** The goal rule switches to
  `execute_recovery` on `coherence < 0.10 or prediction_error > 0.85`, which is
  character-for-character the severity ladder's L3 condition, so the graded
  L0–L2 path is unreachable from that trigger. The two thresholds need to be
  decoupled.
- **The RL agent is repeatedly reset by its own recoveries.** L3 zeroes the
  Q-table by definition, and the demo's NOISY phase triggers L3 on most cycles,
  so the agent restarts learning each time. It accumulates experience only in
  runs that stay off the recovery path.
- **The Isolation Forest cannot rank severity** (see the component map above).
- **The autoencoder's anomaly threshold is a moving target**: it is recomputed
  from the recent error distribution, so a slow drift into a degraded regime is
  eventually absorbed as the new normal.
- Lint is scoped to ruff's `E`/`F`/`B` rules; import sorting and pyupgrade are
  not yet enabled because they rewrite every file at once.

---

## Ideas not implemented

Kept here because they are the direction of the work, and marked so nobody
mistakes them for code that exists.

### Federation as cognitive emergence — *proposed, not implemented*

The claim the design is built around:

> *No node contains the function, but the function emerges from the network.*

Three properties are argued to follow, none of them demonstrated by this
repository:

**Biographical reconciliation (BRP).** When two nodes that operated
autonomously reconnect, each holds experience the other lacks; reconciliation
constructs a timeline neither had individually. *What exists today:* nodes in
one process copy each other's recorded events (`federation/federation.py`).
That is event replication, not the construction of a shared account.

**Federated CUF.** A node can compute triage priority for its own entities but
not global priority across the federation, because that needs information
distributed across all nodes. *What exists today:* nothing — `core/bia.py`
holds entities and their continuity parameters; no triage ordering is
implemented.

**Absolute biographical resilience.** No localised event can erase a memory
distributed across physically disjoint nodes. *What exists today:* nothing —
the nodes share a process and, in the demo, a machine.

A formal claim about the integrated information Φ (Tononi, 2008) of a
federation exceeding the sum of its parts appeared in earlier versions of this
README as though it were established here. It is a theoretical conjecture about
the architecture. Nothing in this repository computes Φ, and no such
measurement has been made.

### Directions under investigation

| Approach | Description |
|---|---|
| Hub-and-spoke topology | Lightweight nodes at the periphery, one orchestrator for federated decisions |
| Async BRP with vector clocks | Nodes publish events to a shared queue; reconciliation never blocks the cognitive loop |
| Outcome-labelled goal learning | Record whether a goal switch restored coherence, then train on that instead of on the rule |
| Real sensor adapters | Depth cameras, IMU, GPS behind the existing sensor interface |

---

## Memory pruning engine

`waaa_memory/engine.py` provides the primitive the MAAA level builds on: MAAA
does not receive raw environmental noise, it receives events that have passed
WAAA's internal coherence threshold. Design notes:
[`docs/DESIGN_memory_pruning.md`](docs/DESIGN_memory_pruning.md). Run it with
`python examples/run_demo.py`.

---

## Relation to MAAA

WAAA focuses on **self-preservation of the agent's perceptual capacity**. MAAA
extends this to **shared embodiment with a human**:

- Human state monitoring (stress, panic, cognitive overload)
- Regulatory Engine with 4-filter cognitive entropy reduction
- 3-level autobiographical memory
- AR overlay guidance in real-time emergency scenarios

→ [MAAA Repository](https://github.com/MarBeo-cyber/MAAA)

---

## The autopoietic agent family

WAAA is the first agent in a family of autopoietic systems — an **artificial
ontogenesis** developing by stages analogous to biological cognitive
maturation. The later agents are separate projects at their own stages of
completeness.

| Agent | Full name | Core function | Biological analogy |
|---|---|---|---|
| **WAAA** | Weak Autopoietic Artificial Agent | Perceptual self-monitoring | Sensory reflex calibration |
| MAAA | Metacognitive Autopoietic Adaptive Agent | Embodied cognition in emergency | Acute stress response |
| PAAA | Personal Autopoietic Adaptive Agent | Neurofunctional continuity | Homeostasis / immune surveillance |
| SAAA | Sapient Autopoietic Adaptive Agent | Learning consolidation | Myelination / synaptic plasticity |

**Repositories:**
- WAAA: https://github.com/MarBeo-cyber/waaa
- MAAA: https://github.com/MarBeo-cyber/MAAA
- PAAA: https://github.com/MarBeo-cyber/PAAA
- SAAA: https://github.com/MarBeo-cyber/SAAA

Each agent extends the previous: MAAA uses WAAA's perceptual loop in shared
embodiment scenarios; PAAA can share its neurofunctional profile with MAAA;
SAAA integrates the PAAA profile to optimise learning sessions and feeds
consolidated knowledge back to MAAA.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

Contributions are welcome; there is no CONTRIBUTING.md yet, so open an issue
first and describe what you intend to change.

---

## License

MIT — see [LICENSE](LICENSE).

*Developed in collaboration with Claude (Anthropic)*
