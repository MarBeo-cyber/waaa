[![AURA Framework](https://img.shields.io/badge/AURA-Level%201%20%7C%20waaa-1F3864)](https://github.com/MarBeo-cyber/AURA)

# 🤖 WAAA — Weak Autopoietic Artificial Agent

> **Federated ML system for real-time environmental anomaly detection, autonomous threshold calibration, and autopoietic recovery — Architecture A (Full ML)**

[![CI](https://github.com/YOUR_ORG/waaa/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/waaa/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/YOUR_ORG/waaa/badge.svg)](https://codecov.io/gh/YOUR_ORG/waaa)

---

## What is WAAA?

WAAA is an autopoietic agent: a system that continuously monitors its own perceptual capacity, detects degradation, and autonomously recovers — without rule-based thresholds or hard-coded logic. Every decision component is a **learned model** that improves with experience.

WAAA is the predecessor of [MAAA](https://github.com/MarBeo-cyber/MAAA) — the Metacognitive Autopoietic Adaptive Agent.

### Architecture A — ML Component Map

| Component | Model | Replaces |
|-----------|-------|---------|
| Scene anomaly detection | **MLP Autoencoder** | Weighted average formulas |
| Threshold calibration | **Q-learning RL agent** | EMA + fixed delta |
| Goal switching | **Random Forest classifier** | `if coherence < 0.35` |
| Recovery assessment | **Isolation Forest** | Fixed-threshold comparisons |
| Episodic memory | **Vector store** (cosine similarity) | SQL queries by type |

---

## Project Structure

```
waaa/
├── main_ml.py               # Entry point
├── core/
│   └── ml_node.py           # MLWaaaNode — integrates all ML models
├── ml/
│   ├── autoencoder_scene.py # MLP Autoencoder anomaly detector
│   ├── rl_aptc.py           # Q-learning threshold calibration
│   ├── goal_classifier.py   # Random Forest goal switcher
│   ├── recovery_detector.py # Isolation Forest recovery assessor
│   └── vector_biography.py  # Cosine similarity episodic memory
├── sensors/
│   └── ml_webcam_sensor.py  # Webcam sensor (real or synthetic)
├── api/
│   └── ml_rest_api.py       # REST API :5001
├── tests/                   # Test suite (pytest)
└── scripts/
    └── run_ml.sh            # Launch script
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_ORG/waaa.git
cd waaa

# Install
pip install -e .

# Run the four-phase ML demo
bash scripts/run_ml.sh demo

# Or start the REST API
bash scripts/run_ml.sh server
```

### Docker

```bash
bash scripts/run_ml.sh docker
# API available at http://localhost:5001
```

---

## ML Model Lifecycle

Each model follows the same bootstrap → active → persist lifecycle:

```
Bootstrap (rule fallback) → Calibration (20-30 samples) → Active (learned model)
         ↑                                                         |
         └─────────────── Incremental retraining ◄────────────────┘
                          (every 50 cycles, saved to disk)
```

On restart, models load from `waaa_models/` — the system picks up with prior knowledge.

---

## REST API

```bash
# All ML model status
curl http://localhost:5001/ml/status

# Semantic memory search
curl "http://localhost:5001/ml/similar?coherence=0.2&prediction_error=0.8"

# Force a degraded scene and observe goal switching
curl -X POST http://localhost:5001/scene/NOISY
curl -X POST http://localhost:5001/tick/n/20
curl http://localhost:5001/goal_log

# Persist models immediately
curl -X POST http://localhost:5001/ml/save

# Reset a single component
curl -X POST http://localhost:5001/ml/reset/autoencoder
```

Full API reference: [`api/ml_rest_api.py`](api/ml_rest_api.py)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas:

- [ ] Real hardware sensor adapters (depth cameras, IMU, GPS)
- [ ] Federated learning across multiple WAAA nodes
- [ ] Edge deployment (NVIDIA Jetson, Raspberry Pi 5)
- [ ] MAAA bridge — feeding WAAA scene data to the MAAA Regulatory Engine

---

## Relation to MAAA

WAAA focuses on **self-preservation of the agent's perceptual capacity**. MAAA extends this to **shared embodiment with a human**, adding:
- Human state monitoring (stress, panic, cognitive overload)
- Regulatory Engine with 4-filter cognitive entropy reduction
- 3-level autobiographical memory
- AR overlay guidance in real-time emergency scenarios

→ [MAAA Repository](https://github.com/MarBeo-cyber/MAAA)

---

## License

MIT — see [LICENSE](LICENSE).

*Developed in collaboration with Claude (Anthropic)*

---

## Federation as Cognitive Emergence

The most theoretically significant property of the WAAA model is not the individual node but the federated network. The central claim:

> *No node contains the function, but the function emerges from the network.*

This is not a claim about distributing a task that could be performed by a centralised system. It is a claim about genuine emergence — properties that exist in the network but not in any of its parts.

### Why Federation Produces Emergence

**Biographical Reconciliation (BRP):** When two nodes that have operated autonomously reconnect, each has accumulated experience the other lacks. Reconciliation constructs a unified timeline and collective memory that neither node had individually — irreducible to the sum of the parts.

**Federated CUF:** A single node can compute triage priority for its own entities. It cannot compute global priority across the federated system, because that requires distributed information about all entities across all nodes. The optimal triage decision is a property of the network, not the node.

**Absolute biographical resilience:** No geographically localised extreme event can erase the collective memory of the system, because that memory is distributed across physically disjoint nodes. Restoring a node to its initial configuration is a local event — other nodes preserve the collective biography and contribute to reconstruction.

**Formal grounding:** The integrated information Φ (Tononi, 2008) of a federation of WAAA nodes is structurally greater than the sum of the individual Φ values. The connections between nodes create irreducible integration — states of the federated system that cannot be described as a sum of individual states.

---

## Known Architectural Limitations

Empirical testing of three-node federations on independent machines with local Ollama models has revealed two structural bottlenecks:

### 1. Latency

A complete LLM reasoning cycle requires **2–8 seconds** on consumer hardware (even with quantised models). Synchronous BRP across three peer-to-peer nodes multiplies this latency — potentially beyond the sub-second response requirements of many operational BIA scenarios.

### 2. Hardware Requirements

Each Ollama instance requires a minimum of **16 GB RAM** for models of sufficient quality. 4-bit quantisation reduces this but degrades structured JSON output reliability below a critical threshold for the reasoning loop.

### Directions Under Investigation

| Approach | Description |
|---|---|
| **Hub-and-spoke topology** | Lightweight ML nodes (Architecture A) at the periphery; single LLM orchestrator for federated decisions |
| **Async BRP with vector clocks** | Nodes publish biographical events to a shared queue (Redis/NATS); reconciliation runs in background without blocking the reasoning cycle |
| **Model specialisation by function** | Small fast models (Phi-3 mini, Qwen 1.5B) for synchronisation and routine monitoring; larger models only for critical decisions |

> These are **engineering constraints, not theoretical ones**. The emergent properties of federation are formally demonstrable regardless of the current hardware ceiling. Local LLM inference quality sufficient for WAAA reasoning did not exist three years ago — it improves every quarter.

---
###   Memory Pruning Engine
The Memory Pruning Engine provides the base primitive that the MAAA level uses to operate on an already-filtered stream. MAAA does not receive raw environmental noise: it receives events that have passed the WAAA's internal coherence threshold. This architectural dependency was implicit in previous versions; v1.4 makes it explicit.

---

## The Autopoietic Agent Family

WAAA is the first agent in an evolving family of autopoietic systems — an **artificial ontogenesis** that develops by stages analogous to biological cognitive maturation:

| Agent | Full Name | Core Function | Biological Analogy |
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

Each agent extends the previous: MAAA uses WAAA's perceptual loop in shared embodiment scenarios; PAAA can share its neurofunctional profile with MAAA; SAAA integrates the PAAA profile to optimise learning sessions and feeds consolidated knowledge back to MAAA.
