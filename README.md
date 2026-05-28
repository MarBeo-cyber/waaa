# 🤖 WAAA — Weak Autopoietic Artificial Agent

> **Federated ML system for real-time environmental anomaly detection, autonomous threshold calibration, and autopoietic recovery — Architecture A (Full ML)**

[![CI](https://github.com/YOUR_ORG/waaa/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/waaa/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/YOUR_ORG/waaa/badge.svg)](https://codecov.io/gh/YOUR_ORG/waaa)

---

## What is WAAA?

WAAA is an autopoietic agent: a system that continuously monitors its own perceptual capacity, detects degradation, and autonomously recovers — without rule-based thresholds or hard-coded logic. Every decision component is a **learned model** that improves with experience.

WAAA is the predecessor of [MAAA](https://github.com/YOUR_ORG/maaa) — the Metacognitive Autopoietic Adaptive Agent.

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
