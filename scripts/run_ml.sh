#!/usr/bin/env bash
# run_ml.sh — WAAA Architecture A launch script
set -euo pipefail

MODE="${1:-demo}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          WAAA — Weak Autopoietic Artificial Agent            ║"
echo "║                  Architecture A (ML)  v1.0                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

case "$MODE" in
  demo)
    echo "▶ Starting four-phase ML demo..."
    python main_ml.py demo
    ;;
  server)
    echo "▶ Starting REST API server on :5001..."
    python main_ml.py server
    ;;
  both)
    echo "▶ Starting server + demo simultaneously..."
    python main_ml.py both
    ;;
  docker)
    echo "▶ Building and running Docker container..."
    docker build -t waaa:latest .
    docker run -p 5001:5001 -v "$(pwd)/waaa_models:/app/waaa_models" waaa:latest
    ;;
  test)
    echo "▶ Running test suite..."
    pytest tests/ -v
    ;;
  *)
    echo "Usage: $0 [demo|server|both|docker|test]"
    exit 1
    ;;
esac
