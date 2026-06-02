#!/usr/bin/env bash
# One-command helper to start the experiment dashboard

set -e

cd "$(dirname "$0")"

echo "=== Starting GPUOS Experiment Dashboard (for baselines, agent ranking, benches, PhaseN) ==="

# Ensure data dir exists
mkdir -p results

echo "[1/2] Checking SurrealDB (optional, for full parity with sudoku-era tracking) on port 8001..."
if curl -s -I http://localhost:8001 > /dev/null 2>&1; then
    echo "    SurrealDB is running."
else
    echo "    (Optional) Starting SurrealDB via docker compose... (json results/ always work even without)"
    docker compose up -d 2>/dev/null || echo "    (docker not required; continuing with local JSON)"
    sleep 3
fi

echo "[2/2] Starting Streamlit dashboard on http://localhost:8502 (GPUOS port to avoid conflicts) ..."
echo ""
echo "Press Ctrl+C to stop the dashboard."
echo "After start, open http://localhost:8502 and explore Runs, Agent Analysis, VRAM/Counters tabs."
echo "New results/*.json dropped in experiments/results/ will appear on reload."
echo "Companion: in another shell, 'python ../viz_gpu_activity.py --latest' (or --track jit-baseline) for terminal pop-up of GPU activity from the same internal data."
echo ""

exec python3 -m streamlit run dashboard/app.py --server.port 8502 --server.headless false