# GPUOS Dashboard

Streamlit UI for experiment results (baselines, agent ranking recall/latency, benches, counters, VRAM).

See ../README.md for usage (start via start_gpuos_dashboard.sh on :8502).

Companion terminal visualizer (no deps, quick after-run "pop up"): ../viz_gpu_activity.py --latest (or --track jit-baseline / agent-ranking). Shows the same internal data (stages, vram deltas, SyncState counters, agent metrics) in ascii boxes + bars + timeline.

Adapted from Percepta sudoku dashboard to give the same visibility for GPUOS phases and the "road with twists and turns".

Local JSON preferred (results/ + benchmark_results/). No new deps beyond streamlit/pandas.
