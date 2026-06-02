"""
GPUOS Experiment Dashboard — adapted from Transformer-VM sudoku dashboard v2/v2.5

Run:
    cd experiments
    ./start_gpuos_dashboard.sh
    # or streamlit run dashboard/app.py --server.port 8502

Local JSON (experiments/results/ + benchmark_results/) preferred.
Supports agent utility (recall, cands/turn), Phase0/1 counters, VRAM, hygiene.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from data import (
    fetch_runs, apply_filters, compute_track_summary,
    get_status_badge, STATUS_BADGES, CORE_DISPLAY_COLS, load_observations_for_run,
)

st.set_page_config(
    page_title="GPUOS Experiments • Phase0/1/2+",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session hygiene (archive, tag baseline for phase, exploratory)
if "archived_ids" not in st.session_state:
    st.session_state.archived_ids = set()
if "baseline_ids" not in st.session_state:
    st.session_state.baseline_ids = set()
if "exploratory_ids" not in st.session_state:
    st.session_state.exploratory_ids = set()
if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None

st.title("GPUOS Experiments Dashboard")
st.caption("Tracking baselines, agent ranking utility/accuracy, benches, SyncState counters, VRAM. JSON-first (results/ + benchmark_results/). Adapted from sudoku-era Percepta dashboard for the GPUOS road ahead.")

df = fetch_runs()
if df.empty:
    st.warning("No results yet. Run the harness or demos to populate experiments/results/ (or benchmark_results/).")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
all_tracks = sorted(df["track"].dropna().unique().tolist())
sel_tracks = st.sidebar.multiselect("Track", all_tracks, default=all_tracks[:6] if len(all_tracks)>6 else all_tracks)
all_phases = sorted(df["phase"].dropna().astype(str).unique().tolist())
sel_phases = st.sidebar.multiselect("Phase", all_phases, default=all_phases)
success_only = st.sidebar.checkbox("Success only", value=True)
show_archived = st.sidebar.checkbox("Show archived", value=False)

fdf = apply_filters(df, sel_tracks, sel_phases, success_only)
if not show_archived:
    fdf = fdf[~fdf.get("id", fdf.index.astype(str)).isin(st.session_state.archived_ids)]  # rough

st.sidebar.metric("Runs visible", len(fdf))
if st.sidebar.button("Reset hygiene (session)"):
    st.session_state.archived_ids.clear()
    st.session_state.baseline_ids.clear()
    st.session_state.exploratory_ids.clear()
    st.session_state.selected_run_id = None
    st.rerun()

# Tabs
tab_overview, tab_runs, tab_agent, tab_vram = st.tabs(["Overview", "Runs Explorer", "Agent Analysis", "VRAM & Counters"])

with tab_overview:
    st.subheader("Track Summary")
    summ = compute_track_summary(fdf)
    st.dataframe(summ, use_container_width=True)
    st.caption("Extend data.py for more GPUOS-specific aggregates (e.g. p95 recall, vram budget pass %).")

    # Simple charts (streamlit native)
    if "kernel_ms" in fdf and len(fdf):
        st.scatter_chart(fdf, x="num_candidates", y="latency_per_turn_ms", color="track", size="recall_at_k")
    if "vram_delta_mib" in fdf:
        st.bar_chart(fdf.groupby("track")["vram_delta_mib"].mean())

with tab_runs:
    st.subheader("Runs")
    disp_cols = [c for c in CORE_DISPLAY_COLS if c in fdf.columns]
    event = st.dataframe(
        fdf[disp_cols],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    if event and event.selection and event.selection.rows:
        sel_idx = event.selection.rows[0]
        sel_run = fdf.iloc[sel_idx].to_dict()
        st.session_state.selected_run_id = sel_run.get("id") or sel_run.get("experiment_name")
    # Simple selector
    run_ids = fdf.get("id", fdf.get("experiment_name", pd.Series())).astype(str).tolist()
    chosen = st.selectbox("Inspect run", run_ids, index=0 if run_ids else None, key="run_sel")
    if chosen:
        match = fdf[fdf.get("id", fdf.get("experiment_name")).astype(str) == chosen]
        if not match.empty:
            run = match.iloc[0].to_dict()
            st.markdown(f"### Run: {run.get('experiment_name')} — {run.get('track')}")
            st.json({k: run.get(k) for k in ["kernel_ms", "vram_delta_mib", "recall_at_k", "sync_processed", "success", "config"] if k in run})
            obs = load_observations_for_run(run)
            if obs:
                st.write("Observations:", obs[:5])
            # hygiene actions
            rid = str(run.get("id") or run.get("experiment_name"))
            c1, c2, c3 = st.columns(3)
            if c1.button("📦 Archive" if rid not in st.session_state.archived_ids else "Unarchive"):
                (st.session_state.archived_ids.add if rid not in st.session_state.archived_ids else st.session_state.archived_ids.remove)(rid)
                st.rerun()
            if c2.button("⭐ Mark baseline"):
                st.session_state.baseline_ids.add(rid)
                st.rerun()
            if c3.button("📝 Exploratory"):
                st.session_state.exploratory_ids.add(rid)
                st.rerun()

with tab_agent:
    st.subheader("Agent Utility & Accuracy (Phase2 end-state focus)")
    agent_df = fdf[fdf["track"].str.contains("agent|ranking", case=False, na=False)]
    if not agent_df.empty:
        st.dataframe(agent_df[["experiment_name", "num_candidates", "recall_at_k", "latency_per_turn_ms", "speedup"]].dropna(how="all"))
        st.line_chart(agent_df.set_index("num_candidates")[["recall_at_k", "latency_per_turn_ms"]])
    else:
        st.info("Run the agent demo with --log (or harness) to populate agent-ranking tracks. Shows recall vs cands, latency (utility), quality proxy (accuracy). Ties to 'sorting from WASM limits'.")

with tab_vram:
    st.subheader("VRAM & Sync Counters (Phase0/1 foundation)")
    if "vram_delta_mib" in fdf:
        st.bar_chart(fdf.groupby("phase")["vram_delta_mib"].mean())
    if "sync_heartbeat" in fdf:
        st.scatter_chart(fdf, x="kernel_ms", y="sync_heartbeat", color="track")
    st.caption("Target: vram_delta <<10% of idle. Counters should be monotonic, final processed == submitted for clean baselines.")

st.caption("Data from experiments/results/*.json + benchmark_results/. Drop new json -> reload. See experiments/README.md and GPUOS-Baseline-Documented-Arc-2026-06.md")
