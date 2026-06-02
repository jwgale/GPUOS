"""
data.py — GPUOS Experiment Dashboard data layer (v2 style, local JSON first).

Handles:
- experiments/results/*.json (new GPUOS runs from harness, logger, agent demo, jit parses)
- benchmark_results/*.json (existing perf benches, normalized to "track")
- Field normalization, derived (vram_delta, success inference, agent utility)
- Filters, summaries, observations
- Status badges tailored to GPUOS (baseline clean, agent recall, phase tags)
- Caching for streamlit

No heavy deps beyond pandas (streamlit provides UI). Surreal fallback not implemented here (json preferred).

Edit this for new GPUOS fields/metrics.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Paths (relative to dashboard/ -> experiments/ -> root)
ROOT = Path(__file__).parent.parent.parent
GPUOS_RESULTS = ROOT / "experiments" / "results"
BENCH_RESULTS = ROOT / "benchmark_results"

CORE_GPUOS_COLS = [
    "experiment_name", "track", "phase", "status", "engine",
    "kernel_ms", "wall_ms", "vram_delta_mib", "sync_processed", "sync_heartbeat",
    "num_candidates", "recall_at_k", "latency_per_turn_ms", "speedup",
    "success", "config_arch", "created_at",
]

def _load_json_dir(d: Path) -> List[Dict]:
    rows = []
    if not d.exists():
        return rows
    for jf in sorted(d.glob("*.json")):
        try:
            doc = json.loads(jf.read_text())
            exp = doc.get("experiment", {}) or {"name": jf.stem, "track": "unknown"}
            exp_name = exp.get("name", jf.stem)
            track = exp.get("track", "unknown")
            phase = exp.get("phase", "0")
            for r in doc.get("runs", []):
                row = {
                    "source_file": str(jf),
                    "experiment_name": exp_name,
                    "track": track,
                    "phase": phase,
                    "status": "success" if r.get("success") else "failed",
                    "engine": r.get("engine", "unknown"),
                    "kernel_ms": r.get("kernel_ms"),
                    "wall_ms": r.get("wall_ms"),
                    "vram_pre_mib": r.get("vram_pre_mib"),
                    "vram_post_mib": r.get("vram_post_mib"),
                    "vram_delta_mib": r.get("vram_delta_mib"),
                    "sync_processed": r.get("sync_processed"),
                    "sync_submitted": r.get("sync_submitted"),
                    "sync_heartbeat": r.get("sync_heartbeat"),
                    "num_candidates": r.get("num_candidates"),
                    "k": r.get("k"),
                    "recall_at_k": r.get("recall_at_k"),
                    "latency_per_turn_ms": r.get("latency_per_turn_ms"),
                    "speedup": r.get("speedup"),
                    "candidates_per_sec": r.get("candidates_per_sec"),
                    "success": bool(r.get("success", True)),
                    "config": r.get("config", {}),
                    "notes": r.get("notes"),
                    "created_at": r.get("created_at"),
                    "raw_output_snippet": r.get("raw_output_snippet"),
                    "observations": r.get("observations", []),
                }
                # flatten a bit
                cfg = row["config"] or {}
                row["config_arch"] = cfg.get("arch") or cfg.get("GPUOS_NVRTC_ARCH") or "compute_120"
                row["config_mode"] = cfg.get("mode") or cfg.get("jit_mode") or "baseline"
                rows.append(row)
            # also support flat bench-style list of dicts (from ResultsReporter)
            if isinstance(doc, list):
                for item in doc:
                    if isinstance(item, dict) and ("mean_time_ms" in item or "pytorch_mean_ms" in item):
                        ms = item.get("mean_time_ms") or item.get("pytorch_mean_ms")
                        rows.append({
                            "source_file": str(jf),
                            "experiment_name": item.get("name", jf.stem),
                            "track": "bench-" + str(item.get("name", "unknown")).split("_")[0],
                            "phase": "bench",
                            "status": "success",
                            "engine": "pytorch",
                            "kernel_ms": ms,
                            "success": True,
                            "config_arch": "n/a",
                            "created_at": None,
                            "num_candidates": None,
                            "recall_at_k": None,
                        })
        except Exception as e:
            print(f"[data] skip bad json {jf}: {e}")
    return rows

@st.cache_data(ttl=30)
def fetch_runs() -> pd.DataFrame:
    gpuos = _load_json_dir(GPUOS_RESULTS)
    bench = _load_json_dir(BENCH_RESULTS)
    allr = gpuos + bench
    if not allr:
        return pd.DataFrame(columns=CORE_GPUOS_COLS + ["source_file"])
    df = pd.DataFrame(allr)
    # normalize
    for c in ["kernel_ms", "vram_delta_mib", "recall_at_k", "latency_per_turn_ms", "speedup"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df.get("status", "success")
    if "success" in df:
        df.loc[~df["success"].fillna(True), "status"] = "failed"
    df = df.sort_values("created_at", ascending=False, na_position="last")
    return df

def apply_filters(df: pd.DataFrame, 
                  tracks: Optional[List[str]] = None,
                  phases: Optional[List[str]] = None,
                  success_only: bool = False) -> pd.DataFrame:
    if tracks:
        df = df[df["track"].isin(tracks)]
    if phases:
        df = df[df["phase"].astype(str).isin([str(p) for p in phases])]
    if success_only and "success" in df:
        df = df[df["success"].fillna(True)]
    return df

def compute_track_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("track", dropna=False)
    out = g.agg(
        runs=("experiment_name", "count"),
        mean_kernel_ms=("kernel_ms", "mean"),
        mean_vram_delta=("vram_delta_mib", "mean"),
        mean_recall=("recall_at_k", "mean"),
        mean_speedup=("speedup", "mean"),
    ).reset_index()
    return out

def get_status_badge(status: str) -> str:
    return {
        "success": "🟢 clean",
        "failed": "🔴 failed",
        "partial": "🟡 partial",
    }.get(str(status).lower(), "⚪ " + str(status))

def load_observations_for_run(run_row: Dict) -> List[Dict]:
    return run_row.get("observations", []) or []

# For app
STATUS_BADGES = {"success": "🟢", "failed": "🔴", "partial": "🟡"}
CORE_DISPLAY_COLS = [c for c in CORE_GPUOS_COLS if c != "config"]
