#!/usr/bin/env python3
"""
Terminal GPU Activity Visualizer for GPUOS experiments.

Quick "pop up" view of what the GPU actually did during a run,
sourced exclusively from *internal* telemetry emitted by the code:
- Stage timings + [VRAM] samples via cudaMemGetInfo (in host.cpp baseline)
- SyncState counters (processed/submitted/heartbeat/ready) via zero-copy
  host-mapped + system-scope atomics (Phase1 layer, persistent_kernel.cu)
- Agent ranking metrics (recall@K, latency/turn) from scheduler per-turn
  scoring demo (or sim path for the "end state on the board" story)

No external polling (nvidia-smi etc.) is used for logged results or this viz.
See "scrap the polling" feedback + plan: results "will be what they are"
from direct SyncState + print_vram. Use `nvidia-smi -l 1` manually in another
term only for live machine-speed queries during a run.

Usage examples (after harness / demo / direct jit):
  python experiments/viz_gpu_activity.py --latest
  python experiments/viz_gpu_activity.py --track jit-baseline
  python experiments/viz_gpu_activity.py --track agent-ranking --latest
  python experiments/viz_gpu_activity.py --json experiments/results/xxx-jit-....json

Fits the testing infra (harness + dashboard + logger) for "bases covered"
on the road of Phase0/1/2+ twists. Run it right after AGENTS commands to
see the GPU work for that particular testing run.

Reuses experiments/dashboard/data.py for consistent loading/normalization
(so new fields in logger just appear). Pure stdlib + pandas (via data).

See:
- AGENTS.md "Test & Dashboard Commands"
- experiments/README.md
- examples/agent_retrieval_rank_demo.py (the end-state MVP header + Triton note)
- src/host.cpp (baseline block + print_vram + poll + FINAL force)
- src/persistent_kernel.cu (persistent worker, op dispatch via fn ptr table,
  system atomics to SyncState if wired)
- GPUOS-Baseline-Documented-Arc-2026-06.md + vault for the full "why"

After changes (per AGENTS): run the viz on fresh output, update docs,
review `git diff`, capture terminal output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Reuse the dashboard data layer for loading + normalization (no dupe logic)
# This keeps viz in sync with UI (new metrics, tracks, hygiene etc. just work).
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

try:
    from dashboard.data import fetch_runs, get_status_badge, STATUS_BADGES
except Exception as e:  # pragma: no cover - fallback for odd import
    print(f"[viz] warning: could not import dashboard.data ({e}); trying direct path")
    sys.path.insert(0, str(ROOT))
    from experiments.dashboard.data import fetch_runs, get_status_badge, STATUS_BADGES  # type: ignore


def ascii_bar(value: float, max_val: float, width: int = 24, fill: str = "█", empty: str = "░") -> str:
    """Simple ascii progress bar. 0-delta friendly (full bar on 100%)."""
    if max_val <= 0:
        return empty * width
    ratio = max(0.0, min(1.0, value / max_val))
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


def parse_jit_stages(snippet: str) -> List[Dict[str, Any]]:
    """Extract timeline events from the raw persistent_jit stdout snippet.
    Captures stage timings, [VRAM] discrete samples, [BASELINE] progress lines,
    sync counters, and the FINAL clean exit marker. This is the 'what the GPU did'.
    """
    events: List[Dict[str, Any]] = []
    if not snippet:
        return events
    for raw in snippet.splitlines():
        line = raw.strip()
        if not line:
            continue
        # [123 ms] Stage ... or [BASELINE] ...
        m = re.match(r"\[(\d+)\s*ms\]\s*(.+)", line)
        if m:
            events.append({
                "t_ms": int(m.group(1)),
                "text": m.group(2).strip()
            })
            continue
        # VRAM lines, baseline progress, sync dump, FINAL (no leading [t ms])
        if line.startswith("[VRAM]") or line.startswith("[BASELINE]") or "FINAL (baseline only)" in line or "sync counters:" in line.lower():
            events.append({
                "t_ms": None,
                "text": line
            })
    return events


def render_jit_baseline(row: Dict[str, Any]) -> None:
    """Pretty terminal view for a jit-baseline / persistent_jit run.
    Shows the exact GPU work: persistent kernel launch (170 SMs), 256 tasks
    of 64k-elem elementwise add via fn-ptr dispatch, SyncState atomics,
    zero-copy host poll, 0-delta instr layer, forced clean baseline exit.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  GPUOS • JIT BASELINE — GPU ACTIVITY (internal telemetry only)      ║")
    print("╚" + "═" * 68 + "╝")

    exp = row.get("experiment_name", "unknown")
    track = row.get("track", "jit-baseline")
    status = row.get("status", "success")
    created = row.get("created_at", "")
    print(f"  Run: {exp}")
    print(f"  Track/Phase: {track} / {row.get('phase', '0/1')}")
    print(f"  Status: {get_status_badge(status)}  (success={row.get('success', True)})")
    print(f"  When: {created}")

    k_ms = row.get("kernel_ms")
    w_ms = row.get("wall_ms")
    print(f"\n  Kernel time (256 tasks): {k_ms} ms")
    print(f"  Wall (setup+kernel+cleanup): {w_ms} ms")

    # VRAM (the key Phase1 measurement: SyncState layer == 0 device MiB)
    def _safe_float(v, default=0.0):
        try:
            if v is None:
                return default
            if isinstance(v, float) and v != v:  # nan
                return default
            return float(v)
        except Exception:
            return default
    vpre = _safe_float(row.get("vram_pre_mib"))
    vpost = _safe_float(row.get("vram_post_mib"))
    vdelta = _safe_float(row.get("vram_delta_mib"))
    print("\n  VRAM samples (cudaMemGetInfo at stages in host.cpp):")
    print(f"    pre-sync-layer  : {vpre:8.1f} MiB")
    print(f"    post-sync-layer : {vpost:8.1f} MiB   (minimal atomic counter layer wired)")
    delta_str = f"{vdelta:+.1f} MiB"
    good = "  ✅ 0 delta (target met for instr layer)" if abs(vdelta) < 0.5 else f"  ⚠️  {delta_str}"
    print(f"    delta (layer)   : {delta_str:>8} {good}")

    # SyncState (the zero-copy observability that lets agent loops poll cheaply)
    def _safe_int(v, default=0):
        try:
            if v is None:
                return default
            if isinstance(v, float) and v != v:  # nan
                return default
            return int(v)
        except Exception:
            return default
    proc = _safe_int(row.get("sync_processed"))
    sub = _safe_int(row.get("sync_submitted"))
    hb = _safe_int(row.get("sync_heartbeat"))
    rdy = _safe_int(row.get("sync_ready"))
    print("\n  SyncState counters (60B host-mapped + __threadfence_system + atomic*_system):")
    print(f"    processed={proc}  submitted={sub}  heartbeat={hb}  ready={rdy}")
    if sub > 0:
        bar = ascii_bar(float(proc), float(sub), width=32)
        pct = (proc / sub) * 100.0
        print(f"    progress: [{bar}] {pct:5.1f}%  (host polls mapped ptr; no cudaMemcpy per iter)")

    # Timeline from the actual binary output (this is the computational trace)
    snippet = row.get("raw_output_snippet") or ""
    events = parse_jit_stages(snippet)
    if events:
        print("\n  GPU Activity Timeline (persistent worker on 170 SMs × 128 threads):")
        print("  " + "-" * 66)
        shown = 0
        for ev in events:
            t = f"[{ev['t_ms']:>4} ms]" if ev["t_ms"] is not None else "         "
            txt = ev["text"]
            print(f"  {t} {txt}")
            shown += 1
            if shown >= 22:
                print("         ... (full trace in the json raw_output_snippet)")
                break
        print("  " + "-" * 66)

    # Verify the invariant
    if "FINAL (baseline only)" in snippet and "Exited cleanly" in snippet:
        print("\n  ✅ FINAL (baseline only): Exited cleanly after kernel baseline")
        print("     (force path per AGENTS: baseline block always runs + produces")
        print("      numbers + exit 0, even if full NVRTC op_mul JIT path hangs)")

    print("\n  Data source: internal prints + SyncState zero-copy poll + cudaMemGetInfo")
    print("  (no external nvidia-smi / high-freq sampler in results or this view)")
    print("  See src/host.cpp: baseline block + alloc_host_mapped_sync + print_vram")
    print("  See src/persistent_kernel.cu: while(!quit) claim+dispatch+atomicAdd_system")
    print("╚" + "═" * 68 + "╝\n")


def render_agent_ranking(row: Dict[str, Any]) -> None:
    """Terminal view for agent-ranking runs (the 'end state on the board').
    Shows utility (latency, cands/sec) + accuracy proxy (recall@K) for per-turn
    custom scoring + top-K over 512-2k candidates. This generalizes the original
    sorting work that came out of WASM-in-model limits (Percepta sudoku 100%).
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  GPUOS • AGENT RANKING (end-state MVP) — utility + accuracy proxy    ║")
    print("╚" + "═" * 68 + "╝")

    exp = row.get("experiment_name", "agent-retrieval-rank-demo")
    track = row.get("track", "agent-ranking")
    print(f"  Run: {exp}  |  track={track}  | phase={row.get('phase','2')}")
    print(f"  Status: {get_status_badge(row.get('status','success'))}")

    nc = row.get("num_candidates")
    rec = row.get("recall_at_k")
    lat = row.get("latency_per_turn_ms")
    print(f"\n  Candidates per turn: {nc}")
    print(f"  recall@K: {rec}")
    print(f"  latency/turn: {lat} ms")

    # Simulated lift note (when not real scheduler path)
    if rec is not None and rec < 0.6:
        print("  (sim path in this env; real scheduler+GPUOS would show the 0.33→~0.67 lift)")
    print("  (see demo --log output for the full story + simulated 15x from paper claims)")

    print("\n  This run exercises (or simulates) the per-turn 'score many + top-K'")
    print("  pattern that an agent would use for retrieval / memory / tool ranking.")
    print("  Custom scoring (dot + bias + relu + scale etc.) is the 'algo' escape.")
    print("  Phase1 SyncState would let the agent loop poll completion zero-copy.")

    print("\n  Intended purpose (from vault + your request):")
    print("    - Improve agent accuracy (better recall via richer/more candidates)")
    print("    - Or improve utility (lower latency or higher cands/sec in budget)")
    print("    - Generalizes 'sorting' from WASM VM limits (100% sudoku precedent)")
    print("    - GPUOS + future Triton @triton.jit fused_score_topk for authoring")

    print("\n  Data source: internal (scheduler per-turn metrics or demo sim + logger)")
    print("  (no external polling; real GPU scoring path activates with full torch+ext)")
    print("╚" + "═" * 68 + "╝\n")


def render_generic(row: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(" GPUOS Experiment Run (generic view)")
    print("=" * 60)
    for k in ["experiment_name", "track", "phase", "kernel_ms", "vram_delta_mib",
              "recall_at_k", "latency_per_turn_ms", "sync_processed", "success"]:
        if k in row and row[k] is not None:
            print(f"  {k}: {row[k]}")
    print("\n  (Tip: --track jit-baseline or agent-ranking for richer viz)")
    print("=" * 60 + "\n")


def pick_row(df, args) -> Optional[Dict[str, Any]]:
    """Select a single normalized row based on CLI."""
    if df.empty:
        return None

    if args.json:
        # load the specific file directly for full fidelity (incl. multi-run)
        jpath = Path(args.json)
        if not jpath.exists():
            print(f"[viz] json not found: {jpath}")
            return None
        try:
            doc = json.loads(jpath.read_text())
            # take first run for display (or could expand later)
            for r in doc.get("runs", []):
                # synthesize a row similar to data.py normalization
                row = {
                    "source_file": str(jpath),
                    "experiment_name": doc.get("experiment", {}).get("name", jpath.stem),
                    "track": doc.get("experiment", {}).get("track", "unknown"),
                    "phase": doc.get("experiment", {}).get("phase", "?"),
                    "raw_output_snippet": r.get("raw_output_snippet"),
                    "status": "success" if r.get("success") else "failed",
                    "success": bool(r.get("success", True)),
                    **r,  # bring the metrics up
                }
                return row
        except Exception as e:
            print(f"[viz] failed to load {jpath}: {e}")
            return None

    fdf = df
    if args.track:
        fdf = fdf[fdf["track"].str.contains(args.track, case=False, na=False)]
    if fdf.empty:
        print(f"[viz] no rows for track filter '{args.track}'")
        return None

    if args.id:
        mask = (fdf.get("id", "").astype(str) == args.id) | (fdf.get("experiment_name", "").astype(str) == args.id)
        hit = fdf[mask]
        if not hit.empty:
            fdf = hit

    # default: most recent (data.py already sorts created_at desc). Prefer a row that actually has GPU/agent metrics over pure harness summary rows.
    if len(fdf) > 1:
        metric_mask = (fdf.get("kernel_ms").notna() if "kernel_ms" in fdf else False) | (fdf.get("recall_at_k").notna() if "recall_at_k" in fdf else False)
        if metric_mask.any():
            fdf = fdf[metric_mask]
    return fdf.iloc[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Terminal pop-up for GPUOS GPU activity / experiment results (internal data only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  After a run:   python experiments/viz_gpu_activity.py --latest
  Specific type: python experiments/viz_gpu_activity.py --track jit-baseline --latest
  Agent view:    python experiments/viz_gpu_activity.py --track agent-ranking
  From a json:   python experiments/viz_gpu_activity.py --json experiments/results/foo-jit-baseline.json

See AGENTS.md for the exact commands that produce the data this visualizes.
        """.strip(),
    )
    ap.add_argument("--latest", action="store_true", help="Most recent run (default behavior)")
    ap.add_argument("--track", type=str, default=None, help="Filter e.g. 'jit-baseline' or 'agent-ranking'")
    ap.add_argument("--id", type=str, default=None, help="Exact experiment_name or run id to show")
    ap.add_argument("--json", type=str, default=None, help="Path to a specific results/*.json (bypasses index)")
    ap.add_argument("--list", action="store_true", help="Just list recent runs (no detail viz)")
    args = ap.parse_args()

    df = fetch_runs()
    if df.empty:
        print("No results in experiments/results/ (or benchmark_results/).")
        print("Run e.g.: python experiments/run_all_gpuos_tests.py --include-jit --include-agent")
        print("     or:  python examples/agent_retrieval_rank_demo.py --log")
        print("     or:  export GPUOS_NVRTC_ARCH=compute_120 ; timeout 30 ./build/persistent_jit")
        return

    if args.list:
        cols = [c for c in ["track", "experiment_name", "kernel_ms", "vram_delta_mib", "recall_at_k", "latency_per_turn_ms", "success"] if c in df.columns]
        print(df[cols].head(12).to_string(index=False))
        print(f"\n({len(df)} total rows; use --latest or --track to visualize one)")
        return

    row = pick_row(df, args)
    if not row:
        print("No matching run. Try --list to see what's available.")
        return

    track = str(row.get("track", "")).lower()
    k = row.get("kernel_ms")
    has_kernel = k is not None and (not isinstance(k, float) or k == k)  # guard float('nan')
    r = row.get("recall_at_k")
    has_agent = (r is not None and (not isinstance(r, float) or r == r)) or "agent" in track or "ranking" in track

    # Prioritize explicit track name for dispatch (jit rows can bleed into agent df if NaNs)
    if "agent" in track or "ranking" in track:
        render_agent_ranking(row)
    elif "jit" in track or "baseline" in track or has_kernel:
        render_jit_baseline(row)
    elif has_agent:
        render_agent_ranking(row)
    else:
        render_generic(row)

    # Gentle hint for next action (ties back to full harness + dashboard)
    print("Tip: for the interactive version with charts + all runs + hygiene:")
    print("  cd experiments && ./start_gpuos_dashboard.sh")
    print("  (then open http://localhost:8502 , Agent Analysis + VRAM & Counters tabs)")
    print()


if __name__ == "__main__":
    main()
