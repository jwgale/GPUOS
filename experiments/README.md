# GPUOS Experiments, Testing Harness & Dashboard

Adapted from the Percepta Transformer-VM sudoku/A*/ExactCover testing dashboard (SurrealDB + Streamlit v2 with local JSON preference) to cover bases for GPUOS subsequent phases (Phase 2+ agent retrieval/ranking end-state, custom Triton ops, full JIT, long-horizon reliability, config variants, "twists and turns").

## Goals
- Reproducible experiment runs with structured metrics (kernel time, VRAM delta, SyncState counters, agent recall@K / latency utility, success, config).
- Data hygiene (tag baseline/exploratory, archive) and history for multi-session work.
- Live interactive dashboard (charts for scaling, vram, recall vs cands, counter health).
- Automated harness that always protects the "baseline only" clean exit + 5ms numbers (per AGENTS).
- Zero-setup primary path (local JSON in results/; streamlit + pandas only; surreal optional via docker + pip).
- Integrates existing benchmarks/ + agent demo + persistent_jit runs.

## Quick Start (safe, matches AGENTS style)
```bash
# From GPUOS worktree root
cd experiments

# 1. (Optional) Surreal for full cross-project sharing (like sudoku era)
# docker compose up -d   # starts on 8001, rocksdb

# 2. Start dashboard (uses port 8502 for GPUOS)
./start_gpuos_dashboard.sh
# or directly:
# streamlit run dashboard/app.py --server.port 8502

# 3. In another term, run the test harness (exercises jit baseline, agent demo, py tests, logs results)
python run_all_gpuos_tests.py --include-jit --include-agent --timeout 120

# Results appear in experiments/results/ and in the dashboard on reload.
# (Optional, human-driven only for live view e.g. machine speed queries during unattended): nvidia-smi -l 1 in another terminal. Standard results use internal [VRAM]/counters from code (harness + dashboard). See plan.md.
```

See AGENTS.md for the "Test & Dashboard Commands" section (added as part of this infra).

## Structure
- results/ : *.json experiment records (preferred data source; gitignore bulk, commit samples or key runs)
- dashboard/ : app.py + data.py (streamlit, modeled on sudoku v2.5: local-first, @cache, filters, detail inspector, hygiene session state, GPUOS-specific tabs/charts for agent utility/accuracy + Phase0/1 counters/vram)
- gpuos_experiment_logger.py : Python logger (always writes json; optional surreal connect; helpers for parsing jit output, bench results, agent runs)
- run_all_gpuos_tests.py : Master harness (timeout-safe runs of C++ tests, py smokes, agent demo w/ trials+recall, jit baseline with force protection + parse, bench subset; auto-logs; asserts baseline clean + exit 0 + reasonable counters/recall)
- start_gpuos_dashboard.sh + docker-compose.yml (adapted from sudoku setup; port 8502)
- (scripts/ for future log_run etc.)

## Data Shape (example, see logger)
See gpuos_experiment_logger.py and the sudoku schema for fields. GPUOS emphasizes:
- run: kernel_ms, vram_*, sync_*, num_candidates, recall_at_k, latency_per_turn_ms, speedup, phase, config (arch, mode)
- verification: baseline_exit_clean, counters_monotonic, vram_under_10pct, recall_ok, ...
- observations: for heartbeat traces, per-turn details, etc.

The dashboard normalizes benchmark_results/*.json into "bench-*" tracks too.

## Integration
- benchmarks/ ResultsReporter + run_all now support --log-experiments (writes to here)
- agent_retrieval_rank_demo.py supports --trials N --log (multi-seed, logs mean recall/latency as experiment)
- Harness wraps persistent_jit runs (parses the [BASELINE]/[VRAM]/sync lines from stdout for auto logging)
- All respect: exact AGENTS build (g++-12, compute_120), timeout, baseline block always produces numbers+clean exit 0, internal instrumentation ([VRAM] via cudaMemGetInfo + SyncState zero-copy direct poll + structured prints) + harness/logger parse. (External nvidia optional manual for live view only.)

## Adding a new "track" or metric
1. Run your code (demo, bench, jit with env).
2. Use logger in py or have harness parse.
3. Drop json in results/.
4. Reload dashboard; new fields appear in tables/details (edit data.py for derived/summary/charts).

## Why this (for twists and turns)
Gives the same visibility+history the sudoku runs had (when we were debugging 100% acc WASM VM inside model, then sorting, then pivot). Now for GPUOS: we can log "before/after" a new custom op, a scheduler change, a counter addition, different agent scoring; dashboard shows if recall/utility regressed, if vram crept, if counters stayed healthy over long runs. Combined with Phase1 SyncState + baseline force, we stay safe while exploring.

All changes followed AGENTS: re-read required md first, todo, context_record, post-edit rebuild+run+capture+doc update, diff review.

See GPUOS-Baseline-Documented-Arc-2026-06.md and vault for the full arc (this testing closes the "how we track the road" gap).

**Live tested (on feat/testing-infra-dashboard-scrapped-polling, pushed to jwgale/GPUOS):** Ran AGENTS Test & Dashboard commands + direct. Harness: PASS, baseline showed internal [VRAM] 0 delta + "5 ms" + sync counters + "FINAL (baseline only)" clean exit 0 + logs. Dashboard data: 34+ rows, agent-ranking (recall e.g. 0.5) + jit-baseline (5ms, 0 vram) tracks from internal only. Demo --log: showed "on the board" + logged. Direct jit: confirmed 0 delta + FINAL. (Graceful paths in no-torch env; real scheduler in torch+ext env.) git diff reviewed; changes documented in commit 1e25070 + this note. See plan.md for details.

## Status
Built as part of "figure out testing + suite + build missing for subsequent phases". Run the harness + dashboard to see it live with your Phase0/1 + agent data.
