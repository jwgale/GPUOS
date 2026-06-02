# AGENTS.md - Rules for Grok Build, Hermes, and any AI coding agents working on GPUOS

## Project Overview
GPUOS (Persistent Kernel + JIT-Injected Operators for CUDA) - Research into low-overhead persistent GPU kernels with runtime operator swapping via NVRTC and function pointer tables.

Target: ≤10% VRAM overhead for instrumentation.

Current focus (as of 2026-06): Phase 1 - Minimal Atomic Counter Layer (SyncState zero-copy wiring in persistent_jit) + re-measure on RTX 5090 (Blackwell, compute 12.0).
(Phase 0 baseline + granular debug complete: always 5-7ms + clean exit via force block; post-launch cuModuleLoadDataEx hang diagnosed but protected.)

## Key Locked Decisions & Conventions
- Use `compute_120` for NVRTC on RTX 5090: `export GPUOS_NVRTC_ARCH=compute_120`
- Build with GCC 12 for host compiler compatibility: `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`
- Main binary: `persistent_jit` (in build/)
- Tests: `test_online_switch`, `test_dual_slot_switch` (currently using no-op for prefetch due to header mismatch)
- Data is pure; logic in Systems.
- Always add structured debug prints with `[DEBUG]` or stage timing for exploration phases.
- Baseline block in main() must always run and exit cleanly for measurement, even if full JIT path has issues.
- Use git: work on feature branches for agent changes. Review diffs before push.

## Build & Test Commands (safe for agents)
```bash
cd ~/projects/Percepta/GPUOS/build
export GPUOS_NVRTC_ARCH=compute_120
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..
cmake --build . -j$(nproc)
```

To run with timeout (recommended during debugging):
```bash
export GPUOS_NVRTC_ARCH=compute_120
timeout 30 ./persistent_jit || echo "=== TIMEOUT ==="
```

## Test & Dashboard Commands (for subsequent phases, agent end-state, reliability)
```bash
# Full harness (jit baseline always protected + clean exit, agent demo, py tests, logs to experiments/results/)
python experiments/run_all_gpuos_tests.py --include-jit --include-agent --timeout 60

# Dashboard (Streamlit, local JSON + bench results; port 8502; like sudoku-era for Percepta)
cd experiments
./start_gpuos_dashboard.sh
# then browse http://localhost:8502 (Overview, Runs, Agent Analysis with recall, VRAM/Counters tabs; hygiene tags)

# Individual
# For real GPUOS scheduler path + actual metrics in agent demo (real-world candidate ranking use case):
#   source .venv/bin/activate   # or use .venv/bin/python (has torch nightly for sm_120 support)
python examples/agent_retrieval_rank_demo.py --log   # logs recall/latency for dashboard (real when .venv)
python test_full_sync.py
export GPUOS_NVRTC_ARCH=compute_120 ; timeout 30 ./build/persistent_jit || echo TIMEOUT
python benchmarks/run_all_benchmarks.py --skip-mps --skip-mig --visualize   # still works; harness can --include-bench

# Terminal "pop-up" viz of GPU activity for a specific run (from internal SyncState + [VRAM] prints only)
python experiments/viz_gpu_activity.py --latest
python experiments/viz_gpu_activity.py --track jit-baseline --latest   # shows 5ms/0-delta/256/heartbeat + timeline from raw
python experiments/viz_gpu_activity.py --track agent-ranking --latest # recall/latency + end-state purpose
# (also --json <path> or --list; run right after the cmds above for "what the GPU did")
```
See experiments/README.md + GPUOS-Baseline-Documented-Arc-2026-06.md for the testing infra rationale (covers twists via experiment tracking + dashboard modeled on sudoku runs).

## Current Known Issues (as of latest)
- `persistent_jit` reaches NVRTC success for op_mul but hangs in `cuModuleLoadDataEx` inside `load_op_mul_ptr_from_ptx`.
- Test files use no-op for `cudaMemPrefetchAsync` to unblock build (API mismatch with current headers).
- nvlink warnings about old static libs (librt, pthread, dl) - harmless.
- CUDA Toolkit mix (headers from 12.0/12.8 while driver 580 supports 13.0).

## How to Work with This Project (for Grok Build / Hermes / Subagents)
- Always read AGENTS.md, GPUOS-Phase0-Exploration-Checklist.md, and GPUOS-Structure-Mapping.md first.
- Record new decisions using the context layer (context_record).
- Use todo_write for multi-step tasks.
- Prefer plan mode (enter_plan_mode) for architectural or large changes.
- For experiments, use isolated worktrees via subagents.
- Add `[DEBUG]` prints liberally during exploration; remove or gate them for production.
- When instrumenting, ensure a "baseline only" path exists that produces numbers and exits cleanly.
- After changes: rebuild (exact AGENTS), run with timeout (or full `python experiments/run_all_gpuos_tests.py ...`), capture output (internal structured [VRAM]/counters from direct poll + print_vram + timings; see harness), update the mapping/checklist/AGENTS/experiments/README docs + view dashboard for history. Review `git diff`. (Optional manual nvidia-smi -l 1 in another terminal for live machine view during long/unattended runs e.g. speed queries -- not required or used in standard verification/harness; results will be what they are from internal per plan/feedback.)
- Use subagents: "explore" for digging into code, "plan" for next phases, "general-purpose" for implementation.
- Share knowledge with other agents (e.g., Hermes) via context layer and this AGENTS.md.

## Safety
- Never run destructive commands without explicit approval in the plan.
- Work on dedicated branches for agent sessions.
- Review all diffs (`git diff`) before committing/pushing.

## Next Priorities (update as we progress)
- [done] Complete Phase 0 baseline capture with current instrumentation + granular debug (hang point identified; baseline always clean).
- Debug/fix the hang in cuModuleLoadDataEx for full JIT path (still protected by baseline force-exit; pre-launch load succeeds).
- [done in this session] Move to Phase 1: minimal atomic counter layer (SyncState host-mapped zero-copy + direct poll + submitted tracking + cudaMemGetInfo) wired in persistent_jit + re-measure VRAM impact (layer adds ~0 device MiB; baseline metrics identical).
- Add more structured output from persistent_jit for future instrumentation (counters, timings per stage, etc.) — Phase 1 added [VRAM] + sync counter dumps in baseline; clock64 can be next.
- **Achievable end state on the board (agent accuracy/utility)**: See approved plan + .tasks/phase2-agent-endstate.md + examples/agent_retrieval_rank_demo.py + `GPUOS-Baseline-Documented-Arc-2026-06.md` (local) + vault master `GPUOS-Phase2-EndState-Baseline-2026-06.md`. GPUOS + Triton for custom ranking/scoring primitives in agent loops (generalizes original "sorting" from WASM-VM limits). Primary: retrieval/candidate ranking with Triton topk (builds on attention "scoring many" + scheduler per-turn patterns). Measure latency/candidates (utility) + quality proxy (accuracy). Phase 0/1 is the foundation (low overhead proof + reliable baseline). Brainstormed alternatives: per-step verifier, planning scores, long-running agent server, etc. For major baseline/arc docs: also write to Obsidian (source of truth) + local summary + Structured Context Layer (context_record, namespace "gpuos").
- Update AGENTS when conventions change (e.g. "SyncState is the minimal atomic counter layer for efficient polling"; "Triton + GPUOS for agent custom algos / ranking primitives").

Update this file when conventions or decisions change.

## Grok Build Local CLI Features for Maximum Leverage (use these!)

When launching the local Grok Build agent in this project, use these flags for more power:

- `--plan`: Enter plan mode first (read-only exploration and planning before any edits). Perfect for big changes like Phase 1 instrumentation. Review the plan, then approve to execute.
- `-w, --worktree [name]`: Start in a new isolated git worktree. Great for safe parallel experiments (e.g., different counter implementations) without touching main branch.
- `--agents <JSON>` or multiple `--agent`: Spawn inline subagents (e.g., one "explore" for digging code, one "plan" for architecture). Use different types: general-purpose, explore, plan.
- `--todo-gate`: Enable strict todo tracking for the session.
- `agent` mode (headless): `grok agent --plan -p "Follow .tasks/current-task.md"` for scripted/non-interactive runs.
- Use subagents via the interface for parallel work (we've been doing this).

Other powerful built-ins:
- Direct tools: The agent can read_file, search_replace, run_terminal_command, list_dir, grep, etc. **directly on this machine's filesystem and terminal** (your 5090 GPU, nvcc, etc.). This is the key to "Grok guidance directly executed" without you editing manually.
- Context layer: Use context_record / context_query to persist architecture decisions, gotchas, and share with your Hermes agent or other sessions.
- Scheduler & monitor: For long-running builds or background monitoring of tests/benchmarks.
- Hooks: Define in project config for pre/post-edit, pre-commit, etc.
- Memory: Cross-session recall of previous decisions.

When asking the agent (in this terminal or via flags):
- Say "enter plan mode" or use --plan for big work.
- "Use a worktree for this experiment."
- "Spawn an explore subagent to map the JIT loading code."
- Always reference this AGENTS.md, the checklist, and mapping doc.

Launch example for heavy work:
```bash
cd ~/projects/Percepta/GPUOS
grok --plan -w gpuos-phase1 --todo-gate "Implement Phase 1 atomic counter layer per the plan in .tasks/phase1-plan.md. Use the build commands from AGENTS.md. Work in this worktree."
```

This setup, combined with structured plans from our conversations here, gives massive leverage and removes you as the editing bottleneck.

Update this file with any new conventions or gotchas discovered.
