# AGENTS.md - Rules for Grok Build, Hermes, and any AI coding agents working on GPUOS

## Project Overview
GPUOS (Persistent Kernel + JIT-Injected Operators for CUDA) - Research into low-overhead persistent GPU kernels with runtime operator swapping via NVRTC and function pointer tables.

Target: ≤10% VRAM overhead for instrumentation.

Current focus (as of 2026-06): Phase 0 Exploration & Baseline on RTX 5090 (Blackwell, compute 12.0).

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

Monitor VRAM in another terminal:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

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
- After changes: rebuild, run with timeout + nvidia-smi, capture output, update the mapping/checklist docs.
- Use subagents: "explore" for digging into code, "plan" for next phases, "general-purpose" for implementation.
- Share knowledge with other agents (e.g., Hermes) via context layer and this AGENTS.md.

## Safety
- Never run destructive commands without explicit approval in the plan.
- Work on dedicated branches for agent sessions.
- Review all diffs (`git diff`) before committing/pushing.

## Next Priorities (update as we progress)
- Complete Phase 0 baseline capture with current instrumentation.
- Debug/fix the hang in cuModuleLoadDataEx for full JIT path.
- Move to Phase 1: minimal atomic counter layer + re-measure VRAM impact.
- Add more structured output from persistent_jit for future instrumentation (counters, timings per stage, etc.).

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
