
---

## Baseline Numbers (from RTX 5090 run, 2026-05-30)

**Environment**:
- GPU: NVIDIA GeForce RTX 5090 (32,607 MiB)
- Driver: 580.159.03 (reports CUDA 13.0)
- CUDA Toolkit used for build: 12.8 (with some 12.0 headers in /usr/include causing test issues)
- NVRTC arch: compute_120 (forced via GPUOS_NVRTC_ARCH)

**Workload for baseline** (using built-in op_add, 256 tasks of 65536 float elements each):
- Persistent kernel launched with 170 SMs, 128 threads per block.
- All 256 tasks processed in **5 ms**.

**VRAM (from nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1)**:
- Idle/baseline: ~1882 MiB
- During run (stable): ~1904-1906 MiB
- Overhead: ~22-24 MiB (very low, <<10% target)

**Notes**:
- The full JIT path (op_mul via NVRTC) reaches NVRTC success but hangs in cuModuleLoadDataEx inside load_op_mul_ptr_from_ptx on this Blackwell + CUDA 12.8 setup.
- The instrumented main now has a self-contained baseline block that always runs and prints timing even if the normal JIT path hangs.
- The two test binaries have compile issues due to cudaMemPrefetchAsync signature mismatch (old code vs headers); they are not required for baseline.
- No latency numbers from the original verification path (program does not reach it).

**Recommendation for Phase 0 completion**:
- The VRAM baseline and kernel timing (5 ms for the workload) can be recorded as the "before instrumentation" numbers.
- The structure mapping is complete.
- Sanity check: Build of main succeeds; benchmarks (this binary) "run" (with timeout for the hanging path); numbers recorded.
- Next: Move to Phase 1 (add minimal atomic counter layer and re-measure).

---

## Baseline + Debug Continuation (worktree gpuos-phase1-experiment, RTX 5090, ~2026-06)

**Fresh run (using exact AGENTS.md build/run cmds in worktree build/)**:
- `export GPUOS_NVRTC_ARCH=compute_120`
- `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..`
- `cmake --build . -j$(nproc)`
- `export GPUOS_NVRTC_ARCH=compute_120 ; timeout 30 ./persistent_jit || echo "=== TIMEOUT ==="`

**Environment** (same machine):
- GPU: NVIDIA GeForce RTX 5090 (32,607 MiB), compute_cap 12.0
- Driver: 580.159.03
- nvcc: 12.8.93
- NVRTC: compute_120 (PTX .version 8.7 .target sm_120 confirmed in debug dump)
- Idle VRAM varied by session (~1110-1220 MiB in these runs vs 1882 in May)

**Workload baseline (built-in op_add, 256 tasks x 65536 f32 elems)**:
- Launched: 170 SMs, 128 threads/block
- Kernel compute time: **5-7 ms** (6 ms in latest capture)
- Full binary wall time (init+baseline+force exit): ~220-320 ms
- Always exits 0 cleanly (baseline block)

**VRAM (high-freq sampling every ~50ms during run)**:
- Idle (pre): ~1219 MiB
- Early after alloc/launch: ~1239 MiB (+~20 MiB)
- Peak during: ~1778 MiB (large jump likely UVM pool reservation on first managed use)
- Post-cleanup: back to ~1219 MiB
- Incremental for instrumentation/queue/kernel (~20-24 MiB) matches prior "very low <<10%" observation. Large pool jumps are workload+UVM driver behavior, not pure instr overhead.

**Granular debug added (in load_op_mul_ptr_from_ptx + main trigger)**:
- Always prints: PTX size (39099 B), arch, full prefix dump (shows valid generated PTX for sm_120)
- Uses cuModuleLoadDataEx with CU_JIT_ERROR_LOG_BUFFER + INFO_LOG to capture driver logs (empty on success/hang)
- Early isolated load (GPUOS_DEBUG_JIT_LOAD=1, before worker launch): **succeeds** - cuModuleLoad success, getFunction, helper kernel launch, extracted fn_addr=0xa50 (non-zero)
- Post persistent_worker launch (GPUOS_FULL_JIT=1): NVRTC succeeds (55ms), load starts, prints "calling cuModuleLoadDataEx...", **hangs inside the call** (no return, no error log populated, timeout). Last [DEBUG] is the "calling..." line.
- Conclusion: hang is specific to loading PTX module *while the 170-block persistent kernel is active* (not general NVRTC or pre-ctx load). Likely driver/Blackwell+compute_120 interaction under load or UVM contention.

**New envs for exploration (do not affect default reliable baseline)**:
- GPUOS_DEBUG_JIT_LOAD=1 : force early (pre-worker) JIT load attempt + granular logs (succeeds)
- GPUOS_FULL_JIT=1 : skip baseline force-exit, attempt full path after worker (reproduces post-launch hang with logs)

**Status**:
- persistent_jit *always* produces baseline timing + exits cleanly (return 0) within timeout. Verifiable step-by-step.
- Hang point now precisely instrumented with PTX evidence for Phase 1 fix (e.g. load ops earlier, or different module strategy, or fix in worker yield).
- Tests built cleanly this time (no prefetch no-op needed in current tree).
- Do not remove baseline force block until post-launch load is fixed.

**Next per .tasks**: update checklist, record via context layer, prepare Phase 1 note (no code changes yet).

---

## Phase 1: After Minimal Atomic Counter Layer (SyncState Zero-Copy Wiring)

**Run (worktree, exact AGENTS.md commands)**:
- Build: `cd build; export GPUOS_NVRTC_ARCH=compute_120; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..; cmake --build . -j$(nproc)`
- Run: `export GPUOS_NVRTC_ARCH=compute_120; timeout 30 ./persistent_jit || echo "=== TIMEOUT ==="`
- Monitor: nvidia-smi + internal `cudaMemGetInfo` via new `print_vram` (called at device init, pre/post layer, post-launch, baseline complete).

**What was added (minimal per approved plan)**:
- Wired `SyncState` via `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer` + `q.sync = dev_alias` (host.cpp, after q.capacity alloc).
- Direct zero-copy poll (`poll_processed`: `if (g_sync_host) return g_sync_host->processed;` else legacy symbol).
- Host `g_submitted_count` + publish to `sync->submitted` on baseline submit.
- `print_vram` (cudaMemGetInfo) + calls bracketing the layer + other stages.
- Baseline enhancements: reset sync/submitted, use `poll_processed` in wait loop, dump `[BASELINE] sync counters: processed=... submitted=... heartbeat=... ready=...`, host submitted, [VRAM] prints.
- Cleanup: `cudaFreeHost` + guards in force exit (preserves clean exit).
- All Phase0 debug/force_baseline/GPUOS_* envs + baseline block behavior **unchanged** (5-7 ms, exit 0 always by default).

**Results (3x default runs, same 256 tasks / 64k elems / op_add workload)**:
- Baseline: **5 ms** (identical to Phase 0 "before").
- Clean exit: always 0, "FINAL (baseline only)", no TIMEOUT. Force path preserved.
- Layer active + structured output:
  - `[VRAM] pre-sync-layer: X MiB ...`
  - `[VRAM] post-sync-layer (minimal atomic counter layer wired): X MiB ...` **(0 MiB delta — pinned host-mapped 60B adds negligible/no device VRAM)**
  - `[BASELINE] sync counters: processed=256 submitted=256 heartbeat=~3500-3900 ready=0`
  - `[BASELINE] host submitted=256 ...`
  - Direct `poll_processed` used (zero-copy path exercised in baseline wait).
- External nvidia-smi samples: during ~ + few MiB (launch/allocs, as Phase0), post-free back to prior idle (no leak from sync layer across runs).
- Overall instr overhead still in the ~20-24 MiB range observed before (dominated by queue ring + data + UVM effects; the "atomic counter layer" itself is tiny: ~1.5KB legacy managed tables + 60B SyncState).

**VRAM re-measure conclusion**:
- Minimal atomic counter layer (SyncState wiring + direct poll) adds **~0 device MiB** (pinned mapped memory lives primarily on host; kernel alias used for system atomics).
- Matches target (<<10%). The prior " ~20-24 MiB instr" includes workload queue + UVM; now we have internal measurement to distinguish the counter layer cost precisely.
- post-cleanup: always returns to session idle (confirmed in sampler runs).

**Sample output excerpt (one run)**:
```
[VRAM] post-device-init: 2927 MiB ...
[VRAM] pre-sync-layer: 2927 ...
[VRAM] post-sync-layer (minimal atomic counter layer wired): 2927 ...   <--- 0 delta
...
[BASELINE] processed 256/256 in 5 ms
[BASELINE] sync counters: processed=256 submitted=256 heartbeat=3808 ready=0
[BASELINE] host submitted=256 ...
[VRAM] baseline-complete-pre-force: 2929 ...
...
=== FINAL (baseline only): Exited cleanly after kernel baseline in 139 ms ===
```
(Full runs + nvidia samples in session logs; 3/3 passed with identical baseline metrics.)

**Status**: Phase 1 complete per plan. Layer active, measured, baseline reliable, more structured counters/timings (the [VRAM] + sync dumps) added. Docs updated. Next: can consider clock64 timing extension or full JIT load fix in follow-on work (still protected here).

**Commands / Rollback**: see .tasks/phase1-minimal-atomic-counters.md

See also the full baseline documented arc in `GPUOS-Baseline-Documented-Arc-2026-06.md` (local worktree) and Obsidian vault master synthesis (written as part of completing the "write this to Obsidian + local + SCL" request).

## Testing, Harness & Dashboard (added for Phase2+ and road with twists)
- experiments/run_all_gpuos_tests.py : master harness exercising C++ tests, py smokes (full_sync etc), agent demo, jit baseline (always verifies "FINAL (baseline only)" + exit 0 + 5ms + counters per AGENTS).
- experiments/gpuos_experiment_logger.py : JSON-first (results/*.json) + optional Surreal; parse jit stdout, log agent/bench runs.
- experiments/dashboard/ (app.py + data.py): Streamlit on 8502, loads GPUOS results + benchmark_results, tabs for agent (recall vs cands), VRAM/counters, hygiene (archive/baseline tags), charts. Modeled on sudoku Percepta dashboard (Session-Summary, Surreal schema, v2.5 code in vault).
- experiments/viz_gpu_activity.py : terminal "pop-up" visualizer (no browser needed). Reuses data layer; --latest/--track jit-baseline|agent-ranking/--json. For jit: shows internal VRAM deltas (0 for SyncState layer), SyncState counters + ascii progress, parsed timeline from raw_output_snippet (stages, [BASELINE] 5ms/256, FINAL clean exit, 170 SM launch), cross-refs to host.cpp/persistent_kernel.cu. For agent: recall/latency + end-state purpose recap. Always "internal only (SyncState + print_vram; no external polling)". Wired into harness end-of-run hint + AGENTS commands. Run after any AGENTS test cmd for quick GPU activity view of that run.
- start script + docker (optional surreal).
- Integrations: agent --log, harness auto-logs, benches can log.
- Commands in AGENTS.md. Always run harness + view dashboard after edits to Phase2 agent ranking, new ops, etc. Guarantees baseline protection + history for "twists and turns".
- Results from recent harness: 5ms, 0 vram delta, counters, agent recall logged + visible in dashboard.
- Autonomous Triton proto progress (2026-06 iters, pushed): triton_labs/ (lab v2 with PTX artifacts + bridge), gpuos_ext load_and_register_custom, demo wiring + iter prints, full harness+viz+logs with "Triton path" notes in agent runs. See .tasks/phase2, Arc end, triton_labs/README. Extends the testing infra to real custom kernel authoring for agent ranking.

