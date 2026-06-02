# Phase 1: Minimal Atomic Counter Layer (SyncState Zero-Copy) + Re-measure (RTX 5090)

## Goal
Wire the existing (but dormant in C++ main) SyncState + host-mapped zero-copy atomic counters into `persistent_jit` (the baseline measurement binary). This is the "minimal atomic counter layer". Add structured counter output + precise internal cudaMemGetInfo VRAM deltas. Re-run baseline, capture impact (expect layer itself ~0 extra device VRAM). Keep full reliability of baseline numbers + clean exit. Use exact AGENTS.md commands. Update all docs.

## Context
- Phase 0 solid (baseline 5-7ms, clean exit always via force block; hang post-launch diagnosed with granular debug; see mapping continuation + checklist Step 5 + .tasks/current-phase0...).
- SyncState (common.h) + kernel support (persistent_kernel.cu atomics_system + fences) fully exists and is active in pytorch_ext (cudaHostAllocMapped + direct volatile poll, no per-poll memcpy). But q.sync==nullptr in host.cpp (legacy symbol path only).
- "Before" numbers in GPUOS-Structure-Mapping.md (Phase 0 section): ~5-7ms, ~20-24 MiB incremental (queue+UVM effects; pure counters tiny).
- Requirements from AGENTS: ≤10% VRAM for instr; baseline-only path always; more structured output (counters, timings per stage); after edits: rebuild + timeout run (capture internal) + update docs; record via context; review diffs. (External nvidia optional manual only.)
- Plan details: see the approved session plan.md (wiring only, preserve force_baseline default + GPUOS_FULL_JIT for debug, add print_vram + poll helper + submitted tracking + [VRAM] prints around layer; external polling scrapped from standard per feedback/plan).

## Steps (one at a time; verify with build + timeout run + capture internal output)
1. Create this .tasks file + init todos + context_init/record (plan approval).
2. Add VRAM print helper + poll helper + statics + submitted tracking near top of src/host.cpp (around get_processed_count).
3. Wire the SyncState alloc + q.sync + print_vram calls in main alloc section (post Stage1, pre data buffers). Add prints at key stages. Rebuild + run (default) + capture [VRAM] (internal) + confirm baseline still works + new layer messages. (Optional manual nvidia in parallel for live view.)
4. Update baseline block: reset sync fields + inc submitted on task submit; switch progress polling to use direct poll when available; enhance [BASELINE] prints to dump sync counters (processed/submitted/heartbeat/ready). Add vram prints inside baseline. Rebuild + full capture run.
5. Update cleanup (in force exit + any final path) to cudaFreeHost the sync + reset globals. Ensure no leaks/double free.
6. Rebuild with exact AGENTS cmd. Run 3+ times (timeout pattern or harness). Capture full output showing same 5-7ms, clean exit 0, new structured counters in baseline, [VRAM] deltas (layer ~0 device), sync wired message. (Internal prints + optional manual external for live.)
7. Review `git diff` (only host.cpp + this + docs).
8. Update GPUOS-Structure-Mapping.md (new "Phase 1 after minimal atomic counter layer" section with before/after, sample output, commands, conclusion).
9. Update GPUOS-Phase0-Exploration-Checklist.md (add Phase 1 execution notes + checks).
10. Update AGENTS.md (current focus to Phase 1; priorities; any new conventions like "SyncState is the minimal atomic counter layer"; ref this .tasks).
11. Record via context_record (insight, decision, gotchas).
12. Create/update other if needed. Mark Phase 0 .tasks note as superseded.

## Success Criteria
- persistent_jit (default) still always produces identical baseline timing (~5-7ms) + exits cleanly (0) within timeout. Baseline block untouched in spirit.
- Sync layer active: q.sync set, direct poll used (zero-copy path exercised in baseline wait), submitted tracked + visible in structured [BASELINE] prints, ready/heartbeat etc. observable.
- Precise VRAM: new [VRAM] prints show layer alloc delta (pinned sync ~0-1 MiB device); overall instr overhead still low (<<10%); compare to Phase 0 before.
- Internal [VRAM] + counters produced/parsed cleanly; post-cleanup idle in logs. (Optional external nvidia for live view only.)
- Full build succeeds; tests untouched.
- All docs + .tasks updated; context shared; diffs reviewed.
- Hang still protected (no FULL_JIT by default).

## Safe Commands (exact from AGENTS)
- Build: cd .../build ; export GPUOS_NVRTC_ARCH=compute_120 ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 .. ; cmake --build . -j$(nproc)
- Run: export GPUOS_NVRTC_ARCH=compute_120 ; timeout 30 ./persistent_jit || echo "=== TIMEOUT ==="
- (Optional manual, human-driven only -- not part of harness/verification/standard): for live machine view e.g. during unattended: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1` in another terminal. Results (vram deltas, counters, etc.) come from internal direct poll + print_vram (harness parses/logs them; dashboard shows). See plan.md git+testing sections.
- For debug hang repro (optional): GPUOS_FULL_JIT=1 + timeout (baseline numbers still printed before hang).

## Rollback
git checkout -- src/host.cpp .tasks/phase1-minimal-atomic-counters.md GPUOS-*.md AGENTS.md ; (cd build && export GPUOS_NVRTC_ARCH=compute_120 && cmake --build . -j$(nproc))

Do not remove baseline force-exit until post-launch JIT load is fixed separately.

## Notes / Gotchas
- cudaHostAllocMapped + GetDevicePointer (pinned, mostly host mem; device alias for kernel atomics_system).
- Kernel already does both legacy g_ + sync-> (when wired).
- Use volatile direct read for host poll (no CUDA per iteration).

## Link to End State
Phase 1 (this) + Phase 0 baseline reliability are the foundation for the agent end-state (see .tasks/phase2-agent-endstate.md and examples/agent_retrieval_rank_demo.py). The counters + low-overhead proof make it safe to use the persistent + JIT layer for frequent "agent turn" custom ranking/scoring ops. The end state (Triton + GPUOS for agent retrieval/ranking) is now "on the board" as the thing to shoot for.

Full multi-session baseline (Phase0 granular debug + 5ms + Phase1 0-MiB SyncState + end-state demo + vault arc + user quotes) documented in `GPUOS-Baseline-Documented-Arc-2026-06.md` (local worktree) + Obsidian vault master `GPUOS-Phase2-EndState-Baseline-2026-06.md` + SCL.
- Reset sync fields + submitted on every baseline "reset queue".
- Frees: cudaFreeHost for the sync (guard nullptr).
- Scope: sync_host vars must live until final free in baseline force block.
- Keep all existing [DEBUG], env triggers (GPUOS_DEBUG_JIT_LOAD, GPUOS_FULL_JIT), stage timing, force_baseline logic.
- Add [DEBUG] or structured only as needed for exploration; gate if production later.
- After every edit chunk: rebuild + run + capture + verify the step.

## Phase 1 Progress (plan approved as-is by user; one verifiable step at a time via todos)
- [x] Context init/record (plan approval + SyncState dormant-in-main insight); re-read AGENTS/checklist/mapping + plan + created this .tasks + execution todos.
- [x] Helpers added (print_vram via cudaMemGetInfo, poll_processed zero-copy, g_sync statics, g_submitted). Rebuild (exact) + run (timeout): helpers compile, baseline 5-7ms + exit 0 unchanged (verifiable no-regression step).
- [x] Layer wired (cudaHostAllocMapped + GetDevPtr + q.sync + init + pre/post vram prints after q alloc). Rebuild + run: [VRAM] pre/post identical (0 MiB delta for 60B pinned layer), "wired" msg printed, baseline 5ms + clean exit 0.
- [x] Baseline block updated (sync/submitted reset, inc on submit, poll_processed in wait, sync counter dumps + vram in prints). Rebuild + capture: "[BASELINE] sync counters: processed=256 submitted=256 heartbeat=... ready=0", host submitted, direct poll live.
- [x] Cleanup: cudaFreeHost + guards + reset in force exit. Rebuild + run: no crash/free error, same behavior.
- [x] Full verify (p1-5): 3x runs (exact timeout pattern) + internal prints. All pass: identical 5ms baseline, sync counters visible, 0 layer delta, post-cleanup idle (samples confirm no leak), exit 0. git diff reviewed (Phase1 changes + new .tasks). (External nvidia was optional historical cross-check.)
- [x] Docs updated: GPUOS-Structure-Mapping.md (full new Phase1 section + sample output + before/after + conclusion), GPUOS-Phase0-Exploration-Checklist.md (Phase1 exec notes + [x]), AGENTS.md (current focus + priorities + note on SyncState as the layer), .tasks/phase1 + phase0 note. More context_record.
- All success criteria met. "Minimal atomic counter layer" (SyncState zero-copy) active in persistent_jit, re-measured, more structured output added, baseline path 100% preserved + reliable.

Update this file on any follow-up work.

