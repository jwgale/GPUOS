# Current Task: GPUOS Phase 0 - Instrumentation & Baseline (RTX 5090)

## Goal
Make `persistent_jit` produce maximum useful structured output (stages, timings, baseline metrics) even on this Blackwell setup where the full JIT path has issues. Capture solid Phase 0 data for VRAM/latency before Phase 1.

## Context
- See AGENTS.md, GPUOS-Phase0-Exploration-Checklist.md, GPUOS-Structure-Mapping.md (updated with latest baselines).
- Current known: Hangs after NVRTC success in cuModuleLoadDataEx (load_op_mul_ptr_from_ptx).
- We have added heavy debug prints + baseline block that runs with built-in op_add and forces clean exit.
- Build requires: export GPUOS_NVRTC_ARCH=compute_120 ; cmake ... -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..

## Steps (do one at a time, verify with build + run + capture internal)
1. Ensure full clean build (tests may use no-op prefetch; main must succeed).
2. Run with timeout + capture full debug output + VRAM.
3. If still hanging in load, add even more granular prints or try skipping JIT entirely for pure baseline.
4. Record exact numbers in the mapping doc.
5. Update checklist status.
6. Prepare simple plan for Phase 1 (add minimal atomic counter, re-run baseline).

## Success Criteria
- persistent_jit always produces baseline timing + exits cleanly within timeout.
- VRAM overhead documented (<10% ideal).
- Hang point clearly identified with logs for Phase 1.
- Full build succeeds or tests are intentionally no-op'd with comment.

## Safe Commands
- Build: See AGENTS.md
- Run: export GPUOS_NVRTC_ARCH=compute_120 ; timeout 30 ./persistent_jit || echo "TIMEOUT"
- (Optional manual, human-driven only): nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1 (in parallel terminal for live view during run)

## Rollback
git checkout -- src/host.cpp ; rebuild

Do not proceed to Phase 1 changes until this task's baseline is solid and documented.

## Progress (this session, one verifiable step at a time)
- [x] Step1: full clean build in worktree/build using AGENTS cmds (g++-12, compute_120). All targets including tests succeeded (nvlink warnings only).
- [x] Step2: multiple runs w/ timeout + internal VRAM (optional manual nvidia). Captured 5-7ms baseline, clean exits.
- [x] Step3: added granular [DEBUG] (PTX dump+size+CUjit opts in load; debug triggers in main) + envs for repro. Refactored baseline block with GPUOS_FULL_JIT to support hang repro while keeping default reliable. Hang reproduced post-launch only.
- [x] Step4: numbers + full diagnosis + commands + envs recorded in GPUOS-Structure-Mapping.md (new continuation section).
- [x] Step5: checklist updated with Step5 details + status.
- [x] Step6: 3x context_record (insight, gotcha on conditional hang, decision on baseline force). Reviewed git diff (only host+docs; AGENTS reverted as it was pre-existing).
- [x] verify: 3x default runs all exit=0 with "FINAL (baseline only)" message, no TIMEOUT. Baseline block always runs.

All success criteria met for Phase 0 continuation. Baseline solid, documented, hang identified with logs/PTX evidence. Ready for next.

**Phase 1 started**: simple plan prepared (via enter/exit_plan_mode + detailed plan.md), approved "as-is", implemented (SyncState wiring as minimal atomic counter layer). See new .tasks/phase1-minimal-atomic-counters.md + updates to mapping/checklist/AGENTS. (Per prior note, Phase1 changes done only after Phase0 solid.)

**Full baseline documented (multi-session)**: See `GPUOS-Baseline-Documented-Arc-2026-06.md` (local) + vault `GPUOS-Phase2-EndState-Baseline-2026-06.md` for the complete arc (WASM/Sudoku/sorting/pivot + Phase0/1 nums + end state) written to Obsidian + local FS + SCL per user request.

Rollback: git checkout -- src/host.cpp GPUOS-*.md ; (cd build && cmake --build . -j$(nproc) ) 

