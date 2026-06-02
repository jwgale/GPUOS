# Current Task: GPUOS Phase 0 - Instrumentation & Baseline (RTX 5090)

## Goal
Make `persistent_jit` produce maximum useful structured output (stages, timings, baseline metrics) even on this Blackwell setup where the full JIT path has issues. Capture solid Phase 0 data for VRAM/latency before Phase 1.

## Context
- See AGENTS.md, GPUOS-Phase0-Exploration-Checklist.md, GPUOS-Structure-Mapping.md (updated with latest baselines).
- Current known: Hangs after NVRTC success in cuModuleLoadDataEx (load_op_mul_ptr_from_ptx).
- We have added heavy debug prints + baseline block that runs with built-in op_add and forces clean exit.
- Build requires: export GPUOS_NVRTC_ARCH=compute_120 ; cmake ... -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..

## Steps (do one at a time, verify with build + run + nvidia-smi)
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
- Monitor: nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1 (in parallel terminal)

## Rollback
git checkout -- src/host.cpp ; rebuild

Do not proceed to Phase 1 changes until this task's baseline is solid and documented.
