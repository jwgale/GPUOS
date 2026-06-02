
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

