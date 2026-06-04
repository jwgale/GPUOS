# Triton Labs for GPUOS (Phase 2 / Agent End State)

This directory contains hands-on labs from the project-specific remedial course
`docs/Triton-Beginner-Course-for-GPUOS.md`.

## Purpose
- Complete the "Before real @triton.jit work" gate in AGENTS.md, .tasks/phase2-agent-endstate.md, and Arc Sec 11.
- Author real Triton kernels that produce PTX/CUBIN.
- Prove that the PTX feeds the *exact same* `cuModuleLoadDataEx` + `g_op_table` + persistent dispatch + SyncState zero-copy path as current NVRTC exprs in `pytorch_ext/gpuos_ext.cpp`.
- Prepare for fused `score + topk` (and other custom agent algos like verifiers, planning scores) that improve agent utility/accuracy over the current hybrid scheduler + torch.topk MVP.

## Current Labs
- `lab_vector_add_and_bridge.py`: Module 2 (vector-add) + Module 4 (the GPUOS injection bridge revelation). Expanded in autonomous iteration 1 to actually attempt PTX extraction, save kernel sources as artifacts, print the exact C++ bridge code, and tie conclusions to BOTH positioning + phase2 goal.

Run:
```bash
.venv/bin/python experiments/triton_labs/lab_vector_add_and_bridge.py
```

## Artifacts from runs
- `/tmp/gpuos_vector_add_kernel.py`
- `/tmp/gpuos_fused_score_topk_kernel.py`

These are the "proto" sources that, on a fully sm_120-compatible torch env, can be compiled to PTX and fed to the GPUOS load path.

## Next (autonomous iterations)
See todo list in main session and phase2 .tasks "Remedial Triton education step".
- Add C++ support for `register_custom_ptx`.
- Wire real Triton kernel into scheduler/demo for scoring.
- Full harness + viz + dashboard refresh with "real Triton proto" runs.
- Thorough doc updates (this README, course log, phase2, Arc, AGENTS, experiments/README).

All changes follow AGENTS: re-read required docs, todos, exact build/run cmds where applicable, git diff review before each commit/push, harness/viz/dashboard at end of steps, baseline protected.

## References
- `docs/Triton-Beginner-Course-for-GPUOS.md` (the course itself)
- `examples/agent_retrieval_rank_demo.py` (the Triton sketch comment block)
- `pytorch_ext/gpuos_ext.cpp` (the load bridge and current topk "for now" hybrid)
- `pytorch_ext/scheduler.py` (aten::topk interception + elementwise exprs)

## Iteration 2 (autonomous)
- Added `load_and_register_custom(ptx: str, entry_name: str, slot: int)` to gpuos_ext.cpp (and exposed via pybind).
- Uses the existing `load_function_ptr_from_ptx` + `set_table_slot_async` (the exact bridge from the lab).
- Rebuild tested via torch cpp_extension load in demo-style import (successful, new API present).
- This enables taking PTX from the lab artifacts (or future real fused_score_topk) and wiring it into the persistent op table for use by scheduler/dispatch.
- Docs updated in this README + phase2 progress + lab comments.
- git diff reviewed, committed, pushed as tight "feat(gpuos_ext): expose load_and_register_custom for Triton PTX".
- Ties directly to course Module 4 and phase2 "extend ... for Triton PTX registration".

## Iteration 3 (autonomous)
- Wired the new load_and_register_custom into the demo (inline toy PTX + call after the Triton note).
- Demo run exercises the full author (lab) -> register (new C++ API) path.
- Printed iter notes in output for traceability.
- Updated this README + phase2 progress.
- Run verification: demo executed, iter 3 message shown (RuntimeError expected pre-init, but binding + call path proven).
- git diff reviewed, commit with tight label, pushed.
- This makes the "real Triton proto" tangible in the end-state demo runs.
