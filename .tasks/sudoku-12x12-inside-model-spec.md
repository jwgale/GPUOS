# Sudoku 12x12 Inside the Model Spec (WASM-in-Model Improvements)

**Motivation**: The core magic of the original Percepta Transformer-VM is verifiable 100% correctness on combinatorial tasks (like Sudoku) by embedding executable code inside the model weights, even though the base model itself does not achieve 100% accuracy on the task. This was demonstrated for 9x9 Sudoku. Scaling to 12x12 (or other larger grids) is the natural "outgrew the VM" challenge the user hit.

**Goal**: Define a concrete, implementable 12x12 Sudoku solver that can be lowered to WASM/tokens and executed reliably inside the transformer model (using the existing C++ engine + HullKVCache), achieving 100% correct solutions with measurable token count, runtime, and search metrics. This serves as a driver for improvements in the VM (scheduler, primitives, cache for larger state, long-horizon reliability).

**Why 12x12?**: User specifically mentioned "a bit larger game with a 12x12 grid". 12x12 Sudoku is a standard "larger" variant (often with 3x4 or 4x3 blocks; we'll choose one for simplicity, e.g., rows/cols divided into 3 bands of 4 and 4 stacks of 3 for 12 cells). It increases:
- Cells: 144 vs 81 (~1.78x)
- Domain: still 1-12
- Constraints: more units, larger search space.
- Program complexity: bigger data structures, deeper search/propagation in a constraint solver.

**Baseline (9x9, fresh 2026-06 capture)**:
- C program: Norvig-style constraint propagation (see transformer_vm/examples/sudoku.c).
- Compiled: 7494 instructions.
- Run (successful solve):
  - 2,228,174 tokens
  - 481,777 ops
  - 56.60s wall time
  - ~39k tok/s, ~8.5k wasm-ops/s
  - Search: solved at depth 45, 44 guesses
  - Model: vocab=915, D=38, 7 layers, 19 heads, etc.
  - Hull cache used (dominant time ~44-47% in proj/hull).
- This "just works" for 9x9 with the current lowering + engine.

**12x12 Requirements**:
- Input: 144-char string (0 for empty, digits 1-9,A-C or 01-12 for clarity; use 0-9A-C or similar restricted charset if needed for WASM).
- Output: solved 144-char grid + optional CoT (guesses, eliminations).
- Solver: Adapt/extend the constraint propagation (or backtracking with pruning) to n=12, block size e.g. 3x4.
  - Precompute units (rows, cols, blocks) for 12.
  - Larger cand[12*12*12], cnt[144], ucnt tables.
  - Trail and work stacks sized appropriately (TRAIL_MAX ~10k+, WORK_MAX ~20k+).
  - Same restricted ISA: focus on simple ops, arrays, no floats, minimal recursion if possible (or manage depth).
- Instrumentation (like sorting spec): count guesses, eliminations, depth, backtracks, comparisons.
- Verification: post-run check that grid is valid 12x12 Sudoku (unique 1-12 in rows/cols/blocks).
- Multiple test cases: easy, medium, hard 12x12 puzzles (generate or hardcode a few).
- Token target: keep under ~10-20M if possible for "reasonable" execution (baseline 9x9 is ~2.2M); this will drive optimizations.

**Challenges (known from history)**:
- Program size: C code + tables will be ~2-4x larger or more.
- Token count after lowering: likely 5-15x the 9x9 (or worse), pushing the "outgrew" limit.
- Scheduler/MILP for weight placement: for 10k+ instructions, time/memory may explode (previous larger A* 10x10 already tested limits).
- Execution inside model: deeper search (depth 100+?), more state to track in HullKVCache (KV for program variables), risk of hitting token limits or cache thrashing.
- Correctness: ensure the embedded solver is sound for n=12.

**Proposed Implementation Steps (I will lead/drive)**:
1. **Spec & Skeleton**: Write a 12x12 Sudoku C solver (generalized from 9x9 Norvig-style, with configurable n=12, block dims). Keep restricted ops. Add metrics (guesses, depth, backtracks, elims).
2. **Compile & Tokenize**: Use wasm-compile on it. Measure instructions, then token file size.
3. **Placement Analysis**: Run the scheduler (MILP) on the token spec. Measure time, memory, feasibility. Compare to 9x9.
4. **Run & Metrics**: Attempt run in the engine (python ref first for correctness, then C++). Capture tokens used, time, search stats, success/failure mode. Use --nohull vs hull to isolate cache impact.
5. **Identify Improvements**: Based on data, propose 1-2 concrete changes (e.g., better data layout for larger n, incremental constraint prop, alternative search that lowers better, or hybrid: offload heavy propagation to a GPUOS custom kernel while keeping the verifiable control flow in WASM).
6. **Iterate**: Implement one improvement, re-measure on 12x12 (or stepping stone).
7. **Agent Tie-in**: Show how reliable 12x12 solving inside the model enables better agent behavior (e.g., an agent "subroutine" for puzzle-like subproblems in planning/reasoning with 100% reliability).

**Success Criteria**:
- A working 12x12 solver C program that compiles/lowers without errors.
- At least one successful solve (or clear diagnosis of why not) with full metrics.
- Quantified scaling (token ratio 12x12 vs 9x9, scheduler time).
- At least one documented improvement idea with initial implementation or experiment.
- Clear next inquiry steps (so user doesn't hit "no idea how to inquire further").
- Link to utility: e.g., "this enables agents to reliably solve 12x12-scale subproblems inside their 'thoughts' with verifiable correctness."

**References**:
- transformer_vm/examples/sudoku.c (9x9 baseline)
- Sorting-Algorithms-Spec-2026-05-29.md (style for instrumentation)
- Grok-Build-Reproduce-Sudoku.md (repro process)
- Vault: TrackA-BinaryHeap-AStar-10x10-Spec.md (example of larger grid attempts)
- .tasks/wasm-in-model-improvements.md (parent task)
- GPUOS pause docs (for potential hybrid later)

This is the start of driving the original fascinating path forward concretely. The 12x12 is the user's mentioned "larger game".

Next immediate action (after user feedback): I will write the initial 12x12 C skeleton in the vm/examples/ and attempt compile + token count. Or adjust if user prefers different first step (e.g., pure scheduler scaling experiment on synthetic large program, or A* on 12x12 grid).

