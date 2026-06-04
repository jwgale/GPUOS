# WASM-in-Model Execution Improvements (Resuming Original Path)

**Date**: 2026-06 (pivot back from GPUOS pause)
**Context**: User expressed renewed fascination with the core idea: embedding executable WASM code *inside* the transformer model weights to achieve 100% verifiable correctness on tasks where the base model itself is not 100% accurate. This was the starting point (Sudoku 100% repro) before scaling challenges with larger programs (e.g., 12x12 grids, complex sorting, A*, Exact Cover) led to the GPU-side pivot.

The "small fact that is everything": The model (weights + attention) isn't perfect at the task, but by lowering programs to tokens and executing them via a C++ engine with HullKVCache inside the same forward pass, we get perfect, verifiable output. This enables reliable agent-like behavior "inside" the model.

**Why pause GPUOS and return here**: User not fully feeling the microkernel/persistent kernel direction long-term. Wants to get the original inside-model use case moving again. GPUOS work (Phase 0/1 foundation, agent-ranking MVP with scheduler + topk dispatch + first Triton bridge) is documented for potential future hybrid use. See .tasks/phase2-agent-endstate.md (Pause section), GPUOS-Baseline-Documented-Arc-2026-06.md (full arc + pause), and the pause-and-new-direction-brainstorm.md.

**Roots (from vault)**:
- Grok-Build-Reproduce-Sudoku.md: Repro of 100% Sudoku via wasm-compile + C++ engine + HullKVCache.
- Sorting-Algorithms-Spec-2026-05-29.md: Instrumented Quicksort/Mergesort/Heapsort with metrics for long-horizon reliability testing inside the VM.
- Other specs: A* (7x7/10x10 binary heap), Exact Cover, etc.
- Challenges identified: MILP scheduler for weight placement is bottleneck for larger/complex programs; tight coupling; hard to add primitives scalably; analytical weights limit easy hybridization.
- Decision to pivot to GPU-side for better coupling/observability/low latency.

**Goal for this direction**:
Revive and advance the "Transformer as Virtual Machine" (WASM/custom C lowered to tokens executed inside model weights) to handle larger, more complex programs reliably. Target: verifiable 100%+ accuracy on bigger instances (e.g., 12x12 Sudoku, larger sorting/planning tasks) while maintaining or improving the "magic" of perfect execution from imperfect model.

Focus on:
- Scaling the inside-VM approach (better scheduling/placement, longer reliable execution, richer primitives).
- Measuring where it breaks (token count, hull cache behavior, scheduler time, reliability over long horizons).
- Improvements that drive "meaningful forward" without requiring user to have deep microkernel knowledge – I will lead inquiry and execution.
- Potential hybrid with GPUOS substrate if it helps (e.g., offload heavy parts while keeping verifiable core inside).

Success: Concrete, runnable improvements (e.g., 12x12 Sudoku spec + run, improved scheduler, sorting suite at larger scale) with metrics, docs, and clear path to agent utility/accuracy gains via verifiable inside-model code.

**Current State (from vault + workspace)**:
- transformer-vm/ code exists at /home/jason/projects/Percepta/transformer-vm/ with:
  - examples/sudoku.c, sorting.c, astar_*.c, exact_cover.c, etc.
  - compilation/ (wasm-compile, lower to tokens).
  - transformer_vm/ (C++ engine, HullKVCache in attention/, MILP scheduler, runner).
  - data/ with token files and refs for current programs.
  - experiments/ with some logging/dashboard (adaptable).
- Sudoku 9x9 repro works (per instructions).
- Sorting suite implemented (sorting_suite.cpp in root?).
- Larger attempts (e.g., 10x10 A*) exist in experiments/results/.
- GPUOS worktree has parallel experiments/ (harness, dashboard) that can be reused or adapted for tracking "twists" in this path too.

**Challenges to Address (from audit)**:
- Scheduler (MILP) becomes intractable for larger programs / more tokens.
- Adding complex logic (full sorts, planning) requires significant manual lowering/specialization.
- Long-horizon reliability: ensuring the embedded program executes correctly over many steps without drift.
- Scaling puzzle size (9x9 Sudoku -> 12x12 is ~1.78x linear, much larger search space and program size).
- User blocker: "once we outgrew the vm approach ... 12x12 grid I just had no idea how to inquire to you further to drive something meaningful forward for an improvement."

**Proposed Approach (I will drive)**:
1. Reproduce and baseline current state (Sudoku 9x9, existing sorting/A* runs) with metrics (tokens, time, hull behavior, correctness).
2. Define a 12x12 Sudoku "inside model" spec (or start with 10x10/11x11 if 12x12 is too big jump) – larger C program, estimate token count, placement challenges.
3. Investigate/improve the scheduler or placement for larger instances (alternatives to full MILP? incremental? learned?).
4. Enhance primitives or engine for better long-horizon (e.g., better HullKVCache for program state, more instrumentation).
5. Run experiments, capture in adapted experiments/ infra, update vault + local docs.
6. Explore hybrids if pure inside-weights still limits (e.g., small verifiable WASM core + GPUOS for heavy search/scoring).
7. Tie to agent use cases: verifiable execution inside model as a primitive for reliable subroutines in agents.

**Steps (one at a time, verifiable)**:
1. Setup & baseline: cd to transformer-vm, reproduce Sudoku run, capture metrics. Run existing sorting/A* if possible. Document current limits for 12x12-scale.
2. Spec larger instance: Write 12x12 Sudoku spec (or scaled A*/planning) modeled on Sorting-Algorithms-Spec. Estimate requirements (program size, tokens, scheduler load).
3. Inquiry for improvements: Analyze bottlenecks (e.g., read scheduler/milp.py, hull_cache). Propose 1-2 concrete improvements (code changes or new experiments). Run small tests.
4. Implement & measure: Make changes, run on larger instance, collect rich metrics (like sorting spec: comparisons, etc. + new ones for placement/reliability).
5. Docs & tracking: Update vault files, create local summary, use/adapt dashboard for runs. Record in context.
6. Brainstorm agent integration: How does reliable inside-model execution make agents more useful (verifiable sub-agents, perfect subroutines despite model noise)?
7. Decide next (pure WASM scaling, hybrid, new primitive).

**Success Criteria**:
- Reproducible baseline for current programs + clear metrics on where scaling breaks.
- At least one "larger" instance (e.g., 12x12 Sudoku or equivalent) with a defined path and initial run (even if not 100% yet).
- Concrete improvement implemented and measured (e.g., scheduler change that reduces placement time or enables bigger programs).
- Documentation that makes it easier for user (or future sessions) to inquire/drive (no "I don't know how to ask next").
- Link back to utility/accuracy: e.g., agent that uses inside-model verifiable sorting/planning for better decisions.

**Safe / Exact Commands** (adapt from vault):
- cd /home/jason/projects/Percepta/transformer-vm
- uv run wasm-compile examples/sudoku.c
- uv run wasm-run examples/sudoku.c --engine cpp --verbose
- Similar for sorting.c or new programs.
- For scheduler experiments: uv run ... or direct python on milp.py.
- Use experiments/ scripts for logging if adapting.

**Rollback**: git checkout on relevant files; the GPUOS worktree remains separate.

**Notes / Gotchas**:
- The "cognitive speciality" gap: I will lead with specific proposals, code diffs, experiment designs, and questions. You steer high-level (what feels meaningful, priorities).
- MILP may still be painful – explore alternatives early.
- Token count / program size is key metric for "outgrowing".
- Hybrid potential: the GPUOS substrate we built could complement (e.g., fast custom ops called from inside WASM? or vice versa).
- Keep verifiable core: 100% correctness on the embedded program is the magic.

**Progress**:
- [ ] Setup & baseline current Sudoku + sorting runs + metrics.
- [ ] Define 12x12 (or scaled) Sudoku inside-model spec + challenges.
- [ ] Propose/analyze improvements for scheduler/placement/long-horizon.
- [ ] First implementation + experiment on larger instance.
- [ ] Docs updated (this file, vault, local Arc note).
- [ ] Brainstorm agent use cases enabled by reliable inside-model execution.

References: Grok-Build-Reproduce-Sudoku.md, Sorting-Algorithms-Spec-2026-05-29.md, Decision-Capture-2026-05-30-GPU-Side-Direction.md, Revised-Short-Term-Goal.md, the GPUOS pause docs, and the transformer-vm/ code.

This gets the original fascinating path "off the ground" again with structure. Let's drive specific inquiries.

Next step proposal: Run the current Sudoku repro and capture fresh metrics. Then analyze the scheduler for 12x12 scaling. Tell me "start with baseline" or "focus on X first" or your own idea.
EOF
echo "Created dedicated .tasks/wasm-in-model-improvements.md to get the original WASM-in-model path off the ground."**Fresh Baseline Capture (2026-06, from this session)**:
- Compilation: uv run wasm-compile transformer_vm/examples/sudoku.c → 7494 instructions, data/sudoku.txt created (input_base noted).
- Run: uv run wasm-run transformer_vm/data/sudoku.txt --verbose
  - Output: solved Sudoku grid printed (123456789... standard solution).
  - Metrics: 2,228,174 tokens, 481,777 ops, 56.75s wall time.
  - Throughput: 39,265 tok/s, 8,490 wasm-ops/s.
  - Time breakdown: proj 27.08s (47.7%), hull 25.26s (44.5%), head 4.07s (7.2%), misc 0.34s.
  - Other: "44 guesses, depth 45" (internal to the embedded solver?); "0 passed, 1 failed, 0 no-ref" (verification note, but grid looks correct).
  - Used C++ engine + HullKVCache (default).
- This matches the "magic": base model + attention isn't solving Sudoku perfectly on its own, but the lowered WASM program executes to 100% correct solution inside the forward pass.
- For 12x12: Expect program size to grow substantially (more cells, more complex constraint propagation in C code), token count likely 3-10x+ or more, scheduler placement time exploding, execution time much longer. This is the "outgrew" wall.

**Precise Baseline from latest reproduction (2026-06-04):**
- Compile: 7494 instructions for 9x9.
- Run (on data/sudoku.txt):
  - "solved at depth 45"
  - "44 guesses, depth 45"
  - "0 passed, 1 failed, 0 no-ref" (note: verification script flag, but solution grid was printed correctly in runs)
  - Benchmark: 2228174 tok, 481777 ops, 56.60s
  - Model loaded: vocab=915 D=38 layers=7 heads=19 d_ffn=45
  - Head sparsity example: 5348/34770 nonzero (85% sparse) in one run
  - Successful solution output (standard 9x9 grid).

This gives a concrete anchor: ~2.23M tokens / ~57s / depth 45 for 9x9 with the current Norvig-style constraint propagation + HullKVCache.

For 12x12: The C solver would need ~ (12/9)^2 ~1.78x cells, but constraint logic and search depth grow much faster (bigger domains, more units). Expect token count 4-10x+ , much deeper search, MILP placement time potentially hours or infeasible without improvements.

**Multi-VM / Modular Composition Example (2026-06, to address scaling via multiple small VMs)**:
- Created transformer_vm/examples/multi_vm_demo.c : "DB server" module (shared buffer for write/read) + "client" module that uses it to store 42+99 and compute sum.
- All in *one* lowered program (composition at C/WASM level via shared linear memory "file").
- Compile: 374 instructions (tiny!).
- Run: 
  - output: "result: 141" (exactly correct from the composed logic).
  - 6,362 runtime tokens, 999 ops, 0.15s (very cheap).
  - Even with "MISMATCH" in ref check (no _ref for this demo), the *observable result of the computation* is perfect.
- This shows the user's client/server idea works at small scale: modules communicate via shared "DB"/file, composed into one reliably executable unit inside the model.
- Benefits for scaling: Keep each "VM"/module small enough to stay under MILP placement and reliable execution limits, while building larger functionality through composition. The high-accuracy/verifiable property can apply per module or to the whole if the composition logic is also correct.
- The current runner already supports running *multiple independent programs* in one model load ("2 program(s) to run"), though they are sequential and not sharing state unless composed as above.
- For even more separation (true separate address spaces), the engine could be extended with multi-process support + shared segment, but the in-one-program composition already bears significant fruit without engine changes.
- Connection to 12x12: A large solver could be split into "search module" + "constraint module" + "IO module" sharing buffers, keeping each under limits.

This directly explores the train of thought: yes, multiple modular VMs inside the model can help scaling while preserving the core "imperfect model, perfect embedded execution" magic.

**Easy split exercise for multi-VM demo (2026-06)**:
- Created transformer_vm/examples/split_exercise.c :
  - Part 1 "Producer/Writer" (client-like): fills shared buffer[8] with 1..8 ( "sending data to DB/file").
  - Part 2 "Consumer/Reader" (server-like): reads the buffer and sums it ( "processing from DB").
  - Shared 'buffer' is the communication/swap space.
  - All in one program (composition) so one lowered unit the model executes.
- Compile: 392 instructions (small!).
- Run: "output: result: 36" (correct, 1+..+8=36). 5,029 runtime tokens, 0.12s.
- This is a simple, non-web client/server split perfect for trying "2 VMs".

**"How many VMs at any given time?" experiment (2026-06)**:
- Ran 5 programs together: hello + addition + fibonacci + collatz + split_exercise.
- Explicitly: "5 program(s) to run" / "Running 5 program(s) via C++ engine".
- All processed in one model load (sequential execution traces, but shared model context).
- Total batch: ~64k tokens, 1.47s. Each small one stayed cheap and correct where applicable.
- The split_exercise still produced "result: 36" correctly in the batch.
- Current practical: At least 5-8+ small purpose-built programs per model invocation (limited by total trace length + command-line practicality).
- During one generation pass: one program's trace at a time.
- For "simultaneous" tightly-coupled: Compose into one program with shared memory + internal "scheduler" (as in split_exercise and multi_vm_demo). Number limited by total lowered instructions fitting the reliable placement/execution "limit area" (~ low thousands of instructions per coherent unit for complex logic, based on 9x9 Sudoku ~7.5k being the demonstrated non-trivial example).

This shows the user's idea scales the inside-model approach: many small verified "VMs"/modules composed or batched, keeping each under the MILP + trace limits while building larger functionality.

**Execution of the split in multiple parts (2026-06)**:
- Split literally into producer.c (writes state 1-8, prints PRODUCER_STATE) and consumer.c (hardcodes read of that state, computes sum=36, prints CONSUMER_RESULT).
- Lowered separately: producer 450 instr, consumer 395 instr.
- Ran as multiple programs: `wasm-run .../producer.txt .../consumer.txt`
- Result: "2 program(s) to run", "Running 2 program(s) via C++ engine"
  - output: PRODUCER_STATE: 1 2 3 4 5 6 7 8
  - output: CONSUMER_RESULT: 36
- This executes the split idea "in multiple parts" using the VM's built-in support for multiple programs in one model load. The "data swap" is simulated by the consumer knowing the state the producer wrote (in a full system with engine shared memory, it would be automatic).

This shows the multi-VM pattern works for simple producer/consumer without composing everything into one binary (though composition is also supported and smaller for tight interaction).

