# Triton Beginner Learning Course for GPUOS

**Goal**: Give you (the project lead / researcher) a solid, no-glaring-holes mental model of **what Triton is**, **why it exists in this project**, and **exactly how it fits the GPUOS persistent kernel + scheduler architecture** for the agent end-state (custom scoring/ranking and future "many small custom ops" patterns).

**Why this remedial education matters here**:
- We have already delivered the "achievable end state on the board" MVP using the scheduler + NVRTC expr JIT for custom per-turn scoring (see `examples/agent_retrieval_rank_demo.py` and the top-k extension in `a7b61d8`).
- The next ambitious steps (full fused `op_topk` kernel, real Triton `@triton.jit` example for `fused_score_and_topk`, richer verifiers/planning scores, long-running agent server kernel, etc.) will involve **authoring real custom GPU code**.
- Your explicit request after the architecture education + positioning clarification ("BOTH: computation capabilities substrate + proximity/closeness via the persistent worker + SyncState, with tooling on top"): "i really don't understand Triton quite yet and i really think i need to get some remedial education on what it is we are building so I have a better grasp... I definitely don't want to get to the end and have glaring holes in my knowledge".
- This course is designed as the bridge. It is **project-specific** (not generic Triton tutorial) while pointing to the best external resources.

**How to use this course**:
- Work through the modules in order.
- Do the hands-on labs (they reuse or lightly extend code that already exists in the repo).
- Watch the recommended videos **while** or **after** the reading/labs.
- After Module 6-7 you will be able to look at `gpuos_ext.cpp`, the demo's Triton comment block, the scheduler, and `persistent_kernel.cu` and understand the full picture without mystery.
- Revisit before implementing any real `@triton.jit` kernel for Phase 2 follow-ons.

**Estimated time**: 6-12 hours spread over a few sessions (reading + videos + labs on this 5090 machine or Colab for some parts).

**Prerequisites (remedial if needed)**:
- Comfortable with Python + PyTorch tensors (`.to('cuda')`, broadcasting, `torch.topk`, small elementwise ops).
- High-level understanding of "kernel launch overhead" (why many tiny ops are expensive). Re-read the first 2-3 pages of `paper.md` Abstract + Introduction if rusty.
- Awareness (not mastery) of CUDA concepts: threads, blocks, SMs, global/shared memory, `__global__` functions, grid/block launch. You do **not** need to have written production CUDA C++.
- Phase 0/1 baseline numbers fresh in mind (5 ms for 256 tasks of 64k elements each, 0 MiB delta for SyncState layer, "FINAL (baseline only)" always clean).

---

## Module 0: The "Why Triton Here" Framing (10-15 min)

**Our big picture (from Arc sec 5 + sec 11 and your positioning question)**:

We are building a **GPU-side execution substrate** that gives agents:
- **Computation capabilities** for "many small custom ops" inside high-frequency loops (per-turn scoring of 512-2048 candidates, custom reductions, fused heuristics, verifiers on partial thoughts, planning scores, memory re-rank with domain rules...).
- **Proximity / closeness to the model** (persistent long-running kernel on the same GPU, zero-copy `SyncState` for the agent loop to poll `processed`/`heartbeat` with almost no tax, ≤10% VRAM target).

**Today** we achieve custom scoring with:
- `scheduler_context` (TorchDispatchMode) that intercepts `aten::add/mul/...` for small tensors.
- Builds a simple expr string e.g. `'(A + B)'` or `'(A * B) + 0.1'`.
- `gpuos_ext` turns the expr into a CUDA C source template (`build_elementwise_src`), compiles it with **NVRTC** to PTX, loads the PTX with `cuModuleLoadDataEx` + `cuModuleGetFunction`, extracts the device function pointer, and registers it into the persistent kernel's `g_op_table`.
- The long-running worker (launched once with ~170 SMs × 128 threads) does CAS-claim from the WorkQueue and dispatches via function pointer: `OpFn fn = g_op_table[phys]; fn(s_task);`.

This already gives big wins over naive per-op launches (the 15x-ish numbers in the paper and demo sim for micro-batch scoring).

**Tomorrow (the "Triton path")**:
- For richer, fused, or domain-specific "custom algos" (e.g. a single kernel that does dot-product scoring + bias + relu + partial top-k selection in one pass over the candidates), writing the logic as a giant string of CUDA C inside `build_elementwise_src` / `build_reduce_src` becomes painful and error-prone.
- **Triton** lets you write that logic in a Pythonic way (`@triton.jit`, `tl.load`, `tl.dot`, `tl.reduce`, masks, program_id, etc.).
- You still **compile to PTX/CUBIN**.
- You load it with **exactly the same 3-4 lines of driver code** (`cuModuleLoadDataEx`, `GetFunction` or `GetGlobal` for the ptr) that the NVRTC path uses today.
- You register the resulting device fn ptr into the same `g_op_table`.
- The persistent worker and the scheduler don't care how the op was authored.

This is why the demo header says:
> "Triton authoring: ... compile to PTX, load via the same driver + fn-ptr bridge as NVRTC today; complementary per paper.md."

And paper.md (related work):
> "TorchInductor: Compiles entire computation graphs, launches fused kernels. GPUOS: Compiles individual operators, executes via persistent kernel. Complementary: TorchInductor handles large operations, GPUOS handles small ones."

Triton is the **authoring tool** that makes the "small custom ops" side of the substrate practical and productive for the endless agent business cases.

**Key mental model**:
- CUDA C + NVRTC = powerful but verbose string templating for simple exprs today.
- Triton = Python surface that compiles to the same low-level PTX that the GPUOS injection machinery already knows how to load and dispatch.
- The "magic" for low overhead + closeness lives in the **persistent worker + SyncState + op table**, not in Triton itself. Triton just makes the *content* of the ops you inject much nicer to write.

---

## Module 1: What Triton Actually Is (and Isn't)

**Official origin** (2021):
- OpenAI released Triton 1.0: "an open-source Python-like programming language which enables researchers with no CUDA experience to write highly efficient GPU code—most of the time on par with what an expert would be able to produce."
- Link: https://openai.com/index/triton/

**Core ideas**:
- You write **kernels** (the code that runs on the GPU) in a restricted Python dialect decorated with `@triton.jit`.
- Inside the kernel you use `triton.language as tl`: `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.dot`, reductions, masks for bounds, `constexpr` for block sizes (known at compile time).
- Triton is a **compiler** (based on MLIR/LLVM under the hood) that turns your Python-like kernel into highly optimized PTX (or cubin) for the target GPU.
- You launch from Python with a **grid** (number of blocks) that is analogous to CUDA launch configuration, but Triton helps with autotuning and memory coalescing in many cases.
- Result: performance close to hand-written CUDA for many DNN primitives, but far higher productivity.

**What it is not**:
- Not a full replacement for all of CUDA (you can drop to lower levels when needed; there's even a "Gluon" lower-level model now).
- Not "just PyTorch" — it is for writing the *inside* of custom ops.
- TorchInductor (what `torch.compile` uses by default) **generates Triton kernels** for many fusions. So in a normal PyTorch 2+ program you are often already running Triton code without writing any `@triton.jit`.

**Project tie-in**:
- In GPUOS today the "custom" part for elementwise is a generated CUDA C string + NVRTC.
- Triton gives a much better surface for the *next level* of custom (fused score+topk, learned-lite scoring heuristics, domain-specific reductions that don't map to simple exprs).

**Resources for this module**:
- Official announcement (short read): https://openai.com/index/triton/
- Official "Welcome" + overview: https://triton-lang.org/main/index.html
- GPU MODE Lecture 14 "Practitioners Guide to Triton" (highly recommended first video — practical, compares to CUDA, has notebook): https://www.youtube.com/watch?v=DdTsX6DQk24 + notes https://christianjmills.com/posts/cuda-mode-notes/lecture-014/

**Lab 0 (no GPU required for understanding)**:
- Read the first 2 pages of the vector-add tutorial (see Module 2).
- Skim the demo's Triton note in `examples/agent_retrieval_rank_demo.py:271-289` (the exact sketch we will flesh out later).

---

## Module 2: Your First Triton Kernel — Vector Addition (The Canonical Starting Point)

This is the "hello world" that almost every tutorial starts with. Do it.

**Official tutorial (do this)**:
- https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

You will write:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Then a launcher that computes a 1D grid and calls `add_kernel[grid](...)`.

Key concepts you internalize here:
- `@triton.jit` turns the Python function into a compilable kernel.
- `tl.program_id` gives you the "which block am I" (like `blockIdx.x`).
- `tl.arange` + offsets give you the elements this block owns (like thread + block indexing).
- `mask` is mandatory for safe loads/stores (Triton is strict about this for correctness).
- `BLOCK_SIZE: tl.constexpr` — the block size is a compile-time constant (allows the compiler to unroll, vectorize, etc.).
- Launch grid is a Python lambda that returns how many blocks.

**Project connection**:
- This is exactly the kind of "elementwise" work our current `build_elementwise_src` + NVRTC does for simple `(A + B)`.
- In the agent demo we do many such small scores; a fused Triton version could do richer math in one pass over the candidate vectors.

**Hands-on in this repo (use the .venv)**:
```bash
cd /home/jason/.grok/worktrees/percepta-gpuos/gpuos-phase1-experiment
.venv/bin/python -c '
# paste the vector add from the official tutorial + the benchmark decorator
# run benchmark.run(print_data=True)
# Expect similar GB/s to torch for this simple case (the win comes on fused/custom).
'
```

Note on this machine: the current `.venv` torch is a cu121 prebuilt that warns about sm_120 (RTX 5090). You may see CUDA capability warnings or need to fall back to CPU for some tensor creation in pure-torch baselines. The Triton kernel authoring + compilation itself is host-side and will still produce correct PTX. The full "launch on 5090 + compare to torch" path has the same wheel limitation we hit during validation (we used monkey patches + extra gencode only for the gpuos_ext build, not the base torch).

**Resources**:
- Official vector-add tutorial (above).
- "Triton Beginner Coding Tutorial From Scratch" (Vuk Rosić, 38 min, very step-by-step, covers the same kernel): https://www.youtube.com/watch?v=F46V4XnKaJU
- GPU MODE Lecture 14 (again) for the practitioner view + why the masks and constexpr matter.

**Lab deliverable**: Run the official vector-add + benchmark on whatever device works in your env. Print the GB/s table. Understand every line of the kernel.

---

## Module 3: Deeper Into the Programming Model (Masks, Reductions, Tiling)

After vector add you are ready for the next official tutorials.

**Recommended order from official site** (https://triton-lang.org/main/getting-started/tutorials/index.html):
- 02-fused-softmax
- 03-matrix-multiplication (the classic tiled matmul — this is where the performance magic of Triton becomes visible)
- Later ones on flash attention, etc.

**Key new concepts**:
- `tl.reduce` / `tl.sum` etc. for proper parallel reductions (very important for scoring + top-k style work).
- Tiling for matmul: load tiles into SRAM (shared memory), compute on them, write back. This is how you beat naive global-memory matmuls.
- Autotuning (`@triton.autotune`).
- `tl.dot` for tensor-core friendly matrix multiply accumulate.

**Project tie-in for agent use cases**:
- Candidate scoring is often "many small dots or batched matmuls against a query vector" + non-linearities + top-k.
- A fused Triton kernel can keep intermediate scores in registers/SRAM instead of writing huge score tensors back to global memory and launching another kernel for topk.
- This is exactly the "fused_score_and_topk" sketch in the demo.

**Resources**:
- Official matmul tutorial.
- "Tiled Matrix Multiplication in Triton - part 1" (YouTube, from the rkinas curated list).
- "Flash Attention derived and coded from first principles with Triton (Python)": https://www.youtube.com/watch?v=zy8ChVd_oTM (excellent for seeing how a real high-value fused kernel is built).

**Lab**:
- Implement a simple batched dot-product scorer in Triton (N candidates, d=64 or 128 dim, score = (cands @ query) + bias then relu or similar).
- Compare latency vs a torch loop of small matmuls/dots.
- (Optional harder) Add a simple argmax or partial top-k inside the Triton kernel.

You will start to feel why "Triton for custom agent algos" is powerful.

---

## Module 4: The GPUOS Injection Bridge — NVRTC vs Triton (The Heart of "Why It Fits")

This is the module that removes the "glaring hole".

**Walk the current NVRTC path in our code** (read these while doing the module):

1. `pytorch_ext/gpuos_ext.cpp`:
   - `build_elementwise_src` (around line 814) — generates a big C string with the expr inlined, TensorRef handling, strides, dtypes, etc.
   - `nvrtc_compile_ptx` (line ~329) — calls `nvrtcCreateProgram`, `nvrtcCompileProgram` with arch, gets PTX via `nvrtcGetPTX`.
   - `load_function_ptr_from_ptx` / `load_ptr_from_ptx` (lines 347-359):
     ```cpp
     CUmodule mod = nullptr;
     cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr);
     CUfunction fn = nullptr;
     cuModuleGetFunction(&fn, mod, fn_name);
     // or GetGlobal for the op_impl_ptr symbol
     ```
   - `ensure_elementwise_registered` then calls `set_table_slot_async` to poke the ptr into the managed `g_op_table`.

2. The persistent side (`src/persistent_kernel.cu`):
   - `g_op_table[128]` (device managed function pointer array).
   - In the worker: `OpFn fn = g_op_table[phys]; fn(s_task);`

3. The demo's vision for Triton (read the exact block):
   `examples/agent_retrieval_rank_demo.py:271` (the long ``` comment):
   ```python
   import triton
   import triton.language as tl
   @triton.jit
   def fused_score_topk_kernel(...):
       ...
   # compiled = triton.compile(...)
   # ptx = compiled.asm['ptx']
   # Then load PTX with cuModuleLoadDataEx (same as current NVRTC path in gpuos_ext),
   # extract the device fn ptr ... and register into the persistent jump table.
   # This is exactly the "Triton for custom agent algos" complementary use (paper.md).
   ```

**The revelation**:
- Triton changes only the **authoring + compilation to PTX** step.
- The **load + register + dispatch** steps are identical.
- Therefore everything we already built and measured (persistent launch once, SyncState zero-copy polling, baseline 5 ms + 0 delta, harness, viz, dashboard, scheduler per-turn ergonomics) continues to work for much richer custom kernels.

**Hands-on lab (the "remedial" payoff)**:
- In a temp script in the worktree (or extend the demo), define a small `@triton.jit` fused scorer (even if you can't easily launch on this torch wheel, the definition + any compile attempt succeeds).
- Print that you would take the resulting PTX and feed it to the exact `load_function_ptr_from_ptx` style code (you can even copy the C++ load snippet into a comment).
- Confirm that after registration the worker would call it the same way it calls the NVRTC-generated `op_add` etc.

This is the moment the "Triton mystery" becomes "oh, it's the nice frontend for the backend we already have".

**Resources**:
- Re-read the load functions in `gpuos_ext.cpp` while watching a Triton internals video.
- Lecture 29: Triton Internals — https://www.youtube.com/watch?v=njgow_zaJMw (explains what the compiler is actually doing under the `@jit`).

---

## Module 5: Performance, Measurement, and Our Project Constraints

Triton kernels are fast, but you still have to think about:
- Memory access patterns (coalescing).
- Occupancy / register pressure.
- When to fuse vs. when TorchInductor already did a good job.
- VRAM — every extra kernel you inject still has some code + any persistent state cost (our Phase 1 target was ≤10% and the SyncState layer itself was 0 device MiB).

**Tie to our infra**:
- Use the existing `python experiments/run_all_gpuos_tests.py --include-agent` + `viz_gpu_activity.py --track agent-ranking --latest` + dashboard to see the effect of any new Triton op you add.
- The harness already logs kernel_ms, recall, latency_per_turn, etc.
- Phase 1 SyncState gives you the "is the worker still alive and making progress" signal that an outer agent loop can poll while the custom Triton kernel is running.

**Resources**:
- Triton testing / benchmarking utilities in the official docs.
- The benchmark functions in our own `benchmarks/`.

**Lab**: Add a trivial Triton vector-add as a new registered op (or just time a standalone Triton kernel vs the current NVRTC expr path for the same work) and run it through the harness/viz so you see the numbers appear in the dashboard "from internal prints + SyncState".

---

## Module 6: Advanced / Project-Specific Topics

- Fused top-k / argsort in Triton (harder; look at community implementations or the FlashAttention-style videos for reduction + selection patterns).
- How Triton interacts with the persistent worker's shared memory / atomics (the worker already uses `__shared__ Task s_task;` and system atomics).
- AOT (ahead-of-time) compilation vs. the JIT we do at agent turn time.
- Debugging (Triton has a Python interpreter mode for some things; also the FpSan sanitizer mentioned in recent LLVM talks).
- Gluon (newer lower-level model inside Triton) if you ever need to drop closer to the metal.

**Resources**:
- Triton Puzzles: https://github.com/srush/Triton-Puzzles (great for practicing reductions, masking, etc. Many can run in the interpreter without a GPU).
- rkinas/triton-resources curated list (videos + blogs): https://github.com/rkinas/triton-resources
- 2025 Triton Developer Conference materials (recordings + slides): see the triton-lang GitHub.

---

## Curated Resource List (Best Bang-for-Buck First)

**Must-watch / must-read (in rough order)**:
1. Official docs + vector-add tutorial: https://triton-lang.org/
2. GPU MODE Lecture 14 — Practitioners Guide to Triton (YouTube + excellent written notes + notebook): https://www.youtube.com/watch?v=DdTsX6DQk24
3. OpenAI Triton announcement (historical "why"): https://openai.com/index/triton/
4. "Triton Beginner Coding Tutorial From Scratch" (Vuk Rosić): https://www.youtube.com/watch?v=F46V4XnKaJU
5. Lecture 29: Triton Internals: https://www.youtube.com/watch?v=njgow_zaJMw
6. Curated list with many more (softmax, matmul, FlashAttention in Triton, etc.): https://github.com/rkinas/triton-resources
7. Triton Puzzles (practice): https://github.com/srush/Triton-Puzzles

**For the GPUOS-specific "injection" insight**:
- Re-read our `pytorch_ext/gpuos_ext.cpp` (the nvrtc + cuModuleLoadDataEx functions) and the comment block in `examples/agent_retrieval_rank_demo.py`.
- Cross-reference with Arc sec 5 (Positioning) and the new sec 11 (your verbatim question + BOTH synthesis).

**Bonus**:
- Philippe Tillet (Triton creator) talk: search "THE TRITON LANGUAGE | PHILIPPE TILLET".
- Triton Conference 2025 recordings on the triton-lang GitHub.

---

## Glossary (Project-Relevant Terms)

- `@triton.jit`: Decorator that marks a function as a Triton kernel (will be compiled).
- `tl.*`: `triton.language` — the DSL primitives (load, store, program_id, dot, reduce...).
- `constexpr`: Compile-time constant (block sizes etc.).
- PTX / cubin: The low-level GPU code. Both NVRTC and Triton ultimately produce this.
- `cuModuleLoadDataEx`: The CUDA driver call that takes PTX or cubin bytes and creates a `CUmodule` (used identically for NVRTC output and Triton output in our bridge).
- `g_op_table`: The device-side function pointer jump table in the persistent kernel.
- Persistent worker: The one long-running grid of blocks that never exits (until we force quit); it claims tasks and dispatches through the table.
- TorchInductor: PyTorch's default backend for `torch.compile`; it often emits Triton kernels for graph fusions (complementary to our "many tiny custom injected into persistent" approach).
- Scheduler (our `scheduler_context`): The TorchDispatchMode that makes the micro custom ops transparent so plain Python agent code "just works" with low overhead.

---

## Next Steps After the Course (Tying Back to the Project Handoff)

Once you feel solid:
- Flesh out a real (even if small) `fused_score_topk` in a branch of the demo or a new bench.
- Extend the scheduler / gpuos_ext to accept a pre-compiled Triton PTX path (or a registered Triton kernel object) in addition to the expr string path.
- Use the harness + viz + dashboard to measure the lift on the agent-ranking track.
- Then we can pick the specific ambitious business case you have in mind (verifier, planning scores, long-running server, etc.) with confidence that the Triton authoring piece is no longer mysterious.

**Update log for this course**:
- Created as direct response to the "remedial education on Triton + what we are building" request after the architecture education pass and the BOTH positioning clarification (Arc sec 11, SCL CID ca6c28fffb59c0c1).
- Will be referenced from `.tasks/phase2-agent-endstate.md`, the Arc, and AGENTS "Next Priorities" as a recommended pre-work for real Triton kernel work in Phase 2+.
- Hands-on started 2026-06 in worktree (experiments/triton_labs/lab_vector_add_and_bridge.py): vector-add kernel authored + benchmark intent; Module 4 bridge lab executed (fused_score_topk_kernel educational toy defined, PTX extraction step shown per demo note, *exact* load_function_ptr_from_ptx + set_table_slot + persistent dispatch snippet from gpuos_ext.cpp printed + "the authoring tool changed, the substrate (persistent + table + load + SyncState) is identical" aha captured). Run with .venv (triton 3.1 present; noted torch cu121 sm_120 launch limitation for full end-to-end on this wheel, same as current demo/agent fallback — authoring/PTX gen + conceptual bridge still fully exercised). Lab script is the deliverable artifact. This directly fulfills the "run the vector-add lab together in the .venv" invitation at end of course + the Module 4 "remedial payoff". (git diff will be reviewed.)
  - Autonomous iteration 1 (2026-06): Expanded to lab v2 in experiments/triton_labs/. Added real PTX extraction attempts (triton.compile/asm with fallback to inspect.getsource + saved artifacts in /tmp/), full C++ bridge printout, explicit BOTH/phase2 ties, and experiments/triton_labs/README.md (run cmds, artifacts, next wiring steps). Ran the lab (kernels defined, sources saved, "Triton only changes authoring" conclusion + bridge aha). Thorough updates to this log + phase2 progress. Followed AGENTS (re-read required, todos, planned diff review + commit/push per iteration, harness/viz/dashboard in later iters). This advances the remedial gate and sets up real fused proto.

---

**You now have a complete, project-grounded path from "I don't really get Triton" to "I can look at our load path and the demo sketch and know exactly what the next code increment should do and why it fits the substrate story."**

When you're ready, tell me which module you want to start with, or "let's run the vector-add lab together in the .venv", or "review the current Triton sketch in the demo and propose the minimal extension", or "I picked the mid-generation verifier business case — let's plan the first Triton kernel for it".

All per the spirit of AGENTS (context recorded, docs persisted, cross-refs added, no glaring holes).