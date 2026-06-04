# Pause on GPUOS + New Direction Brainstorm (2026-06)

**Context**: User decided to pause GPUOS work. "i think i need to pause on this one for awhile here. I'm just not feeling this whole thing long term, or maybe i don't have enough understanding of microkernels and all this, but it is still a cool idea. I think i need to brainstorm another use case and get that off the ground and maybe there is something here to come back to later."

See:
- .tasks/phase2-agent-endstate.md (new Pause section with achieved state + resume notes)
- GPUOS-Baseline-Documented-Arc-2026-06.md (new Pause section + full arc)
- Recent dashboard (82+ runs: 10 agent-ranking, 9 jit-baselines) at http://localhost:8502
- experiments/triton_labs/ (lab + README from autonomous iters)
- AGENTS.md (current Next Priorities note the pause implicitly via docs)

## What Was Established on GPUOS (positive closure, for resume)
- **Foundation**: Phase 0 baseline (always 5ms for 256 tasks of 64k elems, clean FINAL exit via force block). Phase 1: SyncState zero-copy (60B host-mapped, 0.0 MiB device delta, direct poll, heartbeat). Persistent kernel + op table + scheduler for small custom ops.
- **MVP on the board**: agent_retrieval_rank_demo.py — per-turn custom scoring (via scheduler exprs/NVRTC) + topk dispatch. Generalizes original "sorting" from WASM limits. 10+ logged runs with recall@K / latency proxies. Full vault story preserved.
- **Triton bridge (autonomous 4+ iters)**: Lab with vector-add + fused scorer + PTX artifacts. load_and_register_custom in gpuos_ext (reuses exact load bridge). Wired into demo. Verified via harness/viz/new logs. Education gate advanced.
- **Testing/Observability**: Harness + logger + dashboard + viz (internal only: SyncState + [VRAM] + raw snippets). "Scrap external polling" — results are what the code emits. Good for twists.
- **Positioning (BOTH)**: GPU execution substrate for *computation capabilities* (custom small/fused ops for agents: scoring, ranking, verifiers, planning) + *proximity/observability* (same device, persistent low-tax, SyncState polls, low VRAM). Tooling (scheduler, viz, harness) on top. Not general agent framework.
- **Key boundary (from last discussion)**: Strong observability into *GPUOS-submitted work* (counters for custom kernels). Not automatic deep view into unmodified model internals (unless model explicitly shares state or uses GPUOS primitives). This is the current limit vs. full "see inside the model on GPU".

All per AGENTS: re-reads, todos, exact cmds, baseline protection, diff reviews, commits/pushes, docs updates, context attempts.

**Resume when ready**: Re-read the pause sections + run viz/harness/dashboard. The substrate + first custom kernel wiring is there.

## Brainstorming New Use Case / Direction
History roots (from vault query in Arc):
- Percepta Transformer-VM: WASM (custom C) *inside* model weights for verifiable execution. 100% Sudoku (repro in Grok-Build-Reproduce-Sudoku.md with HullKVCache + MILP scheduler).
- Then sorting specs (Quicksort etc. with metrics) for long-horizon reliability testing inside the VM (Sorting-Algorithms-Spec-2026-05-29.md).
- Audit: Hard to scale (scheduler bottleneck, tight coupling for new primitives, analytical weights limit hybridization). Agent-like behavior possible but constrained.
- Pivot: To GPU-side substrate (this GPUOS work) for tighter coupling, observability, low latency, custom algos "close to model".
- User request that started end-state: "the end state here really is something that either improves the accuracy of an agent, or offers that agent some increase in utility." "Put an achievable end state on the board."

**Questions to drive brainstorm (reply with thoughts, or "start with X")**:
1. What from the *original WASM-in-model* (Sudoku 100%, sorting for reliability, verifiable execution inside weights) still excites or feels unfinished? E.g., improving the VM (better scheduler, more primitives without MILP pain, hybrid WASM+GPU)?
2. What new agent problems are top-of-mind now (post all the GPUOS exploration)? E.g., observability/steerability in real LLMs, long-horizon planning, verification of thoughts, custom scoring without launch tax, something else?
3. On "end stage use cases for AI Agents more useful":
   - Accuracy: Better selection via richer custom logic (e.g., domain verifiers on CoT, planning candidate eval).
   - Utility: More candidates per turn / lower latency (e.g., on-device re-ranking of retrieval/tool options).
   - Other: Steerability (counters/traces for agent loops), verifiability (high-stakes), energy/latency for always-on agents, hybrid (WASM for verifiable parts + GPU for speed)?
4. Any constraints or preferences for new direction? (e.g., stay in Percepta vault style, focus on one "board" MVP like before, use existing testing infra, avoid microkernels for now, etc.)
5. Interest in going back to WASM VM side (improve it as the "inside model" path) vs. pure new idea vs. hybrid?

## Proposed Scaffolding for New Direction
If you pick something, I can:
- Create .tasks/new-use-case-YYYY-MM.md modeled on phase2 (Goal, Context from vault, Steps, Success Criteria, Safe Cmds, Progress).
- Start with a small MVP/demo (like the ranking one).
- Use the existing experiments/ harness/dashboard for tracking (it's general).
- Update AGENTS.md with new focus (when conventions change).
- Record in context layer + local Arc note.
- Explore code (e.g., the WASM engine from repro if in vault/workspace).

Example starter directions (pick/edit):
- **Back to WASM VM + sorting reliability**: Improve the inside-model VM (e.g., better MILP alternatives, more primitives, long-horizon tests, integrate with current agent ideas). Target: 100%+ on harder puzzles or new tasks with verifiable execution.
- **Hybrid WASM + GPU substrate**: Use WASM for verifiable "core" logic inside weights, GPUOS for fast custom scoring nearby. Addresses both paths' limits.
- **New agent observability/steerability tool**: Focus on the "see inside" gap — e.g., lightweight tracing or sidecar for model internals without full microkernel.
- **Other from history**: A* / Exact Cover as agent primitives, planning scores, long-running agent "server".

Let's get something off the ground. What resonates, or what's a new problem/use case you're excited about? I'll scaffold it (create task file, initial plan, first small step) once you point the way.

All changes will follow AGENTS (reads first, todos, diff review, etc.). Pause on GPUOS respected — docs are there for later.
