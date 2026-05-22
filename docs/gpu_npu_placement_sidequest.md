# Side quest — GPU↔NPU placement for speculative decoding

Scoped slice of roadmap W4.e. Instead of filling the whole 3-phase ×
3-island matrix, this side quest pins CPU as the reference and
explores the **GPU↔NPU corner**: which accelerator runs which part of
spec decode, what happens when they trade or share phases, and where
the crossover lives.

Motivated by `docs/memory_bandwidth_ceiling.md` — the LPDDR5X fabric
saturates at ~2 busy islands (CPU ~156, GPU ~159, NPU ~66 GB/s real;
228 ceiling). The design question is not "use all three" but "pick
the right two, and assign phases well."

## The actual question

The two configs proposed in conversation turn out to be near
**opposites**:

- "draft on NPU, verify on GPU"
- "PP on NPU, TG on GPU"

Spec decode has two op *shapes*, not three phases:

- **TG-shaped** — autoregressive single-token forward. The *draft*
  loop. Small model, bandwidth-bound, dominated by per-call overhead.
- **PP-shaped** — batched multi-token forward. Prompt prefill *and*
  *verify* (verify is a k-token mini-prefill of the target).

So "PP on NPU" implies verify-on-NPU and "TG on GPU" implies
draft-on-GPU — the mirror of "draft NPU / verify GPU."

The pivot is **whether verify is bandwidth-bound or compute-bound**,
and that depends on speculation depth `k`:

- small k → verify reads the target weight set per ~1 useful token,
  behaves like TG → bandwidth-bound → wants the GPU (159 GB/s).
- large k → verify amortizes one target weight-stream over k tokens
  → compute-bound → wants the NPU's quantized-matmul throughput (the
  sidecar already beats Genie at PP by +39%).

**Central deliverable: find the crossover k** where verify flips from
wanting-GPU to wanting-NPU — and whether it lands inside the useful
range of k (typically 3–8). Everything below either feeds that or
gets wacky around it.

## Configs

Baseline = today's `prefill CPU / draft NPU / verify CPU`. Roles are
{prompt-prefill, draft, verify}. ✓ = cheap (mostly a launch-flag
change); ◆ = needs new orchestration code.

| #  | draft | verify | prefill | note |
|----|-------|--------|---------|------|
| C0 | NPU   | CPU    | CPU     | today's config — baseline |
| C1 | NPU   | GPU    | GPU     | ✓ the recommendation: verify→fastest island |
| C2 | GPU   | NPU    | NPU     | ✓ the inverse ("PP NPU / TG GPU") |
| C3 | GPU   | GPU    | GPU     | ✓ GPU-only spec decode, NPU idle |
| C4 | NPU   | NPU    | NPU     | NPU-only spec decode, GPU idle |
| W1 | alt   | alt    | GPU     | ◆ ping-pong: verify alternates GPU↔NPU per round, draft always on the opposite island — no same-island contention, ever |
| W2 | NPU   | split  | GPU     | ◆ target layer-split: verify layers 0..m on NPU, m+1.. on GPU; one pass crosses the unified-memory fence |
| W3 | NPU+GPU | GPU  | GPU     | ◆ dual-draft race: NPU and GPU draft concurrently at different depth/temp; verify consumes the longer accepted run |

C1 vs C2 is the headline A/B. C3/C4 are the single-accelerator
anchors that say what concurrency actually buys. W1–W3 are the wacky
stretch goals.

## Phases

### Phase 0 — anchors + the crossover microbench (cheap, do first)

Everything else depends on these and they need no new orchestration.

1. **Single-island per-phase timing.** Run target PP & TG, and the
   4B draft TG, on GPU-alone and NPU-alone. Fills the GPU/NPU cells
   of the W4.e matrix. Use `scripts/adreno_bench_matrix.ps1`,
   `bench_qwen3_4b_gpu_knobs_bat.py`, `npu_engine/bench_qwen3_4b_ortqnn.py`.
2. **Verify-shape crossover microbench — the key measurement.** Time
   a *k-token batched forward* of the 4B bundle on GPU vs NPU for
   k ∈ {1,2,4,8,16,32}. Plot per-token cost vs k for each island.
   The k where the two curves cross is the answer to C1-vs-C2, and
   it falls out without ever running the full spec loop. The 4B
   bundle stands in for "target" here — it measures op-shape
   behaviour, not model identity.

### Phase 1 — headline end-to-end (cheap)

Run C0–C4 end-to-end through the async loop, sweeping k ∈ {1..8}.
C1 and C2 are the A/B; the k-sweep tests whether the Phase-0
crossover holds under the real loop (acceptance, overhead, contention).

### Phase 2 — wacky (stretch, new code)

W1 ping-pong, W2 layer-split, W3 dual-draft race. Order by appetite;
W1 is the cheapest of the three (a scheduler change, no model
plumbing).

## Metrics

Per config, AC and battery: end-to-end tok/s, TTFT, mean accept
length, per-island idle %, and **parallel-wall vs serial-sum** (the
async loop already accounts `draft_wait_s` / `verify_wait_s` — that
ratio is the contention indicator). Tag each config with its
predicted concurrent bandwidth sum vs the 228 ceiling, e.g. C1 is
draft-NPU(~66) ∥ verify-GPU(~159) ≈ 225 — right at the edge, so
watch for contention clawback in the wall-vs-sum number.

## Harness reality (what's cheap vs not)

- **Verify** is an HTTP call (`verify_via_target` in
  `scripts/npu_spec_outer_loop_async.py`) to a llama.cpp target
  server. Switching verify between CPU and GPU is just *which*
  llama.cpp server you launch — CPU build vs Adreno-OpenCL build
  (`-ngl 99`). That makes C1 nearly free.
- **Draft** is the NPU sidecar (ORT-QNN). A GPU draft (C2/C3/W3)
  needs a small llama.cpp GPU server wired in as an alternate draft
  source — modest adapter work.
- The async draft∥verify overlap already exists
  (`npu_spec_outer_loop_async.py`, Lever A).
- **Dependency / gating risk:** configs that put **verify on the
  NPU** (C2, C4, and W1's NPU half) need an NPU context bundle of
  the *target* model. Only the 4B bundle exists today (8B-on-NPU is
  open — roadmap W1.b). So end-to-end C2/C4 are gated on a larger
  NPU bundle. **But Phase 0's crossover microbench uses the 4B
  bundle and answers the C1-vs-C2 question regardless** — run that
  first; it de-risks the whole side quest before any bundle work.

## Models

Draft: Qwen3-4B NPU bundle (P0 w4a16 / w8a16, already built and on
the device). Target: Qwen3-14B-Q4_K_M for GPU/CPU verify (SQ1
default; ~9 GB, comfortable in 48 GB). Target-on-NPU configs await a
14B (or 8B) NPU bundle.

## Deliverable

One CSV per config in `results/csv/`, and this doc updated with: the
crossover-k, the filled GPU↔NPU cells of the W4.e matrix, and a
recommended placement policy. Feeds roadmap W4.e and the
`async_orchestration.md` deliverable.

---

## Phase 0 results — verify-shape microbench (2026-05-22)

Measured with Qwen3-4B as the op-shape test model. GPU = llama.cpp
OpenCL backend (Q4_0 GGUF) on the Adreno X2-90; NPU = Qualcomm w4a16
Genie bundle via ORT-QNN. AC power. Raw data + findings in
`gpu_npu_sidequest/`.

Per-pass wall time of one k-token forward:

| k   | GPU ms/pass  | NPU ms/pass |
|-----|--------------|-------------|
| 1   | 39 (TG path) | 34 (AR1)    |
| 2   | 163          | 52 \*       |
| 4   | 167          | 52 \*       |
| 8   | 168          | 52 \*       |
| 16  | 168          | 52 \*       |
| 32  | 175          | 52 \*       |
| 64  | 201          | 52 \*       |
| 128 | —            | 52 (AR128)  |

\* The NPU QNN context binary ships only two graphs — `ar1` ([1,1])
and `ar128` ([1,128]); the batch dim is frozen at compile time. Any
k ∈ [2,128] runs on the AR128 graph (k padded up), so they all cost
the one flat ~52 ms. There is no AR2/4/8/16/32 graph.

### Findings

1. **No crossover — the NPU wins verify at every k.** The
   hypothesised "verify flips to the NPU only at large k" is wrong:
   the NPU is ~3× faster than the GPU at the verify (PP) shape
   across the whole useful range (52 ms vs ~165 ms at k = 3–8), and
   faster at k = 1 too. There is no k where the GPU verify wins.

2. **The GPU PP path is software-bound, not silicon-bound.** GPU
   per-pass cost is a flat ~165 ms plateau for k = 2–16. At the
   Adreno's measured 159 GB/s (`memory_bandwidth_ceiling.md`) a
   bandwidth-bound 4B forward should take ≈14 ms — the 165 ms is
   ~10× the hardware floor. This is a llama.cpp **OpenCL-backend**
   ceiling (its mat-mat / prompt path is far less mature than the
   mat-vec / decode path), not an Adreno limit. GPU TG is healthier
   (~40 ms/token) but still ~3× its bandwidth floor.

3. **The NPU is strongly batch-friendly.** 128× the tokens for
   +52% wall time; per-token cost falls ~84×. Qualcomm's bundle runs
   close to the hardware. Batched forward is the NPU's strong suit —
   exactly the verify workload.

4. **NPU shape-lock is a real constraint.** Small-k verify (k = 4,
   8) has no native graph and pads to AR128, paying the full 52 ms.
   A bundle compiled with an AR8 graph would make small-k verify
   cheaper still — a worthwhile follow-up if the NPU hosts verify.

### This refutes C1 and points to C2

"draft NPU / verify GPU" (C1) is refuted: verify on the GPU costs
~165 ms vs ~52 ms on the NPU. **Verify belongs on the NPU.** The
GPU's role in heterogeneous spec decode is therefore the *draft* —
config **C2 ("PP on NPU, TG on GPU")** — not because the GPU drafts
faster (it doesn't: 40 ms vs the NPU's 34 ms) but because running
draft on the GPU is the only way to overlap it with an NPU verify.

Round-time model for Phase 1 — a spec round of depth k is `k`
sequential draft steps then one batched verify; async overlap
(Lever A) makes round time ≈ max(draft_phase, verify_phase):

- draft NPU ∥ verify NPU — can't overlap (one island) → serial
  sum, slowest.
- draft GPU ∥ verify NPU (**C2**) → verify (~52 ms) hides fully
  under the draft phase; the long pole becomes the draft.
- draft NPU ∥ verify GPU (**C1**) → verify (~165 ms) dominates.
  Worst of the overlapped options.

### Caveats

GPU on Q4_0 vs NPU on w4a16 — different quant and byte footprint;
the comparison is op-*shape*, not a like-for-like model. The GPU
numbers reflect today's llama.cpp OpenCL backend — a better GPU
inference path (the silicon has spare bandwidth) would move these
and could revive a verify-GPU story; a genuine W4 follow-up. NPU
small-k figures are AR128-padded.

### Phase 1 direction (revised)

Measure **C2 end-to-end** with a real small draft model, plus C0
(today) and C1 as baselines. Confirm the GPU PP plateau extends
predictably to pp256/pp512 (fixed-overhead model), and add the CPU
verify-shape anchor for the full 3-island picture.
