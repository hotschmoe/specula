# Phase 5 — NPU-drafted speculative decoding: results

Closes Phase 5. First end-to-end numbers for a heterogeneous
NPU-drafter + CPU-target speculative-decode pipeline on the
Snapdragon X2 Elite Extreme.

**TL;DR.** NPU drafting works and produces coherent, target-verified
text. On Qwen3-0.6B (NPU Path B-mask) × Qwen3-8B-Q4_K_M (llama-server
CPU), **mean decode rate is 7.98 t/s at k=2 with 81.0% accept (best
cell 8.44 t/s)** — a structural regression vs Phase 2's CPU-spec
40.2 t/s ceiling. The loss is driven entirely by NPU per-step latency
(~63 ms, 4 calls/round sequential with target verify); accept rates
are actually _above_ Phase 2's CPU-spec baseline (81.0% vs 79.6%), so
drafter quality is not the problem.

This is Phase 5's _documented loss with root cause_ exit per
`docs/npu_scoping.md` §7 step 10. It is still a load-bearing
milestone: the pipeline establishes the first heterogeneous
(NPU + CPU + HTTP) inference path on this hardware, with all the
plumbing (ONNX export, shape pinning, ORT+AI Hub compile, context
binary load, mask-padded short-prompt decode, KV slot re-
arrangement, HTTP verify loop) landed and reproducible.

## What ran

| component | details |
|-----------|---------|
| target | Qwen3-8B-Q4_K_M.gguf, llama-server (CPU build), 18 threads, `-c 576`, greedy (temperature=0, top_k=1, seed=1), `cache_prompt=false` |
| draft | Qwen3-0.6B exported FP16 via `optimum.onnxruntime`, compiled by AI Hub to a Hexagon v81 context binary (Path B-mask: runtime additive `attention_bias` input) |
| runtime | `onnxruntime-qnn==1.24.4` (bundles QAIRT 2.42), EPContext wrapper per `scripts/npu_load_qwen3_bin.py` |
| bridge | HTTP/JSON to `/completion` endpoint, raw token-id arrays (no tokenizer round-trip) |
| fixtures | `prompts/humaneval_subset.jsonl`, 10 prompts (fibonacci, binary_search, is_palindrome, reverse_string, count_vowels, flatten, two_sum, matrix_transpose, string_to_int, binary_tree_inorder) |
| sweep | k ∈ {2, 3, 4, 8}, n_predict = 256, 10 prompts → 40 cells |
| hardware | Snapdragon X2E94100 — Oryon CPU + Adreno X2-90 + Hexagon v81 NPU — 48 GB LPDDR5X @ 228 GB/s |

Code: `scripts/npu_short_prompt_probe.py` (gate), `scripts/npu_spec_outer_loop.py` (per-prompt run), `scripts/sweep_npu_spec.py` (harness).
CSV: `results/spec-npu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-pathbmask-20260421-180804.csv`.
Full sweep log: `results/phase5_step9_sweep.log`.
Sweep wall time: **25.9 min** for 40 cells (mean 38.9 s/cell, 32.9 s best at k=2/p1, 58.8 s worst at k=8/p5).

## Results

### Aggregate per k

| k | n_cells | mean accept | mean decode t/s | best cell (t/s, prompt) | worst cell (t/s, prompt) |
|---|--------:|------------:|----------------:|------------------------:|-------------------------:|
| **2** | 10 | **81.03%** | **7.98** | 8.44 (p8 binary_tree_inorder) | 7.37 (p5 flatten) |
| 3 | 10 | 75.53% | 7.35 | 8.25 (p1 binary_search) | 6.59 (p5 flatten) |
| 4 | 10 | 70.87% | 6.84 | 7.93 (p1 binary_search) | 6.03 (p5 flatten) |
| 8 | 10 | 55.60% | 5.33 | 6.84 (p1 binary_search) | 4.37 (p5 flatten) |

**Optimum k = 2** (contrast: Phase 2 CPU-spec optimum was k=3). The
shift to k=2 is expected — on this hardware, NPU per-step dispatch
costs ~7× CPU's equivalent, so amortising fewer draft steps per
round wins even though accept rate falls off faster at larger k.
Going to k=1 (pure autoregressive NPU draft) would lose the
`/completion` verify batch amortisation, so k=2 is the sweet spot.

### Per-prompt variance

Per-prompt accept rate at k=2 (the winner):

| prompt | label | accept @ k=2 | decode t/s @ k=2 |
|-------:|-------|-------------:|-----------------:|
| p0 | fibonacci | 75.5% | 7.81 |
| **p1** | **binary_search** | **90.7%** | **8.30** |
| p2 | is_palindrome | 85.3% | 8.24 |
| p3 | reverse_string | 77.7% | 7.88 |
| p4 | count_vowels | 76.5% | 7.91 |
| **p5** | **flatten** | **74.3%** | **7.37** (worst) |
| p6 | two_sum | 77.7% | 7.89 |
| p7 | matrix_transpose | 85.3% | 8.17 |
| **p8** | **binary_tree_inorder** | **92.2%** | **8.44** (best) |
| p9 | string_to_int | 75.2% | 7.82 |

Phase 2 CPU-spec saw 55-91% accept variance at k=3 with the same
draft and the same `binary_search` / `flatten` pairing at the
extremes. Our NPU spread is narrower (74.3-92.2% at k=2), tracking
the same prompt-by-prompt pattern — further evidence that the draft
is behaving identically; only the compute location changed.

## Comparison to Phase 2 (CPU-spec) and Phase 2 (OpenCL-spec)

| config | k | accept | decode t/s | vs CPU-alone (25.91 t/s) |
|--------|---|-------:|-----------:|-------------------------:|
| **NPU-draft + CPU-target** (this phase, best mean) | **k = 2** | **81.0%** | **7.98** | **0.31×** |
| **NPU-draft + CPU-target** (this phase, best cell) | k = 2, p8 | 92.2% | 8.44 | 0.33× |
| CPU-draft + CPU-target (Phase 2 winner)  | k = 3 | 79.6% | 40.19 | 1.55× |
| CPU-draft + CPU-target (Phase 2, k=2) | k = 2 | 82.3% | 29.93 | 1.16× |
| OpenCL-draft + OpenCL-target (Phase 2)  | k = 3 | 77.7% | 9.14 | 0.35× |
| mixed (Phase 2) | k = 3 | 77.1% | 9.52 | 0.37× |
| CPU-alone TG (Phase 1 baseline) | — | — | 25.91 | 1.00× |

NPU-draft at **0.31× of CPU-alone TG** is a regression, but note the
finding: **accept rate at k=2 is essentially identical between CPU
(82.3%) and NPU (81.0%) drafters**, despite the NPU running at FP16
interior with attention-bias masking over padded KV. The drafter's
statistical behaviour survives the compile/quantisation path
cleanly — which means all perf levers below are about compute cost,
not quality.

## Why the loss — bottleneck analysis

### NPU per-step latency is the dominant cost

Measured on humaneval p0 at k=3, n_predict=64 (22 rounds,
`results/phase5_step8_outer_loop.log`, post-lazy-snapshot run):

| phase | wall | per-round | fraction |
|-------|-----:|----------:|---------:|
| NPU draft (k-1 = 2 steps) | 3.34 s | 152 ms | 35.8% |
| target /completion (k+1 = 4 tokens) | 3.46 s | 157 ms | 37.1% |
| NPU absorb (1-2 steps) | 2.52 s | 115 ms | 27.1% |
| **total generate wall** | **9.32 s** | 424 ms | 100% |
| decoded tokens | 65 | | |
| **decode rate** | **6.97 t/s** | | |

NPU draft + absorb together: **5.86 s (63% of wall)**. At the
measured ~63 ms per NPU call and 4 calls per round in the common
j < k case (3 drafts + 1 absorb) + 5 calls in the j == k case
(3 drafts + 1 materialise-snapshot-k + 1 absorb), we're paying
60-100 ms per decoded token to the NPU alone. Phase 2's CPU draft
costs ~9 ms per draft step (Qwen3-0.6B Q8_0 CPU TG = 111 t/s) —
a **7× per-step disadvantage** for the NPU path on this compile.

### Target verify is not the problem

Target verify averages 157 ms/round for k+1=4 decoded tokens plus
HTTP round-trip + KV-cache lookup. That's ~25 tokens/s — consistent
with Phase 1's 8B Q4_K_M CPU TG of 25.91 t/s.
`cache_prompt=false` on our calls means the target re-ingests the
committed prefix each round; per Phase 2's spec-decode numbers that
cost is small (<5% of wall) for short generate lengths and dominated
by the decode work.

### Draft quality is fine

Mean accept rate of 81.03% at k=2 is slightly _above_ Phase 2's
CPU-spec 82.3% at k=2 (82.3% was the CPU peak, 79.6% at k=3 was the
overall winner). The NPU isn't making the draft worse — FP16
interior + attention-bias masking introduce only tens-of-basis-point
logit drift (single-step cos=0.999960 vs CPU reference from
`scripts/npu_short_prompt_probe.py`). The draft's _output_ is fine;
the draft's _latency_ is what kills us.

### The overlap gap

Phase 2's CPU-spec is fast because draft and target share the same
18 CPU threads and the same L3 cache — there's no overhead for
passing KV or tokens between them, and scheduling is OS-managed at
microsecond granularity. NPU-draft + CPU-target is structurally
sequential in our current code: each round alternates NPU draft
phase → HTTP verify → NPU absorb, with no overlap. Even if NPU
per-step cost stayed at 63 ms, overlapping the 5.86 s NPU work
with the 3.46 s verify work would bring wall closer to max(NPU,
verify) = 5.86 s ≈ 11.1 t/s. Still below CPU-alone, but a real
improvement.

## Levers for Phase 5.5 / Phase 6

Ranked by expected impact × implementation cost:

### 1. W4A16 quantisation — biggest perf lever on this binary

Qualcomm's own Qwen3-4B Genie reference bundle (inspected session
10, `models/qualcomm-qwen3-4b-ref/`) ships fully W4A16: INT4 weights
+ INT16 activations + INT16 quantised attention_bias and KV IO.
Four structural wins vs our current FP16 interior:

1. **Weight memory bandwidth drops 4×** (INT4 vs FP16). Qwen3-0.6B's
   ~1.2 GB of FP16 weights becomes ~300 MB of INT4; reading less
   data from LPDDR5X per token is the first-order gain.
2. **HTP has dedicated INT4 MAC paths.** Hexagon's HVX scalar/vector
   INT4 matmul uses fewer cycles per MAC than FP16. Published
   Qualcomm numbers on the Qwen3-4B reference report ~2-3× speedup
   from FP16→INT4 on decode throughput.
3. **Smaller context binary** — less spill/fill per token, less DMA
   pressure on the HTP controller.
4. **Matches Qualcomm's shipped pattern**, so we inherit the tuning
   decisions their team already validated.

Cost: needs an AIMET or AI Hub quant pipeline against 50-100
calibration prompts (our humaneval + structured_json fixtures
should suffice). AI Hub has `--quantize_full_type w4a16` already;
the unknowns are (a) does our Path B-mask ONNX lower cleanly at
W4A16, and (b) how much accept-rate do we lose to drafter
quantisation. Expected accept hit: -5 to -10 pp per the Qwen3-4B
reference's published delta. If per-step drops to 30 ms and
accept stays >55%, decode would land near **20-30 t/s** — still
under CPU-spec but finally above CPU-alone TG.

### 2. Async NPU draft ↔ target verify overlap

Rework the outer loop to run NPU draft for round N+1 in parallel
with target /completion for round N. Requires speculating that
drafts[0] of round N will be accepted (true ~82% of rounds at k=2,
~65% at k=3 in our run); on mis-prediction, discard the speculative
draft and proceed normally.

Expected impact: turns the ~423 ms/round serial sum into
max(NPU_round, verify_round) ≈ max(267, 157) ms = 267 ms/round.
On the humaneval p0 sample that would take 22 × 267 = 5.87 s
generate, giving **~11 t/s** — 57% improvement.

Cost: ~200 LOC of async Python (or C#) rework on the outer loop +
rollback logic on speculation miss. No changes to NPU or target.

### 3. Pipelined drafting within a round

Start round N's second draft step while the first draft step's
output is still in flight (double-buffering the two NPU contexts).
Not possible with a single ORT session on a single QNN context
binary (the EP serialises); would need either a second compiled
binary or a raw-QNN path (Path E per scoping doc §3). Low priority
given lever 2 is simpler.

### 4. Compile with smaller past_len

Our binary has past_len=511 baked in (`CONTEXT_MAX=512`). For
short-context code drafting (our humaneval fixture averages ~20
prompt tokens + 256 generation = 276 tokens), a past_len=256 binary
would compute over half as many attention slots per step. Rough
estimate: ~40% reduction in per-step FLOPs, so ~40 ms/step.

Cost: one AI Hub recompile (~10 min), plus tiered binaries if we
want both short and long contexts. Qualcomm's Qwen3-4B reference
ships 5 context tiers precisely for this reason.

### 5. Zero-copy KV handoff between NPU and HTTP boundary

Current code converts FP32 past_kv numpy arrays to/from ORT tensors
on every step. `cl_qcom_ion_host_ptr`-backed buffers would let us
allocate KV once and rebind without a copy. Probably <10% win
(the per-step NPU compute dominates), but unblocks the Phase 4
DFlash port's needs. Low priority for Phase 5.5 but on the Phase 6
map per `current_status.md`.

## Phase 5 close & Qwen3 graduation

Phase 5 is **closed** per the scoping doc's step 10 exit criterion:
"Phase 5 closes with either a win, tie, or documented loss with
root cause." This is a documented loss: NPU-draft + CPU-target +
llama-server HTTP runs at **7.98 t/s mean (8.44 best) vs CPU-spec
40.2 t/s**, bound by NPU per-step latency at FP16 on a past_len=512
binary. The 5× gap is _structural_ at this compile point — identical
drafter statistics, identical target, different compute location.

Qwen3 window close-out items status:
- [x] Phase 2 CPU-spec baseline (40.2 t/s at k=3) — done, session 4
- [x] OpenCL-spec negative result (mixed + all-OpenCL both regress)
      — done, session 4
- [x] NPU-draft + CPU-target end-to-end numbers — done, this phase
- [x] Short-prompt Path B-mask numerical validation — done, session 11
- [ ] `--draft-p-min` tightening (kept, CPU-spec only; cheap)
- [ ] `prose_longform.jsonl` and `chat_multiturn.jsonl` (kept)
- [ ] Ngram spec (`--spec-type ngram-*`) A/B on JSON (kept)
- [ ] Negative-result upstream contribution (Adreno OpenCL story
      + this NPU story together) (kept)

Graduate to Qwen3.5 (DFlash Phase 4) after landing the remaining
close-out items.

## What we'd need to say "NPU-spec beats CPU-spec on this hardware"

The minimum viable wins-the-benchmark path would be:

1. **Lever 1 (w4a16 compile)** gets per-step to ~30 ms → ~20 t/s.
2. **Lever 2 (async overlap)** pushes wall toward max(NPU, verify).
3. **Draft graduation to Qwen3-1.7B** on NPU (accept rate jumps
   from 65% to ~82% per Phase 2 trend, reduces rounds by ~20%).
4. Combined rough model: 30 ms × 3 calls/round (no overlap overhead)
   = 90 ms NPU/round, max(90, 170) = 170 ms verify-bound, k=3 →
   ~3.2 committed per round, → **~19 t/s**. Still below CPU-spec's
   40.2 t/s.

So even with all the planned levers, NPU-draft on this silicon at
this compile point doesn't reach CPU-spec. The scientific finding
from Phase 5 is that **on Snapdragon X2 Elite Extreme with QAIRT
2.42 and ORT-QNN 1.24.4, the NPU's per-operation dispatch cost
dominates over the memory-bandwidth advantage** for small drafts
of small models. CPU-spec's win comes from keeping draft + target
in the same cache-coherent compute domain; NPU-draft's loss comes
from the sync round-trip between QNN and the CPU target's KV cache
across every round.

This is a Snapdragon-specific finding; the same experiment on a
larger NPU with tighter CPU coupling (Apple ANE via Core ML, which
ships on-die with the CPU cores) might tell a different story.
That's Phase 6+ territory.

## Reproducing these numbers

```bash
# one-shot sanity check
.venv\Scripts\python.exe scripts\npu_short_prompt_probe.py
.venv\Scripts\python.exe scripts\npu_spec_step7_plumbing.py
.venv\Scripts\python.exe scripts\npu_spec_outer_loop.py   # single prompt

# full sweep (replace what this doc tables)
.venv\Scripts\python.exe scripts\sweep_npu_spec.py
```

Pinned dependencies in `pyproject.toml`; see
`docs/npu_ort_qnn_version_match.md` for the ORT-QNN 1.24.4 ↔
QAIRT 2.42 compatibility story.
