# NPU engine v0 — prefill optimizations sidequest

Self-contained engine work that takes the chained ORT-QNN runtime
(driver of the Qualcomm-shipped Qwen3-4B w4a16 bundle) from a naive
AR1-everything baseline to a routing-aware AR1-decode + AR128-prefill
engine that **beats Genie's PP throughput on the same binary**.

Driven from `docs/qwen3_4b_baseline_all_backends.md`'s "ORT-QNN
chained" row (PP 25.76 t/s AR1, J/tok 0.959 — 78% worse than Genie's
Genie's 0.537). The remaining gap to Genie was the engineering
target this sidequest closes.

Code: `npu_engine/` package (split out from `scripts/` so it has room
to grow into the real heterogeneous engine). Bench:
`npu_engine/bench_qwen3_4b_ortqnn.py`.

## Headline results — AC, Qwen3-4B w4a16, X2E NPU

| metric                 | pre-sidequest | post-sidequest | vs Genie (1598 PP / 23.3 TG) |
|---|---:|---:|---:|
| PP — AR1 (compute)     | 25.76 t/s | **27.07 t/s** | n/a (Genie uses AR128) |
| PP — AR128 (compute)   | n/a       | **2229 t/s**  | **+39%** over Genie |
| TG — AR1 (compute)     | 25.78 t/s | **27.81 t/s** | **+19%** over Genie |
| Per-step Python overhead | ~12 ms  | ~7 ms (IOBinding) | — |

The AR128 row is the headline: our stack runs Genie's own prefill
graphs **faster than Genie does** by a clean +39%. This was the
"Matching Genie's *throughput* is tractable; matching its
*efficiency* is the real engineering challenge" note in the original
4B baseline doc — throughput is now solved, surpassed even.

## What landed (npu_engine/ package, sequence of commits)

1. **Reorganization** (`c412530`-ish era). 6 ORT-QNN scripts moved
   from `scripts/` into a dedicated `npu_engine/` package. Now has
   room to grow into the speculative-decode sidecar.
2. **AR128 prefill plumbing**. New `attention_mask_quantized_ar128`,
   `half_dim_rope_quantized_ar128`, `KVStore.stitch_batch`, and
   parameterized `build_part_cfg(metadata, ar=)`. The bundle ships
   AR128 graphs (`prompt_ar128_cl{N}_*`) inside the same `.bin`
   files as AR1 (`token_ar1_cl{N}_*`) — no recompile needed.
3. **IOBinding refactor**. Per-session `run_with_iobinding` with
   pre-allocated output buffers; `KVStore` gains an optional
   AR128-shaped mirror buffer so AR128 KV inputs are zero-copy.
4. **Code-simplifier dedup**. Parts 2/3/4 dispatch consolidated into
   `_run_transformer_part` driven by `PART_LAYER_RANGES` lookup
   tables. Hot loops dropped from ~85 to ~45 lines each.
5. **Swap-mode AR128**. Discovered the HTP context-memory ceiling
   blocks 4 AR1 + 4 AR128 sessions coexisting; restructured `main()`
   into Phase A (load AR128 → prefill) → Phase A.5 (teardown) →
   Phase B (load AR1) → Phase C (decode). Swap costs ~36 s but
   amortizes in a long-lived engine.
6. **Threshold routing + per-partition profile**. `--ar128-min-tokens`
   flag (default 512) skips the swap for short prompts where AR1 is
   already faster end-to-end. Per-partition load timing in both
   phases so we can see whether the load cost is I/O or HTP init.
7. **Sidecar (long-lived engine)**. `npu_engine/sidecar.py`. State
   machine swaps only when `target_mode != current_mode`, so
   back-to-back same-mode requests pay zero load cost. Pure-AR1
   workload of 10 short-prompt requests is 51% faster than 10
   standalone bench runs; mixed (5 AR1 + 2 AR128) is 34% faster.
   See "Sidecar architecture" section below.
8. **Phase batching**. `prefill_only` / `decode_only` primitives +
   per-stream KV (Stream type). Caller batches N AR128 prefills in
   one mode then drains N decodes in another — total mode swaps
   drop from N round-trips to two regardless of N. N=5 batch:
   4.00× speedup vs sequential, 169 s saved. See "Phase-batched
   execution" section below.

## Architecture findings

### ORT-QNN matches EPContext inputs by NAME, not position

Tested. A wrapper ONNX with two EPContext nodes (one for AR1, one
for AR128, both targeting graphs in the same `.bin`) and renamed
graph inputs (e.g. `ar128__input_ids`) fails to load with
`GetGraphInputIndex it != model_input_index_map_.end() was false.
Input name not found.` from `qnn_model.h:76`.

This blocks the elegant "combined wrapper" workaround. Both AR1 and
AR128 internally use names `input_ids`, `attention_mask`, etc., at
incompatible shapes — and ONNX requires unique input names per
shape. So combined wrappers can't disambiguate at the wrapper level.

Saved as `reference_ortqnn_session_limit.md` memory.

### HTP context-memory ceiling: ~7 simultaneous sessions

Loading 4 AR1 + 4 AR128 sessions on this Qwen3-4B bundle exhausts
HTP context memory. The 8th session — AR128 part 4, the largest
partition at ~1 GB — fails with:

```
qnn_backend_manager.cc:1523 Failed to create context from binary.
Error code: 1002  (INVALID_GRAPH)
```

Verified a single AR1 set + a single AR128 set (4 sessions each)
fits when loaded **alone**. Hardlinks / byte-identical copies of
the `.bin` don't help (QNN registers by content, not path).
`share_ep_contexts` provider option doesn't either.

So: **swap mode is the only path that fits**. Pay the load cost
to switch between prefill (AR128) and decode (AR1) chains.

### Per-partition load profile — HTP init dominates, not I/O

```
                size   time  effective bandwidth
  AR128 part 1: 742 MB  1.6s   464 MB/s
  AR128 part 2: 637 MB  2.4s   265 MB/s
  AR128 part 3: 637 MB  4.5s   142 MB/s
  AR128 part 4: 1020 MB 6.5s   157 MB/s
                       ----
                       15.0s total
```

Identical pattern for AR1. Three observations:

1. **Bandwidth drops as partitions accumulate.** Part 3 takes 2× as
   long as part 2 at the same size. Part 4 is 1.6× the size of
   part 2 but 2.7× the time. This is not I/O.
2. **Cold/warm cache behavior** is invisible — load times are
   stable across runs in the same shell session (page cache should
   be warm by run 2, but times don't improve).
3. **Effective bandwidth (~150-460 MB/s) is well below NVMe
   throughput** (typically 3+ GB/s). At pure-I/O bandwidth, the
   full 3 GB bundle would load in ~1 s.

**Conclusion: HTP context init dominates load cost, not file I/O.**
Specifically, kernel finalization + weight upload to HTP + allocator
state grows with cumulative session state.

This means **mmap / page-cache optimizations won't help much**. The
fix is architectural: a long-lived engine process that pays the
~15 s load once at boot and amortizes it across thousands of
requests.

### Empirical crossover: AR128-swap beats AR1-only at >576 tokens

Sweep at `tg=128` (cl=512 cap):

| pp | AR1-only total | AR128-swap total | delta |
|---:|---:|---:|---:|
| 128 | 24.1 s | 40.6 s | +16.4 s |
| 256 | 29.1 s | 41.2 s | +12.0 s |
| 384 | 33.7 s | 40.7 s |  +7.0 s |

Linear fit on AR1: `total = 19.4 + 0.0372 × pp` (sec). AR128-swap
is roughly constant at ~40.8 s (compute is just ~5 s; ~36 s is swap
overhead independent of pp).

**Empirical crossover: pp = 576 tokens.** Above that, the AR128
advantage in compute time exceeds the swap penalty.

The default `--ar128-min-tokens 512` is just below this so the
router defaults to AR128 when a prompt is in the >500-token range.

### IOBinding gains

| metric        | before IOBinding | after IOBinding | delta |
|---|---:|---:|---:|
| AR1 PP t/s    | 25.76 | 27.07 | +5%  |
| AR1 TG t/s    | 25.78 | 27.81 | +8%  |

Modest on AR1 (the per-step NPU compute already dominates over Python
dispatch), more impactful on AR128 where the per-call overhead got
amortized over 128 tokens.

## Crossover with how production engines handle this

| Engine | Pattern | Applies to NPU? |
|---|---|---|
| **llama.cpp** | Dynamic-shape backend; same compiled kernels handle prefill (batch) + decode (1) | **No** — QNN context binaries are compiled for fixed shapes. The bundle ships only AR1 + AR128 graphs. |
| **tinygrad** | Lazy JIT — every shape combo gets a compiled kernel, cached | **No** — kernels are pre-compiled multi-hundred-MB context binaries we don't generate at runtime. |
| **vLLM** | Continuous batching (mix prefill chunks + decode tokens in one forward); chunked prefill; PagedAttention; **server-process pattern** | **Server pattern: yes.** Continuous batching: no, no graph compiled for AR=128+N shape. Chunked prefill: yes, we already do it. |

vLLM's server-process pattern is the architectural shape we want.
The 36 s swap cost is **a per-engine-startup cost, not per-request**,
in a real engine that holds sessions alive.

## Sidecar architecture (landed)

`npu_engine/sidecar.py` is a long-lived process that holds the
ORT-QNN sessions across many inference requests. State machine:

```
  current_mode != target_mode  ->  tear down + load target
  current_mode == target_mode  ->  no swap; reuse loaded chain
```

Two run modes:
- `--mode demo` — runs a fixed mixed-request schedule and prints a
  per-request timing table; computes amortized cost vs N standalone
  bench runs.
- `--mode serve` — reads newline-delimited JSON requests from stdin,
  emits responses to stdout. The IPC interface for external drivers
  (future spec-decode glue, opencode session integration).

### Demo: pure AR1 (10 short-prompt requests)

10 × `pp=256, tg=64` requests. No mode boundaries → zero swaps after
startup.

```
  N requests           : 10
  swap events (>0.5 s) :  0  (would be 10 for N standalone bench runs)
  cum compute_s        :  116.8 s
  total wall (sidecar) :  131.6 s   (startup 14.8 + serve 116.8)
  total wall standalone:  267.3 s
  ===> sidecar saves ~136 s, ~51% faster over 10 standalone runs
```

Per-request consistency: pp 9.1-9.6 s (~27 t/s), tg 2.2-2.4 s
(~27 t/s). Once the AR1 chain is warmed up, every request looks
identical — clean amortization signature.

### Demo: mixed workload (5 AR1 + 2 AR128)

7-request schedule: 3 AR1 (pp=128/128/256) → 2 AR128 (pp=384,
forced) → 2 AR1 (pp=128/128). Only 2 mode boundaries, so 2 swaps
across 7 requests.

```
  swap events (>0.5 s) :  2  (would be 7 standalone)
  cum compute_s        :  45.4 s
  cum swap_s           :  84.3 s
  total wall (sidecar) : 144.8 s
  total wall standalone: 219.8 s
  ===> sidecar saves ~75 s, ~34% faster over 7 standalone runs
```

The smaller win on mixed workloads is because AR128 requests still
pay a full ~42 s round-trip swap (AR1 → AR128 prefill → AR1 decode
→ next request needs AR1 again so we end on AR1). Each *individual*
AR128 request is unchanged; the sidecar only amortizes cost across
requests that share a mode.

### What the sidecar can't help (without further work)

- **Within-request swap.** A single request that has both prefill
  AND decode pays a swap regardless: prefill needs AR128 (for long
  prompts), decode needs AR1. The swap-back is in-flight and can't
  be elided without splitting prefill and decode into separate
  request types.
- **Alternating-mode workloads.** ABABAB sequences swap on every
  boundary, not just two — same as standalone. The state machine
  fights pathological access patterns; only grouped patterns win.

These are addressed by the next two items below.

## Phase-batched execution (landed)

vLLM-style separation: `prefill_only(...)` returns a `Stream` (KV
cache + decode position) without running decode; `decode_only(...)`
consumes a stream until N tokens are generated. Caller batches all
prefills into one AR128 mode-batch, then drains all decodes in one
AR1 mode-batch. Total mode swaps drop from N round-trips (one per
request) to exactly two (one each direction).

`--mode demo-phase-batch --n-phase-batch N` runs the same workload
twice — once naive (current sidecar `serve_request`), once phase-
batched — and reports the speedup.

### Workload: N × (pp=384 forced AR128, tg=64)

| N | naive total | phase-batched total | speedup | saved |
|---:|---:|---:|---:|---:|
| 3 | 134.1 s | 50.0 s | **2.68×** | 84 s |
| 5 | 224.8 s | 56.2 s | **4.00×** | 169 s |

Per-stream timing in the N=5 phase-batched pass shows the win
crisply:

```
  prefill stream 0: 21.73 s  (one swap to AR128 + 0.17 s compute)
  prefill stream 1:  0.18 s  (zero swap; same mode)
  prefill stream 2:  0.17 s  (zero swap)
  prefill stream 3:  0.20 s  (zero swap)
  prefill stream 4:  0.22 s  (zero swap)
  decode  stream 0: 23.93 s  (one swap to AR1 + 2.4 s compute)
  decode  stream 1:  2.54 s  (zero swap)
  decode  stream 2:  2.49 s  (zero swap)
  decode  stream 3:  2.41 s  (zero swap)
  decode  stream 4:  2.32 s  (zero swap)
```

Asymptotic speedup approaches `2 × per_request_naive_swap /
per_request_compute` ≈ `84 s / 2.6 s` ≈ 32× as N grows. Real
agentic workloads (3–10 batched prefills) sit in the 2.7–5× range.

The compute cost is **identical** between naive and phase-batched —
all that changes is whether the swap is paid once or N times.

### What this enables

- An agent that prefills 5 parallel sub-task prompts can do them all
  in 56 s instead of 225 s.
- A long-running session that processes a stream of long prompts
  with delayed decode (e.g. tool-use chains where the LLM reads a
  big context, then produces a short answer per turn) gets the
  phase-batched savings naturally.
- For draft-target spec-decode where one big context drives many
  short hypotheses, the prefill is paid once for both target and
  draft — same architecture.

## What's next

1. **CL=1024 / 2048 / 4096 wrappers**. The bundle has these graphs
   too. Needed to support prompts > 512 tokens. Mechanical extension
   of `build_part_cfg` to take a `ctx` parameter alongside `ar`.

2. **Battery J/tok measurement**. The original headline target was
   closing Genie's 0.54 J/tok lead. The 5× lower compute wall in
   AR128 mode should drop J/tok dramatically; needs a battery run.

3. **Async prefill / decode interleave**. Real chat workloads alternate
   prefill (new turn) + decode (model response). Phase batching today
   only wins when prefills queue up; an async scheduler that defers
   AR1 swap-back until either a decode is requested OR the AR128
   queue idles for some timeout would win even on streaming workloads.

4. **C++ sidecar**. The ultimate Genie-parity move — eliminates
   Python's per-step overhead entirely. Only worth doing after the
   Python sidecar establishes the architecture and tells us precisely
   what perf is left on the table.

## Artifacts

- **Permanent**:
  - `npu_engine/` package (entire)
  - `results/csv/qwen3_4b_ortqnn_2026-04-25_ar1_iob_ac.csv` — AR1+IOBinding baseline (no AR128)
  - `results/csv/qwen3_4b_ortqnn_2026-04-25_ar128swap_ac.csv` — first AR128 swap-mode run
  - `results/csv/qwen3_4b_ortqnn_sweep_pp{128,256,384}_{ar1,ar128}.csv` — 6-point sweep
  - This doc.
- **References**: `reference_qualcomm_graph_naming.md`,
  `reference_ortqnn_session_limit.md` (memory).

## Update log

- 2026-04-25: full sidequest landed in 6 commits. AR128 swap-mode
  beats Genie at PP by +39%; TG beats Genie by +19% with IOBinding;
  empirical crossover 576 tokens. HTP context init (not I/O) is the
  dominant load cost — argues for sidecar architecture as the next
  optimization.
- 2026-04-25 (follow-up, same day): sidecar landed
  (`npu_engine/sidecar.py`). Pure-AR1 amortization is perfect
  (10 requests, 0 swaps after startup, 51% faster than standalone).
  Mixed workloads still pay AR128 round-trip per-request (decode
  forces swap-back to AR1) — phase batching is the architectural
  next step for AR128-heavy workloads.
- 2026-04-25 (third pass, same day): phase batching landed.
  `prefill_only` + `decode_only` + Stream type. N=3 batched
  AR128 requests: 2.68× faster than naive. N=5: 4.00× faster
  (169 s saved). Total mode swaps drop from N round-trips to
  exactly two. Identical compute time, swap overhead amortized.
