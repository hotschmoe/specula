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

## What landed (5 commits in the npu_engine/ package)

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

## What's next

1. **Sidecar architecture** — long-lived process holding sessions,
   IPC entry point. Eliminates per-request swap cost in steady state.
   Each user-visible request just pays compute time:
     - Prompt < ~576 tokens: AR1 prefill, ~37 ms/token
     - Prompt ≥ ~576 tokens: AR128 prefill at ~2200 t/s + decode

2. **Multi-prefill batching** — when multiple long prompts queue, do
   them all in one AR128 session before swapping back. Single 36 s
   swap amortizes across N prefills.

3. **CL=1024 / 2048 / 4096 wrappers** — the bundle has these graphs
   too. Needed to support prompts > 512 tokens. Mechanical extension
   of `build_part_cfg` to take a `ctx` parameter alongside `ar`.

4. **Battery J/tok measurement** — the original headline target was
   closing Genie's 0.54 J/tok lead. The 5× lower compute wall in
   AR128 mode should drop J/tok dramatically; needs a battery run.

5. **C++ sidecar** — the ultimate Genie-parity move. Would eliminate
   Python's per-step overhead entirely. Worth doing only after the
   Python sidecar establishes the architecture and we know
   precisely what perf is left on the table.

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
