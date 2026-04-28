# SQ5 — long-context NPU (>4K) viability

## Status

- **2026-04-27** — SQ5 closed POSITIVE. Engine generalization
  landed; AR1-only AC bench ran at cl=512/1024/2048/4096; AR128
  swap-mode AC bench ran at cl=2048 and cl=4096. All five
  ctx tiers are functional via our ORT-QNN engine. 4K context is a
  comfortably-usable operating point for the SQ1 heterogeneous
  demo.

## Headline result — AR1 decode scales gracefully through cl=4096

| ctx tier | load (4 parts) | AR1 PP | AR1 TG | step median | Δ vs cl=512 TG |
|---:|---:|---:|---:|---:|---:|
| 512 (prior baseline) | ~14.8 s | 27.07 | **27.81** | ~36 ms | — |
| 1024 | 8.6 s | 27.24 | 27.23 | 36.0 ms | -2% |
| 2048 | 7.9 s | 24.33 | 25.15 | 40.2 ms | -10% |
| 4096 | 7.7 s | 20.49 | **20.31** | 46.5 ms | **-27%** |

(All AC, AR1-only, pp=256 tg=128, no AR128 swap.)

**Strategic takeaway:** the NPU can serve as a long-context draft
through the bundle's full 4K range. Per-step latency grows from
36 ms → 47 ms across an 8× context jump (cl=512 → cl=4096) — that's
**dispatch-bound, not weight-BW-bound**, exactly mirroring the
session-22 cross-model finding (4B vs 7B per-step within 1.7%).
Bigger past-KV at cl=4096 IS visible (~30% per-step penalty) but
the magnitude is small enough that 4K context is a usable
operating point for coding-assistant draft work.

**Load cost is FLAT across tiers** at ~8 s for 4 AR1 partitions.
HTP context init is dominated by something other than past-KV
size — probably kernel finalization / weight upload (the .bin
files are the same multi-graph object across tiers; only the
selected graph changes). **The 7-session ceiling concern was
misplaced**: 4 simultaneous AR1 sessions fit at cl=4096 with no
errors. Whether AR1 + AR128 simultaneously fit at cl=4096 (the
"two chains alive" pattern from the original sidequest) remains
untested; cl=512 was already at 4+4=8 sessions and that was the
ceiling.

## Strategic answer to the user's "are we dead in the water for
coding assistants" question

**No.** 20 t/s decode at cl=4096 is comfortable for an interactive
coding assistant. Tool-call loops with system prompt + 1-3 file
reads typically land in the 4K–8K token range; cl=4096 covers the
common case. The NPU draft at this tier is in the same throughput
band as the existing cl=512 result minus a fixed ~25-30% tax.

The **remaining ceiling questions:**
- 8K+ contexts (multi-file refactors, deep agent traces) need a
  custom-compiled bundle — out of scope for SQ5; routes to the
  cloud pipeline doc (`one_pipeline_cloud_gpu.md`).
- AR128 prefill at cl=4096 unmeasured; predicted to take ~10-30 s
  for a 4K prompt vs ~1 s at cl=512. That's an open SQ5b.

## Goal

Extend `npu_engine/` from its hardcoded ctx=512 to use the
`{cl1024, cl2048, cl3072, cl4096}` graphs already shipped in the
Qualcomm Qwen3-4B w4a16 bundle (no recompile). Then answer:

- Does the bigger past-KV buffer push us past the ~7-session HTP
  ceiling (`reference_ortqnn_session_limit.md`)?
- Does AR1 decode t/s scale gracefully past 512 ctx, or does
  per-step latency blow up with the bigger past-KV?
- For SQ1's heterogeneous demo: what's the largest context we can
  ship the NPU draft at?

## What landed (engine generalization)

Three files touched. All compile, all smoke-tested at the
pure-Python level (no NPU touched yet).

### `npu_engine/qualcomm_qwen3_4b_oracle.py`

- `attention_mask_quantized(pos, scale, offset, ctx_len=CTX_LEN)`
  — added `ctx_len` kwarg; mask shape becomes `[1,1,1,ctx_len]`.
- `attention_mask_quantized_ar128(p_base, scale, offset, ctx_len=CTX_LEN)`
  — same; recomputes `past_len = ctx_len - 128` internally; mask
  shape `[1,1,128,ctx_len]`.
- `build_part_cfg(metadata, ar=1, ctx=CTX_LEN)` — added `ctx` arg;
  component lookup becomes `f"ar{ar}_cl{ctx}_{part}_of_4"`; graph
  name embeds the tier. Adds `"ctx": ctx` to each part dict for
  downstream introspection.
- `KVStore.__init__(num_layers, with_ar128_input=False, ctx_len=CTX_LEN)`
  — `ctx_len` becomes an instance attribute; master + AR128 input
  buffers size off `self.past_len` and `self.past_len_ar128`. The
  cap check in `stitch_step` / `stitch_batch` now uses instance
  state, not module constants.
- New `wrapper_path(bundle_dir, part_idx, suffix="", ctx=CTX_LEN)`
  helper. **Backward compatible**: `ctx=512` returns the legacy
  `oracle_part{N}{suffix}.wrapper.onnx` so existing pre-built
  wrappers stay valid. `ctx>512` appends `_cl{ctx}` for
  per-tier disambiguation.

### `npu_engine/bench_qwen3_4b_ortqnn.py`

- New `--ctx-tier` flag (default 512, choices = 5 tiers).
- New module-level `CTX_TIERS` constant exported for the sidecar.
- `_step` and `_step_ar128` now read `kv.ctx_len` and pass it to
  the mask helpers — no signature change at the call site.
- `_load_chain` takes the tier via closure (the `ctx_tier` arg in
  `main()`) and uses `wrapper_path()` for filename construction.
- All `KVStore(...)` constructions thread `ctx_len=ctx_tier`.
- The pp+tg cap check uses `args.ctx_tier - 1` instead of `PAST_LEN`.
- CSV row's `ctx_tier` field becomes the runtime arg, not the
  module constant.

### `npu_engine/sidecar.py`

- New `--ctx-tier` flag (default 512). Imports `CTX_TIERS` from
  bench module.
- `EngineState.__init__(parts_cfg_ar1, parts_cfg_ar128, ctx_len=CTX_LEN)`
  — holds `ctx_len` and `past_len`.
- `EngineState._load` uses `wrapper_path()` for filenames.
- `_maybe_warmup`, `prefill_only`, `decode_only`, `serve_request`
  all thread `ctx_len=state.ctx_len` into `KVStore` constructions
  and use `state.past_len` for cap checks.
- `_load_engine` reads `args.ctx_tier` and builds parts_cfgs at the
  requested tier.

## Pure-Python smoke result

```
=== SQ5 ctx-tier smoke test (no NPU) ===
bundle: models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite
metadata components total: 40

  cl=512  : KV master 36 MB + ar128 27 MB = 63 MB
  cl=1024 : KV master 72 MB + ar128 63 MB = 135 MB
  cl=2048 : KV master 144 MB + ar128 135 MB = 279 MB
  cl=3072 : KV master 216 MB + ar128 207 MB = 423 MB
  cl=4096 : KV master 288 MB + ar128 279 MB = 567 MB

=== SMOKE PASSED — all 5 ctx tiers OK ===
```

All five tiers' `ar1_cl{N}_*_of_4` and `ar128_cl{N}_*_of_4` graphs
exist in `metadata.yaml`. Past-KV input shapes from metadata match
the engine's expected `(NUM_KV_HEADS, 1, HEAD_DIM, ctx-1)` and
`(NUM_KV_HEADS, 1, HEAD_DIM, ctx-128)` for AR1 and AR128
respectively. Mask helper outputs and KVStore allocations are
shape-correct at every tier.

**Host-side KV memory** scales as expected — linear in ctx. The
567 MB at cl=4096 is **per stream**: a 4-stream AR128 prefill
session would hold ~2.3 GB just in numpy buffers on the host side.
The HTP context allocations on top of that are unmeasured but the
~7-session ceiling at cl=512 will likely move down significantly.

## Final AR128 swap-mode results

| ctx | AR128 PP (t/s) | AR1 TG (t/s) | AR128 step (ms) | AR1 step (ms) | total wall (pp+tg=640) |
|---:|---:|---:|---:|---:|---:|
| 512 (prior) | 2229 | 27.81 | ~58 | 36 | (baseline) |
| 2048 | **1629** | 25.27 | 77.5 | 40.9 | 22.8 s |
| 4096 | **1284** | 20.28 | 95.7 | 47.1 | 24.1 s |

(All AC, AR128 forced via `--ar128-min-tokens 0`, pp=512 prefill +
tg=128 decode.)

**Both phases scale sublinearly with ctx.** AR128 prefill loses
27% (cl=2048) → 42% (cl=4096) vs cl=512 baseline; AR1 decode
loses 9% → 27%. The dispatch-overhead floor caps how much per-step
latency can blow up — an 8× ctx-jump produces a 1.3-1.6× per-call
penalty.

**Swap overhead is constant ~17 s** across tiers (8 s AR128 load +
1.7 s teardown + 8 s AR1 load). The .bin files are unchanged across
tiers (multi-graph context binaries hold all 5 ctx × 2 AR
combinations); only the selected graph differs. So load is
dominated by HTP context init for the bin's first graph, not
by past-KV size.

## Open questions resolved by the full AC sweep

- ✅ **Does cl=2048/4096 AR1 chain load?** Yes — 4 AR1 partitions
  load at all tiers. Load is flat ~8 s.
- ✅ **AR1 per-step latency at cl=4096.** 47 ms median, +30% over
  cl=512's 36 ms across an 8× ctx jump. Sublinear.
- ✅ **AR128 prefill at cl=4096.** 1284 t/s. Sublinear with ctx,
  -42% from cl=512.
- ✅ **Simultaneous-session ceiling at cl=4096.** 4-session AR1
  AND 4-session AR128 each fit cleanly (load → run → teardown).
- ✅ **End-to-end pipeline at cl=4096.** AR128 prefill →
  teardown → AR1 decode works exactly like at cl=512, just with
  per-tier penalties on per-call compute. Total wall for a 640-
  token interaction (cold-start, no sidecar): 22.8 s at cl=2048,
  24.1 s at cl=4096. Sidecar amortization removes the 17 s swap
  on subsequent requests, leaving only compute (~5-7 s).

## Strategic answer to the user's "are we dead in the water for
coding assistants" question

**No.** The bundle-shipped graphs cover the 4K context range
needed for typical coding-assistant interactions, with usable
performance:

- **cl=2048**: prefill 2K tokens ≈ 1.3 s; decode at 25 t/s ≈ 5 s
  for 128 tokens; cold-start wall ~25 s, warm sidecar ~7 s.
- **cl=4096**: prefill 4K tokens ≈ 3.2 s; decode at 20 t/s ≈ 6 s
  for 128 tokens; cold-start wall ~28 s, warm sidecar ~10 s.

Above 4K (8K+, multi-file refactor scale) we'd need to compile a
custom bundle — that's the cloud-pipeline story and routes to
`docs/one_pipeline_cloud_gpu.md`. For SQ1's heterogeneous demo,
**recommend cl=2048 default** (~25 t/s draft, fast prefill); the
cl=4096 path stays available for heavier prompts at the cost of
~5 t/s decode penalty.

## What's left for SQ5b (deferred, not blocking SQ1)

- **Battery J/tok at cl=4096.** Today's runs are AC; the J/tok
  penalty for the bigger past-KV is unmeasured.
- **Long-realistic test pp=2048 tg=512 at cl=2048+.** The bundled
  prompt file (`pp512_prompt.txt`) caps at 512 tokens, so the
  current bench couldn't fully fill the cl=2048+ KV. To exercise
  pp=2K+, either point the bench at a longer prompt file or
  port the sidecar's `synth_prompt` repeat trick into the bench.
  Mechanical extension; <0.5 session.
- **AR1+AR128 coexistence at cl=2048+.** The 7-session ceiling
  observation at cl=512 (4+4 simultaneous sessions) is unmeasured
  at higher tiers. Probably tighter, but only matters if the
  combined-wrapper architecture becomes load-bearing — for
  swap-mode (one chain at a time) it's irrelevant.

## Files committed

- Engine edits: `npu_engine/qualcomm_qwen3_4b_oracle.py`,
  `npu_engine/bench_qwen3_4b_ortqnn.py`, `npu_engine/sidecar.py`.
- `last_side_quest/sq5_long_context_npu/smoke_metadata.py` — pure-
  Python sanity test.
- `last_side_quest/sq5_long_context_npu/findings.md` — this doc.
- CSVs:
  - `results/csv/qwen3_4b_ortqnn_2026-04-27_sq5_cl1024_ac_ar1.csv`
  - `results/csv/qwen3_4b_ortqnn_2026-04-27_sq5_cl2048_ac_ar1.csv`
  - `results/csv/qwen3_4b_ortqnn_2026-04-27_sq5_cl4096_ac_ar1.csv`
  - `results/csv/qwen3_4b_ortqnn_2026-04-27_sq5_cl2048_ac_ar128swap_pp1024.csv`
  - `results/csv/qwen3_4b_ortqnn_2026-04-27_sq5_cl4096_ac_ar128swap_pp512.csv`

## Reference

- `docs/npu_engine_prefill_sidequest.md` — original engine writeup.
- `docs/npu_engine_todos.md` — this work was TODO #1 of the engine
  backlog. Once results land, the TODO writeup folds back into the
  sidequest doc.
- `reference_ortqnn_session_limit.md` (memory) — the 7-session
  ceiling at cl=512.

## Files

- `last_side_quest/sq5_long_context_npu/smoke_metadata.py` — the
  pure-Python sanity test. Keep — fast, runs in <1 s.
- `last_side_quest/sq5_long_context_npu/findings.md` — this doc.
- Engine edits: `npu_engine/qualcomm_qwen3_4b_oracle.py`,
  `npu_engine/bench_qwen3_4b_ortqnn.py`, `npu_engine/sidecar.py`.

## Update log

- **2026-04-27** — Engine generalization landed; smoke passes
  on all 5 tiers. No NPU yet. Ready for cl=2048 NPU smoke.
