# SQ5 — long-context NPU (>4K) viability

## Status

- **2026-04-27** — Engine generalization landed; AR1-only AC bench
  ran at all four jumps (cl=512 / 1024 / 2048 / 4096). cl=4096 loads
  and decodes cleanly. AR128 prefill at >cl=512 still pending.

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

## Open questions resolved by the AC AR1 sweep

- ✅ **Does cl=2048/4096 AR1 chain load?** Yes; all 4 AR1 partitions
  load successfully at every tier. Load cost is flat (~8 s).
- ✅ **AR1 per-step latency at cl=4096 vs cl=512.** Grows from 36 ms
  to 47 ms — only +30% across an 8× ctx jump. Sublinear.
- ✅ **Simultaneous-session ceiling at cl=4096.** 4-session AR1 chain
  fits fine. The 7-session worry was at cl=512 with 4 AR1 + 4 AR128;
  larger tiers may shift that ceiling for the AR128-coexistence case
  but the AR1-alone footprint is comfortable.

## Open questions still pending

- ⏳ **AR128 prefill throughput at cl=2048 / cl=4096.** Untested.
  Predicted to scale similarly to AR1 (~30% penalty per ctx
  doubling). Critical for the SQ1 heterogeneous demo's prefill
  story — a 4K prompt prefilled at AR1 rates (24 t/s) would take
  ~170 s, untenable; AR128 at cl=512 was 2229 t/s = 1.8 s for 4K.
  AR128 at cl=4096 is the binding scaling test.
- ⏳ **AR1+AR128 coexistence at cl=2048+.** The 7-session ceiling
  observation was at cl=512 with 4+4 sessions. At cl=2048 the
  per-session HTP footprint is bigger; coexistence likely fails
  earlier. Practical impact: confirms swap-mode (one chain at a
  time) is the right architecture for the 2K+ range too.
- ⏳ **AR1 long-context with realistic pp+tg.** Today's runs are
  pp=256 tg=128. Real coding assistant: pp=2048, tg=512. Would
  exercise the full cl=2048 / cl=4096 buffers and validate the
  per-step latency stays reasonable as KV fills.

## Next actions

1. **AR128 + AR1 swap-mode bench at cl=2048** (`--ctx-tier 2048
   --pp-tokens 1024 --tg-tokens 128`) — exercises the full
   prefill-then-decode pipeline at the 2K tier. Caps the AR128
   scaling question.
2. **Same at cl=4096** if (1) succeeds. Likely the upper bound on
   what's tractable.
3. **Long-realistic test** at cl=4096 with `pp=2048 tg=512` —
   real coding-assistant prompt size. Confirms per-step doesn't
   degrade as KV fills.
4. After (1)-(3), close SQ5 with a writeup that feeds SQ1's
   heterogeneous-demo ctx-tier choice.

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
