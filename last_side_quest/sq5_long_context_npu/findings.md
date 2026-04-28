# SQ5 — long-context NPU (>4K) viability

## Status

- **2026-04-27** — Engine generalization landed. Pure-Python smoke
  test (`smoke_metadata.py`) passes at all 5 ctx tiers (512 / 1024 /
  2048 / 3072 / 4096). NPU smoke + bench at cl=2048/4096 still
  pending.

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

## Open questions (resolved by NPU run)

1. **Does the cl=2048 4-partition AR1 chain even load?** Each AR1
   partition's `.bin` is unchanged (the bundle re-uses the same
   .bin for all tiers, switching graphs by name) so file I/O cost
   should be flat. But HTP context-init cost grows with the
   compiled graph's KV shape — unknown by how much.
2. **AR1 per-step latency at cl=2048 vs cl=512.** Today: 36 ms median
   on AR1 at cl=512. Prediction: per-step compute is mostly
   bandwidth-bound on KV concat → linear-ish growth. Could reach
   ~144 ms at cl=2048 (4× past KV) or be sublinear if the
   attention math dominates rather than KV BW. **This is the
   binding number for the SQ1 heterogeneous demo.**
3. **AR128 prefill throughput at cl=2048.** Today: 2229 t/s at
   cl=512. Prediction: stays similar in t/s but adds proportional
   time for the bigger past-KV.
4. **Simultaneous-session ceiling at cl=2048 / cl=4096.** Today:
   ~7 sessions at cl=512. Prediction: each session uses 4× more
   HTP memory for past-KV at cl=2048; ceiling drops to maybe 3-4
   sessions. May force one-session-per-binary load pattern (worse
   swap cost) at higher tiers.

## Next actions

1. **NPU smoke at cl=2048** — load 4 AR1 sessions at cl=2048,
   confirm session creation succeeds. If yes, run a short bench
   (`--ctx-tier 2048 --pp-tokens 1024 --tg-tokens 256
   --no-ar128`) — AR1-only is the cleanest first measurement.
2. If smoke passes → **AR1 sweep across tiers** (cl=512 / 1024 /
   2048) at fixed `pp=256 tg=128` — record per-step latency and
   compare against the cl=512 baseline.
3. **AR128 sweep** at cl=2048 / cl=4096 — measure prefill t/s and
   see if the AR128 advantage extends to higher tiers.
4. If cl=4096 fails to load 4 partitions: document, then either
   try a 1-stream load (single partition at a time) or document
   the ceiling.
5. **Strategic conclusion** for the user's "are we dead in the
   water for coding assistants" question once cl=4096 numbers
   are in.

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
