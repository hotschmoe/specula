# npu_engine — backlog TODOs

Followups from `docs/npu_engine_prefill_sidequest.md`. Ranked by
impact-per-effort.

## 1. ~~CL=N wrappers for prompts > 512 tokens~~ — SHIPPED 2026-04-27

Landed in commits 29853b5 / c9d8242 / 5f30fba via SQ5 of the
last_side_quest umbrella. `--ctx-tier` flag on bench + sidecar;
`build_part_cfg(metadata, ar=, ctx=)`, `KVStore(ctx_len=)`,
mask helpers, and `wrapper_path()` all parameterized.

Per-tier AC measurements:

| ctx | AR1 step | AR1 TG (t/s) | AR128 PP (t/s) |
|---:|---:|---:|---:|
| 512 | 36 ms | 27.81 | 2229 |
| 1024 | 36 ms | 27.23 | — |
| 2048 | 40 ms | 25.27 | 1629 |
| 4096 | 47 ms | 20.28 | 1284 |

Sublinear scaling on both phases; per-step penalty is dispatch-bound
not weight-BW-bound (consistent with session 22's 4B-vs-7B finding).
4-partition load cost is FLAT at ~8 s across all tiers — HTP
context init dominates, mmap/cache won't help.

Memory-ceiling concern was misplaced: 4 simultaneous AR1 sessions
fit cleanly at cl=4096; 4 simultaneous AR128 sessions also fit.
AR1+AR128 coexistence at cl=2048+ untested but irrelevant for
swap-mode (the only architecture in use).

Full writeup: `last_side_quest/sq5_long_context_npu/findings.md`.

**Still deferred (not blocking SQ1):**
- battery J/tok at cl=2048+
- pp=2048+ realistic-fill test (needs longer prompt file or sidecar's
  `synth_prompt` repeat ported into the bench)

## 2. Async prefill / decode interleave

**What.** The sidecar today pays a swap on every transition between
prefill and decode within a single chat turn. Phase batching only
helps when prefills queue up (agent batch jobs, spec-decode), not
streaming chat. An async scheduler that defers AR1 swap-back until
either a decode is requested OR the AR128 queue idles for some
timeout would win on streaming workloads.

**Why.** Real chat: turn N is `<user input>` → prefill → decode
response → turn N+1 `<more input>` → prefill → decode. Each
prefill→decode boundary costs ~21 s today. If we delay the
swap-back during a soft window (e.g. 500 ms), and a follow-up
prefill arrives in that window, we save the round trip.

**How.**
- Sidecar gains an event loop (asyncio or thread + queue).
- Pending requests buffered with arrival timestamps.
- Scheduler state machine:
  - `mode=ar128, pending_decodes>0, idle_for>T` → swap to ar1.
  - `mode=ar1, pending_prefills>0` → swap to ar128.
- Each request returns immediately with a future; engine drains
  the queue in mode-batched waves.
- Knob: `--swap-idle-ms` (default ~500). Tune with realistic chat
  traces.

**Effort.** ~2-3 sessions. The biggest piece is rewriting the
sidecar from synchronous serve_request loop to async scheduler.
Risk: introduces concurrency bugs around shared NPU state and
KV cache lifetime.

**Validation.**
- Chat-trace replay (synthetic): alternating prefill+decode every
  N seconds. Compare wall vs current sidecar.
- The win should be visible when prefills cluster within the idle
  window even though the user pattern looks streaming.

**Dependency.** Best done AFTER (1) since async scheduling on
a CL=512-only engine is artificial — real chat traces don't fit.

## Notes

- `docs/npu_engine_prefill_sidequest.md` is the reference for
  what's already landed. Keep this TODO file lean — when a TODO
  ships, move its writeup to the sidequest doc and delete the
  entry here.
- For the next session: rerun benchmarks first (battery J/tok
  measurement is the original Genie-efficiency target that's
  still open) before picking up TODO 1 or 2.
