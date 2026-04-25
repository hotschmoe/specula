# npu_engine — backlog TODOs

Followups from `docs/npu_engine_prefill_sidequest.md`. Ranked by
impact-per-effort.

## 1. CL=N wrappers for prompts > 512 tokens

**What.** Extend `build_part_cfg(metadata, ar=)` to take a `ctx`
parameter selecting the cl tier. The bundle ships
`{ar1,ar128}_cl{512,1024,2048,3072,4096}_*_of_4` graphs in the
same `.bin` files we already use — no recompile needed.

**Why.** CL=512 caps `pp_tokens + tg_tokens` at 511. Real agentic
workloads run 2-8k token contexts (system prompt + tool outputs +
chat history). Today the engine refuses anything larger. This is
the blocker between "neat sidequest" and "actually usable".

**How.**
- Add `ctx` param to `build_part_cfg(metadata, ar=1, ctx=512)`.
- Update graph_name template to `{prefix}_ar{ar}_cl{ctx}_{N}_of_4`.
- Past-len for AR128 changes per ctx tier: `past = ctx - AR128_BATCH`.
  Already parameterized via `PAST_LEN_AR128_CL512`; generalize to
  a per-ctx constant or computed at runtime.
- AR1 past-len = `ctx - 1`. Same generalization.
- KVStore needs `ctx_len` parameter; mask + RoPE helpers use it.
- Wrappers gain a ctx suffix in the filename (e.g.
  `oracle_part1_ar128_cl2048.wrapper.onnx`).
- Bench + sidecar gain a `--ctx-tier` flag (default 512 for
  parity with current behavior).

**Effort.** ~1 session. Mostly mechanical. Risk: per-ctx tier
might have different IO scales/offsets — verify by re-extracting
from metadata.yaml per tier.

**Validation.**
- Standalone bench at pp=2048, tg=512 on cl2048 wrappers; match
  current AR128 t/s within ~5%.
- Phase-batched demo at the same size; speedup should hold.

**Memory consequence.** Larger ctx tiers = bigger past-KV input
shape = more HTP context memory per session. May further reduce
the simultaneous-session ceiling. Re-test the swap-mode session
budget at cl2048 / cl4096; might force one-session-per-binary
which currently fits.

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
