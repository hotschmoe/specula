# Phase 1 — config C2 (draft GPU ∥ verify NPU)

C2 = draft model on the Adreno GPU (llama.cpp OpenCL) ∥ verify/target on
the Hexagon NPU (ORT-QNN), through the async draft∥verify overlap loop.
Phase 0 predicts C2 is the winning placement: verify (~52 ms on the NPU
AR128 graph) hides fully under the draft phase.

**Stage reached: Stage 2 (partial).** The GPU draft side was assembled
and proven working end-to-end *independently*. The NPU verify side's
bundle + venv were confirmed present and healthy. **Stage 3 (wire & run
C2) was NOT reached** — it is blocked by a structural harness gap: there
is no NPU verify-TARGET HTTP endpoint the async loop can call. This is
exactly the "◆ needs new orchestration code" risk the side-quest plan
flagged for every verify-on-NPU config (C2/C4/W1). No C2 end-to-end
numbers exist; this is an honest Stage 1/2 feasibility report with the
exact wiring C2 needs.

---

## Stage 1 — feasibility map (the wiring C2 needs)

### How the async loop is shaped today (`scripts/npu_spec_outer_loop_async.py`)

The loop hard-wires **draft = NPU, verify = CPU/GPU llama.cpp** — i.e.
config **C0/C1**, the mirror of C2. Per round:

```
f_drafts = executor.submit(draft_k_tokens_pmin, npu_sess, ...)   # DRAFT on NPU
target_ids = verify_via_target(committed_ids, k)                 # VERIFY via HTTP
drafts = f_drafts.result()                                       # join
```

- **DRAFT** is an in-process ORT-QNN call (`draft_k_tokens_pmin` →
  `npu_single_step_short_prompt`) against the NPU Path-B 0.6B binary.
  It runs `k` sequential AR1 steps on the *NPU*.
- **VERIFY** is `verify_via_target(committed_ids, k)` in
  `scripts/npu_spec_outer_loop.py`. It POSTs raw token IDs to
  `{BASE_URL}/completion` with `n_predict=k+1`, `temperature=0`,
  `top_k=1`, `return_tokens=True`, and expects **k+1 greedy TARGET
  tokens** back as a token-id list. `BASE_URL` →
  `http://127.0.0.1:8088` (a **llama.cpp** `llama-server`, spawned by
  `spawn_server()` in `npu_spec_step7_plumbing.py`, CPU build, 8B
  target).

C2 is the *inverse*: draft must move to the GPU, verify must move to the
NPU. Two independent rewires are needed.

### (a) Can the NPU serve as the verify TARGET over HTTP via `http_server.py`? — **NO, not as-is.**

`npu_engine/http_server.py` exposes **only** `GET /v1/models`,
`GET /health`, `POST /debug/reset_stream`, and `POST
/v1/chat/completions` (OpenAI ChatML). It does **not** expose a
llama.cpp-style `POST /completion`.

Even if the loop were pointed at `/v1/chat/completions`, the semantics
are wrong for verify:

| verify needs | `/v1/chat/completions` gives |
|---|---|
| raw token-id `prompt` (committed ids, no chat template) | renders ChatML around `messages`, re-tokenizes text |
| `n_predict = k+1` greedy continuation as **token ids** | `max_tokens` of decoded **text**; ids not returned |
| stateless per-round forward of a *given* committed prefix | stateful single-tenant stream with LCP-diffing; mutates `conv_state.history` every call |

Underneath, `http_server.py` drives `npu_engine/sidecar.py`. The sidecar
*has* the right primitive — `stream_open` (AR128 prefill) + `stream_decode`
(AR1 greedy decode) on the 4B w4a16 bundle, and even a `draft` op
(`serve_draft_request`) that returns N drafted token ids. But there is
**no `verify` op**: nothing that takes `(committed_ids, k)`, runs the
k-token target forward, and returns the target's k+1 greedy ids in the
`/completion` token-id contract. The sidecar is also single-tenant with
an asyncio lock; its stream model assumes monotonic append, not the
per-round "verify an arbitrary committed prefix" pattern.

**Verdict:** the NPU *engine* can compute a verify (AR128 graph,
flat ~52 ms — measured in Phase 0), but no HTTP/JSON surface exposes it
in the shape `verify_via_target` consumes. New code required.

### (b) Can a llama.cpp GPU server act as the DRAFT source feeding the loop? — **YES (the server), but the loop has no GPU-draft path.**

The GPU draft *server* works and was proven this session (Stage 2
below). `build-opencl/bin/llama-server.exe` loads a small Qwen3 GGUF on
the Adreno with `-ngl 99` and serves `/completion` returning token ids —
exactly the shape needed to *draft* k tokens.

But the async loop's draft is **not an HTTP call** — it is the in-process
`draft_k_tokens_pmin(npu_sess, ...)` ORT-QNN path. To make draft come
from a GPU server, the loop needs a new `draft_via_gpu_server()`
adapter (mirror of `verify_via_target`) and `run_spec_decode_async`
must call it in place of the `executor.submit(draft_k_tokens_pmin, ...)`
branch. A subtlety: speculative decoding needs **draft logits/argmax per
step with rollback snapshots** (`past_snapshots`) so any accepted prefix
can be committed. A plain `/completion n_predict=k` gives k greedy
tokens but no per-step state; that is fine for the LCP accept rule (the
loop only needs the k draft *ids* + the ability to resume from the
accepted prefix, and a fresh `/completion cache_prompt=true` re-drives
cheaply), but it changes the absorb/snapshot bookkeeping the current
loop does on `npu_past`. With draft on the GPU there is **no `npu_past`
at all** — the whole NPU-snapshot machinery (`materialize_snapshot_k`,
`absorb_bonus`, `pad_cpu_past_to_npu`) becomes dead code for C2.

### Exact wiring C2 needs (the deliverable of Stage 1)

A C2 driver is a **new script** — `npu_spec_outer_loop_async.py` cannot
be flag-switched into C2; both of its phases are the wrong island. Two
new pieces plus a thin new loop:

1. **NPU verify-target HTTP shim (the gating item).** Add a
   `POST /completion`-compatible endpoint backed by the NPU sidecar that:
   - accepts `{prompt: [token ids], n_predict: k+1, return_tokens: true}`
   - runs the 4B w4a16 bundle's AR128 graph over the committed prefix
     (one ~52 ms padded pass), then k+1 AR1 steps — or reuses
     `stream_open`+`stream_decode` — to emit the target's k+1 greedy ids
   - is **stateless per request** (verify must re-evaluate an arbitrary
     committed prefix; no LCP history mutation), or carefully resets
     `conv_state` each call
   Cheapest form: a ~40-line FastAPI route in `http_server.py` (or a
   standalone `npu_engine/verify_server.py`) that calls
   `sidecar.request("stream_open", ...)` then `"stream_decode", max_new=k+1`
   and returns `{"tokens": [...]}`. ORT-QNN, `.venv`, 4B bundle — all
   confirmed working this session.
2. **GPU draft adapter.** A `draft_via_gpu_server(committed_ids, k)` that
   POSTs to the OpenCL `llama-server`'s `/completion`
   (`n_predict=k`, greedy, `cache_prompt=true`, `return_tokens=true`)
   and returns the k draft ids. ~15 lines.
3. **New C2 async loop.** `scripts/spec_c2_async.py`: per round, fire
   `draft_via_gpu_server` on one thread ∥ the NPU verify shim on the
   main thread, LCP-accept, commit, repeat. No `npu_past`, no CPU
   prefill ONNX, no `absorb_bonus` — drop the entire NPU-snapshot path.
   The `draft_wait_s` / `verify_wait_s` / `parallel_wall_s` accounting
   from `run_spec_decode_async` carries over verbatim (both phases are
   now HTTP, both release the GIL on `socket.recv`).
4. **Two servers up at once.** GPU `llama-server` on one port + NPU
   verify shim on another. Both proven to start independently (below);
   no evidence of a conflict (different processes, different islands),
   but unverified together.

Estimated effort: shim ~1–2 h (the riskiest part — verify semantics on
the stateful sidecar), adapter + loop ~1 h. This is a half-day of new
orchestration code, not a launch-flag change.

---

## Stage 2 — assembling the pieces (what was proven this session)

### DRAFT (GPU) — assembled and proven working independently

- **Model:** `models/Qwen3-0.6B-Q8_0.gguf` (610 MB) — already on disk;
  **no download needed**. `Qwen3-1.7B-Q8_0.gguf` (1.75 GB) is also
  present as the alternate draft. (The task brief said only
  `Qwen3-4B-Q4_0.gguf` was on disk; that is incorrect — both small
  Qwen3 drafts were already there from the Phase 0 `core` model tier.)
- **Server:** `llama.cpp/build-opencl/bin/llama-server.exe -ngl 99`,
  port 8089, ctx 2048. Came `/health` healthy in **~3 s**.
- **Backend confirmed GPU:** log shows
  `Qualcomm(R) Adreno(TM) X2-90 GPU`, OpenCL 3.0 driver 863.0 — not a
  CPU fallback.
- **Measured (`/completion`, greedy, n_predict=32):**
  **TG 92.37 t/s, prefill 52.98 t/s** for the 0.6B draft on the GPU.
  This is the draft-phase rate C2 would overlap against. (Note: this is
  faster than the 4B-Q4_0 GPU TG of ~25 t/s in `phase0_gpu.md` — a
  0.6B draft on the Adreno is genuinely quick.)

### VERIFY (NPU) — bundle + runtime confirmed present, engine not driven this session

- **Bundle:** `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-...`
  present (4-part w4a16 + AR1/AR128 wrapper ONNXs across ctx tiers).
- **venv:** `specula\.venv` — ORT **1.24.4**, `QNNExecutionProvider`
  available. The known-good Phase 0 ORT-QNN setup.
- **Verify cost (from Phase 0, `phase0_npu.csv`):** the AR128 graph runs
  any k∈[2,128] for a flat **~52 ms**; k=1 AR1 is ~34 ms. Not re-measured
  this session — Phase 0's number stands.
- Not exercised end-to-end here because the missing shim (Stage 1a)
  means there is nothing for the loop to call; running the 4B bench in
  isolation would only re-confirm Phase 0.

### Confirm both run independently — partial

GPU draft server: **confirmed running and serving** (independent run,
above). NPU verify engine: bundle + venv confirmed; engine not launched
this session (no new information vs Phase 0, and the timebox favored the
feasibility map). The two were **not** run concurrently.

---

## Stage 3 — wire & run C2

**Not reached.** Blocked on the Stage 1a shim. Consequently:
end-to-end tok/s, mean accept length, TTFT, per-phase wall time, and the
`draft_wait_s`/`verify_wait_s` overlap accounting for k∈{2,4,8} are all
**UNMEASURED**. No numbers were fabricated; `results/phase1_c2.csv`
contains only the Stage-2 component measurements and an explicit
not-run marker.

---

## What C2 *would* look like (Phase 0 projection — unverified)

A C2 spec round of depth k, with async overlap, costs
≈ max(draft_phase, verify_phase) + absorb:

- **draft phase (GPU, 0.6B):** k sequential decode steps. At the
  measured 92 t/s TG that is ~10.8 ms/token → k=2 ≈ 22 ms, k=4 ≈ 43 ms,
  k=8 ≈ 87 ms. (Plus per-`/completion` HTTP + prefill-of-committed
  overhead each round — not measured; could dominate at small k.)
- **verify phase (NPU, 4B AR128):** flat ~52 ms regardless of k∈[2,128].
- **overlap:** verify (~52 ms) hides fully under the draft phase **only
  once the draft phase exceeds ~52 ms**, i.e. around **k ≈ 5+** at
  92 t/s. At k=2 the draft phase (~22 ms) is *shorter* than verify, so
  verify becomes the long pole — the opposite of the Phase-0 headline.
  **C2's "verify hides under draft" claim holds at larger k, not at
  k=2.** This is the single most useful thing Stage 3 would actually
  test, and it nuances the Phase 0 prediction.
- **bandwidth:** draft GPU(~159) ∥ verify NPU(~66) ≈ 225 GB/s — right
  at the 228 ceiling, same edge as C1; watch for contention clawback.

These are projections from component numbers, not an end-to-end C2 run.

## Comparison to C0 baseline

No C0 end-to-end CSV exists in this side-quest workspace. The C0 anchor
from `current_status.md` (draft NPU / verify CPU, via this same async
loop): Phase 5 sync k=2 = **7.98 t/s**; Lever A async-pipelined k=2 ≈
**18.12 t/s** (AC, 0.6B draft / 8B target). A like-for-like C2 vs C0
comparison must wait for Stage 3.

## Blockers

1. **No NPU verify-target HTTP endpoint** (gating). `http_server.py`
   speaks only OpenAI ChatML; the sidecar has no `verify` op. This is
   the one thing standing between here and a C2 run.
2. **Async loop is C0/C1-shaped.** `npu_spec_outer_loop_async.py` cannot
   be flag-switched to C2 — draft is an in-process ORT-QNN call, verify
   is a llama.cpp `/completion` call; C2 inverts both. A new driver is
   required, and the NPU-snapshot machinery (`absorb_bonus`,
   `materialize_snapshot_k`, `pad_cpu_past_to_npu`) is dead code for C2.
3. **NPU shape-lock (carried from Phase 0).** Verify of k∈[2,127] pads
   to the AR128 graph and pays the full ~52 ms — wasteful at small k but
   not a blocker. A native AR8 graph would need a new QNN bundle compile
   (out of scope).
4. **Two-server concurrency unverified.** GPU `llama-server` + NPU
   verify shim running together was not tested.

## Exact next steps

1. **Write the NPU verify shim** — add `POST /completion` to
   `npu_engine/http_server.py` (or new `npu_engine/verify_server.py`):
   accept token-id `prompt` + `n_predict=k+1`, drive sidecar
   `stream_open`+`stream_decode`, return `{"tokens": [...]}`. Make it
   stateless per request. ~1–2 h; this is the critical path.
2. **Write `draft_via_gpu_server()`** — POST to the OpenCL
   `llama-server` `/completion`, `n_predict=k`, greedy, return k draft
   ids. ~15 lines.
3. **Write `scripts/spec_c2_async.py`** — fork
   `run_spec_decode_async`, replace the NPU-draft branch with
   `draft_via_gpu_server`, point `verify_via_target` at the NPU shim,
   delete the `npu_past`/snapshot path. Keep the
   `draft_wait_s`/`verify_wait_s`/`parallel_wall_s` accounting.
4. **Bring up both servers** (GPU draft :8089, NPU verify :8090),
   confirm they coexist, then run k∈{2,4,8}, n_predict≈128, 3+
   humaneval prompts, AC. Fill `results/phase1_c2.csv` with end-to-end
   tok/s, mean accept length, TTFT, per-phase wall, overlap ratio.
5. **Specifically test the k where draft_phase crosses ~52 ms** — that
   is the real C2 question (verify hides under draft only for k≳5 at
   92 t/s draft), and it refines the Phase 0 "C2 wins" headline.
6. Also run **C0** through the same loop for the headline A/B if a
   clean C0 number is wanted in this workspace.

## Artifacts

- `gpu_npu_sidequest/results/phase1_c2.csv` — Stage-2 component
  measurements + explicit not-run marker for the C2 end-to-end row.
- `gpu_npu_sidequest/logs/c2_gpu_draft_smoke.log[.err]` — GPU draft
  server startup + `/completion` run log (Adreno backend confirmed).
