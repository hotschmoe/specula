# SQ6 — Small-model server harness (NPU-served + comparators)

Workspace for SQ6 (per `last_side_quest/last_side_quests.md`).

**Reframed scope (2026-04-28).** The original SQ6 was just "stand up
a small-model llama-server and feel-test opencode against it." After
SQ1.b/c closed and the 7B-NPU bundle proved viable
(`docs/qwen2_5_7b_baseline_all_backends.md`), the user redirected:
the headline is **expose our QNN runtime as an OpenAI-compatible
HTTP server so any agent CLI (pi, hermes, opencode) can drive the
NPU**, then A/B that against llama-server-CPU/Vulkan and the existing
Qwen3.6-35B-A3B daily-driver. This is what was previously scoped
inside `last_side_quests.md` as "SQ6 stage 2 — FastAPI wrapper" — now
the primary deliverable.

## Goal

A working OpenAI-compatible (`/v1/chat/completions`) HTTP endpoint
backed by `npu_engine/sidecar.py` so:

1. **Direct comparison.** A coding-agent CLI (pi.dev / hermes /
   opencode) can hit:
   - Qwen3-4B served on the **NPU** (via this new server)
   - Qwen3-4B served on Vulkan/CPU (via llama-server)
   - Qwen2.5-7B served on Vulkan/CPU (via llama-server)
   - Qwen3.6-35B-A3B daily-driver (existing `serve_daily_driver.ps1`)
   …with **identical client config**, only swapping the base URL.
2. **NPU-as-context-grows.** Real coding-agent prompts grow over
   multi-turn dialogue. Microbenchmarks at fixed PP/TG can't reveal
   how the NPU behaves under organic context fill. The HTTP
   endpoint exposes that.
3. **Pipeline reuse.** The same server, with stateful streams, is
   the natural shape for the SQ1 spec-decode HTTP endpoint (NPU
   draft + CPU target served as one model). One refactor unlocks
   both.

## Why HTTP-wrap our QNN runtime, not Genie

| candidate | runtime model | usable as server? |
|---|---|---|
| `genie-t2t-run.exe` | Single-shot CLI, ~15 s context-init per invocation | No — would re-pay context init per request |
| Genie SDK (C++ APIs) | Long-lived process possible | Yes but ~1 session of integration work to wire up; we'd be reimplementing what our sidecar already provides |
| **`npu_engine/sidecar.py`** | Long-lived ORT-QNN runtime, +39% PP / +19% TG over Genie at 4B per its docstring | **Pick this.** Already long-lived, JSON-over-stdio API, our code (modifiable) |

## Decisions (2026-04-28)

These are committed but reversible. The plan is MVP-then-optimize:
prove the chain end-to-end with the existing stateless sidecar API
before refactoring for stateful streams.

### 1. Chat template

Lift Qwen3 ChatML directly from upstream HF
`Qwen/Qwen3-4B/tokenizer_config.json`:

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user}<|im_end|>\n
<|im_start|>assistant\n
```

EOS markers: `<|im_end|>` (id 151645) and `<|endoftext|>` (id 151643).
Verify by encoding a known message + decoding round-trip via the
bundle's `tokenizer.json` before first generation.

Qwen2.5 uses the same ChatML pattern with the same special-token
vocab, so this template will carry to the 7B-NPU when its sidecar
adapter lands.

### 2. Stop sequences

Honor:
- Built-in EOS ids `{151643, 151645}`
- Client-supplied `stop: [...]` strings — tokenize each on the
  bundle's tokenizer and scan generated token IDs for the byte
  sequence (not subsequence-match on decoded text — token-id match
  is faster and avoids re-tokenization round-trips).

### 3. Conversation continuity

OpenAI's API has no `conversation_id`; clients re-send full
message history. Server-side strategy:

- One persistent stream per `(client_session_addr, model)` tuple,
  or just one stream for the whole server in single-tenant mode.
- On each `/v1/chat/completions` request: render the message
  history → token IDs. Compare against the active stream's
  `committed[0..L)`. If new request's first L tokens match, ingest
  delta `[L..]` and start drafting. Else: truncate to longest
  common prefix and ingest from there.
- This is robust to "user sends same conversation with one new
  turn appended" (the common case), as well as "user edits earlier
  message" (truncate to divergence point).

This is **deferred to Phase B** (after stateful streams refactor).
Phase A re-prefills every turn; correct but slow.

### 4. Concurrency

Serialize. NPU concurrency-4 is unstable per the 7B baseline doc
(QNN error 1003, HTP context ceiling) — single-tenant server,
asyncio lock around the inference call, return 503 if a second
request arrives while one is in flight. Acceptable for SQ6's
"feel it out" purpose; multi-tenant serving is a separate problem
that would need the W4 sidecar work in any case.

## Plan

### Phase A — MVP HTTP wrapper (stateless)

Goal: prove the chain end-to-end. Slow (full re-prefill per turn)
but catches tokenization / chat-template / stop-sequence bugs with
the simplest possible infrastructure.

Files:
- `npu_engine/http_server.py` — FastAPI app, single endpoint
  `/v1/chat/completions`, talks to a long-lived sidecar subprocess
  via the existing JSON-over-stdio `serve_request` op (the
  stateless one).
- `npu_engine/chat_template.py` (or inline) — Qwen3 ChatML render +
  stop-sequence handling.

Validation:
- `curl` test with a single-turn prompt → expect a Qwen3-shaped
  reply, EOS at `<|im_end|>`, latency ~5–10 s (4B model, ~50–80
  output tokens, ~22 t/s NPU TG).

### Phase A.5 — SSE streaming

OpenAI clients prefer streaming responses. Add SSE chunk emission
so each token (or small batch) flows incrementally. Test with
`curl --no-buffer -N`.

### Phase B — Stateful streams

Refactor `sidecar.py` to expose:
- `stream_open` — initial prefill, store Stream by `stream_id`
- `stream_append` — ingest N new tokens (decode_only or AR128
  prefill if N ≥ 128)
- `stream_truncate` — drop KV state at positions >= P; reset
  `stream.position` / `next_token` / `last_logits`
- `stream_decode` — N decode steps, return the N decoded tokens
- `stream_close`

Wire `http_server.py` to use these; one stream per server (single-
tenant). Adds the longest-prefix-match continuity logic.

### Phase B validation — Path C demo retrofit

Port `demo_path_c.py` to the new stream API. **Validation
target**: JSON K=8 / 4 rounds, expect 91% accept reproduces (same
target argmax sequence as today since target side hasn't changed),
per-round NPU wall drops from ~4 s to ~0.4 s. Append result to
SQ1's `findings.md` as the `NPU rewind op landed` entry.

### Phase C (follow-up session) — Agent A/B matrix

Out of scope for this session, but the deliverable shape:

- `scripts/serve_small.ps1` — parameterized `serve_daily_driver.ps1`
  fork for Qwen3-4B-Q4_K_M and Qwen2.5-7B-Q4_0 on Vulkan.
- WSL Ubuntu: bun + pi (or hermes / opencode), point at Windows
  host endpoints via `host.docker.internal:<port>`.
- Workload: real HTML coding task (build a static page with
  responsive CSS, multi-turn refinement). Capture per-turn TTFT,
  TG t/s, ctx fill, subjective verdict.
- Comparison cells:
  | model | backend | served by |
  |---|---|---|
  | Qwen3-4B | NPU | this new server |
  | Qwen3-4B | Vulkan-Q4_0 | llama-server |
  | Qwen2.5-7B | Vulkan-Q4_0 | llama-server |
  | Qwen3.6-35B-A3B | Vulkan-MXFP4 | `serve_daily_driver.ps1` |

## Phase A landed (2026-04-28)

`npu_engine/http_server.py` is a working OpenAI-compatible HTTP
server backed by the existing stateless `sidecar.py serve`. Spawns
the sidecar at FastAPI lifespan startup (~9 s NPU cold load),
holds a single asyncio inference lock, exposes `/v1/chat/completions`,
`/v1/models`, `/health`. Run via:

```
.venv/Scripts/python.exe -m uvicorn npu_engine.http_server:app \
    --host 127.0.0.1 --port 8081 --no-access-log
```

### Sidecar additions

Added one new op `chat` to `sidecar.py` (`serve_chat_request`):
- Pre-tokenized prompt in (HTTP server tokenizes via bundle's
  `tokenizer.json`, sidecar-side stays IDs-only)
- Greedy decode loop with EOS-id check + stop-token-sequence
  suffix-match per step → correct early stop
- Returns `generated_ids` + `stop_reason` + per-phase timings

The existing `serve_request`, `serve_draft_request`, etc. ops are
untouched. The sidecar's stateful refactor is Phase B work.

### Smoke results (2026-04-28, NPU CL=2048, single-tenant)

| test | prompt | output | finish | wall | TG t/s | notes |
|---|---|---|---|---:|---:|---|
| simple greeting (think on) | "Say hello..." 15 tok | thinking-block, hit cap | length | 2.07 s | 23.25 | thinking-mode noisy by default |
| `enable_thinking=false` | "Capital of France?" | `"The capital of France is Paris."` | stop (EOS) | 1.29 s | 26.05 | suppression works |
| stop word | "Spell ... one to ten" + `stop:["five"]` | `"one, two, three, four, five"` | stop | — | — | BPE-variant fix catches ` five` token |
| **HTML coding task** | concise system + "minimal HTML page red button" | full valid HTML, body+style+button, hover state | stop (EOS) | **7.26 s** | **24.31** | 125 tokens; clean termination |
| 6-message multi-turn | Q3.6 system + 5-message Q&A about colors/fruits | `"Strawberry."` | stop (EOS) | 3.12 s | 26.58 | 70-tok prompt re-prefilled in 2.93 s |

**Headline numbers, NPU-served Qwen3-4B at CL=2048, single-tenant:**
- TG steady 24-27 t/s — matches the 22.91 t/s Qwen2.5-7B-NPU and
  23.30 t/s Qwen3-4B-NPU baselines (`docs/qwen3_4b_baseline_*.md`)
- PP small-prompt ~24 t/s (AR1 prefill below 128-token threshold)
- 9 s sidecar cold load (one-time; daily-driver script style)

**Real HTML page generated end-to-end at ~17 wall-t/s.** 125 tokens
in 7.26 s including prefill. That's the headline subjective datapoint
for the agent A/B matrix (Phase C).

### Phase A inefficiency, quantified

The 6-turn test re-prefilled 70 tokens at PP 23.9 t/s (2.93 s of
NPU work) for a single-token-of-output answer. Same conversation
extrapolated to 1500 tokens of context (typical mid-session
coding agent state) → re-prefill alone would take ~62 s per turn.

This is the headline argument for Phase B. For ≤200-token contexts
(early agent turns) Phase A is acceptable; long-context coding
sessions need stateful streams.

### Phase A known limitations

- **Stop-string token-id matching is BPE-fragile.** ` five` (token
  `Ġfive` = 4236) and `five` standalone (52670) tokenize differently.
  Mitigated by tokenizing with-and-without-leading-space variants
  per stop string. Won't catch `\nfive`, `(five`, etc. Phase B
  should switch to incremental text-side suffix matching using the
  tokenizer's decode.
- **No streaming.** `stream=true` returns 501. Phase A.5 adds SSE.
- **No `tools` field.** Coding agents using OpenAI tool-calling
  won't work. Phase C decision: either (a) ignore tools entirely
  and let the agent fall back to text protocol, or (b) wire up a
  Hermes-style XML tool-call parser. Defer.
- **No multimodal content.** `messages[].content` accepts list-of-
  parts but flattens text-only; image parts are dropped. NPU
  bundle is text-only Qwen3-4B regardless.
- **No real concurrency.** asyncio lock serializes; second
  concurrent request gets 503-busy. Per
  `docs/qwen2_5_7b_baseline_all_backends.md`, NPU concurrency past
  ~3 contexts is unstable on this stack — single-tenant is the
  honest model.
- **4B-only.** `sidecar.py` imports `qualcomm_qwen3_4b_oracle`
  hardcoded. Adding the 7B-NPU bundle requires generalizing the
  sidecar (separate session — see `docs/qwen2_5_7b_baseline_all_backends.md`
  line 442 "follow-on workstream").

## Update log

- **2026-04-28** — SQ6 reframed; this doc scaffolded with the
  four open-question decisions and the Phase A/B/C plan.
- **2026-04-28** — Phase A lands. `npu_engine/http_server.py` +
  `serve_chat_request` op in sidecar. Smoke-tested EOS, stop
  sequences, multi-turn, real HTML coding task. NPU TG 24-27 t/s,
  matches the all-backends baseline. Phase A inefficiency
  quantified: 70-token prompt re-prefill = 2.93 s; 1500-token
  conversation extrapolates to ~62 s per turn — headline argument
  for Phase B.
