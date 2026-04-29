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
- **2026-04-28** — Phase B lands. Stateful streams in sidecar +
  HTTP server context retention. Same refactor unlocks SQ1 NPU
  rewind op (next). Probe (`probe_stream_api.py`) confirms stateful
  matches stateless byte-identical on the same prompt; truncate +
  append + decode works coherently. HTTP server stateful wins
  measured: 2.6× faster on prefix re-send, 2.3× faster on a
  realistic 2-turn coding chat. Crossover analysis settled: stay
  in AR1-append for deltas < 1024 tokens; AR128-reopen pays a 22 s
  mode-swap that only amortizes above ~540 token deltas.
  Limitation: `enable_thinking=False` causes a chat-template/KV
  divergence that caps multi-turn LCP at ~lcp_max - 5 per prior
  assistant turn.
- **2026-04-28** — Phase A.5 lands. SSE streaming via OpenAI's
  standard `text/event-stream` protocol. New sidecar op
  `chat_stream` emits one `{event:"token", token_id}` line per
  decoded token; HTTP server's `_do_chat_streaming` async
  generator decodes the rolling token buffer and yields incremental
  text deltas (BPE-correct: re-decodes whole buffer each step so
  multi-byte UTF-8 chars split across tokens render correctly).
  Final chunk has `finish_reason` set; terminator is `data: [DONE]`.
  Smoke-tested with `curl -N` against counting / coding prompts —
  deltas flow live, `finish_reason` populates correctly on `length`
  cap, `[DONE]` always closes the stream.

## Phase B landed (2026-04-28)

Stateful streams in sidecar + HTTP server context retention. The
same refactor unlocks the SQ1 NPU rewind op for spec-decode (next
deliverable).

### Sidecar additions (Phase B)

`EngineState.streams: dict[str, Stream]` holds persistent KV state
across requests. Six new ops:

| op | purpose |
|---|---|
| `stream_open` | initial prefill, store `Stream` by `stream_id` |
| `stream_truncate` | drop KV slots ≥ new_position; reset next_token |
| `stream_append` | feed specific token IDs at current position (AR1 only) |
| `stream_decode` | N greedy decode steps with EOS/stop early-stop |
| `stream_decode_stream` | streaming variant — emits per-token events |
| `stream_close` | drop the stream |

Probe (`probe_stream_api.py`) verifies:
- ✓ Stateful `stream_open + stream_decode` produces **byte-identical**
  output to stateless `chat` op (same prompt, greedy)
- ✓ `truncate + append + decode` rolls back, ingests injected tokens,
  continues coherently
- ✓ Closed stream returns proper error (no crash)

### HTTP server stateful integration

`ConversationState` tracks `(stream_id, history)`. Each chat
completion request:
1. Renders ChatML, tokenizes
2. `_sync_stream_to_prompt`: longest-common-prefix vs `conv_state.history`
   - First call → `stream_open` (AR128 prefill if ≥ 128 tokens)
   - LCP == history len → just `stream_append(delta)` if any
   - LCP < history len → `stream_truncate(lcp)` + `stream_append(delta)`
   - Strict-prefix re-send → `stream_truncate(lcp-1)` + re-feed last token
3. `stream_decode_stream` (streaming) or `stream_decode` (non-streaming)
4. Mirror `generated_ids` (incl. EOS) into `conv_state.history` for
   next turn's LCP

`/debug/reset_stream` for testing. `/health` reports `stream_opened`
and `history_len`.

### Phase B measured wins (2026-04-28, NPU CL=2048)

**Strict-prefix re-send ("send same prompt twice"):**
- Turn 1: open (28 prompt) + decode 8 → 1.16 s wall
- Turn 2 same prompt: truncate(12) + refeed 1 + decode → **0.45 s wall** (**2.6× faster**)
  - prefill_s collapses 0.61 → 0.034 s
  - **identical content** generated → rewind preserves model state correctly

**Realistic 2-turn coding chat (157-token first response):**
- Turn 1: open (system+user, 28 tok) + decode 142 → 7.7 s
- Turn 2: continuation (same system, same user1, actual asst1, new user2):
  - lcp=32 (limited by `<think></think>` injection — see limitations)
  - truncate+append 145 new tokens (5.87 s in AR1) + decode 145 → **11.7 s**
  - Phase A equivalent: AR1→AR128 swap (~14 s) + AR128 prefill + AR128→AR1 swap (~7 s) + decode = ~27 s
  - **2.3× faster** even with suboptimal LCP

**Heuristic note**: `_stream_reopen` (close + AR128-prefill-restart)
threshold raised from delta=128 to delta=1024. Crossover analysis:
AR128 reopen pays ~22 s mode-swap + delta/1500 t/s; AR1 append pays
delta/24 t/s. AR128 only wins above ~540 tokens of delta. For
typical chat continuations (tens-to-low-hundreds of tokens),
AR1-append always beats AR128-reopen.

### Phase B known limitations

- **Chat-template/KV mismatch with `enable_thinking=False`.** The
  `<think>\n\n</think>\n\n` block is injected at the assistant cue
  for the current turn (matches upstream Qwen3 behavior). On the
  next turn, the prior assistant content is rendered WITHOUT a
  thinking-block (also matches upstream). This means the stream's
  KV layout and the next-turn rendered prompt diverge at the
  thinking-block position, capping LCP at ~lcp_max minus ~5 tokens
  per prior assistant turn. Workaround for max LCP: don't use
  `enable_thinking=false` in multi-turn agents — instead instruct
  via system prompt and post-strip `<think>...</think>` on the
  client.
- **AR1-only ingest path.** `stream_append` always uses AR1 (one
  decode step per appended token). For 100-token deltas at 24 t/s
  this is ~4 s. Future optimization: switch to AR128 batched
  ingest when delta >= 128 AND we're already in AR128 mode (i.e.,
  during a freshly-opened stream's prefill phase). For now AR1 is
  the simple, swap-free path.
- **Single-tenant.** asyncio lock + 503-busy on overlap. NPU
  concurrency past ~3 contexts is unstable
  (`docs/qwen2_5_7b_baseline_all_backends.md`).
