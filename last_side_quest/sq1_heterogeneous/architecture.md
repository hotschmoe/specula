# SQ1 — heterogeneous NPU 4B draft + CPU 14B target — architecture

## Goal recap

Demonstrate Qwen3-4B drafting on NPU + Qwen3-14B verifying on CPU
in the same agent loop. Even if naive throughput is BELOW
CPU-only-target — the win is **proving the architectural pattern
works on this silicon**, which no other public project has shown.
Throughput leadership is a stretch goal, gated on which verify
strategy is feasible.

## What we have (after SQ5)

- **NPU side**: `npu_engine/sidecar.py` runs as a long-lived process
  with a stdin/stdout JSON IPC. Already supports
  `prefill_only(state, stream_id, prompt_ids, ...)` and
  `decode_only(state, stream, n_tokens)`. After SQ5 it can run at
  any of cl={512, 1024, 2048, 3072, 4096} with `--ctx-tier`.
- **Target side**: `scripts/serve_daily_driver.ps1` already serves
  Qwen3.6-35B-A3B on Vulkan/CPU. Easy fork for Qwen3-14B-Q4_K_M.
- **Tokenizers**: NPU 4B's `tokenizer.json` (Qualcomm bundle) and
  Qwen3-14B-Q4_K_M's embedded tokenizer are both Qwen3-family, but
  must be MD5-verified before the demo runs.
- **Existing prototype**: `scripts/npu_draft_sidecar.py` is an
  earlier ONNX-MatMul smoke test, not directly reusable; useful
  only as a reference for ORT-QNN session conventions.

## Three implementation paths (pick one)

### Path A — plumbing-only demo (cheapest)

Sketch:

1. NPU sidecar (`--mode serve --ctx-tier 2048`) prefills the full
   prompt and drafts K tokens.
2. Driver script POSTs the same prompt to llama-server's
   `/completion` with `n_predict=K`.
3. Driver compares draft sequence vs target sequence side-by-side,
   prints accept rate at position-1 (first-token match), 2, ..., K.
4. Done.

**What it proves.** Two compute islands successfully exchange
tokens. NPU draft is qualitatively coherent (or it isn't, in which
case we learn the tokenizer assumption was wrong). Accept rate at
position 1 is a real number we can quote.

**What it can't prove.** Any throughput claim. There is no spec-
decode happening — both backends run the prompt independently
and we just diff their outputs.

**Cost.** ~1 hour of code + 30 min of demo runs.

### Path B — naive K-serial spec-decode loop (real but slow)

Sketch:

1. NPU sidecar gains a `rewind` op (just `kv.t = accepted_pos`).
2. Driver loop:
   a. NPU drafts K tokens given current state.
   b. Driver issues K serial `/completion` calls, each with
      `prompt = prefix + draft[:i]`, `n_predict = 1`. Compares each
      target prediction to `draft[i]`. First mismatch sets `m`.
   c. Driver tells NPU "rewind to position m" + "next token is
      whatever target predicted".
   d. Repeat until done or context full.
3. Driver reports total wall, accept rate, draft tokens consumed.

**What it proves.** A real spec-decode loop runs end-to-end with NPU
draft and CPU target. Accept rate is a real measurement. Useful as
the scaffold that swaps to Path C when batched verify lands.

**What it can't prove.** Throughput win. K serial target calls cost
the same as just generating K target tokens directly — no batching,
no parallelism. The NPU draft's only contribution is "tells us
what to feed the target", which we could trivially do without it.

**Cost.** ~1 session. Adds the rewind op to the sidecar
(~20 LOC), the driver loop (~200 LOC), CSV logging.

**The interesting metric falling out of B.** Per-position accept
rate `(p_1, p_2, ..., p_K)` — the headline accept-rate number for
NPU 4B drafting Qwen3-14B target. Tells us what a real-spec-decode
implementation could buy. If `p_1 > 60%` the demo is pointing at
a real future win once Path C lands.

### Path C — batched verify via prompt_logprobs (real spec-decode)

Sketch:

1. Same as Path B but Step b becomes ONE target call with
   `prompt = prefix + draft[:K]`, `n_predict = 1`,
   `prompt_logprobs = 1` (or whatever llama-server's flag is named).
2. Target returns logprobs at every prompt position; we read off
   the predicted token at each draft position in one call.
3. Real spec-decode formula applies: K accepted ⇒ K tokens
   generated in 1 target step + K NPU steps.

**What it proves.** Real heterogeneous spec-decode with throughput
win. The headline result that nobody else has shown on this
silicon.

**Risk.** Stock llama-server may not expose `prompt_logprobs`. Need
to verify on this fork's commit:
- `--n-probs` flag returns top-N logprobs at *generated* steps.
- `prompt_logprobs` is a vLLM concept; uncertain whether
  llama-server has the equivalent.
- Worst case: fork llama-server to add it. That's days of work,
  not a session — same cost as B20 path 1 (in-process
  llama-cpp-python) which would also expose hidden states.

**Cost.** 0.5 session if the flag exists; 2-3 sessions if we have
to fork llama-server.

## Recommended approach

**Phase 1: Path A.** Lands in <1 hour. Validates the tokenizer
assumption + the IPC. Commits and writes a one-page note.

**Phase 2: Path B.** Builds on A's plumbing. Adds the rewind +
loop. Phase 1's writeup gets the headline accept-rate number.

**Phase 3 (stretch, only if user wants):** Investigate Path C's
prompt_logprobs availability. If exists, swap into B's loop. If
not, evaluate whether to escalate to B20 (custom verifier) — that's
a roadmap-level decision, not a same-day side-quest.

## Open design questions (for user pre-coding)

1. **Tokenizer compat check before any model loads.** Diff the NPU
   bundle's `tokenizer.json` vs Qwen3-14B-Q4_K_M's embedded
   tokenizer. Two ways to do this:
   - GGUF → `tokenizer.json` extraction via `gguf-py` Python lib
     (slow, depends on llama.cpp Python bindings).
   - Run a probe: send a fixed prompt through both, ID arrays
     should match exactly. Faster.
   Recommend the probe approach. Drop a `probe_tokenizer_match.py`
   in `last_side_quest/sq1_heterogeneous/`.
2. **What ctx tier for the demo?** SQ5 found cl=2048 is comfortable
   (25 t/s draft) and cl=4096 works at 20 t/s. Recommend cl=2048
   default for first runs; cl=4096 as a stretch test once the loop
   stabilizes.
3. **What K (draft length per round)?** Qualcomm ships AR1 only
   for the 4B (no AR2/AR4 batched-decode graphs in the bundle).
   So drafting K tokens means K serial NPU calls. K=4 is a
   reasonable starting point — accept rate degrades with K but
   round overhead amortizes.
4. **What workload?** A fixed coding-assistant prompt (~1 KB
   system + a function to complete) is the right demo. ~512-1024
   token prompt + 128 token generation. Reuse
   `prompts/humaneval_subset.jsonl` if convenient.
5. **What to compare against?** Two baselines:
   - llama-server CPU 14B alone (no draft) — pure target rate.
   - llama-server CPU 14B with `--model-draft Qwen3-0.6B` — stock
     llama.cpp internal-draft spec-decode.
   The NPU heterogeneous demo's number relative to these tells the
   real story.

## Files to land

- `last_side_quest/sq1_heterogeneous/architecture.md` (this doc).
- `last_side_quest/sq1_heterogeneous/probe_tokenizer_match.py` —
  tokenizer probe (Phase 0, pre-coding gate).
- `scripts/serve_target_14b.ps1` — fork of `serve_daily_driver.ps1`
  for Qwen3-14B-Q4_K_M.
- `last_side_quest/sq1_heterogeneous/demo_path_a.py` — Path A
  driver (plumbing demo).
- `last_side_quest/sq1_heterogeneous/demo_path_b.py` — Path B
  driver (naive spec-decode loop) [Phase 2].
- `npu_engine/sidecar.py` — add `rewind` op [Phase 2].
- `last_side_quest/sq1_heterogeneous/findings.md` — running
  results writeup.

## Update log

- **2026-04-27** — Doc created. Qwen3-14B-Q4_K_M download in flight
  (~9 GB via curl.exe -C -). Awaiting user pick on Phase 1 path.
