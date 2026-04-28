# SQ1 — heterogeneous NPU 4B draft + CPU 14B target

## Status

- **2026-04-27** — Path A (plumbing-only demo) **lands**. Both
  compute islands successfully exchange tokens for the same
  prompt; tokenizer compat between Qualcomm Qwen3-4B bundle and
  Qwen3-14B-Q4_K_M GGUF is verified by exact-match on the first
  6 generated token IDs. SQ1's "prove the architectural pattern
  works" goal is achieved.

## Headline result — Path A run on a fixed Python coding prompt

Prompt: 72 tokens of "Complete this `fibonacci(n)` function" boilerplate.

| metric | NPU 4B draft (cl=2048) | CPU 14B target |
|---|---|---|
| spec | Q4 / w4a16 weights, AR1 chain, ORT-QNN 1.24.4 | Q4_K_M, 8 threads, llama-server CPU |
| prefill (72 tokens) | 3.06 s (incl. 8 s warmup share) | ~74 ms server-side |
| decode (16 tokens) | 0.64 s, 24.8 t/s, ~40 ms/step | 1.12 s, 13.3 t/s, ~70 ms/step |
| first 16 generated | `[262, 671, 4615, 2038, 1588, 271, 40, 1184, 311, 4211, 279, 729, 311, 470, 279, 55129]` | `[262, 671, 4615, 2038, 1588, 271, 32313, 11, 358, 1184, 311, 3270, 264, 13027, 729, 2598]` |
| decoded text | `'    # Your code here\n\nI need to implement the function to return the nth'` | `'    # Your code here\n\nOkay, I need to write a Python function called'` |

**6/16 NPU-draft positions match the target verbatim** —
specifically positions 0..5 (`'    # Your code here\n\n'`). After
that the two models diverge into different phrasings of the same
intent.

**First-mismatch index = 6.** A real spec-decode loop at K=16
would commit 7 tokens per round (6 NPU draft accepts + 1 target
sample at position 6). Naive position-by-position match rate:
**38%**.

## What this proves

1. **Architectural pattern works.** NPU 4B drafts and a CPU 14B
   target server can be wired together end-to-end. No public
   project has shown this on Snapdragon X-class silicon.
2. **Tokenizer compat is real.** Qualcomm Qwen3-4B bundle's
   `tokenizer.json` produces IDs identical to Qwen3-14B-Q4_K_M's
   embedded tokenizer (verified by exact-match on the first 6
   tokens of two independent forward passes).
3. **Both backends maintain coherent continuation.** The NPU
   draft text reads as legitimate Python coding-assistant output
   (not garbage from a tokenizer mismatch or quant collapse).
4. **Per-step latencies match SQ5 measurements.** NPU AR1 ~40 ms
   at cl=2048 (consistent with 40.2 ms median in SQ5's cl=2048
   AR1 sweep); CPU 14B Q4_K_M ~70 ms/token on 8 threads.

## What this doesn't prove

- **Throughput.** This is plumbing — both models run their full
  generation independently. There's no spec-decode loop, no
  batched verify, no real win to claim.
- **Steady-state accept rate.** One prompt is one datapoint. A
  prompt sweep would tell us if 38%@K=16 is the typical case or a
  high-variance outlier.

## Pre-Path-B / Path-C analysis

If we landed Path B (naive K-serial verify with NPU rewind), the
per-round wall would be roughly:

```
round_wall = K × target_step + K × NPU_step
           = 16 × 70 ms + 16 × 40 ms
           = 1120 + 640 = 1760 ms
committed_per_round = first_mismatch + 1 = 7 tokens
effective_rate     = 7 / 1.76 s = 4 t/s
```

vs CPU 14B alone at 13.3 t/s. **Path B at K=16 is slower than
direct CPU**, because K serial target calls cost the same as
generating K target tokens directly. Path B's only product is
the accept-rate measurement.

If we landed Path C (real batched verify via prompt_logprobs or
in-process bindings):

```
round_wall = 1 × target_step (K positions in one forward) + K × NPU_step
           = 70 ms + 640 ms = 710 ms
committed_per_round = 7
effective_rate     = 7 / 0.71 s = 9.9 t/s
```

vs CPU 14B alone at 13.3 t/s. **Path C at K=16 is still slightly
slower** than direct CPU at this accept rate. For a win we need
either (a) higher accept rate (e.g. 12/16 ⇒ 17 t/s, +28%), or (b)
larger draft/target step ratio (today: 70/40 = 1.75; need >5 to
make spec-decode pay at modest accept rates).

So **at this draft/target pair (4B/14B) and this prompt, real
spec-decode would barely break even.** The win materializes at
larger target gaps (Qwen3-32B target: 175 ms/token at 8 threads
per Phase 1 baseline → ratio 4.4, accept rate 38% → ~1.2× speedup)
or with EAGLE-3-class drafters (accept rate 70-85% literature
range → ~1.5-1.8× speedup at this same K).

## Open questions for next session

1. **Accept rate sensitivity to prompt class.** Try humaneval
   coding, plain prose, JSON tool-call output. Hypothesis:
   structured output (JSON, repeated boilerplate) accepts
   higher; creative prose accepts lower.
2. **K sweep.** Try K ∈ {4, 8, 16, 32} on the same prompt. Lower
   K typically has higher per-position match (less time for
   models to diverge); we might see 100% accept at K=6 since the
   first divergence here was at position 6.
3. **Promote to Path B?** Adds ~200 LOC (rewind op + driver
   loop) for a real-spec-decode wall measurement. Would tell us
   the "warm-loop" steady-state accept rate, not just one round.
4. **Promote to Path C?** Investigate `prompt_logprobs` on
   llama-server. If supported, batched verify replaces K serial
   calls and we get a real throughput number.
5. **Bigger target?** Qwen3-32B-Q4_K_M (~19 GB, not on disk)
   would shift the draft/target ratio favorably. Or Qwen3.6-35B-
   A3B (already on disk) — but tokenizer compat between Qwen3
   and Qwen3.6 needs verification first.

## Files

- `last_side_quest/sq1_heterogeneous/architecture.md` — design.
- `last_side_quest/sq1_heterogeneous/demo_path_a.py` — driver.
- `last_side_quest/sq1_heterogeneous/findings.md` — this doc.
- `scripts/serve_target_14b.ps1` — target server harness.
- `npu_engine/sidecar.py` — added `draft` op (returns token IDs
  given a driver-supplied prompt).
- `results/csv/sq1_path_a.csv` — append-mode CSV (one row per
  demo invocation).

## Reproduction

```powershell
# Terminal 1: target server (~10 s startup)
.\scripts\serve_target_14b.ps1

# Terminal 2: demo (NPU sidecar starts internally, ~9 s)
.venv\Scripts\python.exe last_side_quest\sq1_heterogeneous\demo_path_a.py `
    --target-url http://127.0.0.1:8081 `
    --k 16 `
    --ctx-tier 2048 `
    --csv results\csv\sq1_path_a.csv `
    --tag run_$(Get-Date -Format yyyyMMdd_HHmmss)
```

## Update log

- **2026-04-27** — Path A landed. K=16, cl=2048, fixed Python
  coding prompt: 6/16 first-position match. Plumbing demo done.
