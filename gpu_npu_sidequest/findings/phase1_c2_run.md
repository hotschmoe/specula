# Phase 1 — config C2 end-to-end run (draft GPU ∥ verify NPU)

C2 = speculative decoding with the **draft model on the Adreno GPU**
(Qwen3-0.6B-Q8_0, llama.cpp OpenCL) and **verify/target on the Hexagon
NPU** (Qwen3-4B w4a16, ORT-QNN), async-overlapped through a new
standalone driver.

**Stage reached: Stage 3 — C2 ran end-to-end.** Unlike the prior agent's
report (`phase1_c2.md`, blocked at Stage 2), C2 now runs the full
draft→verify→accept loop. The prior "gating blocker" — needing an NPU
verify HTTP shim — was over-estimated: the NPU sidecar is driven
**in-process** for verify, using its existing `stream_open` /
`stream_decode` / `stream_truncate` / `stream_append` primitives. No new
HTTP surface, no sidecar modification.

**Verdict up front: C2 is correct but slow. Best C2 = 7.53 tok/s (k=2),
versus CPU-only spec decode 54.6 and CPU target-alone 50.7. C2 loses by
~7×.** The cause is not the placement idea — it is that the NPU verify,
run through the available stream primitives, executes as a chain of
single-token AR1 steps, **not** the flat ~52 ms AR128 batched verify
Phase 0 measured. The Phase 0 "C2 wins" headline assumed an AR128 verify
that the current sidecar stream API does not expose.

---

## Setup

- Hardware: Snapdragon X2 Elite Extreme, Windows 11 ARM64, **AC power**.
- Draft: `models/Qwen3-0.6B-Q8_0.gguf`, llama.cpp `build-opencl`
  `llama-server.exe -ngl 99 -c 2048`, port 8089. Backend confirmed
  `Qualcomm(R) Adreno(TM) X2-90 GPU`, OpenCL 3.0 driver 863.0 — not a CPU
  fallback (server log `c2_gpu_server.log`).
- Verify/target: `qwen3_4b-genie-w4a16` bundle via `npu_engine/sidecar.py
  --serve`, in-process subprocess, ORT-QNN 1.24.4, ctx tier cl512.
- Driver: `gpu_npu_sidequest/scripts/run_c2.py` (new, standalone).
- Fixed prompt (one technical paragraph, 33 tokens), `n_predict=128`,
  greedy (`temp=0`, `top_k=1`), k ∈ {2,4,8}, 1× each.

## Step 3 — verify sanity check (done BEFORE trusting throughput)

The NPU verify path was checked against the 4B's own greedy generation
*before* any throughput was reported:

- **Test 1 (correct drafts):** feed verify the 4B's own first-k greedy
  tokens as the "drafts" for `committed=prompt`. Verify must echo
  `ref[0..k]`. Result: `verify_out = [1096,5567,18404,264,501]` ==
  `4B_greedy[0..4]` → **PASS**.
- **Test 2 (wrong draft @ pos 1):** corrupt draft token 1. Verify must
  still produce `ref[0]` at pos 0 and the LCP-accept must stop at j=1.
  Result: `verify_out = [1096,5567,374,264,501]`, `lcp_j = 1` → **PASS**.

Both passed on every run. **The NPU verify path is correct** — it
returns the 4B target's true greedy token at each of the k+1 positions,
so the longest-common-prefix accept produces exactly the 4B's greedy
output. (Spot-confirmed by the loop logs: accepted runs reconstruct a
coherent continuation, and at k=8 several rounds hit j=8 — full
acceptance — exactly where the draft tracks the target.)

## Results — C2 end-to-end (AC, n_predict=128)

| k | decoded | rounds | tok/s | accept rate | accept/round | draft_wait_s | verify_wait_s | parallel_wall_s |
|--:|--------:|-------:|------:|------------:|-------------:|-------------:|--------------:|----------------:|
| 2 | 129     | 54     | **7.53** | 69.4%    | 1.39         | 3.93         | 13.19         | 17.12           |
| 4 | 130     | 40     | **6.60** | 56.2%    | 2.25         | 3.26         | 16.42         | 19.68           |
| 8 | 129     | 32     | **4.99** | 37.9%    | 3.03         | 3.50         | 22.33         | 25.83           |

(Raw: `logs/c2_run_k248.log`, `logs/c2_results.json`. CSV:
`results/phase1_c2.csv`.)

## Comparison vs baselines (`phase1_native_specdecode.md`, same box, AC)

| config | tok/s | C2 (best, k=2) vs it |
|--------|------:|---------------------:|
| **C2 best (k=2)**         | **7.53** | —          |
| CPU-only spec decode (k=4)| 54.59    | C2 **0.14×** (7× slower) |
| CPU target-4B alone       | 50.72    | C2 **0.15×** |
| GPU-only spec decode (C3) | 11.00    | C2 **0.68×** |
| GPU target-4B alone       | 27.03    | C2 **0.28×** |

**C2 does not beat CPU-only spec decode (54.6) or CPU target-alone
(50.7). It loses to all four baselines, including GPU-only spec decode.**

## Overlap accounting — the async overlap works, but verify is the wall

`parallel_wall_s ≈ draft_wait_s + verify_wait_s` to within 0.01 s at
every k (17.12 ≈ 3.93+13.19; 19.68 ≈ 3.26+16.42; 25.83 ≈ 3.50+22.33).

Read carefully, this is **not** "overlap failed" — it is "overlap
succeeded but is irrelevant". The driver pre-issues round N+1's GPU
draft the instant round N's commits are known, so it runs concurrently
with round N+1's NPU verify. Because the GPU draft phase
(`draft_wait` ≈ 3–4 s total across the whole run, i.e. ~60–110 ms/round)
is **much shorter** than the NPU verify phase (~250–700 ms/round), the
draft finishes long before `pending_draft.result()` is awaited:
`draft_wait_s` is the *residual* wait, near zero per round. The draft is
**fully hidden under verify**. But hiding a 70 ms draft under a 400 ms
verify saves nothing meaningful — the round time is `max(draft, verify)
≈ verify`. C2's wall time is verify-bound, period.

This is the nuance `phase1_c2.md` predicted ("verify hides under draft
only at k≳5") — **inverted**. In practice the draft is the cheap phase
and verify is the long pole at *every* k, because the NPU verify is not
running on the fast AR128 path.

## Root cause — the NPU verify is AR1-per-token, not AR128-batched

Phase 0 (`phase0_npu.md`) measured an NPU verify of any k∈[2,128] at a
flat **~52 ms** on the **AR128 graph** (k padded up to 128). C2's
projected win rested entirely on that number: a ~52 ms verify hiding
under the draft.

But the verify in this driver is built from the sidecar's **stream
primitives**, which are **AR1-only** (`stream_decode` and `stream_append`
both step one token at a time through the `ar1` graph at ~34 ms/step).
There is no stream op that runs a k-token batch through the AR128 graph.
So per round, verify costs roughly:

```
  (committed_delta) appends           ~accept/round AR1 steps
  + (k+1) decode steps                 k+1 AR1 steps
  + k (truncate + append) for the      k AR1 steps  (truncate is free;
      speculation substitution             append is one AR1 step)
  --------------------------------------------------------------
  ≈ (2k + 3 + accept/round) AR1 steps  per round, ~34 ms each
```

For k=2 that is ~7.4 steps ≈ 250 ms/round × 54 rounds ≈ 13.2 s — matches
the measured `verify_wait_s` exactly. For k=8 it is ~22 steps/round.
**C2's verify pays ~5–10× the 52 ms AR128 figure**, and it pays it every
round.

The earlier *stateless* version of the driver was far worse still
(0.74 tok/s at k=2): it re-prefilled the entire growing committed prefix
via `stream_open` every round. The shipped driver fixes that with a
**persistent verify stream** that only ingests the per-round delta — a
~10× improvement (0.74 → 7.53 t/s) — but the residual AR1 speculation
cost is the floor it cannot get under.

## Why this is a harness limit, not a silicon limit

The NPU silicon *can* verify k tokens in ~52 ms — Phase 0 proved it on
the AR128 graph. The gap is purely software: the sidecar's stream API
was built for *chat* (monotonic single-token decode), not for
*batched verify*. A true-AR128 C2 needs a new sidecar op:

> `verify_batch(committed_ids, draft_ids)` → run the committed prefix +
> k drafts as one **AR128** forward (pad k up to 128), read the target
> argmax at the k+1 logit positions of interest, return them. One ~52 ms
> pass instead of ~2k+3 AR1 steps.

With that op, C2's verify drops to ~52 ms/round flat. Projected C2 at
k=4: ~40 rounds × max(draft≈80 ms, verify≈52 ms) ≈ 40 × 80 ms ≈ 3.2 s
for 130 tokens ≈ **~40 tok/s** — competitive with, though still short
of, CPU-spec's 54.6. That is a projection, not a measurement; it is the
single highest-value follow-up.

## Blockers / honesty notes

1. **No AR128 batched-verify op in the sidecar (gating for a fast C2).**
   The stream API is AR1-only. Building `verify_batch` means editing
   `npu_engine/sidecar.py` — outside this task's write scope
   (`gpu_npu_sidequest/` only) — so it was not attempted. This is *the*
   blocker between the measured 7.53 t/s and a competitive C2.
2. **Single run per k** (timeboxed). The tok/s figures are n=1; treat
   ±10% as noise. The 7× gap to CPU-spec is far too large to be noise.
3. **cl512 ctx tier** caps total KV at 511 slots; runs stopped cleanly
   at the cap (`L+k+1 > 511`), giving 129–130 decoded tokens — close
   enough to the 128 target.
4. **Async overlap is real but moot.** The ThreadPoolExecutor draft∥
   verify overlap works (draft is fully hidden); it just buys nothing
   while verify is 5–10× the draft cost.
5. The prior agent's claimed gating blocker (an NPU verify HTTP shim)
   was **not** real — in-process sidecar verify works fine and is
   simpler. The *real* blocker is one level down: the AR1-only stream
   API.

## Verdict

- **C2 ran end-to-end and is correct** — verify sanity-checked against
  the 4B's own greedy output (2 tests, both pass at every k).
- **C2 as currently runnable loses decisively**: best 7.53 tok/s vs
  CPU-only spec decode 54.6 (0.14×) and CPU target-alone 50.7 (0.15×).
  It also loses to GPU-only spec decode (11.0) and GPU target-alone
  (27.0).
- The loss is **100% the AR1 verify path**, not the heterogeneous
  placement. The GPU draft is fast (~70 ms/round) and fully overlapped.
- **Phase 0's "C2 wins" prediction is not refuted in principle — it is
  unreachable with today's sidecar.** It assumed a ~52 ms AR128 verify;
  the stream API only offers AR1. A `verify_batch` AR128 op in the
  sidecar would be the unblock; projected C2 with it is ~40 t/s (still
  short of CPU-spec, but a real contest). That op is the recommended
  next step and the one thing standing between this 7.53 t/s and a fair
  C2 number.

## Artifacts

- Driver: `gpu_npu_sidequest/scripts/run_c2.py`
- Results CSV: `gpu_npu_sidequest/results/phase1_c2.csv`
- Logs: `gpu_npu_sidequest/logs/c2_sanity.log`,
  `c2_run_k248.log`, `c2_results.json`, `c2_gpu_server.log`
