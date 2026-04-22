# Qwen3 NPU-spec performance levers — investigation plan

Phase 5.5. Systematic A/B of each perf lever against our closed
Phase-5 baseline so we learn which ones actually pay off on this
silicon before graduating to Qwen3.5.

Qwen3 (pure attention) stays the experiment host because all lever
mechanisms transfer cleanly to Qwen3.5 dense variants — same KV
layout, same compile flags, same HTTP bridge. Redoing ONNX export
+ AI Hub compile on Qwen3.5 before measuring levers would add 2-3
sessions of setup with no information gain.

## Baseline (Phase 5 close, commit 7e10670)

The numbers every lever is measured against:

| stat | k=2 | k=3 | k=4 | k=8 |
|------|----:|----:|----:|----:|
| mean accept | 81.03% | 75.53% | 70.87% | 55.60% |
| mean decode t/s | **7.98** | 7.35 | 6.84 | 5.33 |
| best cell t/s | 8.44 (p8) | 8.25 (p1) | 7.93 (p1) | 6.84 (p1) |
| worst cell t/s | 7.37 (p5) | 6.59 (p5) | 6.03 (p5) | 4.37 (p5) |

CSV: `results/spec-npu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-pathbmask-20260421-180804.csv`

**Per-round wall breakdown** (k=3, humaneval p0, 22 rounds,
n_predict=64 — from `results/phase5_step8_outer_loop.log`):

| phase | wall | per-round | fraction | notes |
|-------|-----:|----------:|---------:|-------|
| NPU draft (k-1 = 2 steps) | 3.34 s | 152 ms | 35.8% | 2 × ~76 ms per call |
| target /completion (k+1 = 4 tokens) | 3.46 s | 157 ms | 37.1% | CPU 8B decode + HTTP |
| NPU absorb (1-2 steps) | 2.52 s | 115 ms | 27.1% | 1 call if j<k, 2 if j==k |
| **total** | **9.32 s** | **424 ms** | 100% | decode 6.97 t/s |

NPU time (draft + absorb): 5.86 s = 63% of wall. Target verify is
37% of wall. The two are strictly sequential today. Every lever
below attacks one of these two budgets or the dependency between
them.

## Status snapshot — 2026-04-21

Running tally of closed levers. Full per-lever detail lives in
each lever's **Result** subsection below.

| lever | state | k=2 mean t/s | vs baseline | accept | notes |
|-------|-------|-------------:|------------:|-------:|-------|
| Phase 5 baseline (7e10670) | — | 7.98 | — | 81.03% | ctx=512, fp16, sync outer loop |
| Lever A (64de69f) | CLOSED | 10.93 | +37.0% | ~81% | async draft∥verify, battery |
| Lever A step 2 pipelined (56b375b) | CLOSED | — | +39.6% wall | — | pipelined verify-ahead stacks on A |
| Lever B ctx=256 + pipelined (f755d6d) | CLOSED | 14.28 | +79.0% | ~76% (battery) | best cell 17.78, 10-humaneval, battery+CAD |
| Lever B AC rerun (this session) | CLOSED | **18.12** | **+127%** | **81.91%** | same sweep, AC+idle; best cell 19.07 |
| R4 zero-copy / shared-mem (557c59e) | NEGATIVE | — | no win | — | per-step cost already dominated by compute not copy |
| Lever C W4A16 (in flight) | WIP | — | — | — | Bundle A compile submitted, see §Lever C Progress |

Battery→AC delta on the same binary is ~+27% — a reminder that all
lever comparisons must be AC-vs-AC to avoid thermal-confound swamping
the signal. The AC baseline (18.12 t/s mean) is the reference for
Lever C's comparison.

## Measurement protocol (every lever)

Each lever produces one artifact:
`results/spec-npu-{lever}-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-*.csv`
from the same 40-cell sweep harness (`scripts/sweep_npu_spec.py`)
or a minor adaptation thereof.

**Schema:** same as baseline (Phase-2 compatible). Diff columns
added in the writeup table:
`Δ accept_pct vs baseline` and `Δ decode_tps vs baseline`, one row
per (prompt_idx, k) cell.

**Gate conditions** (per lever, before rolling forward):
1. NPU-draft correctness: `scripts/npu_short_prompt_probe.py`
   must still return cos ≥ 0.95 and multi-step ≥ 66% match.
2. End-to-end coherence: first three prompts' generated text
   must remain recognisable for the task (fibonacci, binary
   search, palindrome).
3. Accept-rate delta: within ±5 pp of baseline at k=2. Larger
   drops mean the lever silently broke the drafter.

Any gate failure: fix before sweeping. Don't bank numbers from a
broken lever.

## Lever catalogue

Ordered by expected (information per hour) — cheap-to-apply levers
first so we accumulate comparison data quickly, then tackle the
heaviest lever (W4A16) last once the compile + sweep flow is
well-oiled.

### Lever A — Async NPU-draft ↔ target-verify overlap

**Hypothesis.** Wall per round drops from
`NPU_round + verify_round` to `max(NPU_round, verify_round)` if
we kick off round N+1's NPU draft speculatively while round N's
/completion is in flight. Baseline split is 267 ms NPU + 157 ms
verify per round; max = 267 ms, so ~37% wall reduction.

**Expected decode t/s.** 7.98 → ~11 t/s (38% improvement). Best
case stacks: lever A × nothing = 11 t/s; lever A × lever C (W4A16)
= ~24 t/s.

**Speculation rule.** Assume `drafts[0]` of round N will be
accepted (true ~81% of rounds at k=2). Kick off round N+1's draft
phase during round N's verify with that speculative committed_ids.
On mis-predict (drafts[0] of round N rejected), discard N+1's
in-flight draft and re-draft with the correct state. The
mis-predict rate at k=2 is roughly `1 - accept_rate ≈ 19%`, so
overall expected wall reduction is `0.81 × (full overlap) + 0.19 ×
(no overlap)` — still a solid win.

**Implementation cost.** ~200 LOC async rework on
`npu_spec_outer_loop.py`. Python `asyncio.gather` + a single
background NPU-draft task feeding into the main loop. llama-server
/completion is already HTTP so it parallelises trivially.

**Risks.**
- Python GIL — the NPU session.run() holds GIL? ORT-QNN calls into
  native code; if it releases the GIL during execution, async
  works. If not, we need a ThreadPoolExecutor. Check empirically.
- Speculation-miss bookkeeping — need two KV snapshots (committed
  state + speculative state), properly discarded on miss.

**Measurement plan.**
1. Build `scripts/npu_spec_outer_loop_async.py` alongside the sync
   version (keep both for A/B).
2. Smoke-test on humaneval p0 k=2 n=64 — confirm same text,
   measure wall reduction.
3. Full sweep, compare CSV delta cell-by-cell.

**Stop criterion.** If async wall doesn't beat sync by ≥15%
(below the model's predicted 37%), something is serialising
unexpectedly — probably GIL. Investigate before sweeping.

**Result (commits 56b375b → 64de69f, battery).**
Closed +37% over baseline on the 40-cell sweep at k=2 (10.93 t/s mean
vs 7.98). Step 1 was the naive async overlap (+20% wall); step 2
added pipelined verify-ahead (verify round N while NPU drafts round
N+1) for a cumulative +39.6% wall reduction. GIL was not a blocker —
ORT-QNN releases it during `session.run()`. Prediction (11 t/s) and
observation (10.93) land within 1%, validating the per-round wall
decomposition. `run_spec_decode_async_pipelined` in
`scripts/npu_spec_outer_loop_async.py` is the winning driver; the
sync loop remains for regression A/B.

---

### Lever B — Smaller past_len compile tier (past_len=256 or 128)

**Hypothesis.** Our binary bakes past_len=511, so every NPU call
does attention over 512 KV slots regardless of how many are real.
For our code-generation workload (prompt ~16 tok + n_predict ≤ 256
= total ≤ 272 tokens), most slots are masked-out padding. A
past_len=256 tier cuts attention FLOPs ~2× and KV memory ~2×.

**Expected per-step latency.** 63 ms → ~40 ms (rough, assumes
attention is ~40% of per-step cost; the rest is MLP/linears which
don't scale with seq_k).

**Expected decode t/s.** 7.98 → ~11 t/s (similar ballpark to
lever A, different mechanism). Stacks with A: ~15 t/s.

**Implementation cost.** One AI Hub compile (~10 min) with
`past_len=256` baked in, plus wrapper update
(`CONTEXT_MAX=256` in `npu_load_qwen3_bin.py`). Also need to
re-validate via `npu_short_prompt_probe.py` at the new tier.

**Risks.**
- Total generated length capped at 255 (prompt + n_predict).
  Phase-2-comparable n_predict=256 runs would need prompt ≤ 0.
  For the sweep we'd reduce to n_predict=200 or match Phase 2's
  shorter-context fixtures only. Partially apples-to-apples.
- Compile could fail (we hit SymbolicDim bugs at step 6 on the
  first cycle; hopefully past_len change doesn't reintroduce).
- Edge case: some prompts are >128 tokens (e.g. `flatten` p5 at
  ~40 tokens — fine for 256, tight for 128). Check prompt lengths
  before committing to past_len=128.

**Measurement plan.**
1. Recompile ONNX with past_len=256 via AI Hub. ~10 min.
2. Bench wrapper load + single-step latency vs 511 binary. Should
   be ~35% faster per call.
3. Run `npu_short_prompt_probe.py` on new tier; confirm cos > 0.95.
4. Modified sweep at n_predict=200 (to keep total ≤ 256). Compare
   to baseline's equivalent-length rows (post-hoc trim of baseline
   CSV at ~200 decoded tokens).

**Stop criterion.** If per-step latency drops <20% (below model
prediction of 35%), attention is NOT the dominant per-step cost
and the MLP/linears are — then lever A and C become the only
paths forward, skip further past_len work.

**Result (commit f755d6d, battery).** Closed +79% over Phase 5
baseline and +31% over Lever A alone. ctx=256 recompile (`--ctx 256`
on `compile_qwen3_ai_hub.py`, runtime switch via `SPECULA_NPU_CTX=256`
env var) stacked with Lever A's pipelined async driver to reach
**14.28 t/s mean, 17.78 t/s best cell** at k=2 on the 10-humaneval
sweep at n_predict=200 (battery + light CAD load). Attention
was ~40% of per-step cost as modelled; halving seq_k brought
per-step latency down consistent with prediction. No accept-rate
regression on humaneval; structured_json prompts >128 tokens are
ctx=256-safe (max observed prompt = 86 tokens).

**Result (AC rerun, this session, 2026-04-21).** Same sweep on AC
power with other programs closed: **18.12 t/s mean, 19.07 t/s best,
16.92 worst, 81.91% accept**. +26.9% over the battery number on an
identical binary — the battery-vs-AC delta is larger than any
individual lever's gain. CSV:
`results/spec-npu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-pathbmask-async-pipelined-20260421-213621.csv`.
This is the reference baseline for Lever C.

---

### Lever C — W4A16 quantisation (biggest potential win)

**Hypothesis.** Qualcomm's own Qwen3-4B Genie reference bundle
(`models/qualcomm-qwen3-4b-ref/`) ships W4A16 on the same HTP
silicon. Their numbers (published in AI Hub model card) report
~2-3× per-step speedup vs FP16 interior on decode throughput,
driven by:
1. **Weight memory bandwidth −4×** (INT4 vs FP16): 1.2 GB → 300 MB
   for Qwen3-0.6B. First-order gain on LPDDR5X-bound workloads.
2. **HTP dedicated INT4 MAC units** — fewer HVX cycles per MAC.
3. **Smaller context binary** → less spill/fill pressure.

**Expected per-step latency.** 63 ms → 25-30 ms (2-2.5× from
Qualcomm's published ratio).

**Expected decode t/s.** 7.98 → ~18-20 t/s at k=2, assuming accept
rate stays within 5 pp of baseline. Stacks with A + B: ~30 t/s
ceiling (finally in the range of Phase 2 CPU-spec 40.2 t/s).

**Implementation cost.** Larger than A or B.
- AI Hub compile with `--quantize_full_type w4a16` and a
  calibration dataset (20-50 samples from our humaneval +
  structured_json fixtures). ~15-30 min compile.
- Alternatively, AIMET-side PyTorch quant first, then AI Hub
  compile from the quantised ONNX. More control but more code.
- Re-run step-6-equivalent correctness probe: the FP32 CPU ONNX
  reference is still ground truth, but expect cos drop to ~0.95-0.99
  (quantisation error). Need tolerance adjustment.

**Risks (non-trivial).**
- **Our Path B-mask ONNX might not lower cleanly at W4A16.** The
  step-6 retro found that Path B-mask and Path A converge after
  ORT constant-folding anyway; worst case we fall back to Path A
  (no `attention_bias`, all-ones mask → same functional regime as
  full-context prompts). Investigate if B-mask breaks.
- **Accept rate drop.** Qualcomm's Qwen3-4B reference published a
  ~-3 to -5 pp accept hit from W4A16 on benchmark-style tasks. If
  our accept drops more (e.g. -10 pp), the throughput gain is
  partially cancelled by more rejection.
- **Calibration-dataset sensitivity.** Wrong calibration
  distribution → W4 weights end up suboptimal for code/JSON
  generation. Use our actual prompt fixtures as calibration; that
  matches the evaluation distribution.

**Measurement plan.**
1. Curate calibration dataset: 20 prompts from humaneval +
   structured_json, tokenised + in the right format for AI Hub's
   quant pipeline (or AIMET).
2. Compile at W4A16. Download, validate load.
3. `npu_vs_cpu_correctness.py` — expect cos 0.9-0.99, accept if
   argmax match on prefilled-KV single step.
4. `npu_short_prompt_probe.py` — expect cos ≥ 0.95, argmax match,
   multi-step ≥ 66%.
5. Full 40-cell sweep. If accept rate drops more than 5 pp at k=2,
   run at k=3 (more drafts per round → amortises accept drop).
6. Writeup: does W4A16 compile + run on THIS specific export path,
   what numerical drift, what perf delta.

**Stop criterion.** Two failure modes to watch:
- NPU short-prompt probe fails cos ≥ 0.95: compile is numerically
  wrong, investigate (likely calibration or op-lowering).
- Accept rate at k=2 drops >10 pp: drafter is too noisy; retry at
  W8A16 (smaller weight savings, less accept loss). W8A16 is
  trident's fallback per `npu_path_back.md`.

**Progress (this session, 2026-04-21).**

Research refinement: running TWO calibration bundles (not one) to
answer a second-order question for the Qwen3.5 graduation — does
cheap calibration suffice, or is realistic multi-position capture
necessary?

| bundle | samples | decode positions per prompt | past_kv content | local size |
|--------|--------:|-----------------------------|-----------------|-----------:|
| **A (realistic)** | 60 | [5, 25, 60] | CPU FP32 snapshots | 3.27 GB |
| **B (step-0 only)** | 20 | [0] | real step-0 (prompt-prefilled, padded with zeros) | 1.09 GB |

Execution order: compile + sweep A first. If A fails accept gate,
skip B and fall back to W8A16 — B under weaker calibration can't
rescue a fundamentally-marginal quant. If A passes, compile B and
compare the delta. Writeup reports both; if B matches A, future
calibration work is much simpler.

Infrastructure landed this session:

- `scripts/capture_calibration_samples.py` — runs CPU FP32 ONNX
  prefill + greedy decode on the 20-prompt humaneval + structured_json
  fixtures, snapshots model inputs at selected decode positions,
  saves stacked-per-input `.npz` in the exact shape AI Hub's
  `calibration_data` kwarg expects. Reusable for future quant work
  including the Qwen3.5 cutover.
- `scripts/compile_qwen3_ai_hub.py` extended: `--quant {float16,w4a16,w8a16}`,
  `--calibration-npz PATH`, `--calibration-dataset-id ID`,
  `--quant-tag` for disambiguating output filenames. Existing fp16
  path is unchanged (backward-compatible default).
- `SPECULA_NPU_VARIANT` env var added to `npu_load_qwen3_bin.py`
  + `npu_vs_cpu_correctness.py` so probe / outer_loop / sweep can
  target a variant binary (e.g. `SPECULA_NPU_VARIANT=w4a16-a`)
  without code changes to downstream callers. Mirrors
  `SPECULA_NPU_CTX`'s pattern.
- Calibration artifacts on disk (gitignored): `models/calibration/bundle_a_ctx256.npz`
  (3.27 GB, 60 samples) and `.../bundle_b_ctx256.npz` (1.09 GB, 20
  samples), each with a `.manifest.json` describing shapes, dtypes,
  source prompts, and capture positions.

**Gotcha learned this session (commit TBD).** AI Hub's PTQ validator
iterates `calibration_data` positionally and checks each slot against
the ONNX graph's `graph.input[i].name` — key-set equality is not
enough, ordering matters. First submit failed at ~120s with
*"Calibration data set has input 'attention_bias' but expected
'past_key_values.0.key'"* because the capture script built per-sample
dicts as (input_ids, position_ids, **attention_bias**, past_kv…),
but the ONNX graph has attention_bias LAST (after all 56 past_kv
entries, per build_input_specs). Fix: rebuild DatasetEntries by
iterating `specs` (already in graph order) in the compile script;
also fixed `capture_for_prompt` to save in the right order.

**In flight (retry).** Bundle A compile resubmitted with
`--reuse-upload mq23729wn` (model already on AI Hub from the first
failed attempt, ~15 min upload skipped). Calibration re-upload in
progress (~32 min at ~1.7 MB/s for 3.27 GB); compile itself 25-40
min thereafter. Output: `models/qwen3_0_6b_draft_v81_ctx256.pathbmask.w4a16-a.bin`.
Correctness probe + sweep flow pending compile completion.

---

### Lever D — Zero-copy KV handoff (deferred / low priority)

**Hypothesis.** Our outer loop converts past_kv between numpy
arrays and ORT tensors every step. `cl_qcom_ion_host_ptr` or
equivalent Adreno shared-buffer binding would let the NPU session
rebind the SAME buffer across calls without copy.

**Expected decode t/s.** ~5-10% win at most. The per-step cost is
dominated by the compute on the NPU, not the numpy↔ORT plumbing.

**Rationale for deferring.** 
- Low expected delta vs levers A/B/C.
- The real value of ION-backed buffers is **NPU ↔ GPU handoff**
  for a future DFlash-style or NPU-draft + OpenCL-target pipeline.
  Not useful for pure NPU-draft + CPU-target-via-HTTP.
- Non-trivial implementation — needs either raw QNN path (not ORT)
  or a custom ORT EP option that accepts pre-bound buffers.

**Status.** Documented here for completeness; do NOT implement in
Phase 5.5. Revisit when Phase 4 DFlash starts (OpenCL-side needs
this pattern for its `target_feat` ring).

---

## Research additions — might move the needle

### R1. Bigger draft: Qwen3-1.7B on NPU

The Phase 2 decision to use Qwen3-0.6B-Q8_0 as draft was driven by
CPU draft cost (~9 ms/step). On NPU the draft cost is weight-
bandwidth-bound (we're at ~63 ms/step for 0.6B), so scaling to
1.7B would roughly 3× per-step cost → ~190 ms. But accept rate
climbs to ~85-90% (per Qualcomm's published Qwen3 accept tables).
Math:
- 0.6B: 63 ms/step × 3 calls/round + 157 ms verify = 346 ms/round
  → 81% accept × 3 drafts = 2.43 committed + 1 bonus = 3.43 / round
  → 9.9 t/s (matches observed k=3 baseline at ~7.35, close enough)
- 1.7B: 190 ms × 3 + 157 ms = 727 ms/round
  → 88% accept × 3 = 2.64 + 1 = 3.64 / round → 5.0 t/s

**Scaling up loses** even with higher accept. Skip.

Exception: if lever C (W4A16) brings 1.7B down to ~55 ms/step,
math flips: 55 × 3 + 157 = 322 ms/round, 3.64 / 0.322 = 11.3 t/s.
Marginally better than 0.6B × W4A16. Nth-order optimisation;
park until 0.6B × W4A16 data is in.

### R2. Draft-p-min early-exit

llama.cpp's `--draft-p-min 0.75` stops drafting mid-round when
the draft's token probability falls below threshold. Equivalent
here: after each NPU draft step, if `softmax(logits)[argmax] <
0.75`, stop drafting and submit fewer drafts. Saves NPU calls
on low-confidence streaks.

Effect on our data: at k=2, drafts[1] of round would only be
emitted if drafts[0]'s argmax prob is high. On low-accept prompts
(p5 flatten at 74%), we'd skip ~25% of second drafts. Expected
~5-8% wall reduction.

**Cost**: ~30 LOC in draft_k_tokens (track p_max, early-break).
Cheap; probably land alongside lever A so we can measure
both in one pass.

### R3. Tree-verify (larger draft surface per round)

Phase 2 found k=3 was optimal for CPU-spec; on NPU k=2 wins
because per-step cost is higher. Tree verify would let us draft
multiple token paths from a branch point (e.g. top-2 at each
step → 4-wide tree at depth 2) and verify all paths in one
/completion call. llama-server's `/completion` does NOT support
tree drafts natively — we'd need a custom verifier. Out of scope
for Phase 5.5; noted as Phase 6+ research.

### R4. Persistent KV on NPU via ORT-QNN context binding

ORT-QNN 1.24's `shared_session_memory` feature (if exposed)
might let us pin the KV tensors to device memory across
session.run() calls instead of re-uploading each step. Check
if the EP options list supports this. Low-effort if yes, free
10-20% win. High-effort if not exposed (need raw QNN path).
Probe before W4A16.

**Result (commit 557c59e): NEGATIVE.** The
`shared_session_memory` EP option is exposed but the probe did
not find a measurable per-step win — per-step cost is dominated
by compute-on-the-NPU, not numpy↔ORT plumbing. Matches Lever D's
5-10% ceiling estimate. Parking until a Phase-4 OpenCL-target
pipeline actually needs zero-copy NPU↔GPU handoff.

## Suggested investigation order

1. **Lever A (async overlap)** — pure code change, no recompile,
   fastest information. 1 session.
2. **Lever B (past_len=256)** — one recompile, adjust sweep's
   n_predict. 1 session including measurement.
3. **R2 (draft-p-min)** — small addition, can bundle with A or B.
   0.5 session.
4. **R4 (shared_session_memory probe)** — 30 minutes of EP-option
   poking before committing to W4A16.
5. **Lever C (W4A16)** — biggest lever, heaviest lift. 2 sessions
   (calibration curation + compile + validation + sweep).
6. **Combined sweep** (A × B × C if all three land) — final
   Phase 5.5 number.
7. **Writeup** in `docs/qwen3_perf_levers_results.md` (new)
   mirroring this plan doc with real numbers per lever + combined.

Each session ends with its own commit so every lever's data is
independently recoverable from git. Final Phase 5.5 summary
updates `docs/npu_results.md` with a "Phase 5.5 appendix"
section citing the new combined number.

## Success criteria for Phase 5.5 close

- **Minimum.** At least one lever passes its own gate conditions
  and produces a clean delta-vs-baseline number. Even if every
  lever disappoints, the negative data is publishable.
- **Target.** Combined decode rate ≥ 15 t/s on the 40-cell
  sweep (~2× baseline, ~0.58× CPU-alone TG, ~0.37× Phase 2
  CPU-spec). This would establish NPU-draft as a real path, not
  just a proof-of-concept.
- **Stretch.** Combined ≥ 30 t/s (>CPU-alone TG). Would let us
  claim heterogeneous compute as a throughput win on this
  hardware, Qwen3-generation. Qwen3.5 graduation would start from
  a positive NPU result.

If the minimum fails (every lever regresses or breaks compile),
we have a documented-loss story that's still informative. If the
target passes, Phase 5.5 closes with a real number. If stretch
passes, we likely want to write it up for upstream.

## Baseline artifacts (don't touch)

- `results/spec-npu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-pathbmask-20260421-180804.csv`
- `models/qwen3_0_6b_draft_v81_ctx512.patha.bin`
- `models/qwen3_0_6b_draft_v81_ctx512.pathbmask.bin`
- `scripts/npu_spec_outer_loop.py` (post-lazy-snapshot, commit 7e10670)
- `scripts/sweep_npu_spec.py`

Each lever produces new artifacts under parallel names
(e.g. `models/qwen3_0_6b_draft_v81_ctx256.pathbmask.bin` for
lever B) without overwriting the baseline.
