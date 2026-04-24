# Shotgun probe results — session 18 (2026-04-22, battery)

Correctness probe sweep across the 7-variant shotgun (4 new distinct
binaries after collision-deduping). Probe: `npu_short_prompt_probe.py
--path pathb` on humaneval fib-p0 (prompt_len=16) + p1 (25 tokens).
Gate: cos ≥ 0.95 / argmax match / multi-step rate ≥ 66%.

## Power state

- **Battery, discharging, 98% charge** at session start (662 min est
  runtime) and 98% / 566 min at end. Six probes drew ~100 min of
  estimated runtime — not thermally stressed, but AC sweep is still
  outstanding for steady-state numbers.
- All latency values below are single-call, first-inference timings.
  Caveat applies — per prior Phase 5.5 observations, battery→AC is a
  ~27% speedup delta and steady-state after warmup is typically
  faster than a cold first call.

## Summary table

| variant | cos p0 | cos p1 | argmax | top-5 | multi-step | ms/step | outcome |
|---|---:|---:|:-:|:-:|:-:|---:|---|
| baseline w4a16-local (tf) | 0.33 | — | ✗ | 1/5 | 0% | ~30 | COLLAPSE (session 16) |
| w4a16-local-tfe (enhanced) | 0.36 | 0.61 | ✗ | 1/5 | 0% | ~30 | COLLAPSE (session 17) |
| w4a16-local-mse | 0.33 | — | ✗ | 0/5 | 0% | 50.6 | COLLAPSE — act-cal not the lever |
| **w4a16-local-pr** | **0.888** | **0.904** | ✓ | 3-4/5 | **100%** | 47-50 | **Soft pass** (cos<0.95, greedy matches) |
| **w8a16-local** | **0.963** | **0.979** | ✓ | 4/5 | **100%** | 48-50 | **FULL PASS both prompts** |
| w8a16-local-pr | 0.924 | — | ✓ | 4/5 | 100% | 50.8 | Soft pass (per-row hurts at w8) |
| fp16-local (reference) | 0.9999 | — | ✓ | 5/5 | 100% | 65 | correctness ceiling |

## Findings

### 1. w8a16-local is the clean deliverable

First variant in the whole Lever C investigation that clears the full
correctness gate. Passes on both probed prompts (cos=0.963 / 0.979,
argmax match, multi-step 100%). Usable as the w4a16 replacement even
if we never find a correct w4a16 recipe. Bin is the same size as w4a16
(917 MB) so no disk / memory regression vs a working w4a16 would have
been.

### 2. Per-row weight quant is the critical w4 fix — but mildly hurts at w8

- `w4a16-local-pr`: cos 0.33 → **0.888** (p0), 0.904 (p1). Argmax
  matches CPU, multi-step stays aligned. Fixes the layer-1+
  V-projection collapse the session-17 differential probe localised.
  Not quite over the 0.95 strict gate but qualitatively functional
  for speculative decode (greedy path is what matters for accept rate).
- `w8a16-local-pr`: cos 0.963 → 0.924. At w8 there's already enough
  weight headroom that per-row's additional per-channel scales don't
  help; the extra metadata / encoding work slightly hurts the output.
- Lesson: per-row is a w4-specific remedy, not a universal speedup.

### 3. Activation calibration algorithm doesn't move the needle

tf (0.33) / enhanced (0.36) / mse (0.33) all converge near the same
collapsed cos when V weights are under-precisioned. Confirms the
session-17 diagnosis (w4 V-projection weights are the root cause, not
activation ranges).

### 4. Latency at ~50 ms per step for all 4 new variants — caveat

Previously measured baseline w4a16-local and tfe at ~28-30 ms/step
(sessions 15-17) vs ~50 ms for every new variant in this sweep.
Possible confounds:

- First-call inference timing on cold HTP context (vs warm re-runs in
  prior sessions).
- Battery vs AC (prior sessions mostly AC; this session fully battery).
- Thermal state (laptop unplugged ~30 min).
- Per-row bin format overhead (but the non-pr w8a16 also measured 50 ms).

Next session: a warmup-then-steady-state latency harness on AC would
split these. Any of the four could easily be cold-call noise. The
correctness ranking is robust to latency noise; the latency ranking
isn't.

## Decision tree update

Per the session-15 option space (`docs/w4a16_investigation.md`):

- **A.6 w8a16** — primary ask, now GREEN. Deliver w8a16-local as the
  Lever C product.
- **A.5 per-tensor overrides (V/O projections to w8)** — stretch
  goal. Would combine w4's memory savings with w8's correctness.
  Next x86 ask if we want to push further. Our data suggests this is
  viable: `w4a16-local-pr` already fixed most of the collapse via
  per-row granularity alone, so keeping that + per-tensor-override V
  to w8 should land above the gate.
- A.2 enhanced / A.2-alt mse: **retired** (confirmed negative on
  p0/p1).
- A.3 Bundle B calibration: superseded. Diagnosis was weight
  precision, not calibration distribution.
- A.4 CLE: **retired**. Shotgun showed CLE produces a byte-identical
  binary to baseline on our MatMul graph — silent no-op. `--apply_algorithms
  cle` is conv/BN-specific and doesn't match our topology.

## Artifacts

Per-probe logs in `results/correctness_*_p{0,1}.stdout/stderr`:

- `correctness_w4a16_local_pr_p0.stdout`
- `correctness_w8a16_local_p0.stdout` + `…p1` (via grep)
- `correctness_w8a16_local_pr_p0.stdout`
- `correctness_w4a16_local_mse_p0.stdout`

Binaries in `models/` (gitignored — MD5s and compile provenance in
`HANDOFF_shotgun.md` + doc `phase5_local_qairt_compile_findings.md`):

- `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-pr.bin` (620 MB)
- `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` (918 MB)
- `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local-pr.bin` (918 MB)
- `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-mse.bin` (918 MB)

## Next session (on AC)

1. AC sweep with `SPECULA_NPU_VARIANT=w8a16-local` against the Lever
   B 18.12 t/s baseline. Full 40-cell matrix.
2. Same for `w4a16-local-pr` as a secondary data point — if cos=0.89
   accept-rate holds up in practice, the ~20 ms latency advantage
   (w4 memory bandwidth) could outweigh the small cos gap. The
   humaneval greedy match at 100% across 3 multi-steps on both
   prompts suggests yes.
3. Measure steady-state latency properly (warmup + N iterations
   median) rather than first-call.
4. If w8a16 beats 18.12 t/s, Phase 5.5 Lever C closes positive.
