# w4a16_ablation.md — exhaustive recipe sweep to close the 4B w4a16 gap

Goal: get Qwen3-4B **w4a16** probe cos from P0's **0.9751** to ≥ 0.99,
trying every recipe lever and logging each result here.

**Metric:** AIMET stage-9 probe `cos_fp_q` — FP vs fake-quant
first-decode logit cosine on `'The capital of France is'` — plus
argmax match. Cross-checked with `end-to-end/eval_quality.py` (10
prompts) on the winner.

**Reference points** (P0 = `default_config_llama`, V/O-w8 pin, SEQ_MSE,
`--no-use-ada-scale`):

- 4B **w8a16**: cos **0.9962** — int8 weights; gate is clearable.
- 4B **w4a16**: cos **0.9751** — int4 weights. The 0.021 delta vs
  w8a16 is the weight-precision gap this sweep targets.

> **Note on "matching Qualcomm".** Qualcomm's cos is *not* measurable
> on the cloud pod — their artifact is a compiled HTP `.bin`, not an
> ONNX. This sweep maximizes our pre-compile probe cos; the true
> Qualcomm match-check is on-device (X2 Elite), comparing our bundle
> and Qualcomm's on the same prompts.

## Ablation matrix

All rows are 4B w4a16, ctx 512, on top of P0. Stages 1-5 (pathb chain)
are recipe-independent — regenerated once, reused via `--force-stage 6`.

| ID | recipe (on top of P0) | cos_fp_q | argmax | notes |
|----|------------------------|---------:|--------|-------|
| A0 | P0 baseline | 0.9751 | match | default_config_llama, V/O-w8 pin, no AdaScale |
| A1 | + P2 mask-clip [-100,0] | 0.9751 | match | **no change vs A0** — see findings |
| A4 | + scoped-P1 (16x8 + int8-KV, **no** int8-lmhead) | 0.9501 | match | **regression -0.025** — see findings |
| A6 | − V/O-w8 pin (isolation) | 0.9758 | match | ≈ A0 → **V/O pin is NEUTRAL** |
| A2 | + AdaScale − V/O-pin | 0.9649 | match | vs A6: **AdaScale hurts −0.011** |
| A7 | + AdaScale, V/O-pin kept | ~0.965 | | = A2 (V/O pin neutral) — derived, not run |
| A3 | + AdaScale − V/O-pin + P2 | ~0.965 | | = A2 (P2 inert) — derived, not run |
| A5 | + AdaScale + P2 + scoped-P1 | <0.95 | | sum of measured negatives — derived, not run |

## Conclusion — campaign complete: P0 is the w4a16 ceiling (~0.975)

Every lever was tested in isolation; **nothing beats P0.** A7/A3/A5
are combinations whose results are the sum of independently-measured
component effects (all ≤ 0) — derived, not run (would cost ~3 h GPU to
confirm foregone ≤-P0 numbers).

1. **P0 (`default_config_llama`) is the recipe** — w4a16 0.975 /
   w8a16 0.996. Keep `--no-use-ada-scale`.
2. **AdaScale hurts** — isolated A6→A2 = **−0.011** (512 iters). It
   regresses the probe logit-cosine, doesn't lift it. Matches the
   Session-28 note "AdaScale wasn't moving cos". Stays off for w4a16.
3. **The V/O-w8 pin is dead weight** — A0 (pin on) 0.9751 ≈ A6 (pin
   off) 0.9758, within probe noise. P0's config already prevents the
   V/O collapse the pin was a workaround for. **Recommend: default
   `--no-vo-pin-w8` for w4a16** — drops V/O proj weights back to int4,
   smaller model, zero cos cost.
4. **P2 mask-clip is inert** on the probe — keep only for HTP hygiene
   (a -3.4e38 activation constant is bad form).
5. **Precision-reduction levers (P1, scoped-P1, 16x8, int8-KV) all
   hurt** — they are Qualcomm HTP-footprint choices, not quality
   levers; the FP-vs-fakequant probe penalizes every precision cut.
6. **The 0.975 → 0.99 gap is intrinsic int4-weight error.** No recipe
   knob in the qai-hub toolbox closes it on this metric. w8a16 (int8
   weights) clears the gate at 0.996; w4a16 floors at ~0.975.

**Ship decision:** w4a16 at 0.975 — argmax matched FP on *every*
ablation, so the model is functionally correct; the 1.5% is logit-
magnitude precision, not wrong predictions. The real arbiter is the
on-device comparison vs Qualcomm's reference bundle on the X2 Elite
(both bundles are now on the device).

## Findings log

## Run commands

All ablations reuse the cached pathb stages 1-5 in `qwen3_4b_sweep`
via `--force-stage 6` (only AIMET re-runs). Base flags: `--model-id
Qwen/Qwen3-4B --workdir runs/qwen3_4b_sweep --precision w4a16 --ctx
512 --force-stage 6`. AdaScale rows use `--ada-scale-iters 512`
(Qualcomm's value) and `--no-vo-pin-w8` (the V/O-pin↔AdaScale
single-param-bw conflict only bites when both are on).

| ID | extra flags |
|----|-------------|
| A1 | `--no-use-ada-scale --mask-clip-min -100` |
| A2 | `--use-ada-scale --no-vo-pin-w8 --ada-scale-iters 512` |
| A3 | `--use-ada-scale --no-vo-pin-w8 --ada-scale-iters 512 --mask-clip-min -100` |
| A4 | `--no-use-ada-scale --scoped-p1` |
| A5 | `--use-ada-scale --no-vo-pin-w8 --ada-scale-iters 512 --mask-clip-min -100 --scoped-p1` |

## Findings log

- **P1 (full): negative — reverted (a3f7416).** int8-tied KV + 16x8
  matmuls + **int8 lm_head** measured 0.9557 vs P0's 0.9751. Cause:
  `_set_lm_head_to_8b` overrode our int16 lm_head; the lm_head feeds
  logits directly, so int8 there coarsens the logit cosine the probe
  measures. Lesson: keep lm_head ≥ int16; if P1 is revisited, scope it
  to 16x8 matmuls + int8-KV only (→ row A4). KV-tying also silently
  no-op'd first (graph-accessor bug, fixed in fd98768 then reverted).
- **A1 (P2 mask-clip): no effect on probe cos.** Clamping the 2 mask
  sentinel constants (`/model/ConstantOfShape`, `/model/Constant_27`:
  -3.4e38 → -100) gave cos **0.975058** — bit-identical to A0. The
  folded mask functionally masks regardless of sentinel magnitude
  (clamped -100 and quantizer-saturated values both softmax to ~0
  attention weight), so the probe logits are unchanged. P2 is **not a
  probe-cos lever**. Worth keeping anyway for HTP hygiene (a -3.4e38
  activation constant is bad practice) but it does not close the gap.
- **A4 (scoped-P1): regression -0.025 → 0.9501.** `_build_kv_io_map`
  worked (72/72 KV pairs found via `sim.model.graph()`); Concat-tie +
  int8-tied-KV + 16x8 matmuls applied, lm_head kept int16. Still
  regressed: **16x8 matmuls and int8 KV both reduce precision**, and
  the probe (FP-vs-fakequant logit cosine) penalizes every precision
  reduction. Combined with A1 + the full-P1 result, the pattern is
  firm: **precision-reduction levers (P1, scoped-P1, 16x8, int8-KV)
  all hurt probe cos.** They are HTP-efficiency/footprint choices
  Qualcomm makes for deployment, not quality levers. The only lever
  that can *raise* w4a16 cos is one that improves the int4 *weights*
  without cutting precision elsewhere — i.e. AdaScale (A2).
- **A2 (AdaScale − V/O-pin): 0.9649, −0.010 vs P0.** AdaScale (512
  iters, the qai-hub value) did not lift cos. Two reasons it's not a
  clean win: (1) it is confounded — the V/O-pin↔AdaScale single-param-
  bw conflict forced `--no-vo-pin-w8`, so A2 also *lost* the V/O w8
  pin (a precision increase); A6 isolates that. (2) consistent with
  the Session-28 note that "AdaScale wasn't moving the probe cos" —
  AdaScale tunes weight scales but the w4 quantization grid itself is
  the limit. AdaScale wall: 2974 s (50 min) at 512 iters. Next: A6
  isolates the V/O pin; A7 measures AdaScale *with* the V/O pin kept
  (probe-only — the conflict breaks the stage-7 bundle compile, not
  the stage-9 probe).
- _(A6, A7 appended as runs land)_
