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
| A2 | + AdaScale | _TBD_ | | needs V/O-pin↔AdaScale conflict resolved |
| A3 | + AdaScale + P2 | _TBD_ | | |
| A4 | + scoped-P1 (16x8 + int8-KV, **no** int8-lmhead) | _TBD_ | | full-P1 hurt — see findings |
| A5 | + AdaScale + P2 + scoped-P1 | _TBD_ | | kitchen sink |

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
- _(rows A2-A5 appended as runs land)_
