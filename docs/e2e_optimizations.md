# e2e_optimizations.md — closing the quantization quality gap

Living plan for taking the `end-to-end/` pipeline from *structurally
correct* to *quality-correct*: a quantized NPU bundle whose first-decode
logits match the FP model (cos ≥ 0.99) and that reproduces Qualcomm's
shipping Qwen3-4B bundle.

Status started: 2026-05-21 (Session 29). See `current_status.md` for
session-level progress; this doc holds the plan and the experiment
matrix.

## North star

1. **Replicate Qwen3-4B exactly.** Qualcomm ships a known-good
   `qwen3_4b-genie-w4a16` bundle (`reference/qwen3_4b_qualcomm/`).
   Reproducing it — same structure *and* same output quality — proves
   the pipeline.
2. **Generalize the pipeline.** Everything must stay model-parametric
   (layer count, hidden size, num_parts, ctx) so it carries forward
   without per-model surgery.
3. **The holy grail: Qwen3.6-27B on the NPU.** 27B is the real target.
   4B is the rehearsal — nail the recipe at 4B, scale the same
   pipeline up. 27B implies more split parts, larger ctx, tighter
   memory; the pipeline must already be clean before we get there.

## Source of truth

**`qai-hub-models`** is the authority. Qualcomm's shipped bundle was
produced by it; its Qwen3 recipe (export path, quantsim config,
calibration, op handling) *is* "the Qualcomm recipe". When our recipe
and qai-hub-models disagree, qai-hub-models wins until proven
otherwise.

Two reference artifacts on disk:
- `reference/qwen3_4b_qualcomm/…/` — the shipped 4-part genie bundle +
  `metadata.json` (per-part I/O, dtypes, scales).
- `qai-hub-models` (pulled per `end-to-end/COLD_START.md`) — the recipe
  source.

Comparison tooling: `end-to-end/compare_to_qualcomm.py` already does
the structural/file-system audit. We need a **numerical** companion —
see Track 0.

## The quality gap (diagnosis recap)

Probe cos today: 0.6B w8a16 0.56, 4B w8a16 0.44, 4B w4a16 0.51 — gate
is 0.99. Root cause (Session 28, confirmed):

transformers 4.51 exports RMSNorm **decomposed** (Pow / ReduceMean /
Add / Sqrt / Div / Mul). AIMET QuantSim then puts an activation
quantizer on every intermediate — including the `x²` tensor (dynamic
range ~5e6, the residual stream is huge) and internal causal-mask
constants (~4e37). int16 per-tensor over a 5e6 range → ~150-unit
granularity → small values annihilated → model destroyed.

Proven levers:
- Disable **all** activation quantizers → cos 0.99 (so it is purely an
  activation-quant problem, not weights, not precision).
- fp16 for the offending tensors → cos 0.9994 — **but HTP-incompatible**:
  every fp16 quantizer becomes an `fp32→fp16 QNN_Convert` the HTP
  graph-prep rejects.

Key insight (Session 29): Qualcomm's `metadata.json` boundary tensors
are plain **uint16 asymmetric per-tensor** (+ uint8 KV) — nothing
exotic. Qualcomm does not win on activation precision; they win by
**not quantizing normalization internals at all** (HTP runs them in
float). And the qairt-converter log already shows HTP doing
`OPs fallback to float` for some ops — **float fallback is HTP-native
and needs no convert op.** That is the seam to exploit.

## Optimization tracks

Ordered by expected value / effort. Each track is an experiment with a
measured cos delta, validated 0.6B-w8a16 first then 4B.

> **Update (Session 29 — Track 4 landed).** The qai-hub-models diff
> (`docs/qai_hub_recipe.md`) found the root cause and the principled
> fix: we built QuantSim with `config_file=None`; Qualcomm passes
> `default_config_llama.json` (now vendored into the repo), whose
> `RMSNormalization` supergroup pass auto-disables the norm-internal
> activation quantizers — and brings the LayerNorm pass, the op-type
> exclusion set, and Softmax/Sigmoid range constraints with it. This
> **collapses Tracks 1, 2 and 3 into one change**: adopt that config
> file (the **P0** fix). Track 3 (a hand-written RMSNorm fusion pass)
> is no longer needed — AIMET's pass already does it. Track 5
> (percentile observer) is dropped — the diff showed our observer
> config (`min_max` + SEQ_MSE) already matches Qualcomm. The remaining
> work is the P1-P4 refinements; see `docs/qai_hub_recipe.md` §(c).
>
> - **P0** — adopt `default_config_llama.json` + the Concat-tie /
>   Slice-Constant-ignore quantsim flags. Expected: cos 0.5 → ~0.99.
> - **P1** — int8-symmetric, in/out-tied KV cache + 16x8 matmuls
>   (`_set_matmul_second_input_to_8b`); likely removes the need for
>   our `_bump_vo_to_w8` workaround.
> - **P2** — clip the additive attention mask to `[-100, 0]` (we
>   currently feed `-65504`, which wrecks int16 granularity nearby).
> - **P3** — AdaScale `NUM_RMSNORM_PER_BLK = heads + kv + 1` (=41 for
>   4B), only if AdaScale is re-enabled.
> - **P4** — verify calibration data is real text, not random tokens.

### Track 0 — quality baseline harness (do first; everything needs it)
Without a trusted metric the other tracks are guesswork.
- Numerical baseline: FP-vs-quant first-decode logit cosine + argmax
  on a fixed oracle prompt set, per stage (post-AIMET probe is the
  cheap proxy; on-device logits are the real thing, X2E-only).
- Wire it to compare against the Qualcomm reference where possible.
- Extend / complement `compare_to_qualcomm.py`.

### Track 1 — selective fp32 unquantize of catastrophic tensors
Highest value, ~½ day. The fp16 attempt failed only because fp16
forces a `QNN_Convert`. **Omitting the encoding entirely** leaves the
op in HTP's native float fallback — no convert, HTP-compatible.
`lib/aimet.py` already identifies the ~53 offending tensors (the
reverted fp16 code). Change: disable those activation quantizers
(→ fp32 fallback) instead of converting them. Expected: cos → ~0.99.

### Track 2 — align AIMET QuantSim op-config with QAIRT HTP
Make Track 1 principled. AIMET's QuantSim takes a config JSON deciding
which op types get activation quantizers. Derive an op-type-based
config (exclude RMSNorm-internal Pow/ReduceMean/Sqrt/Div, mask
Where/ConstantOfShape) that matches what HTP expects — instead of a
range threshold.

### Track 3 — RMSNorm fusion in the pathb rewrite
Most robust. Add a pathb rewrite pass collapsing the
Pow/ReduceMean/Add/Sqrt/Div/Mul cluster into a single normalization
op — then there are no internal intermediates to quantize and the
graph shape matches Qualcomm's. More work; supersedes Tracks 1-2 if it
lands.

### Track 4 — diff against qai-hub-models recipe
Pull qai-hub-models' Qwen3 quantsim config + export settings and diff
against ours field by field. Adopt every delta. This is the
ground-truth check on Tracks 1-3.

### Track 5 — activation observer
SEQ_MSE forces `min_max` (outlier-dominated) for the weight search.
Activations could use a **percentile** observer (e.g. 99.99%) to clip
outliers. Lower priority — likely subsumed by Tracks 1-3, but cheap to
A/B once the harness exists.

## ctx-length (and ar) sweep

Qualcomm ships `ar{1,128} × cl{512,1024,2048,3072,4096}`. Our pathb
pins `ctx=512`, `ar=1`. Two axes:
- **ctx sweep** — build Qwen3-4B bundles at ctx 512/1024/2048/3072/4096
  via the `--ctx` flag. Confirms the pipeline scales parametrically;
  larger ctx grows the KV cache + mask. Needed for 27B (long context).
- **ar128 prefill** — Qualcomm ships an ar128 prefill graph alongside
  ar1 decode. We are ar1-only. Real deployment needs prefill; scope
  this as a follow-on once the ar1 recipe is nailed.

## Model & precision test strategy

- **0.6B w8a16 — fast iteration only.** ~20 min/run. Use it to smoke-
  test whether a recipe change moves cos. **Caveat:** a 0.6B model is
  more sensitive to quantization (less redundancy) — do *not* treat
  0.6B w4a16 numbers as representative, and don't gate on 0.6B
  absolute quality. The RMSNorm-decomposition bug is size-independent,
  so 0.6B w8a16 is a valid *directional* proxy.
- **Qwen3-4B — the replication anchor.** Both w8a16 and w4a16 (Qualcomm
  ships w4a16). The recipe is "done" when 4B matches the Qualcomm
  reference. This is the gate before scaling.
- **Qwen3.6-27B — later.** Only after 4B is locked. Validates split
  part-count scaling, memory, and long ctx.

## Success criteria

1. 4B w4a16 post-AIMET probe cos ≥ 0.99, argmax matches FP.
2. `compare_to_qualcomm.py` structural diff vs the reference bundle is
   clean (part count, dtypes, KV/mask layout).
3. Bundle compiles end-to-end (split → qairt → qnn) — already true.
4. On-device (X2E) first-decode logit cos ≥ 0.99 — final gate, needs
   hardware.
5. ctx 512→4096 all build from the one parametric pipeline.

## Baseline vs Qualcomm — structural (Session 29)

`compare_to_qualcomm.py` on our w4a16 4-part bundle vs the shipped
reference (both w4a16):

- **Matches:** part1 .bin exact (778 MB == 778 MB), parts 2-3 within
  3.8%, every `genie_config.json` / `htp_backend_ext_config.json`
  field, tokenizer sha256. The pipeline is structurally sound.
- **Deltas:** `ctx` 512 vs 4096 — our build param, not a defect (the
  ctx sweep covers it). **part4 +38%** (1475 vs 1070 MB) — part4
  carries the lm_head; ours is int16 (the `_pin_embedding_w16` pin),
  Qualcomm's is int8 (`_set_lm_head_to_8b`). **P1 closes this** — it
  is both a quality and a size fix.
- Total 3642 vs 3186 MB (+14%), almost all of it the part4 lm_head.

Re-run after P0/P1 land to confirm convergence; numerical (on-device
cos) parity still needs X2E hardware.

## Task index

See the session task list. Tracks 0→4 are sequential-ish (0 first,
then 1, with 2-4 refining); Track 5 and the ctx sweep run alongside
once the harness exists.
