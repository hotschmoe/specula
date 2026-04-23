# W4A16 on X2 Elite — path to a working binary

Phase 5.5 Lever C side-investigation. The pathb rotary-hoisted ONNX
compiles cleanly to w4a16 via AI Hub (job `jg93r1jqg`, SUCCESS at
6050s), but the resulting `.bin` fails at ORT-QNN runtime with a
tensor-size mismatch. This doc captures the bug, the full option
space, and the investigation order we decided on.

Target outcome: a working w4a16 draft binary we can sweep end-to-end
vs the 18.12 t/s fp16 AC baseline. W4A16 is Qualcomm's blessed
precision for X2 Elite HTP (see `results/ai_hub_model_zoo_check.md`
— every shipped X2E LLM uses w4a16; Llama-v3.2-1B showed w4a16 at
90.36 t/s vs w4 at 27.95 t/s, a ~3× hardware fast-path).

## The bug (AI Hub preserve_io_datatype drop)

From `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`:

- **qairt-converter invocation:** `--preserve_io_datatype` list has
  **116 names** — full set (56 past_kv + attention_bias +
  position_ids_cos/sin + 57 outputs). First name: `past_key_values.0.key`.
- **qairt-quantizer invocation (same log):** `--preserve_io_datatype`
  list has **115 names**. First name: `past_key_values.0.value`.
  `past_key_values.0.key` is silently dropped.

Consequence: layer-0 key is uint8-quantized at the IO boundary
while every other past_kv stays fp32. ORT wrapper declares fp32 (4
bytes/elem); the binary expects uint8 (1 byte/elem) → 4× byte-count
mismatch → `ExecuteGraph ORT Tensor data size does not match QNN
tensor data size`.

fp16 binaries are unaffected — the qairt-quantizer step is not
invoked when `--quantize_full_type float16`, so no PTQ preserve list
is constructed.

Verification commands (reproducible):

```bash
# Count unique past_kv names per invocation — 56 expected, 55 in quantizer
awk 'NR==51' jg93r1jqg.log | grep -aoE "past_key_values\.[0-9]+\.(key|value)" | sort -u | wc -l
awk 'NR==72' jg93r1jqg.log | grep -aoE "past_key_values\.[0-9]+\.(key|value)" | sort -u | wc -l
# Confirm which one is missing
awk 'NR==72' jg93r1jqg.log | grep -aoE "past_key_values\.0\.(key|value)"
```

## Why the pipeline is worth fighting for

The rest of the pipeline is validated:

| stage | status | evidence |
|---|---|---|
| optimum export | ✓ cos=1.0 | x86 session 13 probes |
| pathbmask rewrite (additive mask) | ✓ cos=1.0 | x86 session 1 |
| pathb rewrite (rotary hoisted) | ✓ cos=1.0 | x86 session 13, pos=0 + pos=5 probes |
| prep_onnx_for_ai_hub + ORT-BASIC fold | ✓ | shape-pin + dim_param resolve green on all three variants; ORT fold 7131→2054 nodes on pathb |
| AI Hub fp16 compile | ✓ | patha + pathbmask at ctx=512 and ctx=256 |
| AI Hub w4a16 compile (structural) | ✓ | rotary hoist cleared the op-validation failure that killed j563xme75; jg93r1jqg SUCCESS |
| AI Hub w4a16 preserve-list construction | ✗ | drops first preserve-worthy input |
| ORT-QNN EPContext runtime | ✓ for fp16 | pathbmask sweep hit 18.12 t/s AC |

One bug in Qualcomm's orchestrator, between two green stages. Every
other engineering artifact — x86 rewrites, calibration capture,
runtime cos/sin feed, EPContext wrapper builder — transfers to
whichever path we pick below.

## Option space (five paths, ordered by decided investigation order)

### 0. Side-quest — characterize Qualcomm's Qwen3-4B Genie w4a16 bundle

**Not a fix; a decision-input.** `qualcomm/Qwen3-4B` ships a w4a16
context binary for Snapdragon X2 Elite:
`https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b/releases/v0.50.2/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite.zip`

Three questions it answers (that change the choice among Options 1-4):

1. **Can ORT-QNN 1.24.4 load a fully-quantized-IO binary at all?**
   Qualcomm's bundle uses uint8 past_kv + uint16 attn-mask +
   uint16 cos/sin per its `metadata.yaml`. If ORT-QNN can load this
   bundle (same QAIRT 2.42 binary format) and we build a wrapper
   declaring those dtypes, we validate the full-quant-IO path.
2. **What's the real w4a16 t/s ceiling on X2E?** The Phase 5.5
   Lever C throughput goal (~18→30 t/s stacked with A+B) implicitly
   assumes the Qualcomm-published ~3× w4a16 fast-path applies.
   Qualcomm's Qwen3-4B running on the same silicon calibrates that
   assumption.
3. **Is `--preserve_io_datatype fp32` a perf foot-gun?** Qualcomm's
   reference uses full quant IO for a reason — per-call memory
   bandwidth. If Qualcomm's w4a16 numbers are dramatically higher
   than a preserved-IO compile would be, all workaround options
   that keep IO at fp32 are band-aids.

**Cost:** ~2-3 GB download + ORT-QNN load attempt (~30 min). Poke
at its `metadata.yaml` (already on disk at
`models/qualcomm-qwen3-4b-ref/.../metadata.yaml`) for quant-param
structure reference. Optionally run a decode step via Genie SDK
for a t/s number.

**Kill criteria for moving on:** after we know whether ORT-QNN
loads the Qualcomm binary and what t/s it produces, side-quest
is done. Findings captured in `results/qwen3_4b_genie_w4a16_probe.md`.

### 1. AIMET-side pre-quantization (Option 3a in the picks-list)

**Use AIMET on x86 to emit a pre-quantized ONNX, then AI Hub just
transcodes.** `--quantize_full_type float16` (no PTQ step), so the
buggy preserve-list code path never runs.

- **Mechanism:** AIMET's `QuantizationSimModel` walks the ONNX,
  inserts QuantizeLinear/DequantizeLinear (QDQ) pairs with per-tensor
  or per-channel scales/offsets chosen from calibration. Save as
  ONNX+encodings; AI Hub accepts pre-quantized ONNX natively.
- **Pro:** we own the quantization choices end-to-end. Every
  tensor's dtype/scale/offset is explicit in our ONNX. AI Hub is
  just a transcoder to QNN context binary.
- **Pro:** bypasses the preserve-list bug entirely.
- **Pro:** reusable for Qwen3.5 graduation (same AIMET recipe works).
- **Con:** AIMET install + learning curve (~0.5–1 session).
- **Con:** AIMET's QDQ-ONNX format has to match what AI Hub's
  converter accepts — novel debug surface, one ghost cycle likely.
- **Con:** we lose AI Hub's automatic activation-range calibration
  (we do it ourselves via AIMET).

### 2. Local QAIRT toolchain (Option 3b)

**Install QAIRT 2.42 SDK on x86 (we have a dedicated compile box),
run qairt-converter + qairt-quantizer + qairt-context-binary-generator
ourselves. Skip AI Hub entirely.**

- **Mechanism:** QAIRT SDK is the same toolchain AI Hub runs server-
  side. We pass `--preserve_io_datatype` ourselves (correctly), with
  full list control.
- **Pro:** full pipeline ownership. No AI Hub orchestration bugs
  can touch us.
- **Pro:** dedicated x86 box → compile time isn't a session-cost
  concern.
- **Pro:** turns every future re-quant (Qwen3.5, different
  calibration bundles, ablations) into a local fast loop.
- **Con:** one-time SDK install + tool learning. Probably ~1
  session to get a fp16 binary landing, another for w4a16.
- **Con:** we're on our own for bugs (no Qualcomm support).
- **Attractive partner to Option 1:** AIMET emits QDQ ONNX; local
  QAIRT converts to QNN context binary. Composing both gives us
  total control.

### 3. Sacrificial preserve-guard via ONNX surgery (cheap workaround)

**Prepend a dummy float32 graph input that occupies the "first
preserve-worthy" slot so the bug eats the dummy instead of
past_kv.0.key.**

- **Mechanism:** x86 rewrite script adds input `_aa_preserve_guard`
  shape `[1, 1]` float32, wired through a no-op like
  `attention_bias = attention_bias + 0 * _aa_preserve_guard` so
  it's structurally alive (AI Hub warns on unused inputs; we don't
  know if unused inputs get dropped from the preserve list).
- **Pro:** ~0.5 session x86 surgery + 100 min AI Hub retry.
- **Pro:** if the bug hypothesis holds, this just works.
- **Con:** **N=1 evidence base.** We've observed the bug once. The
  drop could be "first item" (hypothesis), or "item at a specific
  index", or driven by some graph-input property we haven't
  inspected. If wrong, another 100-min compile wasted.
- **Con:** fragile — relies on an observed-but-unexplained AI Hub
  behavior. A future AI Hub release that fixes the bug could
  silently break our workaround (guard input survives quantization
  incorrectly).

### 4. ORT-side uint8 quant of past_kv.0.key only (narrow runtime patch)

**Keep the buggy binary; patch our wrapper to declare
`past_key_values_0_key` as uint8 and do per-call quant/dequant.**

- **Mechanism:** extract the scale/offset AI Hub baked in for that
  tensor, update `describe_inputs` to declare uint8 with a matching
  QuantizeLinear wrapping op, update runtime to quantize layer-0
  key on feed and dequantize its present output (`output_1`).
- **Pro:** no re-compile. Works with the binary we already have.
- **Pro:** ~50–100 LOC runtime-only change.
- **Con:** **getting the scale/offset is the hard part.** The
  `.bin` is an opaque QNN context binary. Options:
  - Re-upload to AI Hub with verbose options to try to get the
    intermediate DLC back (unclear if the API supports it).
  - Install local `qairt-dlc-to-json` (implies doing Option 2
    anyway).
  - Guess from our calibration: compute min/max of
    past_kv.0.key samples in `bundle_a_pathb_ctx256.npz`, pick
    a scale/offset. Unlikely to match AI Hub's choice exactly
    → quant error + likely failed accept-gate.
- **Con:** brittle — if a future compile quantizes a DIFFERENT
  tensor due to the same bug (or a different bug), we're patching
  one-off. Does not scale.
- **Con:** doesn't address the perf question (does fp32 IO vs
  quant IO matter for speed?) that the side-quest surfaces.

## Decided investigation order (2026-04-22)

Agreed with the user: do them in this order, commit findings
between each, kill on whatever works.

1. **Side-quest first** — Qualcomm Qwen3-4B Genie w4a16 bundle.
   Findings go to `results/qwen3_4b_genie_w4a16_probe.md`. Budget
   ~0.5 session. Answers whether ORT-QNN can load a full-quant-IO
   binary and sets the perf calibration we measure against.

2. **Option 1 — AIMET pre-quant** (investigation option 3a). If
   the side-quest shows full-quant-IO runs in ORT-QNN and is
   meaningfully faster than preserved-IO, this becomes the primary
   path — we emit a pre-quantized ONNX with the same IO dtype
   convention Qualcomm's reference uses.

3. **Option 2 — local QAIRT toolchain** (investigation option 3b).
   Most attractive long-term because we own the whole pipeline and
   AI Hub orchestration bugs can't touch us. User has a dedicated
   x86 compile box so per-compile wall time isn't a blocker.
   Composes nicely with Option 1 (AIMET emits QDQ ONNX → local
   QAIRT produces the context binary).

4. **Option 3 — sacrificial preserve-guard surgery.** Cheap, but
   N=1 on the bug hypothesis. Falls out naturally if everything
   above fails.

5. **Option 4 — narrow runtime uint8 patch.** Last resort because
   it's brittle and doesn't scale. Only if all else fails and we
   just need *any* w4a16 binary working for a single benchmark.

**Kill-switch on w4a16.** If all four options fail, close Lever C
negative. Lever B's 18.12 t/s AC baseline stands as Phase 5.5's
high-water mark; the finding itself (rotary hoist works but AI
Hub's quantizer orchestration mis-formats the preserve list) is
publishable and filed upstream. Strong prior that we'll get it
working before falling through.

## Update 2026-04-22 (session 15) — Option 2 executed, w4a16 binary loads

Picked Option 2 (local QAIRT) per the decided order. Outcome: **the
Phase 5.5 Lever C runtime block is cleared.** The w4a16 binary now
loads on ORT-QNN 1.24.4 and runs a zero-feed forward pass in 30.66 ms
(structural validation only; correctness gate still ahead).

### x86 compile path (handoff-doc companion)

- Plan: `docs/phase5_local_qairt_compile.md`.
- Findings: `docs/phase5_local_qairt_compile_findings.md` — five
  deviations from the plan, all documented with causes, none blockers.
- First try used QAIRT **2.45.40** (the only SDK on the x86 box at the
  time). Built in ~60 seconds. Binary handed off to
  `Z:\exposed\junk\phase5_step15_local_qairt_out\`.
  - ARM64 load: `LoadCachedQnnContextFromBuffer Error 5000` — Qualcomm's
    compat matrix is explicit that 2.43+-built binaries are not
    backward-compatible with 2.42 runtime (`docs/QAIRT-Docs/QNN/general/htp/htp_backend.html`).
- Escape-hatch probe: ORT-QNN **2.1.0** in isolated `.venv-ort21/`
  (bundles QAIRT 2.45.40, matches binary). Three load attempts, all
  fail (summary: `results/preflight_w4a16_local_ort21_summary.md`):
  - legacy `providers=[(...)]` → silent CPU fallback (2.x is plugin-EP)
  - plugin-EP + `disable_file_mapped_weights=1` → option not honoured
    by the EP; crash at same file-mapping warning
  - plugin-EP + `embed_mode=1` → farthest yet, past "Disabling file
    mapping for this node", crash in `LoadCachedQnnContextFromBuffer`
  - Confirms the existing ORT-side bug catalogued in
    `docs/npu_ort_qnn_version_match.md`. Revisit when 2.1.1 / 2.2.x ships.
- Second try: x86 downloaded QAIRT **2.42.0.251225** from
  `https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip`
  (Qualcomm gateway 403s on HEAD / curl UA — use `curl -sL -A "Mozilla/5.0"`).
  Required `onnx<1.15` downgrade (2.42 tools import removed
  `onnx.mapping`) and `shared_library_path` repointed to the 2.42
  extensions DLL. Rebuilt in ~72 seconds. New NAS drop at
  `Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242\`.

### ARM64 runtime wiring

- Variant-aware schema in `scripts/npu_load_qwen3_bin.py`:
  `SPECULA_NPU_VARIANT=w4a16-local` routes `describe_inputs` /
  `describe_outputs` through `_describe_*_pathb_w4a16_local` with
  **60 inputs** (position_ids dropped by x86's `--remove_unused_inputs`),
  all quantized IO as `TensorProto.UINT16`, only `input_ids` as int32.
- `LOGITS_OUTPUT_NAME` now conditional: the local-compile pipeline
  did NOT apply qairt-converter's output-rename pass, so the binary
  exposes `logits` / `present_N_{key,value}` instead of
  `output_0..output_56`.
- `build_zero_feed` extended with `tensor(uint16)` / `tensor(uint8)`
  dtype entries for smoke-test purposes.

### Naming gotcha — dots vs underscores

`qnn-context-binary-generator` normalises dotted tensor names to
underscored when emitting the `.bin`, **even though the intermediate
DLC and `qairt-dlc-to-json` / `dlc-info` keep the dots**. So:

- DLC info text + `encodings.json` → `past_key_values.0.key`,
  `present.0.key`
- Binary's internal graph IO map → `past_key_values_0_key`,
  `present_0_key`
- ORT-QNN binds by literal name, so the wrapper ONNX must use the
  underscored form or `GetGraphInputIndex` fails with "Input name not
  found".

Verified by scanning printable strings in the `.bin` tail. First
ARM64 load attempt after the 2.42 rebuild hit exactly this mismatch;
underscore-swap in `_describe_inputs_pathb_w4a16_local` /
`_describe_outputs_pathb_w4a16_local` fixed it.

### Preflight result

```
session providers   : ['QNNExecutionProvider', 'CPUExecutionProvider']
inputs (60): input_ids int32, past_key_values_*_{key,value} uint16,
             attention_bias uint16, position_ids_{cos,sin} uint16
outputs (57): logits uint16, present_*_{key,value} uint16
run latency       : 30.66 ms   (first call; includes warmup)
logits shape      : (1, 1, 151936)  finite, non-constant
```

Log: `results/preflight_w4a16_local_qairt242_v2.stdout`.

### QuantSpec runtime layer — DONE

Added `QuantSpec` dataclass + `load_quant_specs` / `quant_to_uint16`
/ `dequant_from_uint16` / `quantized_zero` to `npu_load_qwen3_bin.py`.
Encodings-name translation (binary uses underscored names, encodings
JSON keeps dotted) is handled in `_dot_form`. The smoke-test feed
gets the quant-zero uint16 per-tensor instead of literal uint16=0,
and logits are dequanted for reporting.

Quant formula (verified against every declared min/max in
`dlc_info_w4a16.txt`):

```
x_fp32   = (q_uint16 + offset) * scale
q_uint16 = clip(round(x / scale) - offset, 0, 65535)
```

Round-trip diagnostic (`scripts/probe_w4a16_quant_roundtrip.py`) on
fib-p0: worst per-tensor RMS error 0.001%, zero out-of-range
clipping on all 56 past_kv + attention_bias + cos + sin. Quant layer
is numerically healthy.

### Correctness probe — FAILS with cos≈0.29–0.33 (not a quant issue)

`npu_short_prompt_probe.py --path pathb` extended to accept
optional `quant_specs`; position_ids dropped from feed (not an
input in this binary), fp32 inputs quantized at the session
boundary, uint16 logits dequanted on return, KV chain dequant-
and-requanted between steps.

Results:

| scenario | cos | argmax match | top-5 overlap | multi-step |
|---|---:|:-:|:-:|:-:|
| pathbmask fp16 (reference, same probe) | 0.9999 | ✓ | 5/5 | 100% |
| **pathb w4a16-local, fib-p0 (prompt_len=16)** | **0.33** | ✗ | 1/5 | 0% |
| **pathb w4a16-local, pos=0 identity rotary, BOS-only** | **0.29** | ✗ | 0/5 | n/a |

The pos=0 identity-rotary case is the decisive isolator. At pos=0,
`rope_tables(0)` produces cos=all-1.0, sin=all-0.0 so rotary becomes
an identity. Past_kv is zero. Attention_bias masks all past slots.
Only `input_ids=1` at position 0 matters. Even this trivial case
hits cos=0.29 vs the optimum CPU reference.

What this rules out:

- Probe infrastructure regression — pathbmask still 0.9999 through
  the same code.
- Quant formula — round-trip 0.001% RMS, every fp32 input fits
  inside its calibrated range.
- rope_tables formula — pos=0 makes rotary identity; formula is
  bypassed.
- Rotary hoist equivalence — x86's `probe_pathb_equivalence.py`
  already validated NEW-ONNX vs REF-ONNX at cos=1.0 on pos=0 + pos=5.
- Wrapper schema — 60 uint16 inputs + `input_ids` int32 match the
  binary exactly; pos=0 probe feeds every required tensor.

What this points at (in the x86 compile pipeline):

1. **PTQ calibration picked bad activation ranges on some interior
   tensor.** encodings.json captures IO scale/offset; the ~2200
   internal activation tensors each have their own calibration-
   derived scale that we can't inspect cheaply from ARM64.
2. **qairt-converter or qairt-quantizer did something subtly wrong
   on the hoisted-rotary graph.** Compare-and-contrast: pathbmask
   (rotary inline) works, pathb w4a16 (rotary hoisted, quantized)
   doesn't. Two things differ simultaneously.
3. **A specific op's HTP kernel behaves differently under w4a16
   quant than we'd expect from fp16.** QAIRT's HTP quant has known
   corner cases around Neg/StridedSlice (rotate_half lowering) and
   RmsNorm.

Diagnostics used:

- `scripts/probe_w4a16_quant_roundtrip.py` — per-tensor quant round
  trip on real fib-p0 feed.
- `scripts/probe_ort21_w4a16_local.py` — escape-hatch attempt on the
  2.45 binary (negative, preserved for future re-runs).
- Binary string-scan — confirmed the dot→underscore name
  normalisation `qnn-context-binary-generator` applies.
- pathbmask sanity rerun via the same probe — green.

### Next-session options (ordered by cheapness)

1. **x86 fp16-pathb rebuild.** Re-run the local QAIRT pipeline with
   `--float_bitwidth 32` (no PTQ) on the same pathb ONNX. If fp16-
   pathb passes cos ≥ 0.95, the compile pipeline is correct and the
   issue is specifically w4a16 PTQ. If fp16-pathb also fails, the
   pathb rewrite or prep pipeline changed something a CPU probe
   wouldn't catch. ~80 seconds of x86 compile time.
2. **Different PTQ algorithm.** `qairt-quantizer` supports
   `--act_quantizer {tf,tf_enhanced,cle}`, `--bias_bitwidth`,
   `--use_adjusted_weights_quantizer`, etc. Default was tf. Try
   `tf_enhanced` (per-tensor range enhancement) or `cle`
   (cross-layer equalization) as the most likely accuracy-move-up
   knobs. Each is one x86 recompile.
3. **Calibration bundle swap.** Our current bundle is Bundle A
   (realistic multi-position). Try Bundle B (step-0 only, 20
   samples). The research question from the perf-levers plan was
   "does cheap calibration suffice?" — this session inadvertently
   started answering no, at least for realistic samples.
4. **AIMET-side pre-quantization.** Explicit per-tensor activation
   range control — heavier but gives us a knob on the interior
   activations that encodings.json only reports post-hoc.
5. **qairt-accuracy-debugger.** Linux-only QAIRT tool that compares
   per-op outputs between the DLC and a reference runtime. If the
   user has Linux access on the x86 box or WSL2, this is the
   definitive localiser for "which op went numerically wrong."

### Standing evidence + artifacts (updated)

## Standing evidence + artifacts

- Compiled binary (AI Hub, broken): `models/qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-a.bin`
- AI Hub log (bug evidence): `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`
- Compiled binary (x86 QAIRT 2.45, rejects on ORT-QNN 1.24.4): NAS at
  `Z:\exposed\junk\phase5_step15_local_qairt_out\`
- Compiled binary (x86 QAIRT 2.42, **loads but numerically wrong**):
  `models/qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin`
  + `.encodings.json`. Loads on ORT-QNN 1.24.4 cleanly, 28 ms/step,
  logits finite but cos≈0.3 vs CPU fp32.
- Calibration bundle (reusable): `models/calibration/bundle_a_pathb_ctx256.npz`
  (60 samples × 61 inputs, 3.27 GB)
- pathb ONNX (reusable): `models/qwen3-0.6b-pathb/` — 61 inputs,
  rotary hoisted, cos=1.0 vs optimum source
- X2E plumbing (reusable): commit `1423f6c`; variant-aware w4a16 plumbing commit `0357aa6`
- Qualcomm reference metadata: `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/metadata.yaml`
- Runtime diagnostics: `scripts/probe_w4a16_quant_roundtrip.py`,
  `scripts/probe_ort21_w4a16_local.py`
- Correctness evidence: `results/correctness_w4a16_local_p0.stdout`
  (cos=0.33 fib-p0), `results/preflight_w4a16_local_qairt242_v3.stdout`
  (structural load ok)

## Open questions

These get answered by the investigation, not up front:

- **Does ORT-QNN 1.24.4 accept a graph whose inputs are uint8/uint16
  quantized?** Our current wrapper declares everything as FLOAT
  (float32). If ORT-QNN supports TensorProto.UINT8 / UINT16 inputs
  with explicit scale/offset in an EPContext wrapper, Option 1/4 are
  unlocked.
- **Does AI Hub accept an AIMET-QDQ-annotated ONNX without running
  its own PTQ?** If yes, Option 1 (AIMET) is clean. If it always
  tries to re-quantize, the pre-quant step is wasted and we need
  Option 2 (local QAIRT).
- **Is the AI Hub preserve-list bug deterministic?** A quick
  probe — recompile with a different input ordering and see if
  the dropped name changes — could confirm the "first item"
  hypothesis and make Option 3 (guard) safe.
- **What scale/offset does AI Hub pick for past_kv tensors?** Per-
  tensor asymmetric uint8 (offset -128 per Qualcomm's reference)
  vs per-channel symmetric int8 — we don't know AI Hub's default.

Each answered question either unlocks the next option or eliminates
one, so the investigation converges monotonically.
