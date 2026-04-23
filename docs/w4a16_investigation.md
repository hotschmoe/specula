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

### Update (session 17, 2026-04-22) — A.2 tfe barely moves; differential localises to V-projection weights

x86 shipped `w4a16-local-tfe` at
`Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_tfe\` (MD5
`96667934cbf9dfdcbddf2f1fe93f13a9`). 2.42 calls the option `--act_quantizer
enhanced`, not `tf_enhanced` (see `HANDOFF_tfe.md` §"Terminology
correction"). Runtime contract identical to w4a16-local; refactored
our IS_LOCAL_COMPILE dispatch to pattern-match (`"-local" in VARIANT`)
so any `w4a16-local-*` suffix flows through the same schema.

Correctness probe on fib-p0:

| variant | cos vs CPU | argmax | top5 | multi-step |
|---|---:|:-:|:-:|:-:|
| fp16-local | **0.9999** | ✓ | 5/5 | 100% |
| w4a16-local (tf) | 0.33 | ✗ | 1/5 | 0% |
| **w4a16-local-tfe (enhanced)** | **0.36** | ✗ | 1/5 | 0% |

cos shifted 0.33 → 0.36. Enhanced activation calibration is doing
something measurable but nowhere near the 0.95 gate. Prompt-1 also
ran: cos=0.606 (tfe). So the error is prompt-dependent (activation
distribution matters) but dominated by a deeper issue than activation
range.

**Differential probe (`scripts/probe_w4a16_vs_fp16_differential.py`)
— same feed, fp16-local vs w4a16-local-tfe binaries side by side, walked
per-layer present_N_{key,value}:**

```
layer    key_cos  value_cos   k_maxabs   v_maxabs
    0   0.985804   0.957463     50.149      0.125
    1   0.982467   0.130328     35.275      0.602
    2   0.968868   0.069804     47.292      0.712
    3   0.881869  -0.001491     11.599      2.928
    ...
    8   0.226160   0.102045     23.842      3.194
   21   0.295060   0.113408     24.617     26.873
   27   0.446465   0.181096     18.297     45.679
```

Key tensors degrade gracefully from cos=0.99 at layer 0 to ~0.45 at
layer 27. **Value tensors COLLAPSE at layer 1** (0.957 → 0.130) and
stay near-random (0.0 to 0.18) for every subsequent layer. V-tensor
absolute range also explodes — layer-0 max 0.125, layer-27 max 45.6
— a ~350× dynamic-range growth across depth.

Value projection is a pure `W_v × x` linear projection with no rotary
folding. Keys survive better because rotary + MatMul smooths some
error. Values don't. **If layer-1+ V-projection weights can't be
represented cleanly at w4 precision, every downstream value tensor is
garbage and accumulated error corrupts the logits.** This is a known
failure mode for low-bit quantization of attention V/O projections,
especially in small (0.6B) models where each layer matters.

Enhanced activation calibration would not address this — it tunes
ranges on activations that the W_v × x MatMul emits, but if W_v is
already quantized too aggressively, no activation range tweak helps.

**Implication for the decision tree:** the remaining options split
into weight-precision fixes (new primary) and activation fixes
(secondary):

- **Weight precision:** A.6 w8a16 (all weights 8-bit — brute sanity),
  A.5-style per-tensor overrides targeting V and O projections
  specifically, or A.4 CLE (redistributes weight magnitudes but
  doesn't raise bitwidth).
- **Activation fixes:** A.3 Bundle B, A.2-alt 2.42 calibrators
  (`sqnr` / `mse` / `percentile`). Lower expected impact given the
  diagnosis; park unless weight-side fixes don't land.

Updated x86 ask in `docs/phase5_lever_c_x86_ask.md` — A.6 w8a16 is
now primary, A.4 CLE stays backup, A.5 per-tensor overrides stacked
for after the w8a16 result.

### Update (later, same day) — A.1 result: fp16-pathb is fully correct

x86 shipped the fp16-pathb rebuild at
`Z:\exposed\junk\phase5_step15_local_qairt_out_fp16\`. See
`docs/phase5_local_qairt_compile_findings.md` §"fp16-pathb rebuild (A.1
isolator)" for the pipeline.

Runtime wiring extended to support both local variants under a single
`IS_LOCAL_COMPILE = VARIANT in ("w4a16-local", "fp16-local")` flag.
Shared `_describe_inputs_pathb_local(cfg, dtype)` /
`_describe_outputs_pathb_local(cfg, dtype)` route the schema with
`UINT16` for w4a16-local / `FLOAT` for fp16-local. Probe's position_ids
skip + `present_N_*` output-name lookup gated on `IS_LOCAL_COMPILE`
(previously only quant-specs-bearing w4a16-local).

Probe result on fib-p0, `SPECULA_NPU_VARIANT=fp16-local`:

```
argmax match      : True  (cpu=262, npu=262)
top-5 overlap     : 5 / 5
cosine sim        : 0.999959
max |delta|       : 0.1404
multi-step match  : 100% (3/3)   stream: '    if n'
npu step latency  : 64.7 ms      (vs w4a16-local 28-30 ms)
```

**Full green.** Same probe through the same code returns cos=0.9999
for fp16-pathb; for w4a16-pathb-local it still returns cos=0.33. The
branch is therefore:

- Pathb rewrite: correct. ✓
- prep_onnx_for_ai_hub + ORT-BASIC fold: correct. ✓
- qairt-converter on pathb: correct. ✓
- qnn-context-binary-generator on fp16 pathb: correct. ✓
- qairt-quantizer w4a16 PTQ: **breaks numerical correctness**. ✗

D.1 (rewrite pathb with rotary inline) is **eliminated** — not needed.
Next action is A.2 (tf_enhanced), A.3 (Bundle B), A.4 (per-tensor
overrides), or C.1/C.2 depending on results.

Bonus observation — the fp16-pathb binary runs at ~65 ms/step, same
ballpark as fp16 pathbmask (no latency win from the rotary hoist
alone). So fp16-pathb is not a usable product deliverable; it was a
diagnostic. The throughput payoff of Lever C still depends on
landing a correct w4a16 binary, which the ~2× latency gap to
w4a16-local (28-30 ms) confirms is the real target.

Current x86 ask: `docs/phase5_lever_c_x86_ask.md` §"Follow-up asks" —
A.2 `tf_enhanced` PTQ variant is the cheapest next attempt.

### Next-session options — decision tree

Grouped by who owns the work. Cost = wall time to get the answer.
Cut branches as they go negative.

**Active ask to x86 team:** see `docs/phase5_lever_c_x86_ask.md` for
the focused one-pager handoff with exact commands + NAS drop paths.

#### A. x86-side rebuilds (primary path — these answer "what does the compile pipeline do wrong?")

- **A.1 fp16-pathb rebuild** — re-run local QAIRT with
  `--float_bitwidth 32` (no PTQ) on the same pathb ONNX. **Decisive
  isolator:** if fp16-pathb passes cos ≥ 0.95, w4a16 PTQ is the
  culprit (proceed to A.2–A.4); if fp16-pathb also fails, the pathb
  rewrite or prep pipeline changed something a CPU-level probe
  missed (skip to D.1).
  Cost: ~80 s of x86 compile time. **Do this first.**
- **A.2 PTQ algorithm variants.** `qairt-quantizer` supports
  `--act_quantizer {tf,tf_enhanced,cle}`, `--bias_bitwidth`,
  `--use_adjusted_weights_quantizer`. Default on our run was `tf`.
  Try `tf_enhanced` (per-tensor range enhancement) and `cle`
  (cross-layer equalisation) — either can rescue accuracy on a
  transformer with wide activation distributions. Cost: one ~80 s
  x86 recompile per variant.
- **A.3 Calibration bundle swap.** Our run used Bundle A (60
  samples × multi-position). Try Bundle B (20 samples × step-0
  only) at `models/calibration/bundle_b_pathb_ctx256.npz`. Answers
  the perf-levers research question "does cheap calibration
  suffice?" as a side-effect. Cost: ~80 s x86.
- **A.4 Per-tensor quant overrides.** `qairt-quantizer
  --quantization_overrides <json>` lets us pin specific tensors to
  higher bitwidth. If A.2/A.3 narrow the issue to one subgraph
  (e.g. rotary MatMul), override that subgraph to 16-bit weights
  or stay fp. Cost: ~2 h to author the overrides JSON + 80 s
  compile per iteration.

#### B. ARM64-side diagnostics (cheap, done while waiting on A)

- **B.1 Intermediate-activation diff.** Run the pathb ONNX on CPU,
  dump a layer-0 output (e.g. `/model/layers.0/self_attn/o_proj/...`);
  compare against a version where we feed the same prefix to the
  NPU and ask for the intermediate via an ONNX subgraph extraction.
  Would localise "which layer's output diverged first". Cost: ~2 h
  scripting + ORT partition dance.
- **B.2 Quant layer sanity re-use.** We know the quant layer works
  (0.001% round-trip). Skip this.

#### C. Heavier tooling paths (come back if A + B don't localize)

- **C.1 AIMET pre-quantization.** x86-side AIMET `QuantizationSimModel`
  emits a pre-quantized ONNX with explicit QDQ pairs, giving us
  direct control over every activation's scale/offset. Heavier
  lift (~1 session) but makes the interior quant choices visible
  and edit-able. Original Option 1 from this doc's option space.
- **C.2 `qairt-accuracy-debugger`.** Linux-only QAIRT tool that
  compares per-op outputs between the compiled DLC and a reference
  runtime, flagging the first op where numerical drift exceeds a
  threshold. The definitive localiser for "which op went wrong,"
  but needs Linux (WSL2 on the x86 box would work).

#### D. Nuclear options (if A.1 fails, i.e. fp16-pathb also wrong)

- **D.1 Rewrite pathb to keep rotary inline.** Skip the rotary
  hoist; use only the additive-mask trick. The pathbmask binary
  already works at fp16; w4a16-of-pathbmask would skip the whole
  rotary-hoist risk surface. Cost: ~1 session x86 (revert the
  `rewrite_qwen3_pathb.py` change; re-run the compile pipeline).
  Downside: we lose the rotary-hoist as a reusable piece for
  Qwen3.5 graduation, and we're back to testing whether AI Hub's
  preserve-list bug still bites a pathbmask-w4a16 compile.
- **D.2 Close Lever C negative.** Ship the 18.12 t/s Lever B AC
  baseline as Phase 5.5's final number. File the PTQ correctness
  finding upstream with Qualcomm. Move to Phase 6 / Qwen3.5
  graduation with a CPU-only spec decoder.

### Standing evidence + artifacts

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
