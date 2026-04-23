# Phase 5.5 Lever C — x86 team ask (session 16, 2026-04-22)

Focused one-pager for the x86 compile box. Companion to
`docs/w4a16_investigation.md` §"Next-session options — decision tree"
and `docs/phase5_local_qairt_compile_findings.md` (the pipeline you
already ran).

## Status from ARM64 side

Your QAIRT 2.42 rebuild landed and is **structurally green**:

- Binary MD5 `b8d8f3b7a4df9a6825af9b969f631228` — matches your
  `HANDOFF_qairt242.md`, copied into `models/`.
- Loads cleanly on ORT-QNN 1.24.4 (no error 5000, no
  "Input name not found" after we fixed the dot→underscore name
  mismatch on our side).
- Forward pass completes in ~28 ms, logits finite, non-constant.

But it's **numerically wrong**:

| probe | cos vs CPU fp32 | argmax match |
|---|---:|:-:|
| pathbmask fp16 (reference, same probe code) | 0.9999 | ✓ |
| pathb w4a16-local, humaneval fib-p0 (prompt_len=16) | **0.33** | ✗ |
| pathb w4a16-local, pos=0 + identity rotary + BOS-only | **0.29** | ✗ |

The pos=0 isolator is decisive: at pos=0 `rope_tables(0)` produces
cos=all-1.0 / sin=all-0.0 so rotary is identity; past_kv is zero;
attention_bias masks every past slot. Only `input_ids=1` at
position 0 matters. Even this trivial case hits cos=0.29.

ARM64-side ruleouts (every "could it be our runtime?" answer):

- Probe infrastructure: pathbmask still passes cos=0.9999 through
  the same code.
- Our quant formula: per-tensor round-trip RMS error 0.001% across
  all 56 past_kv + attention_bias + cos + sin on the real fib-p0
  feed (`scripts/probe_w4a16_quant_roundtrip.py`). No clipping.
- rope_tables formula: pos=0 bypasses it entirely.
- Wrapper schema: 60 uint16 inputs + int32 input_ids match the
  binary exactly per the `dlc_info_w4a16.txt` you shipped.
- Rotary-hoist math equivalence: your
  `probe_pathb_equivalence.py` already validated NEW-ONNX vs
  REF-ONNX at cos=1.0 for pos=0 and pos=5.

So the error is inside the compile pipeline you own
(`qairt-converter` → `qairt-quantizer` → `qnn-context-binary-generator`),
not in our runtime or the ONNX inputs.

## Primary ask — fp16-pathb rebuild (decisive, ~80 s)

Re-run your pipeline with **no PTQ** on the same pathb ONNX. This
is the one diagnostic that splits the remaining possibilities:

- If fp16-pathb passes cos ≥ 0.95 on ARM64 → **w4a16 PTQ is the
  culprit**. We move to A.2 / A.3 / A.4 in the investigation doc
  (PTQ algorithm variants, calibration bundle swap, per-tensor
  overrides).
- If fp16-pathb ALSO fails → the rewritten pathb graph or the
  prep pipeline lost something our CPU-level probe didn't catch.
  We fall back to option D.1 (rewrite pathb to keep rotary
  inline) or revert to pathbmask w4a16 via AIMET pre-quant.

### Commands (from your `docs/phase5_local_qairt_compile.md` pipeline, with PTQ skipped)

Reuse the same ONNX + working directory. Only step 3 changes:
drop the weight/activation bitwidth flags and feed no calibration
data so no PTQ runs. Step 1 (converter) and step 4
(context-binary-gen) are identical.

```bash
# Step 1 — ONNX -> DLC (unchanged)
qairt-converter \
    --input_network qwen3-0.6b-pathb-ai-hub-ctx256/model.onnx \
    --output_path qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc \
    --preserve_onnx_output_order \
    --remove_unused_inputs \
    2>&1 | tee qairt_compile_log.fp16.txt

# Step 3' — NO PTQ; convert fp32 DLC to fp16 activations/weights
# via the fp-bitwidth knob. No --input_list, no bitwidth flags
# that invoke the asymmetric-uint quantizer.
qairt-quantizer \
    --input_dlc qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc \
    --output_dlc qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.dlc \
    --float_bitwidth 16 \
    2>&1 | tee -a qairt_compile_log.fp16.txt

# Step 4 — DLC -> context binary (same config files you used for w4a16)
qnn-context-binary-generator \
    --model libQnnHtpV81Prepare \
    --backend libQnnHtp \
    --binary_file qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local \
    --dlc_path qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.dlc \
    --config_file config_main.json \
    2>&1 | tee -a qairt_compile_log.fp16.txt
```

Note: if `qairt-quantizer` refuses to run without a calibration
list when you pass `--float_bitwidth 16` alone, the alternative is
to pass `--float_bitwidth 32` (the default) and just run
`qairt-converter` → `qnn-context-binary-generator` directly,
skipping the quantizer step entirely. That's the AI-Hub fp16
pathbmask pathway we know works.

### NAS drop location

```
Z:\exposed\junk\phase5_step15_local_qairt_out_fp16\
├── qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.bin
├── qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.encodings.json  (if any)
├── dlc_info_fp16.txt
├── qairt_compile_log.fp16.txt
└── HANDOFF_fp16.md   (just MD5s + compile timings)
```

ARM64 will:

1. `Get-FileHash -Algorithm MD5` verify.
2. Copy `.bin` into `models/` as
   `qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.bin`.
3. Rerun `npu_short_prompt_probe.py --path pathb` with
   `SPECULA_NPU_VARIANT=fp16-local`. Same naming convention as
   w4a16-local; our wrapper builder is already variant-aware and
   handles FLOAT dtypes trivially (fp16 binaries have always been
   our working case).
4. Report back: cos vs CPU ground truth + single-step latency.

## Follow-up asks (only if fp16-pathb passes, localising w4a16 PTQ)

Each is another ~80 s compile; drop each as a new suffix under
`Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_<tag>\`:

### A.2 — tf_enhanced quantizer

Same commands as your w4a16 run, plus:

```bash
qairt-quantizer ... --act_quantizer tf_enhanced \
    --weights_bitwidth 4 --act_bitwidth 16 --input_list input_list.txt
```

`tf_enhanced` picks a tighter per-tensor range than `tf` default
on distributions with outliers — common accuracy win on
transformer activations.

### A.3 — Bundle B calibration

The ARM64-side calibration capture built two bundles:

- Bundle A (currently used): 60 samples × multi-decode-position.
  ~3.3 GB, `models/calibration/bundle_a_pathb_ctx256.npz`.
- Bundle B: 20 samples × step-0-only, ~1.1 GB,
  `models/calibration/bundle_b_pathb_ctx256.npz`.

If you want Bundle B, we'll push it to the same NAS
`phase5_step15_local_qairt_inputs\` staging folder your Bundle A
came from. Ping when ready.

### A.4 — CLE (cross-layer equalisation)

```bash
qairt-quantizer ... --use_adjusted_weights_quantizer --act_quantizer cle \
    --weights_bitwidth 4 --act_bitwidth 16 --input_list input_list.txt
```

Likely the heaviest-hitting PTQ improvement when activations vary
per-layer (common in rotary-heavy graphs). Try after A.2.

### A.5 — Per-tensor overrides

If A.2–A.4 narrow the issue to a specific subgraph (e.g. rotary's
rotate_half StridedSlice/Neg chain), we can hand-author a JSON
that pins those tensors to higher bitwidth. ARM64 will build the
JSON once we know which layer's range is off.

## Questions welcome

- If `qairt-quantizer --float_bitwidth 16` misbehaves on our
  graph, what's the 2.42-SDK-native path to produce a fp16 context
  binary? (AI Hub's `--quantize_full_type float16` equivalent.)
- Does `qairt-accuracy-debugger` run on 2.42 Windows, or do we
  need a Linux host? That tool is the definitive "which op went
  wrong" localiser and saves us if A.1–A.4 all fail.

ARM64 contact: `docs/w4a16_investigation.md` + commits `0357aa6`
(w4a16 plumbing) and `5ed37ac` (correctness probe findings).
