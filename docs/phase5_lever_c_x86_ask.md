# Phase 5.5 Lever C — x86 team ask (session 16, 2026-04-22)

Focused one-pager for the x86 compile box. Companion to
`docs/w4a16_investigation.md` §"Next-session options — decision tree"
and `docs/phase5_local_qairt_compile_findings.md` (the pipeline you
already ran).

## Update: A.1 fp16-pathb is CORRECT on ARM64

Your fp16-pathb rebuild at `Z:\exposed\junk\phase5_step15_local_qairt_out_fp16\`
(MD5 `a4bf0c4e0f8ee9994d0ea2cec998186b`) passes the correctness gate:

| metric | fp16-local | w4a16-local |
|---|---:|---:|
| cosine vs CPU fp32 | **0.999959** | 0.33 |
| argmax match | ✓ | ✗ |
| multi-step (3 steps) | 100% | 0% |
| latency per step | 65 ms | 28 ms |

fp16-pathb decodes `'    if n'` identical to the CPU reference on
fib-p0. So the whole pipeline up to and including
`qnn-context-binary-generator` on an fp16 DLC is numerically correct.
**The error is localised to `qairt-quantizer` in w4a16 mode.**

New ask: the follow-ups below, specifically **A.2 `tf_enhanced`** as
the next single diagnostic to run. A.1 stays retired unless something
surprises us later.

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

## A.1 fp16-pathb rebuild (COMPLETE — retained for history)

Ran green. Section kept so the commands below are re-runnable if we
ever need to sanity-check the pipeline on a different pathb ONNX.

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

## Primary ask now — A.2 `tf_enhanced` w4a16 rebuild (~80 s)

Same pipeline as the QAIRT 2.42 w4a16 run, only change is the
activation quantizer algorithm. Default was `tf` (min/max per
tensor); `tf_enhanced` uses a tighter range-enhancement heuristic
that commonly wins on transformer activations with outliers. Step
3 becomes:

```bash
qairt-quantizer \
    --input_dlc qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc \
    --output_dlc qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.dlc \
    --input_list input_list.txt \
    --weights_bitwidth 4 \
    --act_bitwidth 16 \
    --act_quantizer tf_enhanced \
    2>&1 | tee qairt_quantizer.tfe.log
```

Step 4 unchanged, just point at the new DLC and emit a new `.bin`.

NAS drop:

```
Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_tfe\
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.bin
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.encodings.json
├── dlc_info_w4a16_tfe.txt
├── qairt_compile_log.tfe.txt / qairt_quantizer.tfe.log / qnn_ctx_bin_gen.tfe.log
└── HANDOFF_tfe.md  (MD5s + compile timings)
```

ARM64 will consume under `SPECULA_NPU_VARIANT=w4a16-local-tfe` —
plumbing already supports any `*-local` suffix by convention.

## A.3 backup ask — Bundle B calibration

If A.2 doesn't move cos, swap the calibration bundle. Bundle B is
20 step-0-only samples vs Bundle A's 60 multi-position. Two
research questions at once: "is multi-position calibration
necessary?" (perf-levers research Q) and "is our calibration
distribution the root cause?" (this investigation).

Bundle B npz lives at `models/calibration/bundle_b_pathb_ctx256.npz`
on ARM64 (~1.1 GB); we'll push to `phase5_step15_local_qairt_inputs\`
when A.2 result is in, or push now if you want to run them in
parallel — just say.

Commands identical to A.2 except `--input_list` points at a Bundle
B-derived raw layout, and swap `-tfe` suffix for `-b`.

## A.4 backup ask — CLE (cross-layer equalisation)

Heaviest-hitting PTQ rescue for transformer-style graphs with
per-layer activation-range variance. Try after A.2 if A.2 still
shows cos < 0.95.

```bash
qairt-quantizer ... --act_quantizer cle \
    --use_adjusted_weights_quantizer \
    --weights_bitwidth 4 --act_bitwidth 16 --input_list input_list.txt
```

NAS suffix `-cle`.

## A.5 backup ask — Per-tensor overrides

If A.2–A.4 narrow the issue to a specific subgraph (rotary's
rotate_half StridedSlice/Neg chain is the usual suspect), ARM64
will author a `--quantization_overrides <json>` that pins those
tensors to higher bitwidth. Needs localisation evidence first; A.2
and A.3 don't provide that, C.2 (accuracy-debugger) would.

## Questions welcome

- Does `qairt-accuracy-debugger` run on 2.42 Windows, or do we
  need a Linux host? That tool compares per-op outputs between
  the compiled DLC and a reference runtime — the definitive
  "which op went wrong" localiser. If it's Linux-only, a WSL2
  invocation on your box would unblock it; if you already have
  WSL2 + an Ubuntu image, that's the fastest path.
- Any observations from the compile logs that caught your eye?
  Our end scanned `qairt_quantizer.log` / `qnn_ctx_bin_gen.log`
  for `tiling.h:242` chunking warnings on rotate_half
  StridedSlice, but couldn't identify anything out-of-pattern
  vs fp16 compile.

ARM64 contact: `docs/w4a16_investigation.md` (full decision tree +
ruleouts) + commits `0357aa6` (w4a16 plumbing), `5ed37ac` (probe
findings), and the fp16-local variant wiring landing with this
update.
