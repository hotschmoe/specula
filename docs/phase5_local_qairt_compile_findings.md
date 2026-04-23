# Phase 5.5 Lever C — local QAIRT compile **findings** (x86 side, 2026-04-22)

Companion to `docs/phase5_local_qairt_compile.md`. That doc was the plan;
this one records what actually happened on the x86 box and what the
ARM64 side should know before loading the binary.

## TL;DR

- **Local compile works.** Binary produced, magic bytes identical to
  Qualcomm's shipping reference, encodings extracted, handed off to
  `Z:\exposed\junk\phase5_step15_local_qairt_out\`.
- **Pipeline ran in ~60 seconds total** (not the 2.5 h the plan budgeted).
- **IO dtypes are all uint16**, not uint8/uint16 mix — plan-level doc was
  wrong about that. Runtime contract for ARM64 is uniform uint16 in and
  out.
- **Five deviations from the plan** — all documented below with reasons.
  None are blockers; several are simplifications.

## Setup on the x86 box (`99-Luftballons`)

- QAIRT 2.45.40.260406 copied from the ARM64 box into `C:\Qualcomm`.
  Multi-arch SDK; the `x86_64-windows-msvc` tree has working `.exe`s and
  Python scripts.
- Python 3.10.20 via `uv python install 3.10`. Separate venv at
  `C:\work\specula-qairt\.venv` (cannot share the repo's 3.12 venv —
  QAIRT's `check-python-dependency` hard-rejects anything other than 3.8
  or 3.10).
- `check-python-dependency` installed all 33 pinned deps cleanly on first
  try. The `setuptools<81` pin is required (the script imports
  `pkg_resources`, which was removed from setuptools in 82+).
- `onnx==1.21.0` added after (not in SDK's dep list, but qairt-converter
  needs it). Ended up upgrading `protobuf` from 3.19.6 → 7.34.1 as a
  transitive — no downstream tool complained.
- Environment shim: `C:\work\specula-qairt\env.sh` sources PATH +
  PYTHONPATH + `PYTHONIOENCODING=utf-8`. The UTF-8 forcing is load-bearing:
  without it, `qairt-converter --help` crashes because the stock cp1252
  console can't encode `‑` from the help text.
- **No QAIRT 2.42 available.** Compiled with 2.45. Whether ORT-QNN 1.24.4
  (which bundles 2.42) can load a 2.45-produced binary is **the** open
  question. Magic-byte equivalence with Qualcomm's reference suggests yes;
  confirm on ARM64.

## Deviations from the compile plan

### 1. `qairt-converter` must pass `--remove_unused_inputs`

The plan-doc invocation was:

```
qairt-converter --input_network ... --output_path ... --preserve_onnx_output_order
```

With that alone, the pathb ONNX's unused `position_ids` input stays in
the DLC as a declared-but-unconsumed input. `qairt-quantizer` then fails
with one of two internal-inconsistency errors depending on which layer
counts first:

```
attempt 1 (61 keys in input_list):
  [ ERROR ] Graph contains 61 inputs, but only found input data for 60 inputs from user
attempt 2 (60 keys, position_ids skipped):
  [ ERROR ] [QNN_CPU] Expected number of inputs for Graph is 60 instead 61 provided
```

Fix: add `--remove_unused_inputs` to the converter invocation. Then drop
`position_ids` from the calibration npz (we added `SKIP_KEYS` logic in
`qairt_prep_calibration.py`). Third attempt succeeded.

### 2. Calibration raw files must match DLC dtypes, not ONNX/npz dtypes

`qairt-converter` silently downcasts int64 inputs to int32
(`keep_int64_inputs=False` default). The npz has `input_ids` /
`position_ids` as int64 (8 bytes/element) but the DLC expects int32
(4 bytes/element). Feeding int64 raws leads to misaligned reads during
calibration — observable as weird activation ranges, if it doesn't
crash outright.

Fix in `qairt_prep_calibration.py`: cast int64 → int32 on write. Every
other dtype pass-through unchanged.

### 3. All IO quantizes to uint16, not uint8/uint16 mix

Plan-doc predicted the binary would follow Qualcomm's Qwen3-4B Genie
w4a16 convention: uint16 hidden/mask/cos/sin, **uint8 past_kv**. With
bare `--weights_bitwidth 4 --act_bitwidth 16` and no per-tensor
overrides, PTQ picked **uFxp_16 for every single IO tensor** including
past_kv and present_kv.

Practically this is a simpler ARM64 contract — one dtype for all IO,
no uint8 KV chain to build. Logits also uint16. Full scale/offset per
tensor in `encodings.json`.

If an ARM64 side prototype explicitly needs the uint8 KV pattern (for
memory/bandwidth reasons), two options:
- Re-run PTQ with explicit quantization overrides via
  `--quantization_overrides <json>` telling it past_kv target is 8-bit.
  Untested here.
- Fall back to the Qualcomm reference bundle (validated in
  `results/qwen3_4b_genie_w4a16_probe.md`) and chain through that, not
  our draft model.

### 4. `htp_backend_ext_config.json` needs to be inside a wrapper in 2.45

Plan-doc recipe:

```
--config_file htp_backend_ext_config.json
```

with that file directly containing `devices` / `memory` / `context` keys.
QAIRT 2.45's `qnn-context-binary-generator` rejects every key:

```
Unknown Key = devices/0/soc_model passed in config
Unknown Key = devices/0/dsp_arch passed in config
...
Unknown Key = context/weight_sharing_enabled passed in config
```

Because 2.45 wants a two-level structure:

- `--config_file` points at `config_main.json`
- `config_main.json` is `{ "backend_extensions": { "shared_library_path":
  ".../QnnHtpNetRunExtensions.dll", "config_file_path":
  ".../htp_backend_ext_inner.json" } }`
- `htp_backend_ext_inner.json` is the real config with
  `devices`/`cores`/etc.

Both files are in the handoff bundle as reference.

### 5. `weight_sharing_enabled` is overhead for single-graph binaries

The plan-doc example config had `"context": {"weight_sharing_enabled":
true}`. That flag is meaningful only when compiling a **bundle** of
context binaries that share weights (like the Qualcomm 4-part llama
reference). Our draft model is one graph. Leaving the flag on produces
non-fatal but noisy errors during compile:

```
wtshare_operation.cc:1057::ERROR:small.is_shared()  (x3)
```

Dropped it. Clean compile. Dropping it might change the on-disk layout
slightly vs. Qualcomm's reference, but the magic bytes still match the
doc's pattern.

## What on-x86 validation actually tells us

The plan-doc's validation step (`qnn-net-run --backend libQnnCpu
--retrieve_context ...`) does not work: the QnnCpu backend cannot
deserialize an HTP-compiled context binary (fails immediately at
`[QNN_CPU] Context de-serialization failed`). The plan was incorrect
about this.

Substituting `--backend QnnHtp.dll` forces the on-host HTP simulator.
That path **does** load the binary successfully (got past
"Creating context from binary file" with no deserialization error) and
entered "Executing Graphs" — so the format is at least structurally
valid. But then it hangs with no log output and no result files for
25+ minutes. Host-sim of a 600 M-param w4a16 graph on x86 appears to be
effectively unworkable. Killed and moved on.

**What the x86 side DID confirm:**

- DLC → context-binary compile completes with no errors (only the
  expected `libcdsprpc.dll couldn't open` warning: that DLL is the
  Hexagon RPC bridge and does not exist on x86 — we are not running
  on-device and we don't need it).
- Binary size 876 MB (vs. plan's 800–900 MB estimate, close enough).
- Magic bytes (first 16):
  `00 00 00 02 00 00 00 03 00 00 00 00 00 00 00 01` — identical to the
  reference pattern quoted in the compile plan's risk register.
- `qnn-net-run` can open + deserialize the binary under HTP host-sim
  without error.
- Encodings JSON extracts cleanly; 2227 tensors, 1686 nodes.

**What only ARM64 can confirm:**

- Whether ORT-QNN 1.24.4 (QAIRT 2.42 runtime) can load a 2.45-produced
  binary. If it fails with the load-error-5000 pattern from
  `docs/npu_ort_qnn_version_match.md`, the next move is to install
  QAIRT 2.42 SDK on the x86 box and re-run steps 1–5. The ONNX +
  calibration npz stay on the NAS for this scenario. Alternatively,
  bump the ARM64 runtime to ORT-QNN 2.1.0 (bundles 2.45), which might
  just work, but `reference_ort_qnn_qairt_match.md` notes 2.1.0 has
  unrecoverable bugs on the X2E94100 driver — so 2.42 rebuild is the
  more likely fix.
- Numerical correctness post-quant (cos vs. CPU fp32 target, target
  ≥0.95 per the plan's gate).
- Actual step latency on real HTP (projection from the
  `qwen3_4b_genie_w4a16_probe.md` side-quest: ~17 ms/step, giving
  30–45 t/s at k=4–8 speculative).

## Artifacts + locations

**On the x86 box** (preserved for potential debug or re-run):

```
C:\work\specula-qairt\
├── .venv\                                    # py3.10 + SDK deps
├── env.sh                                    # source to get PATH/PYTHONPATH right
├── scripts\qairt_prep_calibration.py         # npz → raw files + input_list.txt
├── qwen3-0.6b-pathb-ai-hub-ctx256\           # input ONNX (2.9 GB)
├── bundle_a_pathb_ctx256.npz                 # input calibration (3.3 GB)
├── calibration_raw\                          # 60 × 60 raw files (3.51 GB)
├── input_list.txt                            # 60 lines, tensor_name:=path
├── qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc           # pre-quant DLC (2.9 GB)
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc    # post-PTQ DLC (867 MB)
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin    # THE binary (876 MB)
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json  # 3.1 MB
├── config_main.json / htp_backend_ext_inner.json       # working 2.45 config
└── {qairt_compile_log.txt, qairt_quantizer.log,
     qnn_ctx_bin_gen.log, qairt_selftest.{cpu,htp}.log} # per-step logs
```

**On the NAS** (handoff to ARM64):

```
Z:\exposed\junk\phase5_step15_local_qairt_out\
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json
├── HANDOFF.md                   # the one-pager for the ARM64 team
├── dlc_info_w4a16.txt           # per-tensor dtypes + encodings summary
├── qairt_compile_log.txt        # step 1 stdout
├── qairt_quantizer.log          # step 3 stdout (PTQ run)
├── qnn_ctx_bin_gen.log          # step 4 stdout (binary compile)
├── qairt_selftest.cpu.log       # qnn-net-run QnnCpu attempt (deserialize-fail, expected)
├── qairt_selftest.htp.log       # qnn-net-run QnnHtp attempt (loads, hangs in exec)
├── config_main.json             # working 2.45 backend-ext wrapper
└── htp_backend_ext_inner.json   # working 2.45 backend-ext inner config
```

**MD5s** (verify on receive):

```
b49d3e1299c1565260a27f8dfa6ebe54  qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin
aab34c8bb69b16189f42078831a822c6  qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json
```

## Update 2026-04-22 — 2.45-built binary failed, rebuilt with QAIRT 2.42

ARM64 team reported load failure (error 5000) on the 2.45-built binary —
matches Qualcomm's own compat matrix prediction exactly (prepared on
2.43+ → 2.42 runtime = Not Supported, full matrix in
`docs/QAIRT-Docs/QNN/general/htp/htp_backend.html` inside the SDK
install).

### Obtaining QAIRT 2.42

Version string: **2.42.0.251225**. Direct download URL:

```
https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip
```

Gotcha: the Qualcomm API gateway returns **403 on HEAD requests and
non-browser user agents**. Plain `curl -I` or `curl --head` fails. A
GET with a Mozilla UA works:

```bash
curl -sL -A "Mozilla/5.0" -o qairt.zip <url>
```

Download size: 1,543,955,191 bytes (1.44 GB). The zip extracts to a
`qairt/2.42.0.251225/` inner directory; unzip then flatten one level.

### Changes needed to run 2.42 tools in our existing venv

- 2.42's `check-python-dependency` requires the **same** pinned deps as
  2.45 — nothing new to install.
- **onnx must be downgraded from 1.21.0 → 1.14.1.** 2.42's
  `qti.aisw.converters.onnx.util` imports the `onnx.mapping` submodule,
  which was removed from onnx after 1.14. The Qualcomm script has a
  bare `except:` around the onnx imports and silently sets
  `onnx = None` on failure, so the symptom surfaces downstream as a
  confusing `AttributeError: 'NoneType' object has no attribute
  'AttributeProto'`. `pip install 'onnx<1.15'` fixes it.
- The `shared_library_path` in `config_main.json` had to be repointed
  from 2.45's `QnnHtpNetRunExtensions.dll` to 2.42's equivalent. Same
  filename, different install root.

All other deviations from the plan-doc (remove_unused_inputs,
position_ids skip, int64→int32 calibration cast, uniform uint16 IO,
wrapped config, no weight_sharing) applied identically.

### 2.42 pipeline timings

| step | 2.45 timing | 2.42 timing |
|---|---:|---:|
| qairt-converter | ~22 s | ~22 s |
| qairt-quantizer PTQ | ~33 s | ~39 s |
| qnn-context-binary-generator | ~7 s | ~10 s |
| qairt-dlc-to-json | ~1 s | <1 s |
| **total** | **~63 s** | **~72 s** |

Output sizes are identical (876 MB bin, 867 MB quantized DLC). Encodings
JSON grew 3.1 → 3.4 MB — 2.42 writes slightly more per-tensor metadata.

### New NAS drop

```
Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242\
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin           876 MB
│   MD5: b8d8f3b7a4df9a6825af9b969f631228
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json  3.4 MB
│   MD5: 9b9457c39e4e2139c76bcb9fe0b124b9
├── HANDOFF_qairt242.md
├── dlc_info_w4a16.txt
├── qairt_compile_log.txt / qairt_quantizer.log / qnn_ctx_bin_gen.log
└── config_main.json / htp_backend_ext_inner.json
```

The earlier 2.45-built drop at `phase5_step15_local_qairt_out\` is
retained as a reference (known to fail ORT-QNN 1.24.4 with error 5000).

## Update 2026-04-22 (later) — fp16-pathb rebuild (A.1 isolator)

Ask: `docs/phase5_lever_c_x86_ask.md` § "Primary ask — fp16-pathb
rebuild". Re-run the same pipeline with no PTQ, so the cos≈0.3
failure observed on the w4a16-local build can be split between
"w4a16 PTQ is the culprit" (fp16-pathb passes) vs "pathb rewrite /
prep pipeline lost something" (fp16-pathb also fails).

### Pipeline executed

Reused the existing fp32 DLC on disk (deterministic from identical
step-1 flags). Only new work:

| step | tool | duration |
|---|---|---:|
| 3' | `qairt-quantizer --float_bitwidth 16` (no `--input_list`) | ~5 s |
| 4 | `qnn-context-binary-generator` (same config_main.json) | ~79 s |

Quantizer logged `Skipping quantization, no input_list provided` and
wrote a DLC whose IO tensors all declare `Float_32` — matching the
working fp16 pathbmask binary's runtime contract. Step 4 was 8× slower
than the w4a16 build (79 s vs 10 s); the prepare pass does more work
on a fp16 graph than a w4a16 one. Compile completed cleanly through
every stage; magic bytes `00 00 00 02 00 00 00 03 00 00 00 00 00 00 00 01`
match the Qualcomm reference.

One Windows-quirk correction to the compile plan: the ask doc's
step-4 invocation used the Linux-style `--model libQnnHtpV81Prepare
--backend libQnnHtp`. On our x86 box those don't resolve — the real
filenames in the SDK's `lib/x86_64-windows-msvc/` are `HtpPrepare.dll`
and `QnnHtp.dll`. Passing full DLL paths works.

One venv-quirk correction: QAIRT tools shebang `#!/usr/bin/env python3`
doesn't find python3 on Windows by default (Store app-execution-alias
intercepts first). Fix: `cp python.exe python3.exe` inside the venv's
`Scripts\` dir. Applies to both 2.42 and 2.45 trees.

### NAS drop

```
Z:\exposed\junk\phase5_step15_local_qairt_out_fp16\
├── qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.bin    1.41 GB
│   MD5: a4bf0c4e0f8ee9994d0ea2cec998186b
├── HANDOFF_fp16.md
├── dlc_info_fp16.txt
├── qairt_compile_log.fp16.txt
└── config_main.json / htp_backend_ext_inner.json
```

No `encodings.json` — the DLC declares fp32 IO, so no per-tensor
quant params exist to extract. ARM64 runtime contract is float32 in,
float32 logits out (same as the working fp16 pathbmask binary).

## Update 2026-04-22 (later still) — A.2 enhanced-calibration w4a16 rebuild

Ask: `docs/phase5_lever_c_x86_ask.md` § "Primary ask now — A.2
`tf_enhanced` w4a16 rebuild". Same pipeline as w4a16-local-qairt242,
only change is the activation quantizer calibration algorithm.

Terminology note — `tf_enhanced` is pre-2.42 SDK syntax. In QAIRT 2.42,
`--act_quantizer` accepts `{tf, enhanced, adjusted, symmetric}`. First
attempt with `--act_quantizer tf_enhanced` failed with
`modeltools::IrQuantizer: Invalid activation quantizer type: tf_enhanced`.
Retried with `--act_quantizer enhanced`, which 2.42 documents in-source
as "an algorithm useful for quantizing models with long tails present in
the weight distribution" — exact match for the ask's intent.

Modern 2.42 alternative (not used here, for future iteration):
`--act_quantizer_calibration {sqnr, entropy, mse, percentile}` with
`--act_quantizer_schema {asymmetric, symmetric, unsignedsymmetric,
signedasymmetric}` — these replace the legacy flag with finer control.

### Pipeline executed

| step | tool | duration |
|---|---|---:|
| 3 | `qairt-quantizer --weights_bitwidth 4 --act_bitwidth 16 --act_quantizer enhanced` | ~49 s (≈42 s QNN_CPU sim × 60 samples) |
| 4 | `qnn-context-binary-generator` (same config_main.json, full DLL paths) | ~9.6 s |
| 5 | `qairt-dlc-info` + `qairt-dlc-to-json` | <1 s |

DLC metadata echo confirms `act_quantizer=enhanced` took effect.
Binary-gen log is quieter than baseline — no
`q::rms_normvals_temp` / `q::Add_flat` "no properties registered"
warnings this time, implying the `enhanced` calibration assigned
scales to a couple of scratch ops that `tf` was leaving unquantized.
Otherwise warning pattern is identical to baseline.

### NAS drop

```
Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_tfe\
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.bin              875 MB
│   MD5: 96667934cbf9dfdcbddf2f1fe93f13a9
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.encodings.json   3.32 MB
│   MD5: 3dda18331e8fff2e11e9c2ee144bef12
├── HANDOFF_tfe.md
├── dlc_info_w4a16_tfe.txt
├── qairt_compile_log.tfe.txt / qairt_quantizer.tfe.log / qnn_ctx_bin_gen.tfe.log
└── config_main.json / htp_backend_ext_inner.json
```

Runtime contract unchanged from w4a16-local-qairt242: 60 uint16 inputs
(+ `input_ids` int32), 57 uint16 outputs, dotted names in encodings.json
and underscored names in the binary. Suggested variant:
`SPECULA_NPU_VARIANT=w4a16-local-tfe`.

### What this closes

- Answers the open question from the original findings: **2.45-built
  binaries are not backward-compatible with 2.42 runtime**, as
  Qualcomm's compat matrix explicitly states. Our load-failure
  observation matches.
- Confirms the SDK-version guess-and-check approach is viable when AI
  Hub's orchestrator-level flags (`--qairt_version 2.42`) aren't
  available — just install the matching SDK and rebuild. On our box
  this cost ~72 s of compile time once the SDK was in place.

## Next steps (ARM64 side)

1. `Get-FileHash -Algorithm MD5` both files against the values above.
2. Copy `.bin` into `models/`. Name per whatever convention fits
   `scripts/npu_load_qwen3_bin.py`.
3. Update `describe_inputs` — all 60 inputs are uint16 (not uint8/uint16
   mix). Parse `encodings.json`'s `graph.tensors` dict; for each input
   name, pull `quant_params.scale_offset.scale` / `offset` /
   `bitwidth` and compare against `dlc_info_w4a16.txt`'s summary table
   for sanity.
4. Wrap `session.run()` with per-tensor float32 → uint16 feed and
   uint16 → float32 dequant on logits.
5. `npu_short_prompt_probe.py --path pathb` and
   `npu_vs_cpu_correctness.py --path pathb` with
   `SPECULA_NPU_VARIANT=w4a16-local`. Gate: cos ≥ 0.95.
6. If load fails with error 5000 or similar: **don't debug the ARM64 side
   first**. Ping x86 team to install QAIRT 2.42 SDK and re-run steps 1–5
   locally. 2.45 → 2.42 compatibility was the known risk going in.
