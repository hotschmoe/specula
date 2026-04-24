# Phase 5.5 Lever C — local QAIRT w4a16 compile

**Status:** drafted 2026-04-22 after AI Hub's compile orchestrator
was caught dropping the first entry of the quantizer's
`--preserve_io_datatype` list, leaving our pathb w4a16 binary's
`past_key_values.0.key` unintentionally uint8-quantized (runtime
load fails with a 4× byte-count mismatch). Rather than bandage
around AI Hub's bug, we run the exact same QAIRT toolchain
ourselves on a dedicated x86 box and produce a **full-quant-IO
w4a16 binary** matching Qualcomm's shipping Qwen3-4B reference
pattern — which we've already validated loads cleanly in our
ORT-QNN 1.24.4 runtime (see `results/qwen3_4b_genie_w4a16_probe.md`).

Read top-to-bottom if you're the x86 team picking this up cold.

## Why this document exists

Three facts drive the plan:

1. **AI Hub's w4a16 pipeline is currently broken for our graph.**
   Compile job `jg93r1jqg` reached SUCCESS at 6050s but the output
   `.bin` fails at ORT-QNN `session.run()` because the driver
   mis-formats `--preserve_io_datatype` for `qairt-quantizer`:
   converter-list=116 names, quantizer-list=115 names, the first
   item (`past_key_values.0.key`) is silently dropped. Evidence:
   `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`,
   writeup in `docs/w4a16_investigation.md`.
2. **QAIRT 2.42's qairt-converter + qairt-quantizer +
   qnn-context-binary-generator can produce the binary locally.**
   AI Hub's log reveals the exact invocations they use — we can
   reproduce them without their orchestrator.
3. **Qualcomm's reference Qwen3-4B Genie w4a16 bundle uses
   full-quant IO (uint8 past_kv, uint16 hidden/mask/cos/sin) and
   loads cleanly in our ORT-QNN 1.24.4.** Probed 2026-04-22; both
   partitions we tested (embed-only and 12-layers-full-quant)
   load + run, 7.22 ms median for 12 layers on AC. Binary magic
   bytes are identical to our AI Hub-compiled binaries — same
   QAIRT 2.42 format. So "do it Qualcomm's way" is directly
   supported by our runtime and gives us a cleaner IO convention
   than AI Hub's preserve-everything-at-fp32 approach.

Net: **drop `--preserve_io_datatype` entirely, let qairt-quantizer
fully quantize the IO**, and our existing EPContext wrapper gets
updated to declare uint8/uint16 inputs (we already validated this
works via `scripts/probe_qualcomm_qwen3_4b.py`).

## Deliverable

Two files handed back to the ARM64 team via the NAS drop at
`Z:\exposed\junk\phase5_step15_local_qairt\`:

- `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin` — the QNN
  context binary.
- `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json`
  — per-input and per-output scale/offset/dtype, extracted from the
  quantized DLC via `qairt-dlc-to-json`. ARM64 runtime uses these
  to quantize inputs and dequantize logits per step.

Plus the intermediate artifacts retained on the x86 box (for
debug if the binary misbehaves):

- `qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc` — pre-quant DLC.
- `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc` — post-PTQ DLC.
- `qairt_compile_log.txt` — combined stdout/stderr of all three
  tool invocations.

## Prerequisites on the x86 box

- x86_64 Linux or Windows. Linux slightly preferred — the QAIRT
  Linux SDK is Qualcomm's reference configuration and some tools
  (notably `qairt-accuracy-debugger`) only exist as Linux shell
  scripts in the Windows SDK. All three tools we use
  (`qairt-converter`, `qairt-quantizer`,
  `qnn-context-binary-generator`) work on both platforms.
- Python 3.10 or 3.12 (x86_64). The SDK ships Python package
  dependencies in `lib/python/qti/aisw/`. Create a clean venv.
- **QAIRT SDK 2.42** preferred; 2.45 acceptable if you pass
  `--qairt_version 2.42` to `qnn-context-binary-generator` (AI
  Hub does exactly this — their runtime was QAIRT 2.45 but they
  targeted 2.42 binaries for ORT-QNN 1.24.4 compatibility).
  - Our ARM64 runtime is ORT-QNN 1.24.4 which bundles QAIRT 2.42.
  - Loading a 2.45-format binary in 1.24.4 fails with loader
    error 5000 (memory: `reference_ort_qnn_qairt_match.md`).
    Always target 2.42 unless the ARM64 side is upgraded to
    ORT-QNN 2.1.0 (which bundles 2.45).
- ~15 GB free disk: 3 GB input ONNX + 3 GB calibration bundle +
  ~900 MB output bin + intermediate DLCs (~900 MB each, two of
  them).
- Network share or SCP access to our NAS at `Z:\exposed\junk\`.

No ONNX re-export needed — we ship a pre-staged ONNX at the right
shapes from the ARM64 side (see §Inputs).

## Inputs handed off from ARM64 side

Drop point on the NAS: `Z:\exposed\junk\phase5_step15_local_qairt\`

| file | size | source | purpose |
|---|---:|---|---|
| `qwen3-0.6b-pathb-ai-hub-ctx256/model.onnx` | ~0.6 MB | our staging step | ONNX graph (shapes already pinned; ORT-BASIC folded 7131→2054 nodes; zero symbolic dims) |
| `qwen3-0.6b-pathb-ai-hub-ctx256/model.data` | ~2.87 GB | our staging step | external weights sidecar |
| `bundle_a_pathb_ctx256.npz` | 3.27 GB | our calibration capture | 60 samples × 61 inputs (input_ids, position_ids, attention_bias, position_ids_cos, position_ids_sin, 56 past_kv) |
| `bundle_a_pathb_ctx256.manifest.json` | ~5 KB | our calibration capture | per-sample metadata (prompt source, decode position, etc.) |

The ONNX is already AI-Hub-upload-shape-ready (per `prep_onnx_for_ai_hub.py`
on ARM64); nothing to re-prep. The calibration npz shape schema:

```
input_ids              [60, 1, 1]             int64
position_ids           [60, 1, 1]             int64
attention_bias         [60, 1, 1, 1, 256]     float32
position_ids_cos       [60, 1, 1, 128]        float32
position_ids_sin       [60, 1, 1, 128]        float32
past_key_values.N.key    [60, 1, 8, 255, 128] float32   (N = 0..27)
past_key_values.N.value  [60, 1, 8, 255, 128] float32
```

## The pipeline

Three tool invocations. The flags below are distilled from AI Hub's
actual compile log (`jg93r1jqg.log`) with the preserve-list
*removed* — that's the key change.

### Step 0 — environment + working dir

```bash
# Linux example. Windows PowerShell equivalent uses envsetup.ps1.
source /opt/qcom/QAIRT/2.42.0/bin/envsetup.sh
#   → sets QAIRT_SDK_ROOT, prepends SDK bin + lib to PATH, sets PYTHONPATH

mkdir -p /work/specula-qairt && cd /work/specula-qairt
cp -r /mnt/nas/phase5_step15_local_qairt/qwen3-0.6b-pathb-ai-hub-ctx256 .
cp /mnt/nas/phase5_step15_local_qairt/bundle_a_pathb_ctx256.npz .
```

Verify the SDK is live:

```bash
qairt-converter --version
qairt-quantizer --version
qnn-context-binary-generator --version
```

### Step 1 — ONNX → DLC (qairt-converter)

```bash
qairt-converter \
    --input_network qwen3-0.6b-pathb-ai-hub-ctx256/model.onnx \
    --output_path qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc \
    --preserve_onnx_output_order \
    2>&1 | tee qairt_compile_log.txt
```

What's changed vs AI Hub's invocation:

- **No `--preserve_io_datatype`.** We want qairt-quantizer to
  fully quantize IO per Qualcomm's reference convention, not keep
  it at fp32. The AI Hub bug specifically mis-formats this flag;
  dropping it eliminates the bug surface.

Expected duration: 1–2 min (it's a pure ONNX-to-DLC transcode,
not a PTQ run).

Sanity-check the DLC:

```bash
qairt-dlc-info -i qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc | head -40
# Should show 61 inputs, 57 outputs, ~2054 ops.
```

### Step 2 — Build the calibration input_list.txt

`qairt-quantizer` wants `--input_list` — a text file pointing at
raw binary files, one per input per sample. We have a .npz from
our capture pipeline; convert to the on-disk raw-bin layout qairt
expects.

Write this as `scripts/qairt_prep_calibration.py` on the x86 side:

```python
#!/usr/bin/env python3
"""Expand bundle_a_pathb_ctx256.npz into raw binaries + input_list.txt.

Layout produced:
    calibration_raw/
        sample_00/
            input_ids.raw
            position_ids.raw
            attention_bias.raw
            position_ids_cos.raw
            position_ids_sin.raw
            past_key_values.0.key.raw
            ...
        sample_01/
            ...
    input_list.txt   ← one line per sample, names in ONNX graph-input order

input_list.txt format (one line per sample, space-separated, TENSOR_NAME:=PATH):
    input_ids:=calibration_raw/sample_00/input_ids.raw position_ids:=calibration_raw/sample_00/position_ids.raw ...
"""

from pathlib import Path
import numpy as np
import json

NPZ = Path("bundle_a_pathb_ctx256.npz")
OUT = Path("calibration_raw")
LIST = Path("input_list.txt")

data = np.load(str(NPZ))
keys = list(data.files)                # iteration order = insertion order = graph order (our capture script is careful about this)
n_samples = data[keys[0]].shape[0]
OUT.mkdir(exist_ok=True)

with LIST.open("w") as f_list:
    for s in range(n_samples):
        sample_dir = OUT / f"sample_{s:02d}"
        sample_dir.mkdir(exist_ok=True)
        parts = []
        for k in keys:
            arr = data[k][s]           # drop the sample dim
            arr = np.ascontiguousarray(arr)
            raw_path = sample_dir / f"{k}.raw"
            arr.tofile(str(raw_path))
            parts.append(f"{k}:={raw_path}")
        f_list.write(" ".join(parts) + "\n")

print(f"wrote {n_samples} samples to {OUT} and {LIST}")
```

Run:

```bash
python scripts/qairt_prep_calibration.py
# Expected: "wrote 60 samples to calibration_raw and input_list.txt"
# Disk footprint: ~3.27 GB across 60 × 61 .raw files.
```

### Step 3 — DLC → quantized DLC (qairt-quantizer)

```bash
qairt-quantizer \
    --input_dlc qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc \
    --output_dlc qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc \
    --input_list input_list.txt \
    --weights_bitwidth 4 \
    --act_bitwidth 16 \
    2>&1 | tee -a qairt_compile_log.txt
```

What's present vs AI Hub:
- `--weights_bitwidth 4 --act_bitwidth 16` — the "w4a16" spec.
- `--input_list` — path to calibration data (we built it in step 2).

What's **removed**:
- `--preserve_io_datatype` — intentionally not passed. Every IO
  tensor gets quantized per calibration. AI Hub auto-adds this
  flag and that's where their bug lives; we just don't.

Expected duration: 60–90 min on a modern desktop CPU (this is the
PTQ calibration run — runs the graph over all 60 samples in
simulation and picks per-tensor quantization params).

Sanity-check the output DLC:

```bash
qairt-dlc-info -i qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc | head -30
# Should show input dtypes as uint16 / uint8 per Qualcomm's convention.
```

### Step 4 — DLC → QNN context binary (qnn-context-binary-generator)

```bash
qnn-context-binary-generator \
    --model libQnnHtpV81Prepare \
    --backend libQnnHtp \
    --binary_file qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local \
    --dlc_path qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc \
    --config_file htp_backend_ext_config.json \
    --qairt_version 2.42 \
    2>&1 | tee -a qairt_compile_log.txt
```

Where `htp_backend_ext_config.json` is a one-line file matching
Qualcomm's reference bundle:

```json
{"devices": [{"soc_model": 88, "dsp_arch": "v81", "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}], "memory": {"mem_type": "shared_buffer"}, "context": {"weight_sharing_enabled": true}}
```

Copy it verbatim from
`Z:\exposed\junk\phase5_step12_pathb\qwen3-0.6b-pathb\` — wait,
that file is in the Qualcomm Qwen3-4B bundle at
`models/qualcomm-qwen3-4b-ref/.../htp_backend_ext_config.json`.
Use that one (same SoC + arch + perf profile as our target).

**`--qairt_version 2.42` is critical.** Without it, the SDK
defaults to emitting 2.45-format binaries that our ORT-QNN 1.24.4
runtime can't load (error 5000).

Expected duration: 1–3 min. Fastest step.

Final artifact: `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin`
(~800–900 MB).

### Step 5 — Extract per-tensor encodings for the ARM64 runtime

```bash
qairt-dlc-to-json \
    -i qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.dlc \
    -o qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json \
    2>&1 | tee -a qairt_compile_log.txt
```

This JSON contains every tensor's quantization parameters. The
ARM64 runtime reads it to build the per-input quantization step
(float32 → uint8/uint16 on feed) and per-output dequantization
(uint16 logits → float32 for argmax).

Expected format (matches the pattern in
`models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/metadata.yaml`):

```json
{
  "graph": {
    "inputs": {
      "input_ids": {"dtype": "int32", "shape": [1,1]},
      "attention_bias": {"dtype": "uint16", "shape": [1,1,1,256], "quantization_parameters": {"scale": 0.00152, "offset": -65535}},
      "past_key_values.0.key": {"dtype": "uint8", "shape": [1,8,255,128], "quantization_parameters": {"scale": 2.34, "offset": -128}},
      ...
    },
    "outputs": {
      "output_0": {"dtype": "uint16", "shape": [1,1,151936], "quantization_parameters": {"scale": 0.00138, "offset": -26872}},
      ...
    }
  }
}
```

Exact JSON structure may differ between QAIRT 2.42 and 2.45 — the
ARM64 side has a parser that's schema-tolerant, just include
every per-tensor scale+offset+dtype.

## Validation on x86 before handoff

Don't just ship the .bin blindly — do a structural sanity check
with `qnn-net-run` on the x86 box (CPU simulation of the HTP
backend, slow but correct).

```bash
# Pick one calibration sample, use its raw tensors as the input.
# qnn-net-run expects --input_list in the same format as
# qairt-quantizer's calibration input_list.
head -1 input_list.txt > input_list_single.txt

qnn-net-run \
    --model libQnnHtpV81Prepare \
    --backend libQnnCpu \
    --retrieve_context qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin \
    --input_list input_list_single.txt \
    --output_dir qairt_selftest_out \
    2>&1 | tee qairt_selftest.log
```

Expected outcome:
- No errors during `retrieve_context` (binary loads in QAIRT's own
  runtime — if this fails, the binary is malformed).
- 57 output files land in `qairt_selftest_out/Result_0/` with the
  expected sizes (output_0 is 151936 × 2 bytes = ~297 KB uint16;
  output_1..56 are per-layer present KV slices).
- `output_0` values are non-constant (min ≠ max).

If `qnn-net-run` crashes or produces all-zero logits, the binary
is bad — **don't transfer it**. Debug before handoff:
1. Re-run qairt-quantizer with `--log_level verbose`; look for
   per-op quantization-range warnings.
2. Try with fewer calibration samples (edit input_list.txt to 5
   samples) to narrow down whether a specific sample is degenerate.
3. Check if the pathb ONNX has any dim_param that slipped past
   our prep step (the ARM64 side's prep is supposed to handle all
   dim_params, but if 2.45 SDK handles them differently than AI
   Hub's 2.45 did, that's a new surface).

## Handoff back to ARM64

NAS drop: `Z:\exposed\junk\phase5_step15_local_qairt_out\`

```
phase5_step15_local_qairt_out/
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.encodings.json
├── qairt_compile_log.txt                 # combined stdout+stderr
├── qairt_selftest.log                    # qnn-net-run output
├── qairt_selftest_out/                   # sample inference outputs (for cross-check)
│   └── Result_0/
│       ├── output_0.raw
│       └── ... (57 total)
└── HANDOFF.md                            # one-pager: versions used, MD5s, any warnings
```

MD5 both the `.bin` and `.encodings.json` in HANDOFF.md. ARM64
side verifies before loading.

## What ARM64 does after receiving the binary

(Captured here so the x86 team knows the downstream contract and
can flag concerns before shipping.)

1. Copy `.bin` to `models/qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin`.
2. Update `scripts/npu_load_qwen3_bin.py::describe_inputs` to
   declare uint8/uint16 per the encodings.json (we already
   validated this works in
   `scripts/probe_qualcomm_qwen3_4b.py` — same pattern).
3. Add a `QuantSpec` table (loaded from encodings.json) and a
   quant/dequant layer around `session.run()`:
   - On feed: float32 → quantized per-tensor.
   - On output: uint16 logits → float32 for argmax; uint8 present_kv
     stays quantized (passed straight into next step's past_kv_in).
4. Run `npu_short_prompt_probe.py --path pathb` +
   `npu_vs_cpu_correctness.py --path pathb` with
   `SPECULA_NPU_VARIANT=w4a16-local`. Gate: cos ≥ 0.95 post-quant.
5. AC sweep vs the 18.12 t/s fp16 baseline. Projection per the
   side-quest: ~17 ms/step → realistic 30-45 t/s k=4-8
   speculative.

## Reference — the AI Hub invocations we're reproducing

Extracted from `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`.
Keep these for comparison if our local pipeline ever produces
unexpected output.

**AI Hub converter invocation (line 51, abbreviated):**

```
qairt-converter \
  --input_network /tmp/tmpxm2z8qaj/tmpubsgov7x.onnx \
  --output_path /tmp/tmpxm2z8qaj/graph_8z3ykv_j.dlc \
  --preserve_io_datatype past_key_values.0.key past_key_values.0.value ... \
  --preserve_io_datatype past_key_values.0.key past_key_values.0.value ... \
  --preserve_onnx_output_order
```

(They pass `--preserve_io_datatype` twice with the same list — both
are 116 names. The first instance is for inputs, the second for
outputs; the tool handles the duplication.)

**AI Hub quantizer invocation (line 72, abbreviated — with the bug):**

```
qairt-quantizer \
  --input_dlc /tmp/tmpxm2z8qaj/graph_8z3ykv_j.dlc \
  --output_dlc /tmp/tmpxm2z8qaj/graph_8z3ykv_j.dlc \
  --preserve_io_datatype past_key_values.0.value past_key_values.1.key ...  ← MISSING past_key_values.0.key
  --input_list /tmp/tmpxm2z8qaj/input_list.txt \
  --weights_bitwidth 4 \
  --act_bitwidth 16
```

115 names, first one dropped. **That's the bug.** Our local version
drops the flag entirely, so "which names get preserved" is a
non-question.

## Timeline estimate

| step | duration | notes |
|---|---|---|
| Install QAIRT 2.42 SDK (one-time) | 30–60 min | if not already done |
| Transfer inputs from NAS | 5–10 min | 6.3 GB |
| Step 1 qairt-converter | 1–2 min | |
| Step 2 build input_list | 2–3 min | writing 60 × 61 small files |
| Step 3 qairt-quantizer (PTQ) | **60–90 min** | single biggest time sink |
| Step 4 qnn-context-binary-generator | 1–3 min | |
| Step 5 qairt-dlc-to-json | ~30 s | |
| qnn-net-run self-test | ~5 min | CPU sim of one sample |
| Transfer back to NAS | 3–5 min | ~900 MB |
| **Total wall clock** | **~2–2.5 hours** | mostly step 3 |

Budget one afternoon.

## Risk register + fallbacks

- **qairt-quantizer rejects the pathb ONNX's rotary hoist.**
  Unlikely — AI Hub's 2.45 tools accepted it at the structural
  level (job succeeded; the failure was in the orchestration of
  preserve_io_datatype, not op-lowering). If this fires, the
  fallback is to pass `--preserve_io_datatype` ourselves for all
  116 names (correctly, without dropping one).
- **Output binary loads in qnn-net-run CPU sim but fails in ARM64
  ORT-QNN.** Probably a version mismatch — double-check
  `--qairt_version 2.42`. If that's set and it still fails,
  inspect the first 32 bytes of our binary vs Qualcomm's reference
  (`models/qualcomm-qwen3-4b-ref/.../qwen3_4b_part_1_of_4.bin`) —
  they should be byte-identical in the magic region
  (`00 00 00 02 00 00 00 03 ...`).
- **PTQ calibration produces a numerically-broken binary.**
  Likely cause: calibration sample distribution doesn't cover
  runtime activation ranges. We provide 60 realistic samples from
  Bundle A (prompts × decode-step positions); if PTQ still picks
  bad ranges, try swapping to Bundle B (20 step-0-only samples —
  `bundle_b_pathb_ctx256.npz` in the NAS drop). Or reduce to
  subset of samples that all decode-step-0 to sharpen the ranges.
- **Accuracy degradation too large after PTQ.** Post-quant cos vs
  CPU FP32 target is ~0.95 (per Qualcomm's published ~3–5 pp
  accept-rate hit on benchmarks). If we see cos < 0.85 or accept
  rate drops >10 pp at k=2, try `--act_bitwidth 16 --weights_bitwidth 8`
  (w8a16) as a fallback — larger weights but easier quant, smaller
  accuracy hit. AI Hub accepted this at the options level; qairt
  tools should too.

## Contact points on the ARM64 side

- Lead doc: `docs/w4a16_investigation.md` (option ordering,
  investigation log, why local QAIRT became the primary path).
- Bug evidence: `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`.
- Runtime side-quest: `results/qwen3_4b_genie_w4a16_probe.md` —
  validates ORT-QNN loads full-quant-IO binaries; contains the
  latency projections we're chasing.
- Existing compile plumbing for reference:
  `scripts/compile_qwen3_ai_hub.py` (we're replacing the cloud
  part with local QAIRT but the pre-prep + staging is reused).
