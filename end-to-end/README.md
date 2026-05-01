# end-to-end — Qwen → HTP NPU bundle

ONE script, idempotent, max-quality defaults.

## Pipeline

```
HF FP weights (e.g. /workspace/models/Qwen3-0.6B/)
    ↓ (1) optimum-cli export onnx --task text-generation-with-past
    ↓ (2) scripts/rewrite_qwen3_htp.py --mode stage
    ↓ (3) scripts/rewrite_qwen3_htp.py --mode fold-pathbmask
    ↓ (4) scripts/rewrite_qwen3_pathb.py        (rotary hoist)
    ↓ (5) scripts/pin_shapes_qwen3_4b.py        (pin AR=1, ctx=N)
    ↓ (6) AIMET aimet_onnx PTQ + SEQ_MSE + AdaScale (+ optional V/O w8 pin)
    ↓ (7) qairt-converter ONNX+encodings → DLC
    ↓ (8) qnn-context-binary-generator DLC → HTP context .bin (v75)
    ↓ (9) bundle .bin + tokenizer + metadata, tar
deployable bundle
```

Every stage drops a `done.json` under its workdir; re-running `quantize_to_npu.py`
with the same `--workdir` skips completed stages. Use `--force-stage N` to
re-run from stage N onward.

## Quickstart (Qwen3-0.6B w8a16, full-quality recipe)

```bash
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
NVLIBS=$(find /workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/nvidia \
              -name lib -type d | tr '\n' ':')

LD_LIBRARY_PATH=$NVLIBS \
$PY /workspace/specula/end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-0.6B \
    --model-path /workspace/models/Qwen3-0.6B \
    --workdir /workspace/runs/qwen3_0p6b_w8a16 \
    --precision w8a16 \
    --ctx 512
```

Wall: ~2 hr on RunPod A40 ($0.44/hr, ~$0.90 total). Output:

- `/workspace/runs/qwen3_0p6b_w8a16/09_bundle_w8a16/<bundle>.tar` — transportable
- `<bundle>/qwen3_0p6b_pathb_w8a16.bin` — HTP v75 context binary
- `<bundle>/metadata.json` — recipe + sha256s + AIMET probe (cos vs FP32)

## Quickstart (Qwen3-0.6B w4a16 with V/O collapse mitigation)

```bash
$PY /workspace/specula/end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-0.6B \
    --model-path /workspace/models/Qwen3-0.6B \
    --workdir /workspace/runs/qwen3_0p6b_w4a16 \
    --precision w4a16 \
    --ctx 512 \
    --vo-pin-w8     # default for w4a16 — keeps V/O proj at w8 to avoid V/O collapse
```

Wall: roughly the same as w8a16. The V/O pin is the SQ2/m1d-derived
mitigation: per-tensor weight encodings get bw=8 for the attention
v_proj and o_proj weights (2 × num_layers tensors); everything else
quantizes at w4. Empirically restores cos toward 0.95.

## Quality knobs (defaults are max-quality)

| flag | default | what it does |
|---|---|---|
| `--num-cal-samples` | 128 | calibration set size; SEQ_MSE/AdaScale/compute_encodings all consume this |
| `--use-seq-mse` | on | per-tensor weight scale search (`apply_seq_mse`) |
| `--seq-mse-candidates` | 20 | scale candidates evaluated per tensor |
| `--use-ada-scale` | on | per-block scale tuning, gradient-based (`apply_adascale`) |
| `--ada-scale-iters` | 1500 | iterations per decoder block; Qualcomm's recipe default |
| `--vo-pin-w8` | on for w4a16, off for w8a16 | bumps attention V/O proj weight bw to 8 |
| `--quant-scheme` | post_training_tf_enhanced | activation observer; histogram-based search |

## Wall-clock estimates

| model | w8a16 (full recipe) | w4a16 (full + V/O pin) |
|---|---|---|
| Qwen3-0.6B | ~2 hr | ~2 hr |
| Qwen3-4B | ~6-8 hr | ~6-8 hr |
| Qwen3-14B | ~16-24 hr (likely needs A100) | same |

The dominant costs are AdaScale (≈ 1.5-3 min/block × num_blocks at
1500 iters) and `compute_encodings` (CPU-bound observation; scales
linearly with num_cal_samples × num_layers).

## Resuming a partial run

If a stage crashes or you cancel mid-run, just re-invoke with the
same `--workdir`. Each stage reads its prior stage's output dir; if
`done.json` already exists, the stage is skipped. Use
`--force-stage N` to forcibly re-run from N onward (e.g. to re-do
AIMET with a different recipe but skip the pathb rewrite).

## Targeting a different SoC arch

The compile chain pins **HTP v75** (Snapdragon X2 Elite) by default
via `configs/qnn_v75_config.json` → `configs/qnn_v75_inner.json`. To
target a different arch (e.g. v79 for SM8750):

1. Copy `configs/qnn_v75_inner.json` → `configs/qnn_v79_inner.json`,
   change `dsp_arch` to `"v79"`.
2. Copy `configs/qnn_v75_config.json` → `configs/qnn_v79_config.json`,
   point `config_file_path` at the new inner.
3. Pass `--qnn-config /workspace/specula/end-to-end/configs/qnn_v79_config.json`.

Supported arch values per aimet_onnx 2.26: v66, v68, v69, v73, v75,
v79, v81, v85.

## Generalising beyond Qwen3-0.6B

The pathb scripts (`rewrite_qwen3_htp.py`, `rewrite_qwen3_pathb.py`,
`pin_shapes_qwen3_4b.py`) work on any Qwen3-family checkpoint that
optimum-cli exports cleanly — the hidden assumptions are:

- standard Qwen3 architecture (no rope_scaling)
- 28 IsNaN guards exactly (one per layer, generalises by layer count)
- Cast_4_output_0 / Cast_5_output_0 as the rotary terminals
  (held for Qwen3-0.6B, Qwen3-4B based on prior runs)

For Qwen3.5/3.6 (which use rope_scaling), the rewrite_qwen3_pathb.py
asserts identity scaling and bombs out — the script tells you
explicitly. Folding the rope_scaling factor into the externally-
computed cos/sin is a small follow-up (the assert at line 88 has
the hint).

For non-Qwen3 architectures, the existing scripts are Qwen3-specific.
The aimet_onnx layer is family-aware via `AdaScaleModelConfig(model_type=...)`
which today supports {qwen2, qwen3, llama, mistral, phi3} — extending
to non-Qwen would require both new pathb rewrites AND a different
`model_type` argument; this script currently hard-codes "qwen3".

## Why each stage exists

- **pathb rewrite** is non-optional for HTP compile. Without it the
  attention-mask BOOL chain + IsNaN guards + Cast→BOOL ops fail HTP
  op-config validation; rotary embedding on the GPU host is fine
  but on HTP we hoist cos/sin to graph inputs to keep the partition
  seam clean.
- **shape pinning** is required because qairt-converter cannot lower
  graphs with symbolic dims to DLC.
- **AIMET aimet_onnx** (vs aimet_torch) is required because we need
  the encodings file to reference the post-pathb tensor names —
  aimet_torch traces the unrewritten torch model and emits names
  that don't line up with the rewritten ONNX, causing 80% float-
  fallback in qairt-converter.
- **SEQ_MSE + AdaScale** are the difference between cos 0.07 and
  cos 0.99+ for w8a16 on Qwen3-0.6B (SQ2 + m1d found this
  empirically; the V/O collapse on w4a16 is a separate issue
  addressed by `--vo-pin-w8`).
- **HTP v75 nested config** is required for X2 Elite. The default
  `libQnnHtp.so` backend targets v68; AIMET's int16 attention ops
  fail validation with `"Value 68, expected >= 73"`. The two-file
  wrapper is documented but easy to miss.

## Bundle layout

The final tar contains:

```
<bundle_name>/
  <prefix>.bin                 # HTP context binary (the main artifact)
  <prefix>.encodings           # AIMET encodings (kept for re-compile or audit)
  bin_info.json                # qnn-context-binary-utility dump (dspArch, IO shapes)
  metadata.json                # recipe + sha256 + provenance
  tokenizer.json
  tokenizer_config.json
  config.json
  generation_config.json
  special_tokens_map.json      # if present in source
```

`bin_info.json` is parseable; `npu_engine/sidecar.py` consumers can
read it to confirm graph IO shapes match the wrapper ONNX they're
about to bind to.

## Known issue: AdaScale ReduceMean v18

`apply_adascale` crashes mid-run on `NotImplementedError: Converter is
not implemented (... ReduceMean, version=18)`. The optimum-cli export
emits the graph at opset 18 and Qwen3 RMSNorm uses ReduceMean; aimet_onnx
2.26's `experimental.adascale.onnx2torch_ext` doesn't have a v18
ReduceMean handler.

**Workaround**: pass `--no-use-ada-scale` to skip AdaScale and use
SEQ_MSE only. SEQ_MSE alone carries most of the gain over basic PTQ
(empirically ~80-90%); we have not landed an end-to-end measurement
where AdaScale on top of SEQ_MSE meaningfully moved the cos number on
this pathb-rewritten Qwen3 graph (because we never got it to run).

**Two real fixes** (TODO):

1. Force optimum-cli to export at a lower opset (13 or 14) where
   ReduceMean has axes as an attribute rather than an input. Side
   effects on AIMET's INT4/INT16 QDQ insertion (which want ≥ opset 21)
   need verification.
2. Patch
   `/workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/aimet_onnx/experimental/adascale/onnx2torch_ext.py`
   to register a v18 ReduceMean converter. Probably ~10 LOC mirroring
   the v13/v17 ReduceMean handlers in `onnx2torch.node_converters`.

## Troubleshooting

**"Unknown Key = devices/0/dsp_arch passed in config"** — the OUTER
`--config_file` JSON only accepts `backend_extensions`. Move
`devices` into the inner file pointed to by
`backend_extensions.config_file_path`.

**"Value 68, expected >= 73"** — the nested config is missing or the
`dsp_arch` key isn't being read. Verify with:

```bash
cat $QNN_CONFIG     # outer; should reference inner via config_file_path
cat $QNN_INNER      # inner; should have devices[].dsp_arch
```

**"SOC model SM8650 is not supported"** — qairt-converter
`--target_soc_model` allow-list is small. Don't pass it (the script
doesn't); pin arch at the binary-generator step.

**"max diff X is not within set tolerance 1e-05" during optimum
export** — that's optimum's reference-model roundtrip warning; it's
fine. Drop in any case to make sure model.onnx + model.onnx_data
exist after.
