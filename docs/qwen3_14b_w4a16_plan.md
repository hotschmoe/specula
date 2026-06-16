# Qwen3-14B w4a16 — how to quantize the saved 14B outputs (research + plan)

Created 2026-06-16. We have a **complete w8a16** 14B NPU bundle (built on the
Threadripper build server, `docs/threadripper_build_server.md`). We also want
to try **w4a16** (half the weight bytes → ~7 GB resident, faster decode). This
doc researches *how*, given what's already on the box.

## What's already saved on the box (the cheap re-entry points)

`/mnt/vm_8tb/specula-build/runs/qwen3_14b_w8a16/` keeps:
- `05_pathb_ctx512/` — the pinned pathb ONNX (split input)
- `06_split/part1..part10/` — the **10 per-part fp32 ONNX sub-graphs** (the
  exact converter inputs; the balanced split: embed + 8×5-layer + lm_head)

So w4a16 does **not** need to re-run export/rewrites/split — just re-quantize
from `06_split`.

## Option A — `qairt-quantizer` native w4a16 (no AIMET, runs on the box NOW) ★ try first

The w8a16 build used `--weights_bitwidth 8`. **w4a16 is the same chain with
`--weights_bitwidth 4`** — qairt-quantizer takes 4 or 8 (verified from its
`--help` + `scripts/qairt_quantize_4b_parts.py`). Per part, in the
`specula-qairt:2.45` image:

```
qairt-converter --input_network 06_split/partN/model.onnx --output_path partN.dlc --preserve_io_datatype
qairt-quantizer  --input_dlc partN.dlc --output_dlc partN_q.dlc \
    --weights_bitwidth 4 --act_bitwidth 16 --bias_bitwidth 8 \
    --use_per_channel_quantization --use_per_row_quantization \
    --act_quantizer_calibration min-max --act_quantizer_schema asymmetric \
    --apply_algorithms cle
qnn-context-binary-generator ... (then bundle)
```

- **Free, local, reuses `06_split`.** Reuse `build_qairt_14b.sh` /
  `build_part8_9.sh` with `--weights_bitwidth 4` + a `w4a16` output run dir.
- **Quality caveat:** 4-bit is far more sensitive than 8-bit. Basic PTQ at
  w4a16 likely needs the quality levers we learned on the 4B
  (`docs/qai_hub_recipe.md`): **per-channel + per-row weights, CLE, the
  RMSNorm-skip config** (`default_config_llama.json` — keeps RMSNorm internals
  in float; without it 4B w4a16 collapsed to cos 0.51), and **calibration**
  (`--input_list` of real activation samples — w8a16 built fine without it,
  but w4a16 probably needs it; capture via the `_4b` calib scripts adapted to
  40 layers). Validate per-part cos vs the fp reference before trusting it.
- This is the **basic-PTQ ceiling** (~0.975 cos on the 4B); the last ~0.02 gap
  to Qualcomm-grade needs Option B.

## Option B — AIMET w4a16 (higher quality, needs a GPU)

AIMET's **SEQ_MSE + AdaScale** are what close the basic-PTQ → shipping-quality
gap at w4a16 — but they are **CUDA-only** (V100/A100 per Qualcomm's own
`quantize.py`). AIMET emits an `encodings.json` → `qairt-converter
--quantization_overrides` (the e2e pipeline's normal path, which we *skipped*
for w8a16).

- The box is **x86_64 Linux**, so `aimet_onnx`'s manylinux wheel installs
  there — but **only basic PTQ runs on CPU**; SEQ_MSE/AdaScale still demand a
  CUDA GPU. So on the current (GPU-less) box, AIMET gives no quality edge over
  Option A.
- Real Option B = run AIMET on a CUDA box: the planned **DGX Spark** (after the
  aimet_onnx→aimet_torch ARM port, or via PyTorch sbsa CUDA), or a **RunPod
  A100 rental** (~$5–30 for a 14B SEQ_MSE pass). Then bring the `encodings`
  back to the box and re-run qairt-converter (`--quantization_overrides`) +
  quantizer + context-bin from the saved `06_split`.

## Option C — Qualcomm AI Hub — NOT viable for w4a16

`submit_quantize_job` is **int8-weights only** (no int4), plus the
`preserve_io_datatype` drop bug ([[reference_ai_hub_preserve_io_bug]]). AI Hub
can compile a pre-quantized w4a16 ONNX, but it cannot *produce* w4a16. Out.

## Recommended path

1. **Now / free:** Option A on the box — re-quantize `06_split` at
   `--weights_bitwidth 4` with the quality levers, build a `w4a16` bundle,
   measure per-part cos vs fp. This tells us if basic-PTQ w4a16 is good enough.
2. **If quality short:** Option B — AIMET SEQ_MSE/AdaScale on the DGX Spark /
   RunPod, encodings → qairt-converter overrides → rebuild from `06_split`.
3. Bundle + deploy exactly like w8a16 (same 10-part layout, same session-
   ceiling deploy concern — see `docs/next_session_npu_engine_14b.md`).

(Note: w4a16 ~7 GB still splits into the same ~10 parts for the HTP per-context
ceiling; the `.bin`s shrink (~half) but the part *count* is set by layer count,
not byte size — revisit whether fewer/larger parts fit once weights are 4-bit.)
