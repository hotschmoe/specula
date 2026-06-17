# llama.cpp Hexagon — our work plan (qwen35 + w4a16)

Working branch: **`hotschmoe-npu-work`** in
`C:\Users\hotschmoe\Documents\GitHub\llama.cpp` (off master 45cac7c). This is
where we add the Qwen3.6 (`qwen35`) architecture and, as a stretch, a
w4a16 integer-HMX path. Background: `docs/llama_hexagon_build_setup.md`,
`docs/htp_memory_ceiling_problem.md`, [[reference_llamacpp_hexagon_npu_works]].

## Verified perf landscape (2026-06-16, X2E)

| path | runtime | quant | PP t/s | TG t/s | scales to 27B? |
|------|---------|-------|--------|--------|----------------|
| Qualcomm 4B AR128 | ORT-QNN | **w4a16** | **2224** (peak 2310) | 26 | ❌ (~10 GB cap) |
| Qualcomm 4B AR1 | ORT-QNN | w4a16 | ~26 | ~26 | ❌ |
| 4B | **llama.cpp HTP** | Q4_0 | 102 | 18 | ✅ |
| 14B (hybrid -ngl34) | **llama.cpp HTP** | Q4_0 | 41 | 11 | ✅ |

The w4a16+AR128 PP (2224) is **~22× the llama.cpp HTP Q4_0 PP (102)**. Root
cause (verified in `ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c`): llama.cpp
**dequantizes Q4_0/MXFP4 → fp16 then does an fp16 HMX matmul**; Qualcomm does
**int4×int16 integer HMX matmul (no dequant)**. HMX int throughput is ~2–4×
fp16, plus no dequant overhead → the gap. (AR128 batching helps too, but the
integer-HMX path is the bulk.)

**Strategic tension:** ORT-QNN w4a16 = fast PP but capped ~10 GB (no 27B);
llama.cpp = scales (14B runs, 27B will) but fp16-HMX PP is 22× slower. Goal:
**llama.cpp + w4a16 integer-HMX = fast AND scalable.**

## Hardware facts (must design around)
- **Exactly 4 HTP sessions** (HTP0–3, domains 3/7/11/15; 5th → `error 0x200`),
  **~2 GB each → ~8 GB max resident on the NPU.** This IS the ceiling.
- >8 GB models → hybrid `-ngl` (overflow to Adreno GPU / CPU over unified
  48 GB). 14B (8.5 GB) → ~6 layers off-NPU; 27B (16 GB Q4_0) → ~50/50.
- Runtime needs `ADSP_LIBRARY_PATH` → skel+signed-cat dir; signed skel
  catalog via WDK `inf2cat` (see build doc).

## Workstream A — `qwen35` arch support (PRIMARY, for the 27B)
`Qwen3.6-27B` GGUF is `general.architecture=qwen35`; llama.cpp can't load it.
- Add the `qwen35` arch: hparams, tensor map, graph build. It's a hybrid
  **full-attention + linear-attention (gated-delta-net SSM)** model
  (16 full-attn layers @ [3,7,…,63] + 48 linear; partial rotary 0.25;
  mRoPE) — see `end-to-end/lib/model_config.py` + the AI-Hub op-compilability
  work (`docs/qwen3_6_27b_op_compilability.md`) for the exact arch.
- The gated-delta-net op has **no HTP kernel** → runs on CPU/GPU via the
  hybrid scheduler (fine to start). Full-attn + FFN layers run on HTP.
- Validate: load + coherent decode on CPU first, then hybrid `-ngl` with
  HTP for the supported layers. Tokenizer is the qwen3.6 generation
  ([[reference_qwen_tokenizer_generations]]).

## Workstream B — w4a16 integer-HMX path (STRETCH, the perf prize)
1. **Baseline first:** bench Q4_0 vs MXFP4 vs Q8_0 on HTP (4B, one session);
   profile with `GGML_HEXAGON_PROFILE=1|2` to confirm dequant/fp16 is the
   bottleneck. MXFP4 (E8M0 scales) may already be the best fp16-path option.
2. **Add w4a16:** a new packed weight type + an **int4×int16 HMX matmul
   kernel** in `htp/hmx-matmul-ops.c` (skip the fp16 dequant LUT; quantize
   activations to int16 in-graph). This is the Qualcomm-blessed path.
3. Measure PP/TG vs Q4_0; target closing the 22× PP gap.

## Why not just patch ORT-QNN for big models? (decided 2026-06-16)

ORT-QNN runs **whole pre-compiled HTP context binaries** (`.bin` = HTP-only
machine code); it has **no per-op multi-device scheduler**, so it cannot
interleave HTP + CPU within a graph the way ggml does. You can hand-roll a
part-level hybrid (we did: `engine_14b_q.py`/`engine_14b_swap.py`) but it's
strictly worse: HTP still caps at 4 PDs/~8 GB, >4 parts churn-crashes the DSP
transport, and **CPU-overflow layers can't run the HTP `.bin`** (they'd need a
separate CPU graph → lose the w4a16 speed there anyway). Decisive point: the
**w4a16/HMX speed is a property of HTP-resident layers**; ANY runtime running a
>~8 GB model offloads the rest to CPU/GPU, where there's no HMX win. So
ORT-QNN-w4a16 is already the fastest option for models that *fit* the HTP
(≤~8 GB), and for bigger models the runtime choice doesn't fix the CPU
bottleneck. ⇒ Don't patch ORT-QNN. Bring **integer-HMX into llama.cpp**
(scalable hybrid) instead — task #11 — so the HTP-resident layers get the HMX
speed inside the engine that also handles big models.

## #11 detail — integer-HMX matmul (int8×int8 stepping stone → int4×int16)

Current HTP matmul (`ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c`) dequantizes
Q4_0/MXFP4 → **fp16** then runs an **fp16 HMX** matmul (`q4_0_to_fp16_lut`,
`core_dot_chunk_fp16` using `_hf` intrinsics). HMX **integer** throughput is
~2–4× fp16 + skips the dequant ⇒ the 22× PP gap. v81 exposes integer HMX
intrinsics (`_b`/`_h` suffix; `Q6_mxclracc_b/h`).

**Two-step plan, landed on separate branches off `hotschmoe-npu-work`:**
- **int8×int8** (`npu-int8-hmx`, stepping stone): reuse Q8_0 weight packing,
  quantize activations to int8 per-row, `core_dot_chunk_int8` (`_b` HMX
  intrinsics, int32 accumulate), rescale int32→fp32 by (act_scale×wt_scale).
  Simpler (no 4-bit unpack); validates the integer-HMX path + the rescale.
- **int4×int16** (`npu-int4a16-hmx`, the prize, parallel agent): int4 weights
  kept packed, int16 activations, int4×int16 HMX, rescale. The Qualcomm-blessed
  w4a16. Builds on the int8 learnings.

De-risk order: confirm dequant cost via `GGML_HEXAGON_PROFILE`; int8×int8
first; validate numerics (cos vs fp16 path) + PP on the 4B at each step.
Detailed file-level plan: subagent report captured in this session's log.

## #11 kernel — concrete spec + findings (2026-06-16)

**De-risk DONE — integer HMX is real on v81.** `hmx_hexagon_protos.h`
(SDK 6.6 / HEXAGON_Tools 19.0.07) exposes: activations `Q6_activation_ub`
(uint8), `_hf` (fp16), `_f8` (fp8) — **no int16 activation**, so "w4a16" =
int4-weight × **fp16**-activation; weights `Q6_weight_b` (int8),
`Q6_weight_ubit/sbit` (int4 unsigned/signed), `_hf` (fp16); accumulators
`Q6_mxclracc` (int) / `_hf` (fp16). HMX supports INT4/INT8/INT16/FP16 and
**applies per-output-channel scale+bias in hardware** (the `Q6_bias_mxmem2`
path the fp16 kernel already uses for Q4_0/Q8_0 scales).

**The proven fp16 template to mirror** (`hmx-matmul-ops.c::core_dot_chunk_fp16`):
```c
Q6_bias_mxmem2_A(scales);            // per-channel scales
for r,c tiles:
    Q6_mxclracc_hf();                // clear fp16 acc
    for k: Q6_activation_hf_mxmem_RR(act); Q6_weight_hf_mxmem_RR(wt);  // matmul-acc
    Q6_mxmem_AR_after_hf(out, 0);    // readout fp16
```
Two candidate integer paths:
- **int8×int8 (stepping stone, `npu-int8-hmx`):** `Q6_mxclracc()` +
  `Q6_activation_ub_mxmem_RR` (uint8 act, needs per-row fp32→uint8 quant +
  zero-point) + `Q6_weight_b_mxmem_RR` (int8 wt, reuse Q8_0 x4x2 quants, skip
  the fp16 dequant) → int32 readout → rescale by act_scale×wt_scale. Clearest
  integer semantics; more host code (activation quant + rescale).
- **int4×fp16 (the prize, `npu-int4a16-hmx`, agent):** keep `Q6_activation_hf`
  (no activation quant!) + `Q6_weight_ubit/sbit_mxmem` (int4, no dequant, ¼
  weight bandwidth). Minimal-change vs fp16 if int4-wt × fp16-act is a valid
  HMX mode with fp16 accumulate.

**BLOCKED ON: the HMX matrix-instruction semantics** — valid activation×weight
pairings, accumulator/readout per pairing, int8 tile layout, and exact
scale/zero-point application. NOT in the SDK (the `qhl_hmx` sample was removed;
`haozixu/htp-ops-lib` is fp16-only). Authoritative source = **Hexagon V79
Programmer's Reference, matrix (HMX) instructions**
(docs.qualcomm.com/.../80-N2040-60/instructions.html). **Do NOT write+run HMX
intrinsics on the DSP without these — wrong intrinsics crash the FastRPC
transport (machine-level).** Plan: confirm semantics from the V79 manual →
compile-only draft → isolated single-tile numeric test → integrate.

## Suggested order
1. ✅ Quant baseline bench (#9, done — Q4_0 for dense).
2. ✅ `qwen35` arch (#10, already in llama.cpp; 27B+35B run hybrid on NPU).
3. ▶ integer-HMX (#11): int8×int8 stepping stone, then int4×int16 (w4a16).
4. Then: SSM HTP kernel (gated-delta-net) + `-ngl`/NHVX tuning.
