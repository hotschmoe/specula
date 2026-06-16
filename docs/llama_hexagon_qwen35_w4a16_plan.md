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

## Suggested order
1. Quant baseline bench (A is unblocked regardless).
2. `qwen35` arch → get the 27B loading + running hybrid on the NPU (the goal).
3. w4a16 integer-HMX kernel for PP speed once correctness is proven.
