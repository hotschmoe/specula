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

## Suggested order
1. ✅ Quant baseline bench (#9, done — Q4_0 for dense).
2. ✅ `qwen35` arch (#10, already in llama.cpp; 27B+35B run hybrid on NPU).
3. ▶ integer-HMX (#11): int8×int8 stepping stone, then int4×int16 (w4a16).
4. Then: SSM HTP kernel (gated-delta-net) + `-ngl`/NHVX tuning.
