# gemma-pipeline — status

Created 2026-05-22. Honest current state of the Gemma 4 → Hexagon NPU
pipeline. Read with `README.md` (overview) and `ARCHITECTURE_NOTES.md`
(the Qwen3-vs-Gemma difference matrix).

## What this is

A new pipeline subdirectory, sibling to `end-to-end/` (the Qwen3 → HTP
pipeline), retargeted at **Gemma 4 E2B** — the smallest Gemma 4
(35-layer text decoder, 128k native context). Goal: a loadable w4a16
NPU bundle at ctx 32768 for Snapdragon X2 Elite (HTP v75).

## Built (this session) — runs on the current x86 dev box

- `lib/model_config.py` — Gemma-4-aware `ModelInfo`. Parses the nested
  `text_config`, surfaces dual-RoPE thetas, the per-layer attention
  map, PLE width, KV-shared layer count, logit soft-cap. Pure stdlib —
  testable now: `python -m lib.model_config <gemma-dir>`.
- `configs/` — HTP v75 + v81 QNN compile configs (copied from
  `end-to-end/`, paths repointed to `gemma-pipeline/`).
- `quantize_to_npu.py` — orchestrator. Runs stage 1 (optimum export),
  resolves + prints the model plan, and **stops with an explicit,
  actionable error** at the first unbuilt Gemma stage. Does not fake
  success.
- `submit_ai_hub.py` — Qualcomm AI Hub cloud-path preflight + launcher.
- `ARCHITECTURE_NOTES.md`, `README.md`, `scripts/README.md` — the
  design + the spec for the four scripts still to build.

## Not built — scoped, not started

The four Gemma-specific graph-surgery scripts (`scripts/README.md`):

1. `rewrite_gemma4_htp.py` — HTP-illegal-op strip + dual mask fold
2. `rewrite_gemma4_pathb.py` — dual+partial rotary hoist + PLE preserve
   *(highest uncertainty — PLE has no Qwen3 precedent)*
3. `pin_shapes_gemma4.py` — per-layer-type shape pinning
4. `strip_multimodal_wrapper.py` — only if stage 1 fails on the wrapper

Stages 6–9 (AIMET → QAIRT → bundle) reuse `end-to-end/lib` unchanged —
they are ONNX-level and architecture-agnostic — but are unreachable
until 1–5 produce a pinned pathb ONNX.

## Blockers — why this cannot finish on the current machine

The task ("get Gemma 4 onto the Hexagon NPU, do it all on this x86
box") has three hard physical blockers on **this** machine (x86 AMD,
Intel B50 GPU):

1. **AIMET SEQ_MSE/AdaScale needs a CUDA GPU.** The Intel B50 (Arc Pro
   B50) is not CUDA. `aimet_onnx`'s GPU wheels are CUDA-only. Stage 6
   cannot run here → rent a RunPod A100 (`end-to-end/COLD_START.md`,
   ~$3–10) **or** use the AI Hub cloud path.
2. **The Hexagon NPU is on the other machine.** This repo's NPU is the
   Snapdragon X2 Elite laptop. On-device load/throughput validation
   (`npu_engine/`) happens there, not here.
3. **No Qualcomm AI Hub token configured.** `~/.qai_hub/client.ini`
   is absent. `submit_ai_hub.py --check` confirms this. AI Hub job
   submission needs `qai-hub configure --api_token <TOKEN>` first.

What *does* run here: the model-config adapter, the rewrite scripts
once written (ONNX/CPU), QAIRT compile (QAIRT 2.42 + 2.45 are installed
under `C:/Qualcomm/AIStack`), bundle assembly, and — with a token — AI
Hub job submission.

## AI Hub investigation (session 33 — see AI_HUB.md)

Dug into whether Qualcomm AI Hub shortcuts any of this. Result:

- **No Gemma recipe** in `qai-hub-models` (Llama/Qwen/Phi/Mistral
  only). The one-command path does not exist for Gemma 4.
- **BYO path is architecture-locked** — can't feed Gemma 4 into the
  Llama recipe. The graph surgery is unavoidably ours.
- **AI Hub server-side jobs are still useful**: `submit_compile_job`
  (ONNX → QNN binary) and `submit_inference_job` (validate on a real
  cloud Snapdragon) run with no local CUDA / NPU — token only.
- **`submit_quantize_job` is int8-weights only — no int4.** So a
  full-AI-Hub flow yields **w8a16**; **w4a16 still needs local AIMET**.

## Route 1 progress (session 33 — in flight)

w8a16-via-AI-Hub route. Environment + model are done; now iterating
AI Hub compile of the fp16 decoder to find every blocker.

Done:
- `qai-hub` installed; token configured + verified.
- **`Snapdragon X2 Elite CRD` confirmed as an AI Hub device** — compile
  + validate directly for our v75 target, no device mismatch.
- optimum-cli cannot export Gemma 4 → switched stage 1 to the ungated
  pre-export `onnx-community/gemma-4-E2B-it-ONNX` (fp16 decoder 4.76 GB,
  downloaded).
- Decoder inspected; architecture predictions confirmed (30 KV inputs
  = 15 KV-owning layers, per-layer head_dim 256/512, PLE as an input).

### AI Hub compile-blocker ledger (ctx=512, AR=1, fp16 decoder)

| # | job | blocker | fix | status |
|--|--|--|--|--|
| 1 | jgkr100n5 | external weights not uploaded | repack to qai-hub model dir (one .onnx + one .data) | **cleared** |
| 2 | jgoex7kdp | `SimplifiedLayerNormalization` op unknown | decompose to primitives (`rewrite_gemma4_htp.py`) | **cleared** |
| 3 | jgkr1wkn5 | int64 inputs | compile flag `--truncate_64bit_io` | **cleared** |
| 4 | jpe4q0qv5 | `RotaryEmbedding` (com.microsoft) unsupported | decompose to standard rotary | **TODO** |
| 5 | (expected) | `GroupQueryAttention` (com.microsoft) unsupported | decompose to MatMul SDPA + KV concat | **TODO** |

Blockers 1–3 were quick iterative fixes. Blockers 4–5 are the real
graph surgery — effectively the `rewrite_gemma4_pathb.py` work
ARCHITECTURE_NOTES.md flagged as hardest:

- **RotaryEmbedding** ×50 — non-interleaved; `cos_cache_local` /
  `cos_cache_global` ([131072, N]); some nodes are **partial rotary**
  (global layers, partial_rotary_factor 0.25). Decompose to
  reshape→gather cos/sin→rotate-half→concat. ~tractable.
- **GroupQueryAttention** ×12 — `num_heads=8, kv_num_heads=1,
  local_window_size=512, softcap=0`. The additive mask is already
  supplied as input[10] (`gqa_attention_bias`), so decomposition is
  KV-cache concat + GQA head-expand + scaled MatMul-softmax-MatMul.
  The `local_window_size` sliding band is a **no-op at ctx ≤ 512**
  (whole context fits the window) — only needs an explicit band mask
  at ctx > 512.

Verification: both decompositions must be checked numerically
(`onnxruntime` runs the original — it *has* the com.microsoft ops — vs
the rewritten graph; compare logits) before trusting an AI Hub compile.

## Recommended path to a bundle

Gemma 4 is a **better** long-context target than dense Qwen3-4B ever
was — native sliding-window attention removes the 32k VTCM wall the
Qwen3 project could not pass (`ARCHITECTURE_NOTES.md` §long-context).
Order:

1. **Download `google/gemma-4-E2B-it`**; run `quantize_to_npu.py
   --dry-run` to confirm the model-config resolves.
2. **Build the four `scripts/` rewrites**, validating each at cos ≥
   0.99 with `probe_gemma4_equivalence.py`. Budget: PLE + partial RoPE
   are ~1 session each; the htp/pin scripts are lighter ports. This is
   the gating work — AI Hub does not shortcut it.
3. **Route 1 (w8a16, no CUDA, no rent):** `submit_quantize_job` +
   `submit_compile_job` + `submit_inference_job` — full bundle via AI
   Hub on a token alone. Validates the whole chain on real silicon.
4. **Route 2 (w4a16, if quality short):** AIMET SEQ_MSE on a RunPod
   A100 for the w4a16 encodings, then `submit_compile_job` (or local
   QAIRT). ~$3–10. Reuses every other stage.
5. **Deploy** to the X2 Elite laptop `npu_engine` (compile for HTP
   v75 — AI Hub's cloud device is X Elite / v73; see AI_HUB.md caveat).

## Open questions

- ~~Does `qai-hub-models` have a Gemma 4 recipe?~~ **No** (resolved).
- ~~Does `qai-hub list-devices` show an X2 Elite target?~~ **Yes** —
  `Snapdragon X2 Elite CRD` (resolved).
- ~~Does `optimum-cli` export Gemma 4?~~ **No** — use the
  `onnx-community` pre-export instead (resolved).
- Can AI Hub `submit_compile_job` ingest the (shape-pinned) merged
  decoder ONNX and produce an X2 Elite QNN context binary?
- Does the `onnx-community` decoder need our HTP rewrites, or does AI
  Hub's lowering handle the mask/rotary ops?
- Does w8a16 server-side PTQ clear a usable quality bar (no SEQ_MSE)?

All resolved by doing — capture answers here as they land.
