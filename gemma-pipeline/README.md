# gemma-pipeline — Gemma 4 → Hexagon NPU bundle

Sibling of `end-to-end/` (the Qwen3 → HTP pipeline). Same shape — one
idempotent script, max-quality defaults — but retargeted at **Gemma 4**,
whose decoder differs from Qwen3 in ways that touch every graph-surgery
stage. Read `ARCHITECTURE_NOTES.md` before touching any rewrite script.

**Target model:** `google/gemma-4-E2B-it` — the smallest Gemma 4
(2.3B effective / ~5.1B param table, 35-layer text decoder, 128k native
context). **Target SoC:** Snapdragon X2 Elite, Hexagon HTP v75.
**Goal:** a loadable w4a16 (or w8a16) NPU bundle at ctx 32768.

## Why Gemma 4 is the *right* long-context target

The Qwen3 project hit a structural wall: dense global attention can't
tile a 32k KV slice into the NPU's VTCM (`current_status.md` sessions
31–32), and Qwen3-4B was never sliding-window-trained. Gemma 4 is
**natively sliding-window** — 28 of 35 layers use a 512-token window,
only ~7 are global — plus 8:1 GQA, 1 KV head, and 20 KV-shared layers.
The 32k wall the Qwen3 pipeline could not pass is **absent by
construction** here. See `ARCHITECTURE_NOTES.md` §"the long-context
goal".

## Pipeline

```
HF FP weights (google/gemma-4-E2B-it)
    ↓ (1) optimum-cli export onnx  — text decoder only (strip vision/audio towers)
    ↓ (2) scripts/rewrite_gemma4_htp.py --mode stage          [TO BUILD]
    ↓ (3) scripts/rewrite_gemma4_htp.py --mode fold-pathbmask  [TO BUILD]
    ↓ (4) scripts/rewrite_gemma4_pathb.py  (dual+partial rotary hoist, PLE) [TO BUILD]
    ↓ (5) scripts/pin_shapes_gemma4.py     (pin AR=1, ctx=N, per-layer head_dim) [TO BUILD]
    ↓ (6) AIMET aimet_onnx PTQ + SEQ_MSE (+ AdaScale if ReduceMean-v18 patched)
    ↓ (7) qairt-converter ONNX+encodings → DLC          (reused, arch-agnostic)
    ↓ (8) qnn-context-binary-generator DLC → HTP .bin    (reused, configs/)
    ↓ (9) bundle .bin + tokenizer + metadata, tar         (reused)
deployable bundle
```

Stages 7–9 are reused unchanged from `end-to-end/`. Stages 2–6 are the
Gemma-specific work; the `[TO BUILD]` scripts do not exist yet —
`scripts/README.md` specifies them. `quantize_to_npu.py` orchestrates
and **stops with a clear error** at the first unbuilt stage rather than
pretending to run.

## Where each stage can run

| stage | needs | runs on current x86 dev box? |
|---|---|---|
| 1 export | torch + optimum, CPU | **yes** |
| 2–5 rewrites | onnx (CPU) | **yes** (once the scripts exist) |
| 6 AIMET SEQ_MSE/AdaScale | **CUDA GPU** (V100/A100 class) | **no** — Intel B50 is not CUDA; rent RunPod A100 or use the AI Hub cloud path |
| 7–8 QAIRT compile | QAIRT SDK, CPU (x86 Linux or ARM) | **yes** (QAIRT 2.42 + 2.45 installed under `C:/Qualcomm/AIStack`) |
| 9 bundle | tar, CPU | **yes** |
| on-device validation | Snapdragon X2 Elite + Hexagon NPU | **no** — that is the other (ARM) machine |

The AIMET step is the only hard CUDA gate. Two ways past it:

1. **RunPod A100** — the `end-to-end/COLD_START.md` runbook. Rent ~3–8 h,
   ~$3–10. The Gemma rewrites must be built first so the cloud run has
   something to execute.
2. **Qualcomm AI Hub cloud** — `submit_ai_hub.py`. AI Hub compiles +
   (int8-only) quantizes + validates on Qualcomm's own cloud devices;
   needs only an API token, no local CUDA and no local NPU. AI Hub has
   **no Gemma recipe** (investigated — `AI_HUB.md`), so the graph
   surgery is still ours; but AI Hub can take our ONNX and produce a
   **w8a16** bundle end to end on a token alone (w4a16 still needs
   local AIMET). **Needs a token configured:** `qai-hub configure
   --api_token <TOKEN>` — none is set on this box.

## Quickstart (once the rewrite scripts exist)

```bash
# on a RunPod A100 pod, after end-to-end/COLD_START.md setup:
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
$PY /workspace/specula/gemma-pipeline/quantize_to_npu.py \
    --model-id google/gemma-4-E2B-it \
    --model-path /workspace/models/gemma-4-E2B-it \
    --workdir /workspace/runs/gemma4_e2b_w4a16_ctx32768 \
    --precision w4a16 \
    --ctx 32768
```

## Smoke test that runs *now* (no GPU, no NPU)

The model-config adapter is pure stdlib and is fully exercisable on the
x86 dev box against a downloaded Gemma `config.json`:

```bash
python -m lib.model_config <path-to-gemma-4-E2B-it-dir> google/gemma-4-E2B-it
```

It prints the normalized `ModelInfo` — layer-type split, KV-owning
layer count, PLE width, dual-RoPE thetas — i.e. everything the
downstream stages key off.

## Status

See `STATUS.md` for the honest current state, what is built vs scoped,
and the blockers between here and a loadable bundle.

## Files

```
gemma-pipeline/
├── README.md              # this file
├── ARCHITECTURE_NOTES.md  # Qwen3 vs Gemma 4 difference matrix — READ FIRST
├── AI_HUB.md              # what Qualcomm AI Hub can/cannot do for Gemma 4
├── STATUS.md              # current state, blockers, next actions
├── quantize_to_npu.py     # orchestrator (stops cleanly at unbuilt stages)
├── submit_ai_hub.py       # Qualcomm AI Hub cloud path (token required)
├── configs/               # HTP v75 / v81 QNN compile configs (reused)
├── lib/
│   ├── __init__.py
│   └── model_config.py    # Gemma-4-aware ModelInfo — BUILT, runnable now
└── scripts/
    └── README.md          # spec for the 4 Gemma-specific rewrite scripts
```
