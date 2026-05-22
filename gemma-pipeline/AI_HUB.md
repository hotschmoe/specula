# Qualcomm AI Hub — what it can and cannot do for Gemma 4

Investigation 2026-05-22 (session 33). Question: can Qualcomm AI Hub
get us a Gemma 4 NPU bundle without the RunPod-A100 / local-CUDA path?

**Short answer:** AI Hub has *no* Gemma support of any kind, so we must
do the graph surgery ourselves. But once we have the ONNX, AI Hub can
compile + quantize + on-device-validate it server-side — giving a
**w8a16 bundle with no local CUDA and no physical NPU**. w4a16 still
needs local AIMET.

## Finding 1 — no Gemma recipe (Path A is dead)

`qai-hub-models` ships pre-built export recipes for these LLMs:

- Llama: v2-7B, v3-8B, v3.1-8B, v3.2-1B, v3.2-3B (+ regional variants)
- Qwen: Qwen2-7B, Qwen2.5-7B, Qwen2.5-VL-7B, **Qwen3-4B**, Qwen3-4B-Instruct-2507
- Phi: Phi-3.5-Mini
- Mistral: Mistral-7B-v0.3

**No Gemma — not Gemma 2, not Gemma 3, not Gemma 4.** Gemma 4 released
2026-04-02; Qualcomm has not added a recipe yet.

So the one-command path (`python -m qai_hub_models.models.<x>.export`)
does not exist for us.

## Finding 2 — the BYO-LLM path is architecture-locked

The custom-LLM tutorial (`ai-hub-models/tutorials/llm/quantize_llama3.md`)
lets you feed your own checkpoint to an *existing* recipe — but:

> "The architecture, model type, number of hidden layers, hidden size,
> number of attention layers, number of key value layers cannot be
> changed."

That means the BYO path only accepts a *different checkpoint of the
same architecture* (e.g. a Llama-3.2-3B fine-tune into the Llama recipe).
Gemma 4 (`gemma4_text` — dual RoPE, Per-Layer Embeddings, KV sharing,
sliding-window) is not Llama/Qwen/Phi/Mistral. **We cannot ride an
existing recipe.** Confirmed: the surgery is ours to do.

## Finding 3 — AI Hub's server-side jobs (the useful part)

Three job types run entirely on Qualcomm's cloud — **no local CUDA, no
local NPU, just an API token:**

| job | input | output | notes |
|---|---|---|---|
| `submit_quantize_job` | unquantized ONNX + calibration data | quantized ONNX | **int8 weights only — NO int4.** Activations int8/int16. Beta. 500–1000 cal samples. |
| `submit_compile_job` | ONNX (or QNN context binary) | QNN context binary | targets `hub.Device("Snapdragon X Elite CRD")`; HTP graph-prep done offline |
| `submit_profile_job` / `submit_inference_job` | compiled binary | latency / outputs on a **real Snapdragon device in Qualcomm's cloud** | lets us validate without the physical X2 laptop |

## The decisive limitation: AI Hub server-side quantize is int8-only

`submit_quantize_job` does **not support int4**. So:

- **w8a16** → fully achievable through AI Hub, zero CUDA, zero rent.
- **w4a16** → NOT achievable via AI Hub quantize. Needs local AIMET
  (the `end-to-end` recipe: SEQ_MSE + AdaScale on a CUDA box / RunPod
  A100). AI Hub can still do the *compile* of an already-w4a16 ONNX.

This reframes the original "w4a16 or w8a16" choice: **w8a16 is the
zero-infrastructure path; w4a16 buys ~2× weight-bandwidth but costs a
RunPod A100 run.**

## Two concrete routes to a Gemma 4 bundle

Both require Finding-2's conclusion first: **build the 4 Gemma rewrite
scripts** (`scripts/README.md`) on the x86 dev box to produce a pinned
Gemma 4 pathb ONNX. That stage needs no GPU.

### Route 1 — w8a16, all via AI Hub (no CUDA, no rent, token only)

```
gemma-4-E2B-it (HF)
  → stages 1–5  (our surgery, x86 CPU)            → pathb ONNX
  → submit_quantize_job  w8/a16 + calibration     → quantized ONNX   [AI Hub cloud]
  → submit_compile_job   → QNN context binary     [AI Hub cloud, X Elite]
  → submit_inference_job → on-device numerics     [AI Hub cloud]
  → fetch bundle → npu_engine on the X2 laptop
```

Cost: an AI Hub token. Quality: w8a16 server-side PTQ (no SEQ_MSE) —
likely below our 0.99-cos bar but a real, fast first bundle.

### Route 2 — w4a16, hybrid (AIMET local + AI Hub compile)

```
  → stages 1–5  (our surgery, x86 CPU)            → pathb ONNX
  → AIMET SEQ_MSE (+ V/O w8 pin) on RunPod A100   → w4a16 ONNX + encodings
  → submit_compile_job (or local QAIRT)           → QNN context binary
  → submit_inference_job → on-device numerics     [AI Hub cloud]
```

Cost: ~$3–10 RunPod + token. Quality: the full `end-to-end` recipe.

## Recommendation

Do **Route 1 first.** It produces an end-to-end working Gemma 4 NPU
bundle with nothing but an AI Hub token — it validates the whole
surgery + compile chain and the on-device load on real Snapdragon
silicon. If w8a16 quality is short, Route 2 swaps in the high-quality
w4a16 quantize for one RunPod run, reusing every other stage.

Either way the gating work is the same and is ours: the 4 Gemma
rewrite scripts. AI Hub does not shortcut that.

## The ONNX-source problem (and its solution)

Producing the Gemma 4 ONNX ourselves via `optimum-cli` does **not work
today**: `optimum` has no `gemma4` exporter config, and `transformers
4.57.6` (the export venv) does not even recognise `model_type:
gemma4`. Patched community attempts to route Gemma4 through the Gemma 3
exporter produce a graph that fails at load with a ShapeInferenceError
(incompatible matmul dims) — the variable head dims (256 vs 512) and
PLE input are not handled.

**Solution:** `onnx-community/gemma-4-E2B-it-ONNX` — an ungated,
pre-exported ONNX of Gemma 4 E2B. It splits the model into
`embed_tokens.onnx`, `decoder_model_merged.onnx` (the text decoder —
our compute graph), plus vision/audio encoders. The decoder export
already exposes the `per_layer_inputs` (PLE) tensor as a graph input.
Sizes: fp16 decoder 4.76 GB, fp32 decoder 9.12 GB. We take the fp16
decoder as the pipeline's stage-1 input — this *replaces* the
optimum-export stage.

## Caveats

- `submit_quantize_job` is beta — schema may shift.
- All AI Hub LLM context binaries are built with QAIRT 2.42, which
  matches our ORT-QNN 1.24.4 pin (`docs/npu_ort_qnn_version_match.md`).
- The `decoder_model_merged.onnx` is a merged prefill+decode graph
  with dynamic seq/ctx dims. AI Hub's QNN-context-binary compile needs
  fixed shapes — shape pinning (`pin_shapes_gemma4.py`) is still
  required before `submit_compile_job`.

## Token + device status (2026-05-22, session 33 — VERIFIED)

- **Token configured and working.** `qai-hub configure` done;
  `~/.qai_hub/client.ini` written. `qai-hub list-devices` succeeds.
- **`Snapdragon X2 Elite CRD` IS an AI Hub device**
  (`qualcomm-snapdragon-x2-elite, sc8480xp`, Windows 11). This is the
  project's exact target silicon (HTP v75) — so AI Hub can compile AND
  on-device-validate directly for X2 Elite. The earlier "v73 vs v75
  mismatch" caveat is **resolved** — no mismatch, no physical laptop
  needed in the AI Hub loop.
