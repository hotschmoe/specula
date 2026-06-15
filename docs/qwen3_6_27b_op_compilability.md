# Qwen3.6-27B op-compilability — can the SSM op leave PyTorch?

Created: 2026-06-15. Investigation doc for the Qwen3.6-27B NPU port
(charter: `qwen3_6_27b_npu_kickoff.md`). Tracks the single make-or-break
question for the hybrid: **do the `linear_attention` (gated-delta-net /
Mamba2-style) layers export to ONNX and compile to the Hexagon HTP at
all?** Quantization (w8a16/w4a16) is moot if the op can't leave PyTorch
as a static graph.

## Why a proxy, and which one

Qwen3.6-27B is `model_type: qwen3_5` (arch `Qwen3_5ForConditionalGeneration`,
a VLM — LLM dims nested under `text_config`). **transformers 4.57.6 does
not ship `qwen3_5`** (`KeyError: 'qwen3_5'` from `CONFIG_MAPPING`); it
needs transformers-from-source. But it *does* ship **`qwen3_next`**, whose
`linear_attention` layers are the **same gated-delta-net op family** —
identical `partial_rotary_factor=0.25`, `head_dim=256`, and the `linear_*`
dim schema. So a tiny random `qwen3_next` is a faithful proxy for the one
op we care about, with **no 54 GB download and no GPU**.

Probe: `end-to-end/probes/op_compilability_probe.py` (run in
`.venv-arm-export`). Tiny model: 0.79M params, `hidden=128`, 4 layers,
`layer_types=[linear, linear, linear, full]`.

## Stage 1 result (2026-06-15): exports FAIL with stock exporters

| step | outcome |
|---|---|
| build tiny qwen3_next | ✅ OK |
| **eager forward** (CPU, pure-torch SSM fallback) | ✅ OK — logits `(1,8,256)` |
| **dynamo / `torch.export`** | ❌ `GuardOnDataDependentSymNode: Could not guard on Eq(u0, 1)` |
| **legacy / TorchScript** | ❌ `RuntimeError: invalid unordered_map<K, T> key` |

Two **distinct, named** walls:

1. **Data-dependent control flow** (dynamo). `_update_linear_attn_mask`
   (`modeling_qwen3_next.py:1029`) calls `.item()` on a tensor →
   `torch.export` can't guard the unbacked symint. This is mask-construction
   convenience, **not** the SSM math — analogous to what the dense pathb
   additive-mask rewrite already neutralizes. **Likely patchable.**
2. **vmap + custom `autograd.Function` chunked delta rule** (legacy). The
   gated-delta-net recurrence uses a vmapped custom autograd function the
   TorchScript tracer rejects. This is **the core SSM op** and the real
   research target.

Note: the eager run warns "fast path not available … flash-linear-attention
/ causal-conv1d" and falls back to pure torch — which is what we *want*;
the CUDA kernels would never export anyway. The blocker is the pure-torch
chunked-scan, not the missing kernels.

## Path forward (options, not yet chosen)

- **A — patch + force the recurrent path.** Monkeypatch
  `_update_linear_attn_mask` to drop the `.item()` (static mask, like
  pathb), and force the gated-delta-net to its **non-chunked recurrent
  `torch_forward`** (no vmap) before export. Cheapest next probe; may clear
  both walls at once. **Try this next.**
- **B — export-friendly scan.** Replace the chunked delta rule with an
  explicit `Scan`/`Loop` (or unrolled fixed-length) recurrence that
  `torch.export` can trace. More work; produces HTP-questionable `Loop`/
  `Scan` ops (see HOSTILE_OPS in the probe).
- **C — roll our own aarch64/HTP gated-delta-net op.** Hand-author the
  recurrence as a QNN/HTP-friendly op graph (matmul + conv + elementwise),
  bypassing the PyTorch export entirely for the SSM layers. Highest effort,
  highest payoff, most "research clout."
- **D — punt to AI Hub.** Once *something* exports, `submit_compile_job`
  on `Snapdragon X2 Elite CRD` tells us if their compiler accepts the
  resulting ops. (AI Hub can't help with the export itself.)
- **E — wait for upstream.** transformers-from-source `qwen3_5` + an
  optimum exporter for the hybrid. Out of our control; don't block on it.

## On-device / AI-Hub split (operating principle, per user)

Everything except the AIMET-GPU calibration runs **on the X2E** (Prism
x86): export, rewrite, split, qairt-convert, qairt-quantize, ctx-bin-gen,
ORT-QNN load. **w8a16 first** — basic PTQ is competitive at w8a16 (4B hit
cos 0.996 with the RMSNorm-skip config) and runs on-device or on AI Hub's
free quantize job, so a first w8a16 bundle needs **zero GPU/cloud**. SSM
op-compilability is the blocker, identical for on-device and AI Hub, so it
gets solved first (options A–C). The DGX Spark / RunPod only re-enter for
w4a16 SEQ_MSE/AdaScale *quality*, later.

## Environment notes

- Probe venv: **`.venv-arm-export`** (ARM64, torch 2.10.0, transformers
  4.57.6, onnx 1.21). Added **`onnxscript 0.7.0` + `onnx-ir 0.2.1`** this
  session (the dynamo exporter needs onnxscript; without it the dynamo path
  silently no-ops with `ModuleNotFoundError`).
- **Do NOT blind-update `.venv-qairt` / `.venv-ort21`** — version-pinned
  (ORT-QNN ↔ QAIRT 2.45.40 must match per
  [[reference_ort_qnn_qairt_match]]); bumping them breaks HTP load.
  `.venv-arm-export` and `.venv` are safe to keep current.

## Next

1. Option A probe: patch `.item()` + force recurrent gated-delta-net, retry
   dynamo export; tally ONNX ops (flag `Scan`/`Loop`/`If`).
2. If A exports → qairt-converter + AI Hub `submit_compile_job` (X2 Elite
   CRD) on the single-layer ONNX → first real HTP op-validation signal.
3. In parallel, the dense **Qwen3-14B w8a16** all-local run (no SSM) as the
   scaling stepping stone.
