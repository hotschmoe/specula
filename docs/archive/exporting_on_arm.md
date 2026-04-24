# Exporting Qwen3 ONNX on Windows-on-ARM (X2E)

Verified 2026-04-23 on Snapdragon X2 Elite Extreme + Windows 11 +
Python 3.12.10 ARM64. This doc supersedes the "x86 is required" claim
in `docs/phase5_export_on_x86.md` for any export that doesn't need
`torch>=2.11`.

## Why this works now

PyTorch 2.7 (April 2025) was the first stable release to publish
native `cp312-cp312-win_arm64` CPU wheels via the Microsoft + Arm
collaboration. Versions 2.7, 2.7.1, 2.8, 2.9, 2.9.1, and 2.10 all
ship win_arm64 wheels on `download.pytorch.org/whl/cpu/`. They are
**not on PyPI**, which is why a vanilla `pip install torch` on the
X2E still fails — the wheel selector never sees them.

The rest of the export toolchain (optimum, optimum-onnx, transformers,
onnx, onnxruntime, onnx-graphsurgeon, tokenizers) is either pure
Python or already publishes win_arm64 cp312 wheels on PyPI.

## Install recipe

```bash
# From the repo root. Creates an isolated venv so it doesn't disturb
# the inference-side .venv or .venv-ort21.
uv venv .venv-arm-export --python 3.12

# Install the export stack. The --extra-index-url is critical — it's
# the only place torch's win_arm64 wheels are published.
VIRTUAL_ENV=.venv-arm-export uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.10.0" \
    "optimum==2.1.0" \
    "optimum-onnx==0.1.0" \
    "transformers==4.57.6" \
    "onnx==1.21.0" \
    "onnxruntime==1.24.4" \
    "onnx-graphsurgeon==0.6.1" \
    "huggingface_hub==0.36.2" \
    "tokenizers==0.22.2" \
    "numpy==2.4.4" \
    "safetensors" "sentencepiece"
```

The only Rust build in the chain is `safetensors`; the win_arm64
toolchain handles it cleanly (~15 s build).

Sanity check after install:

```bash
.venv-arm-export/Scripts/python -c "import torch, platform; \
    print('torch:', torch.__version__); \
    print('machine:', platform.machine()); \
    print('matmul:', torch.matmul(torch.randn(3,4), torch.randn(4,3)).shape)"
# Expected: torch: 2.10.0+cpu / machine: ARM64 / matmul: torch.Size([3, 3])
```

## Run the export

Identical CLI to the x86 path:

```bash
.venv-arm-export/Scripts/python -m optimum.exporters.onnx \
    --model Qwen/Qwen3-0.6B \
    --task text-generation-with-past \
    --no-post-process \
    --dtype fp16 \
    --cache_dir models/.hf_cache \
    models/qwen3-0.6b-optimum-arm
```

Wallclock on the X2E (12P/4E ARM cores, batch-1 trace): comparable
to the Intel Core Ultra 7 155H reference run (~90 s for the trace
itself, plus HF download time on the first run).

## Verification (2026-04-23 smoke test)

Compared a fresh ARM export against the x86 reference at
`models/qwen3-0.6b-optimum/`.

**Topology** (loaded with `onnx.load(..., load_external_data=False)`):

| metric | ARM | x86 | match |
|---|---:|---:|:-:|
| nodes | 7,667 | 7,667 | yes |
| graph inputs | 59 | 59 | yes |
| graph outputs | 57 | 57 | yes |
| initializers | 311 | 311 | yes |
| opset | 18 | 18 | yes |
| `com.microsoft` ops | 0 | 0 | yes |
| `model.onnx` size | 1,430,583 B | 1,430,583 B | yes |
| `model.onnx_data` size | 3,006,529,792 B | 3,006,529,792 B | yes |

**File MD5s differ** between the two host CPUs — `model.onnx` differs
because protobuf field ordering for unnamed nodes is not deterministic
across torch builds; `model.onnx_data` differs because the fp16 cast
of weights has small bit-level variation depending on the host's
floating-point ordering. Both are functional non-issues, confirmed by
the cos probe below.

**Numerical equivalence** (CPU-ORT, same probe protocol as
`scripts/probe_cos_vs_source.py` — past=511 zeros + BOS at position 511):

| metric | value |
|---|---:|
| cos(ARM logits, x86 logits) | **0.9999999995** |
| max abs diff (fp32 logits) | 0.000666 |
| argmax | 133724 == 133724 |
| top-5 overlap | 5/5 |

Comfortably above the cos≥0.9999 gate that
`docs/phase5_export_on_x86.md` uses for "rewrite is numerically
equivalent." Downstream rewrites (`scripts/rewrite_qwen3_htp.py`
stage / fold-patha / fold-pathbmask, `scripts/rewrite_qwen3_pathb.py`)
should produce the same artifacts whether they run on the x86 or ARM
ONNX, modulo the same fp16-cast noise floor.

## Known limitations

1. **No torch 2.11 win_arm64 wheel** at time of writing. If you need
   2.11 specifically (e.g. for a feature in optimum-onnx that requires
   it), use the x86 path. The MS/Arm cadence is roughly one minor
   version behind upstream stable.
2. **CPU-only.** Win_arm64 torch wheels do not ship CUDA / DML / Vulkan
   / OpenCL backends. The trace runs on the X2E's CPU cores only. This
   is fine for export (it's a one-shot static trace, not training); it
   would not be fine for any iterative / GPU-accelerated flow.
3. **fp16 cast nondeterminism vs x86.** The two `model.onnx_data`
   blobs differ at the byte level. If a downstream tool (e.g. a
   binary-diff CI check) requires byte equality with a reference, this
   will trip it. Use cos probes instead — they're what we already use
   to gate the surgical folds.

## When to use which path

- **ARM (this doc) — default.** Same machine as inference, no NAS
  round-trip, no second venv to maintain across boxes.
- **x86 (`docs/phase5_export_on_x86.md`) — fallback.** Use when:
  - You specifically need `torch==2.11.x` or newer.
  - You want a CUDA-accelerated trace (the export itself runs in
    seconds, but a 4B+ model trace can be RAM-bound on the X2E).
  - You need to reproduce the existing `models/qwen3-0.6b-patha/` /
    `pathbmask/` / `pathb/` artifacts byte-for-byte (they were built
    on x86 with `torch==2.11.0`).
