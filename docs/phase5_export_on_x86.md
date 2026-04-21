# Phase 5 — ONNX export on an x86_64 machine

Revised 2026-04-20 session 7 (first execution on the Intel x86 box).
Read top-to-bottom if you're picking this up cold.

## Why this document exists

Phase 5 of the specula project needs a QAIRT-compatible ONNX of
`Qwen/Qwen3-0.6B` so we can submit it to Qualcomm AI Hub and receive a
Hexagon v81 QNN context binary. We're doing this on a Snapdragon X2
Elite Extreme laptop running Windows on ARM64.

Three attempts at producing that ONNX on the X2E itself have already
been ruled out:

1. **Local `optimum.exporters.onnx` export on the X2E.** Requires
   torch. torch has no `cp312 win_arm64` wheel (older torch 2.1.2
   ships cp38-cp311 win_arm64 only; nothing newer ships win_arm64 at
   all). Hard wall.
2. **Use the `onnx-community/Qwen3-0.6B-ONNX` pre-export from HF Hub.**
   Works as an artefact, but the graph uses four ORT-internal fused
   ops that QAIRT's QNN converter doesn't recognise:
   - `SimplifiedLayerNormalization` × 57
   - `SkipSimplifiedLayerNormalization` × 56 (com.microsoft)
   - `RotaryEmbedding` × 56 (com.microsoft)
   - `GroupQueryAttention` × 28 (com.microsoft)

   AI Hub rejected the upload at 100 s into the `OPTIMIZING_MODEL`
   phase with `No Op registered for SimplifiedLayerNormalization with
   domain_version of 14`. Full evidence in
   `results/ai_hub_compile_attempt3.log` and
   `results/npu_env_snapshot.txt`.
3. **`onnxruntime_genai.models.builder` with `execution_provider="qnn"`.**
   The previous version of this document recommended this path as
   Microsoft-official. **It is not functional** in the current PyPI
   release (`onnxruntime-genai` 0.13.1, 2026-04-20). The `"qnn"`
   string is silently accepted but there is no `qnn` entry in the
   builder's `ep_attrs` dict (only cpu/cuda/dml/webgpu/trt-rtx), and
   `grep -ri qnn` across the installed package finds zero matches.
   The produced ONNX still contains `com.microsoft::RotaryEmbedding`,
   `SkipSimplifiedLayerNormalization`, and `MultiHeadAttention` — the
   exact ops we were trying to eliminate. The call then crashes at
   `make_genai_config` with `KeyError: 'qnn'`. The previous export
   script `scripts/export_qwen3_qnn.py` is retained in the repo with
   a deprecation note for historical reference.

**Resolution:** use the fallback path, `optimum.exporters.onnx` with
`--no-post-process`. `--no-post-process` disables the ORT fusion pass
that would otherwise introduce the `com.microsoft` ops, so the exported
graph stays in the default ONNX domain and QAIRT can lower every node.
This has been verified to produce a clean graph (see "Verify the
output" below — 7,667 nodes, opset 18, zero `com.microsoft` ops).

This document describes running that export on an x86_64 machine
(Linux or Windows) and transferring the output back to the X2E for the
AI Hub compile step.

## Prerequisites on the x86 machine

- Python 3.10–3.12, x86_64.
- 10 GB+ free disk (HF cache + ONNX output + workspace; the optimum
  fp16 export alone is ~3 GB because `--no-post-process` also skips
  the weight-deduplication pass).
- Internet access (downloads the HF checkpoint from
  `huggingface.co/Qwen/Qwen3-0.6B`, ~1.2 GB).
- Ideally a GPU (CUDA or ROCm) to accelerate the trace. CPU works
  fine too — the reference run completed in ~90 s on an Intel Core
  Ultra 7 155H.
- `git`.

No QAIRT SDK install needed on the x86 machine — we only need it to
*produce* the ONNX. AI Hub does the Hexagon compile in the cloud from
the X2E side.

## Setup (one-time)

```bash
# Clone specula
git clone https://github.com/hotschmoe/specula.git
cd specula

# Fresh venv
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install the export toolchain. `optimum-onnx` is a separate package
# from `optimum` as of optimum 2.x; both are needed.
pip install optimum "optimum-onnx" torch transformers huggingface_hub onnx
```

Versions known to work (reference run on 2026-04-20): `optimum 2.1.0`,
`optimum-onnx 0.1.0`, `torch 2.11.0+cpu`, `transformers 4.57.6`,
`huggingface_hub 0.36.2`, `onnx 1.21.0`. Note that installing
`optimum-onnx` will downgrade `transformers` from 5.x to 4.57.x — this
is intentional; optimum-onnx hasn't caught up to transformers 5 yet.

## Run the export

```bash
python -m optimum.exporters.onnx \
    --model Qwen/Qwen3-0.6B \
    --task text-generation-with-past \
    --no-post-process \
    --dtype fp16 \
    --cache_dir models/.hf_cache \
    models/qwen3-0.6b-optimum
```

What each flag does:

- `--model Qwen/Qwen3-0.6B` — HF model id, auto-downloaded.
- `--task text-generation-with-past` — generative decoder with KV cache
  inputs/outputs. Required; auto-inference picks up only `feature-
  extraction` for this model.
- `--no-post-process` — **the critical flag.** Skips the ORT fusion
  pass so `com.microsoft::` ops never get introduced. Without this, the
  graph looks like the HF pre-export from attempt #2 above.
- `--dtype fp16` — HTP runs fp16 natively, halves the file size.
- `--cache_dir models/.hf_cache` — keeps the HF download inside the
  repo tree (the `.gitignore` already excludes it).

Expected duration:

- With a GPU: ~1 min.
- CPU-only: ~90 s on Core Ultra 7 155H; a few minutes on lower-end
  parts. Most of the wall-clock is the HF download, not the trace.

At the end, optimum runs a PyTorch-vs-ONNX validation sweep and prints
max-diff values per output. fp16 max diffs in the ~1e-3 range are
normal and expected.

## Verify the output

**This is the critical step — confirm the ONNX has no `com.microsoft`
ops.** Without this check you'll burn another 15+ min upload to AI Hub
before discovering a problem.

```bash
python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-optimum/model.onnx
```

What you want to see:

```
opset imports:
  <default>            version 18

... op histogram here ...

all ops are in the default onnx domain
```

What you do **not** want to see:

```
NON-STANDARD ops (will need decomposition or QAIRT custom op):
  com.microsoft::...
```

Reference run histogram (top entries, for comparison): Constant 2759,
Unsqueeze 884, Mul 567, Shape 429, Gather 372, Cast 351, Concat 340,
Add 256, MatMul 254, Reshape 228, Div 169, Transpose 141, Slice 140,
Where 114, Pow 113, ReduceMean 113, Sqrt 113, ConstantOfShape 59,
Equal 58, Expand 58, Neg 56, Softmax 28, IsNaN 28, Sigmoid 28, Range 3,
And 2, LessOrEqual 1, Flatten 1, Cos 1, Sin 1. Total 7667 nodes.

If `com.microsoft` ops somehow survive:

1. Double-check `--no-post-process` is on the command.
2. Retry with `--optimize O1` explicitly off (it should be off by
   default without `--optimize`).
3. Escalate with the inspect_onnx_ops.py output — may need a targeted
   decomposition pass.

## Transfer the output back to the X2E machine

The fp16 export is ~3 GB (larger than the old ort-genai builder claim
of 1-2 GB because `--no-post-process` disables weight de-duplication).
Pick your transfer mechanism:

```bash
# scp (simplest over SSH)
scp -r models/qwen3-0.6b-optimum/ <user>@<x2e-host>:<path>/specula/models/

# rsync (resumable, good for slow links)
rsync -av --progress models/qwen3-0.6b-optimum/ \
    <user>@<x2e-host>:<path>/specula/models/qwen3-0.6b-optimum/

# Mapped Google Drive / cloud sync (the reference run used
# G:\Shared drives\MAIN\Junk\qwen3-0.6b-optimum\ as the drop site)
cp -r models/qwen3-0.6b-optimum "/path/to/mounted/drive/"

# USB / your tool of choice
```

Do **not** commit the ONNX to git. The repo's `.gitignore` excludes
`*.onnx`, `*.onnx_data`, and `*.safetensors` for exactly this reason
(model weights don't belong in VCS; see the download_qwen3_onnx.py
pattern for the preferred "script the download" approach).

## Resume on the X2E side

Once the directory lands at `<specula>/models/qwen3-0.6b-optimum/` on
the X2E machine:

1. **Sanity-check the transfer** (file sizes match what you sent —
   `model.onnx_data` should be ~3.0 GB).

2. **Point the staging + compile scripts at the new source.** In the
   current session these scripts hardcode `models/qwen3-0.6b-onnx/` as
   the source; they need a small update to accept the
   `qwen3-0.6b-optimum/` path. Options:
   - Edit `scripts/prep_onnx_for_ai_hub.py`'s `SOURCE_ONNX` /
     `SOURCE_DATA` constants to match the new filenames.
   - Or (cleaner) add `--source` argparse flags to both scripts. The
     ARM-side user will hit this first and can refactor in-place.

3. **Re-stage and compile:**

   ```powershell
   .venv\Scripts\python.exe scripts\prep_onnx_for_ai_hub.py
   .venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --check    # dry-run
   .venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --submit   # ~15-25 min
   ```

4. **Expected outcome:** `OPTIMIZING_MODEL` phase completes, followed
   by quantization (if applicable) and final context-binary
   generation. Downloaded artefact:
   `models/qwen3_0_6b_draft_v81_ctx512.bin` (~500–800 MB FP16).

5. **If compile fails again on a different op:** capture the failure
   reason, inspect the ops via `inspect_onnx_ops.py`, and either
   regenerate with different optimum flags (on the x86 machine) or
   escalate for a targeted decomposition. See
   `results/npu_env_snapshot.txt` for the prior failure-mode log as
   a template for what to capture.

## Where this fits in the larger plan

- **Scoping doc:** `docs/npu_scoping.md` — full Phase 5 plan, the
  10-step bring-up, toolchain pins, known failure modes.
- **Current session log:** `results/npu_env_snapshot.txt` — steps 1-4
  execution history, including the three AI Hub attempts.
- **Status doc:** `current_status.md` — project-wide phase tracking.

Steps 1-3 of the scoping doc are green. Step 4 is blocked on the work
this doc describes. Step 5 (load the .bin on the X2E via ORT-QNN)
picks up as soon as step 4 produces a working binary — the
`NPUSession` wrapper in `scripts/npu_draft_sidecar.py` is already in
place and was smoke-tested at step 2 against a toy graph. Swap the
model path, rebind to the Qwen3 IO surface, and we're running.

## Reference: alternative paths considered

Documented here so future sessions don't re-derive these from scratch.

- **Manual ONNX graph surgery** to decompose the four fused ops on the
  X2E side. 6-11 hours of work; `GroupQueryAttention` with KV-cache
  semantics is the bulk of the cost and the highest bug risk. Ruled
  out in favour of producing a clean graph at source.
- **`onnxruntime_genai.models.builder` with `execution_provider="qnn"`.**
  Previously recommended as the "Microsoft-official QNN path." Does
  not work in PyPI 0.13.1 — no QNN support present in the package.
  See the "Why this document exists" section and
  `scripts/export_qwen3_qnn.py` for the deprecation note. If a future
  ort-genai release adds QNN support, reconsider.
- **QAIRT custom-op authoring** (Path C in the scoping doc). Ruled out
  at scoping time because Hexagon Skels have to be signed and retail
  Windows blocks testsigning.
- **Pre-compiled Qwen3 in the AI Hub Model Zoo.** A "check first"
  shortcut — if Qualcomm has already published a Qwen3-0.6B QNN
  context binary for Snapdragon X2 Elite, we skip the ONNX step
  entirely. Recommended to browse <https://aihub.qualcomm.com/models>
  before burning the export cycle. See
  `results/ai_hub_model_zoo_check.md` (if present; otherwise perform
  the check yourself).
