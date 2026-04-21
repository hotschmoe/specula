# Phase 5 — ONNX export on an x86_64 machine

Written 2026-04-20 at session 6 close. Read top-to-bottom if you're
picking this up cold.

## Why this document exists

Phase 5 of the specula project needs a QAIRT-compatible ONNX of
`Qwen/Qwen3-0.6B` so we can submit it to Qualcomm AI Hub and receive a
Hexagon v81 QNN context binary. We're doing this on a Snapdragon X2
Elite Extreme laptop running Windows on ARM64.

Two attempts at producing that ONNX on the X2E itself have already been
ruled out:

1. **Local `optimum.exporters.onnx` export.** Requires torch. torch
   has no `cp312 win_arm64` wheel (older torch 2.1.2 ships cp38-cp311
   win_arm64 only; nothing newer ships win_arm64 at all). Hard wall.
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

**Resolution:** re-export on a non-WoA machine using Microsoft's
`onnxruntime_genai.models.builder` with
`execution_provider="qnn"`. That builder decomposes the ORT fused ops
into QNN-compatible primitives as a tested feature — it's the official
Microsoft path for QNN targets, not a workaround.

This document describes running that export on an x86_64 machine
(Linux or Windows) and transferring the output back to the X2E for the
AI Hub compile step.

## Prerequisites on the x86 machine

- Python 3.10–3.12, x86_64.
- 10 GB+ free disk (HF cache + ONNX output + workspace).
- Internet access (downloads the HF checkpoint from
  `huggingface.co/Qwen/Qwen3-0.6B`, ~1.2 GB).
- Ideally a GPU (CUDA or ROCm) to accelerate the trace. CPU works too
  but is slower.
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

# Install the export toolchain. onnxruntime-genai's builder pulls in
# torch+transformers on its own, but we pin explicitly for clarity.
pip install onnxruntime-genai torch transformers huggingface_hub onnx
```

Versions known to work (check `onnxruntime_genai.__version__` if
debugging later): `onnxruntime-genai >= 0.5`, `torch >= 2.3`,
`transformers >= 4.51`.

## Run the export

```bash
python scripts/export_qwen3_qnn.py
```

That command does everything:

1. Downloads `Qwen/Qwen3-0.6B` from Hugging Face into
   `models/.hf_cache/`.
2. Traces the PyTorch model.
3. Runs Microsoft's QNN-targeted ONNX export, which replaces ORT
   fused ops (`SimplifiedLayerNormalization`, `RotaryEmbedding`,
   `GroupQueryAttention`, etc.) with QNN-compatible primitives.
4. Writes the ONNX + external data + tokenizer files to
   `models/qwen3-0.6b-qnn-source/`.

Expected duration:

- With a GPU (CUDA / ROCm): ~5 min.
- With CPU only: ~15–30 min depending on core count.

Flags (all optional):

```bash
python scripts/export_qwen3_qnn.py --precision fp16   # default, recommended
python scripts/export_qwen3_qnn.py --precision int4   # smaller binary
python scripts/export_qwen3_qnn.py --output /some/path
```

## Verify the output

**This is the critical step — confirm the ONNX has no `com.microsoft`
ops.** Without this check you'll burn another 15+ min upload to AI Hub
before discovering a problem.

```bash
python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-qnn-source/model.onnx
```

What you want to see:

```
opset imports:
  <default>            version 17    (or 18, 19, ...)

... op histogram here ...

all ops are in the default onnx domain
```

What you do **not** want to see:

```
NON-STANDARD ops (will need decomposition or QAIRT custom op):
  com.microsoft::...
```

If `com.microsoft` ops remain, the builder's QNN target didn't
decompose them. Possible reasons: the specific model/precision
combination isn't fully supported yet, or the `onnxruntime-genai`
version is old. Try:

1. Newer `onnxruntime-genai`: `pip install --upgrade onnxruntime-genai`.
2. Different precision: `--precision int4` sometimes produces a
   different op set than `fp16`.
3. Escalate back to specula's maintainer with the inspect_onnx_ops.py
   output — may need a targeted decomposition pass written.

## Transfer the output back to the X2E machine

The ONNX + weights is 1–2 GB depending on precision. Pick your transfer
mechanism:

```bash
# scp (simplest over SSH)
scp -r models/qwen3-0.6b-qnn-source/ <user>@<x2e-host>:<path>/specula/models/

# rsync (resumable, good for slow links)
rsync -av --progress models/qwen3-0.6b-qnn-source/ \
    <user>@<x2e-host>:<path>/specula/models/qwen3-0.6b-qnn-source/

# USB / cloud drive / your tool of choice
# (skip git — ONNX files are gitignored and too big anyway)
```

Do **not** commit the ONNX to git. The repo's `.gitignore` excludes
`*.onnx`, `*.onnx_data`, and `*.safetensors` for exactly this reason
(model weights don't belong in VCS; see the download_qwen3_onnx.py
pattern for the preferred "script the download" approach).

## Resume on the X2E side

Once the directory lands at `<specula>/models/qwen3-0.6b-qnn-source/`
on the X2E machine:

1. **Sanity-check the transfer** (file sizes match what you sent).

2. **Point the staging + compile scripts at the new source.** In the
   current session these scripts hardcode `models/qwen3-0.6b-onnx/` as
   the source; they need a small update to accept the
   `qwen3-0.6b-qnn-source/` path. Options:
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
   regenerate with different builder flags (on the x86 machine) or
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
  out in favour of the Microsoft-official path.
- **Fresh export via `optimum.exporters.onnx` with `no_post_process=True`.**
  Skips ORT fusion so standard ops survive. Still needs torch, so
  still requires a non-WoA machine. Less QNN-optimised than the
  builder path, so the AI Hub compile might miss some QNN-specific
  layouts. Kept as fallback if `onnxruntime_genai.models.builder`
  doesn't support a given model.
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
