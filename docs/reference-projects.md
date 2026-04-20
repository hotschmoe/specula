# Reference projects

Local sibling work that informs specula. None are dependencies — they are
prior art and knowledge banks we can raid when the relevant phase lands.

## trident — Zig inference engine for Snapdragon X2

**Location:** `C:\Users\hotschmoe\Documents\GitHub\trident`

Pure-Zig LLM inference runtime targeted at Snapdragon X2 Elite Extreme. Not
llama.cpp-based. Has its own ggml-independent graph execution, CPU kernels
tuned for Oryon v2, and ongoing NPU experiments.

Key artifacts to read when relevant:

- `cpu_optimizations_plan.md` — what worked and what didn't on the Oryon CPU
  path (NEON / DOTPROD / I8MM / SVE). Directly relevant to any SME2 /
  KleidiAI work in specula phase 1.
- `npu_current_status.md`, `npu_optimizations_thoughts.md`, `npu_path_back.md`,
  `postmortem.md` — hard-won notes on the Hexagon NPU path. Read these
  BEFORE starting specula phase 5 (NPU-accelerated drafting). Likely saves
  days of re-discovery.
- `src/` — Zig kernel and graph code. If specula ever spins up its own
  standalone runtime (see README "Zig — optional"), this is the starting
  point, not a greenfield rewrite.
- `research_outline.md`, `spec.md`, `MILESTONES.md` — project framing, for
  cross-reference with specula's phase plan.

**When to revisit:** phase 4 (DFlash block-diffusion drafter — trident's
graph execution could host a standalone drafter if llama.cpp's ggml path
proves awkward for block-diffusion) and phase 5 (NPU kernels — mine the
npu_* docs first).

## voice_project — QAIRT + Qualcomm AI Hub cloud compile, NPU-resident Whisper

**Location:** `C:\Users\hotschmoe\Documents\voice_project`

End-to-end working pipeline: Whisper → ONNX → QNN binary via Qualcomm's
cloud compile (AI Hub) → Hexagon NPU execution. This is the only local
project where the NPU path has been proven out end-to-end.

Key artifacts:

- `aihub_compile.py` — cloud compile invocation. This is the pattern for
  turning an ONNX graph into a Hexagon-resident QNN binary.
- `build_qnn_models.py`, `run_qnn_converter.py` — local QNN converter path
  (non-cloud). Useful when the cloud compile token is gone or when
  iterating on a small kernel.
- `export_onnx.py`, `export_onnx_dynamo.py`, `fix_onnx_int64.py` — HF →
  ONNX export, including the int64 fixups the NPU toolchain requires.
- `convert_whisper.py`, `npu_transcribe.py` — full transcription harness
  running on NPU. Confirms the runtime wiring works.
- `current_status.md` — narrative of the wins and failures getting the
  pipeline working. Read this before phase 5.
- `encoder_info_v81.json` — specific hardware/toolchain targeting info for
  Hexagon v81. Phase 5 custom ops will need the equivalent for v79 (X2E
  shipped with v79 per current hardware notes) — or v81 if the hardware
  actually supports it; verify first.
- `qnn_cache/`, `intermediates/`, `build_test/` — artifacts showing the
  shapes and sizes of what a successful NPU-compiled graph looks like.

**When to revisit:** phase 5, step "Author QNN custom ops via QAIRT;
cross-compile to Hexagon". The README already notes "reuse device-targeting
info from prior transcription project" — this is that project.

## gguf_models — llama.cpp CPU inference on Qwen3.6-35B-A3B

**Location (notes):** `C:\Users\hotschmoe\Documents\gguf_models\LOCAL_LLM_NOTES.md`

**Location (models):** `C:\Users\hotschmoe\Documents\gguf_models\*.gguf`

Production-grade notes from getting llama.cpp running well on this exact
machine. The numbers are the ceiling specula phase 1 needs to match or
beat on its CPU baseline.

Most relevant sections:

- **llama.cpp build recipe** — clang-on-Windows-ARM64 specifics, required
  runtime DLLs next to binaries, helper scripts (`_configure.bat`,
  `_build.bat`, `_patch_kleidiai.py`, `_kill_llama.ps1`). specula's
  `scripts/build_llama_cpp.ps1` should converge with this recipe, not
  diverge.
- **KleidiAI / SME2 status** — KleidiAI built fine but crashed with
  `STATUS_ILLEGAL_INSTRUCTION` on batched matmuls. Root cause suspected to
  be Windows-on-ARM64 not having the full SME2 ZA-tile user-mode state
  enabled. Basic SME smstart/smstop works, ZA-tile does not. Left OFF.
  This is exactly the thing specula phase 1 wants to re-attempt — see
  README phase 1.
- **Benchmarks (18 threads, KleidiAI off)** — PP/TG numbers for Qwen3-4B,
  Gemma-4-31B, Gemma-4-26B-A4B, Qwen3.6-35B-A3B at several quants. Use
  these as the reference CPU ceiling. Any specula CPU baseline run that
  undershoots these is a build / config regression, not a hardware
  characterization.
- **Thread scaling and concurrency tables** — PP compute-bound (scales
  linearly to 18 threads), TG bandwidth-bound (saturates at ~8). Use this
  to pick the thread-count axis in specula's sweep matrix without wasting
  runs.
- **Run parameters for Qwen3.6-35B-A3B Q5_K_M** — `-fa`, `q8_0` KV cache,
  `-c 131072`, `-t 18`. Drop-in for any agentic-coding style workload in
  specula phase 2.

**When to revisit:** phase 0 (build recipe), phase 1 (CPU baselines —
must match these numbers), phase 1/2 (SME2 retry), and any time we're
choosing model / quant / KV-cache config for a workload.

## Summary table

| Project       | Owns                                         | Unlocks specula phase |
|---------------|----------------------------------------------|-----------------------|
| gguf_models   | CPU llama.cpp build recipe, CPU ceiling      | 0, 1                  |
| trident       | Oryon CPU kernel notes, NPU postmortems      | 1, 4, 5               |
| voice_project | Working QNN/AI-Hub pipeline to Hexagon       | 5                     |
