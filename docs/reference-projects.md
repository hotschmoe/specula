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

## lucebox-hub — RTX 3090 DFlash port, the public reference for phases 3 & 4

**Location:** `C:\Users\hotschmoe\Documents\GitHub\lucebox-hub`

Pulled 2026-04-20 after reading the companion article (archived locally at
`new_spec_decode_example_to_research.md`). Luce-Org's standalone C++/ggml
DFlash speculative decoder for Qwen3.5-27B Q4_K_M on a single RTX 3090,
MIT-licensed.

Headline numbers (HumanEval 10-prompt, Q4_K_M target, single 3090, 24 GB):

- **129.5 tok/s mean at DDTree budget=22** (3.43× over AR baseline 37.78 t/s)
- **207.6 tok/s peak** demo (5.46×)
- Average accept length **AL ≈ 8.9** at budget 30 (our CPU best: AL ≈ 2.4 at k=3)
- 128K context on 24 GB via Q4_0 KV + rolling 4096-slot target_feat ring
- ~2000 LOC of C++/CUDA on `libggml*.a` only; no libllama

Two subprojects: `megakernel/` (separate) and `dflash/`. The dflash/ tree is
what we care about for specula phases 3–4. Laid out as:

- `dflash/src/qwen3_dflash_graph.cpp` — **the DFlash drafter graph glue.**
  This is the template for a Qwen3-family DFlash drafter. Most directly
  portable file to specula phase 4.
- `dflash/src/qwen35_target_graph.cpp` — Qwen3.5 hybrid target graph.
  Only relevant if we go hybrid; specula README sticks to pure-attention
  Qwen3 for phases 2–4.
- `dflash/src/delta_net_chunked.{cpp,h}` — Gated DeltaNet kernel wrapper.
  Skip unless we decide to do Qwen3.5-hybrid after Phase 4.
- `dflash/src/dflash_graph.h` — the tree + block-diffusion verify schema.
  **Read this first to understand the DDTree data layout** before looking
  at the graph code.
- `dflash/src/safetensors_draft.cpp` — draft-weight loader. z-lab publishes
  DFlash drafter weights in safetensors, not GGUF; we inherit this.
- `dflash/src/kv_cache.cpp` — Q4_0 KV cache on ggml-cuda FA. Generally
  applicable pattern for long-context; the Adreno/Hexagon analogue is
  still open.
- `dflash/src/gguf_target_loader.cpp` — how they load a GGUF target without
  linking libllama. Relevant if we ever build our own standalone runtime.
- `dflash/RESULTS.md` — full benchmark matrix including budget sweeps
  (AL plateau at 8.9 across budget 20/30/40) and per-prompt AL data.
  Use this as the upper bound when reasoning about what a good DFlash
  port can achieve; gap to our CPU/Adreno numbers is the budget of work
  we have to do.
- `dflash/README.md` — the article body (matches
  `new_spec_decode_example_to_research.md` modulo graphics).

**Platform gap:** CUDA only. Authors explicitly ruled out Metal / Vulkan /
OpenCL. That is the specula-shaped hole in their coverage — and the reason
the Snapdragon/Adreno/Hexagon lane is genuinely uncrowded.

**Key transferable perf lessons from their day-by-day log** (full list in
the article; the short version, with the Snapdragon-X2 implications):

- **f16 intermediate cache halves memory BW** (+5% tok/s). Adreno OpenCL
  should benefit at least as much, since TG on this hardware is
  bandwidth-bound (228 GB/s ceiling).
- **`ggml_gated_delta_net_tree_persist` skips a 9 ms `ggml_cpy`/step (+11%).**
  Pattern: avoid re-materializing persistent state. Applies whenever we
  hold SSM/tree state across verify steps.
- **D2D target_feat copy removed a GPU→CPU→GPU roundtrip (+3.3%).** On
  RTX 3090 that was a PCIe DMA. Our architectural equivalent is the
  OpenCL buffer-model cost diagnosed in session 4 (see "Unified memory
  vs zero-copy" below).
- **Tree-aware `ggml_ssm_conv_tree`**: sibling nodes gather along parent
  chain, not DFS order. Same idea applies to tree-verify attention masks
  on any backend; SGLang has the same pattern in `causal_conv1d_triton`.
- **Silent `verify_logits_buf` overflow**: sized `vocab * q_len`, but DDTree
  reads `vocab * (budget + 1)` past budget 15. Watch for this when we
  port: one-line fix, but the failure mode is memory corruption with no
  crash.

**When to revisit:**

- Phase 3 (EAGLE-3 viability check): skim `dflash_graph.h` + the chain-vs-
  tree AL numbers in `RESULTS.md` to calibrate expectations before sinking
  time into the EAGLE-3 PR.
- Phase 4 (DFlash port to Adreno/Hexagon): `qwen3_dflash_graph.cpp` +
  `safetensors_draft.cpp` are the starting point. Their graph calls into
  ggml directly — so the port effort is mostly "replace CUDA kernels with
  OpenCL / Hexagon equivalents" rather than "reimplement graph."
- Phase 5 (NPU drafting): read how they structure the draft-verify
  handoff. The D2D optimization is what we'd want to beat with
  shared-LPDDR zero-copy between NPU and GPU.
- Any discussion of performance-engineering tactics in the project: the
  day-by-day log in their article is a checklist of specific things to
  watch for.

## Unified memory vs zero-copy — the buffer-model aside

Relevant because session 4 mixed-device runs showed 0.37× regression on
k=3 spec decode, and the lucebox paper's top perf win was eliminating a
GPU→CPU→GPU PCIe round-trip. On Snapdragon X2, CPU + Adreno + Hexagon all
share the same LPDDR5X (228 GB/s). **The PCIe DMA the paper fixed doesn't
exist for us — but a different, smaller version of the cost does.**

What's actually happening in ggml-opencl (verified by reading
`llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`, ~20 call sites):

- All buffers allocated with plain `clCreateBuffer(context,
  CL_MEM_READ_WRITE, size, NULL, ...)`. No `CL_MEM_USE_HOST_PTR`,
  no `clSVMAlloc`, no `cl_qcom_ion_host_ptr`.
- That means the Qualcomm OpenCL driver is free to place the buffer in a
  cached-on-GPU region. `enqueueWriteBuffer` / `enqueueReadBuffer` then
  incur cache-flush on source + driver memcpy + cache-invalidate on
  destination, even though the underlying RAM is physically shared.
- `cl_qcom_large_buffer` is already detected in the backend
  (`ggml-opencl.cpp:3177`), but is a different feature (extends max
  buffer size past 4 GB), gated by `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1`.

Paths to recover "true" zero-copy on this hardware:

1. `CL_MEM_USE_HOST_PTR` with aligned host allocation — eliminates the
   driver memcpy, still pays cache maintenance.
2. OpenCL 2.0 SVM (`clSVMAlloc`) — single pointer valid on both sides,
   lowest overhead path. Adreno supports it.
3. `cl_qcom_ion_host_ptr` (ION-backed buffers) — the vendor-native
   zero-copy path, what QAIRT uses to share buffers between GPU and DSP.

The per-call kernel-launch overhead on Adreno is probably the bigger
cost at k=3 verify batches (4-token workloads don't amortize dispatch);
memory-transfer noise is second-order but not zero. A ground-up runtime
(trident-style Zig, or a lucebox-style standalone C++ harness) would
let us pick one of (1)–(3) as an allocation invariant rather than
fighting llama.cpp's opaque backend dispatch per op.

This is an open optimization lane, not something to do before Phase 4–5
lands. Flagging it here so the lane is visible.

## Summary table

| Project       | Owns                                         | Unlocks specula phase |
|---------------|----------------------------------------------|-----------------------|
| gguf_models   | CPU llama.cpp build recipe, CPU ceiling      | 0, 1                  |
| trident       | Oryon CPU kernel notes, NPU postmortems      | 1, 4, 5               |
| voice_project | Working QNN/AI-Hub pipeline to Hexagon       | 5                     |
| lucebox-hub   | Reference DFlash+DDTree impl, perf playbook  | 3, 4                  |
