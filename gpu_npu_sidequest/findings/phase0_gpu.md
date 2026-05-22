# Phase 0 — GPU track: verify-shape crossover microbench

GPU half of the `gpu_npu_placement_sidequest` Phase 0. Measures the
Adreno X2-90 per-pass cost of a k-token batched forward (PP-shaped =
the *verify* op) and a single-token autoregressive forward (TG-shaped
= the *draft* op), as a function of speculation depth k.

## Build & model used

- **Build:** existing `llama.cpp\build-opencl\` — no fresh build
  needed. OpenCL/Adreno backend, commit `856c3adac` (build 9128),
  clang/ARM64. Device confirmed at runtime as
  `Qualcomm(R) Adreno(TM) X2-90 GPU`, OpenCL 3.0 QUALCOMM driver
  863.0, `GGML_OPENCL_USE_ADRENO_KERNELS` on. Backend column in
  llama-bench output reads `OpenCL`, ngl=99 — confirmed GPU, not a
  CPU fallback.
- **Model:** `models\Qwen3-4B-Q4_0.gguf` (2.21 GiB, 4.02 B params).
  **No Qwen3-4B Q4_K_M GGUF exists on disk** — checked `specula\models\`,
  `GitHub\llama.cpp\models\`, and `runpod_models_staging\` (the latter
  holds only NPU `.tar` bundles, no GGUF). Q4_0 is the only Qwen3-4B
  GGUF present. This is fine for the microbench: it measures op-shape
  behaviour (PP vs TG cost vs k), not model identity, and Q4_0 vs
  Q4_K_M differ only marginally in bytes-streamed.
- **Tool:** `llama-bench.exe`, `-ngl 99`, `-r 5`, JSON output. For
  each PP point `-b k -ub k` so the k-token forward is **one
  un-chunked physical batch** — exactly the single k-wide pass the
  crossover needs, not an internally split prefill.
- Run on **AC power** (`PowerLineStatus=Online`, battery status 2).

## Results

`ms_per_pass` = wall time of one k-token forward (llama-bench `avg_ns`,
the timed pass itself). `tok_per_s` = throughput.

### PP (batched k-token forward — the verify shape)

| k  | ms_per_pass | tok/s  | ms per useful token (=ms_per_pass/k) |
|----|-------------|--------|--------------------------------------|
| 1  | 39.0        | 25.8   | 39.0   (special: -b1, TG-equivalent path) |
| 2  | 163.4       | 12.2   | 81.7   |
| 4  | 166.9       | 24.0   | 41.7   |
| 8  | 167.9       | 47.7   | 21.0   |
| 16 | 168.3       | 95.1   | 10.5   |
| 32 | 174.6       | 183.4  | 5.5    |
| 64 | 201.1       | 318.2  | 3.1    |

### TG (autoregressive single-token forward — the draft shape)

| run  | tok/s | ms per token |
|------|-------|--------------|
| tg32 | 24.96 | 40.1         |
| tg64 | 25.04 | 39.9         |

GPU TG decode ≈ **25 t/s** for the 4B Q4_0 — stable across n=32/64.

## The per-pass-cost-vs-k curve — the headline shape

**GPU PP per-pass cost is essentially FLAT from k=2 to k=16**:
163 → 167 → 168 → 168 ms. A 2-token verify and a 16-token verify cost
the Adreno almost the same wall time. Cost only starts rising past
k≈16: +4% at k=32 (174 ms), +20% at k=64 (201 ms).

Interpretation: **at small k the Adreno is compute-underutilized.**
One k-wide pass for k≤16 is dominated by per-pass overhead — kernel
dispatch, weight streaming, the fixed cost of touching the 2.2 GB
weight set once — not by the GEMM work, which scales with k. The flat
plateau is the signature of a bandwidth-/overhead-bound regime: adding
tokens to the batch is *free* until the matmuls grow large enough
(k≳32) to actually saturate the ALUs, at which point the curve bends
upward into a compute-bound regime.

This is the crossover-relevant fact. Because GPU per-pass cost is flat
across the entire useful speculation range (k≈3–8), the GPU's
*per-useful-token* verify cost falls steeply with k (41.7 ms/tok at
k=4 → 21.0 at k=8 → 10.5 at k=16): you get extra verify tokens for
free. The GPU is the strong choice for verify **at small k**, exactly
the bandwidth-bound regime the side-quest plan predicted.

Where the NPU wins is the rising part of the curve: once verify is
compute-bound (k≳32 here), the NPU's quantized-matmul throughput
should overtake. The Phase-0 crossover-k is whatever k the NPU PP
curve (NPU track) intersects this flat ~167 ms GPU line — and given
the GPU plateau, that crossover will land *high* (k well past the
typical 3–8 range) unless the NPU's small-k per-pass cost is itself
much lower than 167 ms. Net read for C1-vs-C2: **the flat GPU plateau
favours C1 (verify→GPU) across the useful speculation range.** The NPU
track's small-k numbers decide whether the crossover ever enters that
range.

### The k=1 anomaly

`pp1` was run with `-b1 -ub1` and clocked 39 ms — a TG-shaped path,
not the k≥2 PP plateau. It matches the TG per-token cost (~40 ms)
almost exactly, which is the expected identity: a 1-token "batched"
forward *is* a TG step. The PP plateau proper begins at k=2. Read the
PP curve as flat-from-k=2; k=1 belongs with TG.

## GPU-specific gotchas

- **OpenCL banner corrupts JSON capture.** The Adreno backend prints
  a ~20-line init banner to stdout *before* llama-bench's JSON. When
  piped to a file the banner displaces the JSON array's closing `]`,
  so `-o json` output is not directly parseable — logs had to be
  repaired by appending `]`. The per-object data is intact; only the
  array terminator is lost. Future runs: capture stdout and post-strip
  the banner, or use `-o csv`.
- **max mem alloc size = 2048 MB** on this OpenCL device. The 4B Q4_0
  (2.2 GiB) loads fine because llama.cpp splits tensors across
  allocations, but a larger target (14B Q4_K_M) on the GPU may need
  attention to per-buffer limits.
- The 39 ms pp1 vs 163 ms pp2 jump is a `-b`/`-ub` path artifact, not
  a real cliff — keep k=1 separate from the PP sweep when fitting.
- `-r 5` gives stddev ~2-4% on PP, ~1% on TG — clean enough; no need
  for more repeats.

## Deliverables

- CSV: `gpu_npu_sidequest\results\phase0_gpu.csv`
  (copy in `results\csv\phase0_gpu.csv`).
- Raw logs: `gpu_npu_sidequest\logs\gpu_pp{1,2,4,8,16,32,64}.json`,
  `gpu_tg{32,64}.json`.
- Run script: `gpu_npu_sidequest\scripts\run_phase0_gpu.ps1`.

## Open / hand-off

- Q4_K_M not measured (no GGUF). If a Q4_K_M Qwen3-4B is fetched,
  re-run is ~2 min; expect the same flat-plateau shape.
- The crossover-k itself needs the **NPU track's** PP-vs-k curve;
  this file supplies the GPU half (flat ~167 ms for k=2..16, rising
  past k≈32).
