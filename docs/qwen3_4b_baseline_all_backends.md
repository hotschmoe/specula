# Qwen3-4B — all-backends baseline matrix

One comparable PP / TG throughput number per compute island on the
Zenbook A16 (Snapdragon X2 Elite Extreme, Hexagon v81, Adreno X2-90,
48 GB LPDDR5X). Measurement recipe in
`docs/qwen3_4b_baseline_methods.md`.

**Status:** scaffold. Model artifacts in place (Qualcomm w4a16 bundle
+ unsloth Q4_K_M GGUF), llama.cpp builds ready across all four presets
(cpu / cpu-kleidiai / vulkan / opencl), Genie runtime installed. Numbers
pending one measurement session.

## Models under test

| backend family | artifact | weight footprint |
|---|---|---|
| NPU | `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/*.bin` (4 parts) | ~w4a16, 4 × ~600 MB |
| CPU / GPU | `models/Qwen3-4B-Q4_K_M.gguf` (unsloth) | 2.38 GiB (Q4_K_M) |

Quant parity is close but not identical (w4a16 vs Q4_K_M mixed
4/6-bit). See methods doc §"Quant parity caveat" before using any
tie-breaker within ~20% of the leader.

## Headline matrix (AC, ctx=2048, greedy sampling)

| backend | runtime / build | PP512 (t/s) | TG128 (t/s) | notes |
|---|---|---:|---:|---|
| **NPU**            | genie-t2t-run (QAIRT 2.45, bundle compiled 2.42) | — | — | |
| **NPU (alt)**      | ORT-QNN 1.24.4 chained 4-part probe              | — | — | cross-check vs Genie; will use our runtime |
| **CPU**            | llama.cpp `build-cpu` (ARM64 NEON)               | — | — | -t 8, Prime cluster |
| **CPU + KleidiAI** | llama.cpp `build-cpu-kleidiai` (NEON + i8mm)     | — | — | SME2 runtime-fenced |
| **GPU (OpenCL)**   | llama.cpp `build-opencl` (Adreno-tuned kernels)  | — | — | -ngl 99 |
| **GPU (Vulkan)**   | llama.cpp `build-vulkan`                         | — | — | -ngl 99 |

Each cell to be filled with the **median** of 3–5 measured runs
(warmup discarded). Fill the `notes` column with power state (AC/BAT)
+ anything that caused variance > ±5%.

### Reference numbers from prior phases (not directly comparable)

Anchoring points — different model / different config, useful to
sanity-check scale:

| context | value | source |
|---|---|---|
| Qwen3-0.6B Q8_0 PP512 on Adreno OpenCL (Phase 1) | 2674 t/s | `results/phase1_adreno_pp512.csv` |
| Qwen3-8B Q4_K_M TG on CPU -t 18 (Phase 2) | 25.9 t/s | `docs/phase2_*` |
| Qwen3-0.6B w8a16 NPU single-step (Phase 5.5 Lever B) | 22 ms → ~45 t/s standalone | `docs/qwen3_perf_levers_investigation.md` |
| Qwen3-4B w4a16 NPU 12 layers (Lever C side-quest) | 7.22 ms → ~45 t/s extrapolated | `results/qwen3_4b_genie_w4a16_probe.md` |

The Genie row should land near the 45 t/s extrapolation; the Adreno
OpenCL PP row should land proportionally below 2674 t/s (4B has ~6×
the matmul volume of 0.6B). Deviations are signal, not noise — flag
them.

## Per-backend detail (to be populated)

Each backend gets a short subsection below with: raw stdout excerpt,
command invoked, power state, run count, any anomalies.

### NPU (Genie)

```
cmd:    (pending)
build:  QAIRT 2.45.40.260406 / genie-t2t-run / bundle compiled QAIRT 2.42
date:   (pending)
power:  (pending)

PP512 median: — t/s  (runs: —)
TG128 median: — t/s  (runs: —)
TTFT:         — ms
```

### NPU (ORT-QNN chained)

```
cmd:    (pending — extension of scripts/probe_qualcomm_qwen3_4b.py to parts 3+4)
build:  ORT 1.24.4 + QAIRT 2.42 (via onnxruntime-qnn pkg)
date:   (pending)
power:  (pending)

PP (per-call ar=128): — ms  -> — t/s
TG (per-call ar=1):    — ms  -> — t/s
```

### CPU (ARM64 NEON)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -c 2048 -t 8 -r 3
build:  llama.cpp build-cpu, clang-on-vcvarsarm64, commit (see build-cpu/SPECULA_BUILD.txt)
date:   (pending)
power:  (pending)

PP512 median: — t/s
TG128 median: — t/s
```

### CPU + KleidiAI

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -c 2048 -t 8 -r 3
build:  llama.cpp build-cpu-kleidiai (GGML_CPU_KLEIDIAI=ON, SME2 runtime-fenced)
date:   (pending)
power:  (pending)

PP512 median: — t/s
TG128 median: — t/s
uplift vs plain CPU:   PP ×— , TG ×—
```

### GPU (Adreno / OpenCL)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -c 2048 -ngl 99 -r 3
build:  llama.cpp build-opencl (GGML_OPENCL=ON, Adreno kernels ON)
date:   (pending)
power:  (pending)

PP512 median: — t/s
TG128 median: — t/s
```

### GPU (Adreno / Vulkan)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -c 2048 -ngl 99 -r 3
build:  llama.cpp build-vulkan (GGML_VULKAN=ON)
date:   (pending)
power:  (pending)

PP512 median: — t/s
TG128 median: — t/s
```

## Post-mortem — to be written after measurement

Sections to fill once the matrix has numbers. The questions below are
the ones worth answering, not a template to mindlessly complete.

### Which island wins each workload?

Expected answers going in (to be confirmed or refuted):
- **PP**: GPU wins by a wide margin. Prefill is big-matmul-heavy and
  bandwidth-friendly; that's Adreno's regime. NPU second (the batched
  ar=128 prefill graph), CPU last.
- **TG**: NPU wins narrowly over CPU; GPU last. Single-token decode
  has high kernel-launch overhead on GPU (Phase 2 showed this on
  0.6B). NPU's HMX path is optimized for this exact regime.

If either of the above flips in measurement — that's the story. The
number is a hypothesis test, not a confirmation run.

### Does the quant-parity gap matter?

Q4_K_M vs w4a16 are close but not equal. If the NPU number is
unexpectedly low, check:
- Genie's sampler config overriding greedy? (set `temp: 0`)
- Which ctx tier got selected? (should be 1024 for 512+128; 2048
  would add overhead)
- uint16 dequant cost on IO boundary vs our ORT-QNN-measured 17 ms

If the CPU numbers are unexpectedly low, verify `-t 8` actually
saturated the Prime cluster (check with task manager; see
cpu-mask 0xe0 in genie_config.json for the Prime cluster's cpu mask).

### What does this tell us about the 3-island pipeline?

This table is one slice of the roadmap's `W4.e` phase × island
matrix — specifically, the "which island does each phase best in
isolation" row. Fill it in and the next decisions become
sharper:
- If GPU PP dominates by >5× over NPU PP → W1.a (GPU prefill +
  CPU decode split) becomes the highest-priority roadmap step.
- If NPU TG beats CPU TG by >1.5× → stays on the 0.6B draft path
  only if draft-size ratio math still closes (Axis B of
  `docs/w4a16_investigation_continued.md`).
- If GPU TG is competitive with CPU/NPU → Phase 2's "GPU decode is
  a regression" conclusion is model-dependent and needs re-testing
  on 4B target.
- If both GPU backends (OpenCL, Vulkan) land within 10% of each
  other → Vulkan-only is a viable simplification (we drop the
  OpenCL-specific kernel stack).

Each bullet above is a concrete decision the matrix will unblock.

### Does this change the w4a16 investigation priority?

`docs/w4a16_investigation_continued.md` Phase 5.5.1 is currently
executing (local QAIRT A.2 / A.1 compile pending on the x86 box). The
question this baseline answers: is continuing that investigation the
right call, vs pivoting to a different workstream?

- **If NPU TG on 4B w4a16 is strong (> 35 t/s)**: continuing w4a16
  work on Qwen3-0.6B makes sense — we know the hardware is capable,
  and the 0.6B w4a16 cap is a recipe problem, not a silicon problem.
- **If NPU TG on 4B is weak** (< 25 t/s; bandwidth-bound despite
  w4): silicon is the limit. Further w4a16 work on a smaller draft
  is diminishing returns — pivot to W2.d (tree drafts) or W4 (async
  pipelining) which address the bottleneck directly.
- **If CPU TG approaches NPU TG at 4B**: spec-decode's central
  premise on this silicon weakens — draft speedup buys less than
  expected, and the W1 investment (moving prefill, not decode)
  becomes the highest-leverage lever.

## Artifacts

- Methods: `docs/qwen3_4b_baseline_methods.md`
- Raw logs (to be created): `results/qwen3_4b_baseline/*.log`
- Script helpers (to be created if needed):
  - `scripts/bench_genie_pp_tg.py` — parses `--profile` JSON into
    PP/TG t/s.
  - `scripts/probe_qualcomm_qwen3_4b_chained.py` — extends the
    side-quest probe to all 4 partitions for the NPU (alt) row.

## Update log

- 2026-04-23: scaffold created. Model artifacts downloaded (unsloth
  Qwen3-4B-Q4_K_M.gguf, 2.5 GB) + confirmed (Qualcomm bundle already
  on disk). Genie runtime verified at
  `C:/Qualcomm/AIStack/QAIRT/2.45.40.260406/bin/aarch64-windows-msvc/genie-t2t-run.exe`.
  Measurements pending next session.
