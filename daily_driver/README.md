# daily_driver

The model we actually run for production-style local serving on the
Snapdragon X2 Elite Extreme laptop. **Not** the bench-matrix work — the
all-backends side-quests for Qwen3-4B and Qwen2.5-7B live in
`docs/qwen3_4b_baseline_all_backends.md` and
`docs/qwen2_5_7b_baseline_all_backends.md`. This directory is for the
*chosen* model and the optimization sweep around it.

Two purposes only:

1. **Recipe to fetch + load + serve** the chosen model on a fresh
   repo clone (`recipe.md`).
2. **Optimization matrix and findings** — what knobs we sweep, what's
   been measured, what's open, what advanced speed paths to research
   (`optimization.md`).

Models themselves live in the repo's existing `models/` directory
(gitignored, large), not here.

## Current pick (2026-04-26)

**Qwen3.6-35B-A3B** — MoE, 35B total / ~3B active params per token.
Best fit for the 48 GB unified-memory constraint with headroom for
the KV cache: Q4_K_M lands around 21 GB on disk, ~22-24 GB resident,
leaving ~20 GB for runtime + KV. Active-param count of ~3B keeps
per-token decode compute closer to a dense 3B than a dense 35B,
which is the whole point of the MoE pick at this RAM budget.

Backends in scope:

- **CPU** (llama.cpp `build-cpu`, ARM64 NEON, `-t 8`)
- **CPU + KleidiAI** (llama.cpp `build-cpu-kleidiai`, i8mm)
- **GPU OpenCL** (llama.cpp `build-opencl`, Adreno X2-90, `-ngl 99`)
- **GPU Vulkan** (llama.cpp `build-vulkan`, Adreno X2-90, `-ngl 99`)

**NPU is explicitly deferred.** The Workbench compile path for a 35B
MoE bundle is not in scope yet — the W1.b roadmap entry only goes to
8B-class dense. Re-add NPU once the cloud-export pipeline
(`docs/rent_cloud_compute.md`) is wired up for MoE architectures.

## Quants downloaded (per-backend optimal Q4)

The 4B and 7B baselines established that GPU and CPU prefer different
Q4 variants on this silicon, so we keep both on disk:

| file | path | for backend | rationale |
|---|---|---|---|
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `models/` | CPU, CPU+KleidiAI | best K-quant quality at ~Q4 size; CPU has no Q4_0 fast-path advantage |
| `Qwen3.6-35B-A3B-Q4_0.gguf`   | `models/` | GPU OpenCL, GPU Vulkan | Adreno OpenCL kernels are optimized for Q4_0/Q8_0/MXFP4; Q4_K is fallback. Vulkan with `GGML_VK_DISABLE_F16=1` also fastest on Q4_0 (per 7B doc) |

Other quants (Q5_K_M, Q6_K, Q8_0, IQ-class) come later — see
`optimization.md` § Quantization sweep.

## Status

Phase: **kickoff**, 2026-04-26. Models downloading; first AC bench
sweep on the optimization matrix not yet run. Update log lives in
`optimization.md`; broader session log in `current_status.md` at
repo root.

## Pointers

- `recipe.md` — actionable runbook (download, sanity-bench, serve).
- `optimization.md` — variable matrix, results so far, advanced
  speed-path research questions.
- `docs/qwen2_5_7b_baseline_all_backends.md` — the 7B side-quest
  whose findings (Q4_0 for GPU, `DISABLE_F16+PREFER_HOST` for
  Vulkan) directly inform the 35B starting configs.
- `docs/repo_hygiene.md` — rules for what's kept vs staged for
  deletion. CSVs go in `results/csv/`, raw logs in
  `marked_for_deletion/`.
