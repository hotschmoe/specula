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

## Primary use case (drives every decision below)

This rig serves a **long-running coding agent inside a harness** —
opencode, Hermes, or similar. Concrete profile:

- **Concurrency = 1.** One agent at a time. Multi-stream throughput
  is irrelevant for this workload, so the N=4 wins from the 4B/7B
  side-quests do not transfer.
- **Context window: 120k+ tokens.** Multi-turn conversation, dozens
  of tool calls, web-research output pasted in, file contents loaded.
  The context grows monotonically until the agent is restarted.
- **Heavy KV cache reuse.** Each turn appends to the prior turn's
  KV; a fresh prompt is the exception. Prefix-cache / slot handling
  in llama-server is the path that makes this cheap.
- **Highly repetitive output.** Tool-call JSON, file paths, error
  messages, code patterns — the same tokens recur across turns.
  Strongly favors n-gram lookup decoding and any technique that
  exploits redundancy.
- **TG-at-long-context is the headline metric.** Not TG-at-1k-ctx.
  A model that does 25 t/s at 4k but 8 t/s at 120k is unusable for
  this loop; one that does 18 t/s at both is great.
- **TTFT after a tool call** is the UX number. Most of the prompt
  is already in the KV cache; only the delta (tool result +
  the harness's framing) needs prefilling. So PP throughput on
  the *marginal* tokens determines feel, not on the whole 120k.

## Current pick (2026-04-26)

**Qwen3.6-35B-A3B** — MoE, 35B total / ~3B active params per token.
Best fit for the 48 GB unified-memory constraint with headroom for
the KV cache: Q4_K_M lands around 21 GB on disk, ~22-24 GB resident.
Active-param count of ~3B keeps per-token decode compute closer to
a dense 3B than a dense 35B, which is the whole point of the MoE
pick at this RAM budget.

**Architecture confirmed** (via `gguf-dump`, 2026-04-27):

- `qwen35moe` arch, 40 transformer blocks
- **GQA with 2 KV heads** (16 Q heads, 8:1 ratio) — minimal KV cache
- 256 experts, 8 used per token (top-8 routing, ultra-sparse MoE)
- Native context: **262144 (256k)** — twice the 131k we initially
  planned to operate at, so plenty of headroom
- Per-token f16 KV cost: ~40 KB; at 131k that's **5.2 GB**, at full
  256k it's 10.5 GB

This means **memory headroom is NOT the binding constraint** —
even f16 KV at the full native 256k ctx leaves 15+ GB free on this
48 GB system. The original "q8 KV is mandatory at long ctx" rule
(based on the wrong assumption of 8 KV heads / pure full-attn) is
no longer required. q8 KV becomes a pure throughput-vs-quality knob,
not a memory necessity. See `optimization.md` § Phase 2 for the
corrected memory math and the Phase 4 plan to disentangle FA cost
from KV-quant cost.

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

| file | source | size | for backend | rationale |
|---|---|---:|---|---|
| `Qwen3.6-35B-A3B-Q4_K_M.gguf`    | `lmstudio-community` | 21.17 GB | CPU, CPU+KleidiAI | vanilla K-quant; matches the "Q4 to compare against most users" goal |
| `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` | `unsloth`            | 21.71 GB | GPU OpenCL, GPU Vulkan | substitutes for Q4_0 (which doesn't exist for this model in 2026): MXFP4 is on Adreno's OpenCL fast-path list `{Q4_0, Q8_0, MXFP4}`, and this build is MoE-tuned (per-expert MXFP4) |

Other quants (UD-Q4_K_M, UD-IQ4_XS, Q5_K_M, Q6_K, Q8_0) come later —
see `optimization.md` § Tier-2 quant sweep.

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
