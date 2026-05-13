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
- Native context: **262144 (256k)**

This means **memory headroom is NOT the binding constraint** —
f16 KV at 131k = 5.2 GB → 27 GB total on 48 GB system. Even at the
full 256k native ctx it's only 32.5 GB total.

**Canonical config (session 27, 2026-05-13)**: **OpenCL `-ngl 0`
("coprocessor mode") is the new primary; CPU is fallback; Vulkan
is deprecated.**

Why the change from the Phase-10c (2026-04-27) Vulkan recipe:

- **Vulkan PP path broken on `856c3adac`.** Session 25's scheduled
  llama.cpp sweep landed PRs that segfault `GGML_VK_DISABLE_F16=1`
  at model load (STATUS_ACCESS_VIOLATION). Without `DISABLE_F16`,
  Vulkan PP collapses to ~6 t/s — long-ctx cold prefill is
  unusable.
- **OpenCL `-ngl 0` discovered.** Session 26's overnight perf sprint
  found that registering the OpenCL backend with `-ngl 0` (no model
  layers offloaded) lets Adreno act as a coprocessor while weights
  stay in CPU RAM. Yields **+10-20% TG over pure CPU at every
  context depth** with no shader-JIT first-request penalty.
- **Session 27 long-ctx + delta-prefill probes confirm it works.**
  TG at d=32k: OpenCL `-ngl 0` 14.5 t/s vs CPU 13.2 (+10%).
  Delta-prefill via `cache_prompt`: turn-2 wall 13.5 s on a 5k base
  + 524 delta (vs turn-1 cold 86.8 s). Cache reuse: 4941/5461
  tokens. The agent warm path is fast.

| metric (probes 1+3, session 27)        | CPU-kleidiai | OpenCL `-ngl 0` | Δ        |
|---|---:|---:|---:|
| TG @ d=0                                | 24.85 t/s    | **28.70 t/s**  | **+15%** |
| TG @ d=8k                               | 20.77 t/s    | **24.80 t/s**  | **+19%** |
| TG @ d=32k                              | 13.19 t/s    | **14.51 t/s**  | **+10%** |
| Turn-2 delta-prefill wall (5k+524)      | not measured |   13.5 s       | --       |
| Cache reuse on turn 2                   | --           | 4941 toks      | works    |

- **Primary**: `OpenCL build + MXFP4_MOE + -ngl 0 -t 18 + f16 KV`.
  No env vars needed. No `--no-warmup` (no shader-JIT to defer).
- **CPU fallback**: `cpu-kleidiai build + Q4_K_M + -t 18 + f16 KV`.
  Slightly slower (10-20% lower TG) but the simplest env. Same
  memory footprint.
- **Vulkan deprecated** until upstream restores the `DISABLE_F16`
  path. The Phase-10c numbers above are no longer reproducible.

All paths use **f16 KV + no FA**. FA + f16 KV livelocks on llama.cpp
(`f53577432` thru `856c3adac`+) on both CPU and Vulkan — Phase 4 /
Phase 10c documented this. The OpenCL `-ngl 0` path doesn't hit
the FA livelock but we still leave `-fa` unset for consistency.

See `optimization.md` § Phase 10c for the original Vulkan
delta-prefill analysis (since invalidated by session-25 sweep).
See `docs/2026-05-13_overnight_perf_results.md` for the OpenCL
`-ngl 0` discovery context. See `recipe.md` for the canonical
serve invocation.

## Run it (opencode loop)

Three PowerShell scripts (`scripts/`) wrap the canonical flow:

```powershell
# 1. Start the server (Vulkan canonical, port 8080, alias 'daily-driver')
.\scripts\serve_daily_driver.ps1

# 2. In a separate shell, verify it's up + smoke-test a completion
.\scripts\check_daily_driver_status.ps1

# 3. Launch opencode in your project (cwd matters — opencode reads
#    the local repo); it picks up the global config at
#    ~/.config/opencode/opencode.json which is already set to the
#    'llama-cpp-local' provider pointing at http://localhost:8080/v1.
opencode
```

In opencode pick the model **`llama-cpp-local/daily-driver`**. The
alias is set by `--alias daily-driver` in `serve_daily_driver.ps1`
and must match the `models.daily-driver` key in opencode.json — if
you change one, change both.

**First request after server boot is moderate** on OpenCL `-ngl 0`
(~30-90 s depending on prompt length; no shader-JIT hit unlike the
old Vulkan recipe). Subsequent turns at a 5k warm context take
~13 s/turn (probe 3); at d=16k expect ~20-25 s/turn; at d=32k+
~30-45 s/turn (PP path slowdown dominates the cold delta cost).

Backend overrides:

```powershell
# CPU fallback — simpler env, ~10-20% lower TG.
.\scripts\serve_daily_driver.ps1 -Backend cpu

# Vulkan rollback (DEPRECATED — segfaults at load on 856c3adac).
# Only kept for the event that upstream restores the F16-off path.
.\scripts\serve_daily_driver.ps1 -Backend vulkan
```

Same opencode model id (`daily-driver` alias is set by both backends).

### opencode.json (reference)

Lives at `~/.config/opencode/opencode.json` (outside the repo, not
tracked). Re-create from this if you wipe your machine:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-cpp-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local Qwen3.6-35B-A3B)",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "daily-driver": {
          "name": "Qwen3.6-35B-A3B (daily-driver)",
          "limit": {
            "context": 128000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

The `models.daily-driver` key MUST match the `--alias` passed to
llama-server. `serve_daily_driver.ps1` sets it; if you ever invoke
llama-server by hand, pass `--alias daily-driver`.

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

Phase: **canonical config revised 2026-05-13 (session 27).** Phases
1-10c (2026-04-27) picked Vulkan as primary; that recipe broke when
session 25's scheduled llama.cpp sweep landed PRs that segfault the
`DISABLE_F16` env var at model load. Sessions 26-27 re-tested all
backends at long ctx + delta-prefill and switched the primary to
**OpenCL `-ngl 0` coprocessor mode**. opencode integration scripts
unchanged; `serve_daily_driver.ps1 -Backend opencl` is the new default.

Open queues (see `optimization.md` "Things still open"):
- Quality probe at 120k ctx (was open under Vulkan; still open under
  OpenCL `-ngl 0`)
- slot-save/-load to amortize d=120k cold prefill
- FA upstream fix
- ngram_mod cache-reuse workaround test
- **NEW**: TG measurement at d=131k on OpenCL `-ngl 0` (probe 1 ran
  out of budget at d=32k; need to run with `slot_save_path` so the
  d=131k state can be reused across measurements)
- **NEW**: PR #22673 (MTP support) — once it merges to mainline,
  test if MTP self-draft helps 35B-A3B's long-ctx TG (session 27 saw
  +45-55% on Qwen3.6-27B Q4_0/Q8_0 — would be a big agent UX win)

Update log lives in `optimization.md`; broader session log in
`current_status.md` at repo root.

## Pointers

- `recipe.md` — actionable runbook (download, sanity-bench, serve
  command lines).
- `optimization.md` — variable matrix, results so far, advanced
  speed-path research questions.
- `../scripts/serve_daily_driver.ps1` — one-shot server launcher.
- `../scripts/check_daily_driver_status.ps1` — health probe + smoke
  completion.
- `~/.config/opencode/opencode.json` — opencode points at the local
  server; model id is `llama-cpp-local/daily-driver`.
- `docs/qwen2_5_7b_baseline_all_backends.md` — the 7B side-quest
  whose findings (Q4_0 for GPU, `DISABLE_F16+PREFER_HOST` for
  Vulkan) directly inform the 35B starting configs.
- `docs/repo_hygiene.md` — rules for what's kept vs staged for
  deletion. CSVs go in `results/csv/`, raw logs in
  `marked_for_deletion/`.
