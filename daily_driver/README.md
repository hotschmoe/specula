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

**Canonical config (post-Phase-10c, 2026-04-27)**: **Vulkan
everywhere; CPU is fallback only.**

- **Daily-driver default** = `Vulkan + MXFP4_MOE + ngl=99 +
  GGML_VK_DISABLE_F16=1 + GGML_VK_PREFER_HOST_MEMORY=1 +
  --flash-attn off + --no-warmup + f16 KV`. Phase 10c measured
  this against CPU on the actual agent workload (16k cold session
  + 4 multi-turn delta-prefill turns) and Vulkan won every metric:

  | metric                       | CPU    | Vulkan | Δ        |
  |---|---:|---:|---:|
  | Cold session @ d=16k         | 748 s  | 340 s  | **−55%** |
  | Per-turn warm wall (avg 1-4) | 21.8 s | 13.2 s | **−39%** |
  | TG @ d=16k+ (server)         | 15.5   | 20.0   | **+29%** |
  | Cold PP aggregate            | 22.2   | 49.4   | **+123%**|

- **CPU + Q4_K_M fallback**: same `-t 8 + f16 KV + no FA` config,
  used only when Vulkan is unavailable OR for trivially-short
  single-turn chat at d≤4k.

Both use **f16 KV + no FA**. FA + f16 KV livelocks on this llama.cpp
commit (`f53577432`) on both CPU and Vulkan — Phase 4 / Phase 10c
documented this; revisit once upstream fixes it. `--no-warmup` on
Vulkan defers shader-JIT to the first request (otherwise warmup
itself hits the FA livelock and never completes); the first
request after server boot pays an extra ~5 min one-time cost.

See `optimization.md` § Phase 10c for the full delta-prefill
analysis; `recipe.md` for the canonical serve invocation.

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

**First request after server boot is slow** (~5-6 min on Vulkan:
shader JIT + 16k cold prefill folded together because of
`--no-warmup`). Subsequent turns are normal (~13 s/turn at d=16k+).
Plan the first interaction accordingly.

CPU fallback if Vulkan acts up:

```powershell
.\scripts\serve_daily_driver.ps1 -Backend cpu
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

Phase: **canonical config locked, 2026-04-27.** Phases 1-10c done;
Vulkan is the daily-driver backend; opencode integration scripts
landed. Open queues (see `optimization.md` "Things still open"):
quality probe at 120k ctx, slot-save/-load to amortize d=120k cold
prefill, FA upstream fix, ngram_mod cache-reuse workaround test.
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
