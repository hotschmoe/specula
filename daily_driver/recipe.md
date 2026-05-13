# Recipe — fetch, load, serve Qwen3.6-35B-A3B

Fresh-clone-friendly. Assumes a Snapdragon X2 Elite Extreme laptop
running Windows 11 ARM64 with the four llama.cpp builds present in
`llama.cpp/build-{cpu,cpu-kleidiai,opencl,vulkan}/` (the existing
build process is unchanged from the 4B/7B baselines — see
`docs/qwen3_4b_baseline_methods.md` § "Building llama.cpp").

All paths are repo-relative.

## 0. Prerequisites

- ~50 GB free disk for both Q4 GGUFs + scratch.
- llama.cpp builds present (CPU / CPU+KleidiAI / OpenCL / Vulkan).
  Verify: `ls llama.cpp/build-*/bin/llama-bench.exe`.
- Python venv at `.venv/` with `huggingface_hub` (for `hf` CLI).
  If missing: `uv pip install huggingface_hub`.

If any of these fail, fix them first — the 4B baseline doc has the
exact build invocations.

## 1. Download the GGUFs

Two files, two backend paths. Repo `models/` is gitignored, so push
straight there.

```bash
# CPU / CPU+KleidiAI path — vanilla Q4_K_M (lmstudio-community)
# This is also the "comparable to most users" baseline.
.venv/Scripts/hf.exe download \
    lmstudio-community/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --local-dir models/

# GPU OpenCL / Vulkan path — MXFP4_MOE (Unsloth)
# Replaces the Q4_0 plan: nobody ships plain Q4_0 for this model in 2026.
# MXFP4 is on Adreno's OpenCL fast-path list (alongside Q4_0 / Q8_0)
# AND this build is MoE-tuned (per-expert MXFP4).
.venv/Scripts/hf.exe download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    --local-dir models/
```

Notes — sources, naming, the missing Q4_0:

- **No plain Q4_0 exists for this model.** Verified 2026-04-26 across
  Unsloth, lmstudio-community, mradermacher, and bartowski (latter
  hasn't published yet). The community has moved past Q4_0 to
  K-quants / Unsloth-Dynamic-quants / MXFP4 by 2026. The 7B doc's
  "use Q4_0 on Adreno OpenCL" rule still applies in spirit — the
  fast-path quants are `{Q4_0, Q8_0, MXFP4}`, and **MXFP4 is the
  one that exists**, so MXFP4_MOE substitutes.
- **CPU GGUF is from `lmstudio-community`** (not Unsloth) on purpose:
  it's a vanilla Q4_K_M, which is what we want for the
  "comparable to most users" Q4 baseline the matrix opens with.
  Unsloth ships `UD-Q4_K_M` (their dynamic-quant variant) — also
  worth measuring later, but it's an apples-to-pears comparison
  for the baseline.
- **Expected sizes**: Q4_K_M ≈ **21.17 GB**, MXFP4_MOE ≈ **21.71 GB**.
  Confirmed via the HF tree API on 2026-04-26.
- **Optional — DFlash draft model.** z-lab has trained a DFlash
  speculative-decoding draft specifically for this target:
  `z-lab/Qwen3.6-35B-A3B-DFlash`. Pull it now if you want to bench
  spec decoding (see `optimization.md` § DFlash):
  `hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir models/dflash-draft/`.
  Note: DFlash drafts run on z-lab's own runtime, not llama.cpp —
  benching it means standing up that runtime separately.
- **Future quant sweeps** (per `optimization.md` § Tier-2): Unsloth's
  UD-IQ4_XS (17.7 GB), lmstudio-community's Q6_K (28.5 GB), Q8_0
  (36.9 GB) are all options when we widen the quant axis.

## 2. Sanity load (1-shot llama-bench)

A short PP+TG pass on each backend confirms the GGUFs load and the
builds resolve their backends. Use `-p 128 -n 32` for fast feedback
(under 2 min each on this hardware); the full optimization sweep
runs `-p 512 -n 128 -r 3` and lives in a script (TBD — see
`optimization.md` § Runner).

```bash
# CPU
llama.cpp/build-cpu/bin/llama-bench.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -p 128 -n 32 -t 8 -r 1

# CPU + KleidiAI
llama.cpp/build-cpu-kleidiai/bin/llama-bench.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -p 128 -n 32 -t 8 -r 1

# GPU OpenCL (MXFP4_MOE — on Adreno's fast-path list)
llama.cpp/build-opencl/bin/llama-bench.exe \
    -m models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    -p 128 -n 32 -ngl 99 -r 1

# GPU Vulkan (MXFP4_MOE + the F16-off knobs known to fix Adreno PP)
# Note: F16-off was confirmed for Q4_0 on the 7B. Whether it's still
# a win on MXFP4_MOE is an open question — measure with and without
# during the matrix sweep.
GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
    llama.cpp/build-vulkan/bin/llama-bench.exe \
        -m models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
        -p 128 -n 32 -ngl 99 -r 1
```

If a backend OOMs or refuses to offload the full model:

- **OpenCL `clCreateBuffer` failure** at -ngl 99: drop to a partial
  offload — start with `-ngl 32` and walk up. Adreno's address space
  is tight at 35B Q4_0 (~20 GB).
- **Vulkan VRAM headroom**: the Adreno X2-90 reports its budget via
  `vkGetPhysicalDeviceMemoryProperties` — `GGML_VK_PREFER_HOST_MEMORY=1`
  routes through host-pinned memory which on this UMA-style SoC is
  the right path; without it you can hit a phantom budget cap.
- **CPU OOM** is implausible (48 GB RAM, ~22 GB resident). If it
  happens, check Windows working-set caps and `--mlock` use.

## 3. Serve via llama-server

The daily-driver target is a coding agent at 120k+ context (see
`README.md` § Primary use case).

**Recipe revision history.** Phase 1-10c (2026-04-27) picked Vulkan
`+DISABLE_F16+PREFER_HOST_MEMORY` as the primary backend. **That
config is broken on llama.cpp `856c3adac`** (session 25 sweep) — the
`DISABLE_F16` env var STATUS_ACCESS_VIOLATIONs at model load. With
only `PREFER_HOST_MEMORY` set, Vulkan PP collapses to ~6 t/s, making
the long-ctx cold-start unusable. **Session 27 (2026-05-13) re-tested
all available backends at long context** and switched the primary to
**OpenCL with `-ngl 0`** — the "coprocessor mode" discovered in the
overnight perf sprint.

### Canonical config (session 27)

- `-c 131072` — 128k ctx, the realistic operating point.
- **OpenCL build with `-ngl 0`** — Adreno backend registered as a
  coprocessor; model weights stay in CPU RAM, but the GPU still
  accelerates select ops. **Beats pure-CPU TG by 10-20% at every
  context depth measured.**
- `-t 18` — all physical cores. (The `-t 16` rule from session 26
  applies only to ≤14B dense models; for 35B-A3B the extra thread
  wins because the MoE active-param compute scales.)
- **No `-ctk q8_0 -ctv q8_0`** — GQA-2 + KV @ 131k = 5.2 GB f16, no
  memory pressure to quantize. f16+no-FA is faster than q8+FA.
- **No `-fa 1`** — Flash Attention with f16 KV livelocks at this
  commit on both CPU and Vulkan. OpenCL `-ngl 0` is the regular
  attention path; no FA flag needed.
- **`--no-warmup` not needed** — no shader-JIT to defer like Vulkan
  had. First-request startup is reasonable (~30-90 s for first
  long prompt cold prefill).
- **`--spec-type ngram-mod` is intentionally NOT recommended.**
  Phase 9b made it look like a +11-15% short-ctx win, but Phase 10's
  multi-turn delta-prefill bench found ngram_mod adds ~200 tokens of
  spurious re-prefill per turn when `cache_prompt=true` (the agent-
  loop default) — net **−60% UX per turn at d=16k+**. Avoid until
  that interaction is fixed upstream. See `optimization.md` § Phase 10.

### Session 27 measurements

Probe 1 — TG vs ctx depth (`results/csv/probe1_35b_ngl0_longctx_2026-05-13.md`):

| depth | OpenCL `-ngl 0` PP/TG | CPU-kleidiai PP/TG | OpenCL wins by |
|---:|---:|---:|---:|
|     0 | 198 / **28.7** | 176 / 24.9 | TG +15% |
|  8192 | 132 / **24.8** | 126 / 20.8 | TG +19% |
| 32768 |  59 / **14.5** |  59 / 13.2 | TG +10%, PP equal |

(d=65k/131k bench preroll exceeded 60-min budget; trend extrapolates
to TG ~5-8 t/s at d=131k. OpenCL is still expected to hold its
~10-20% lead over CPU at any depth based on the curve.)

Probe 3 — delta-prefill with `cache_prompt=true`
(`results/csv/probe3_delta_prefill_2026-05-13.md`):

| turn | prompt_n | cache_n | wall (s) | PP (t/s) | TG (t/s) |
|---|---:|---:|---:|---:|---:|
| 1 cold (5457 toks) | 5457 |    0 | 86.8 | 64.8 | 27.6 |
| 2 (same prefix + 524 delta) | 524 | 4941 | **13.5** | 44.0 | 28.0 |
| 3 (rerun, full cache hit)   |   4 | 5461 |  **1.6** | 57.8 | 28.4 |

**Prefix-cache works flawlessly on OpenCL `-ngl 0`.** Turn 2 saved
73 seconds by reusing 4941 cached tokens; turn 3 saved another 85
seconds. The agentic loop is fast on the warm path.

### Serve commands

```bash
# Primary (session-27 canonical): OpenCL -ngl 0 + MXFP4_MOE
llama.cpp/build-opencl/bin/llama-server.exe \
    -m models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    -ngl 0 -t 18 \
    -c 131072 \
    --alias daily-driver \
    --host 127.0.0.1 --port 8080

# CPU fallback — simpler env, ~10-20% lower TG, same memory footprint.
# Use when troubleshooting OpenCL or running without GPU drivers.
llama.cpp/build-cpu-kleidiai/bin/llama-server.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -t 18 \
    -c 131072 \
    --alias daily-driver \
    --host 127.0.0.1 --port 8080

# Vulkan rollback (DEPRECATED at this commit) — kept for if/when
# upstream restores the F16-off path. Currently segfaults at model
# load with DISABLE_F16=1. With only HOST_MEMORY=1 set, PP drops
# to ~6 t/s (unusable for cold prefill of long ctx).
GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
    llama.cpp/build-vulkan/bin/llama-server.exe \
        -m models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
        -ngl 99 -c 131072 \
        --flash-attn off --no-warmup \
        --alias daily-driver \
        --host 127.0.0.1 --port 8080
```

All run with default KV (f16) and no `-fa`. KV stays f16 because:

- GQA-2 architecture makes the KV small (5.2 GB at 131k) — no
  memory pressure to quantize.
- Phase 4 found f16 + no-FA is faster than q8 + FA on CPU.
- FA + f16 KV livelocks on this llama.cpp commit on both CPU and
  Vulkan.

**Memory at 131k ctx**: model 22 GB + f16 KV 5.2 GB ≈ 27 GB resident.
Comfortable on 48 GB system. Adreno's default OpenCL 24 GB cap is not
a problem at `-ngl 0` (weights live in CPU RAM, not GPU). If you ever
need `-ngl 99` for a smaller model on 35B-A3B, set
`GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1` to unlock the full 44 GB BIOS
allocation via the `cl_qcom_large_buffer` extension — but `-ngl 99`
on this MoE craters TG (16 vs 28-31 at `-ngl 0`) so don't.

**OpenCL `-ngl 99` is not recommended.** Phase 1 showed it winning PP
(180 t/s on the old commit; 210 today) but TG drops by half (15-17
t/s at `-ngl 99` vs 28-31 at `-ngl 0`). Per-token kernel-launch
overhead and the bandwidth penalty of GPU-resident weights for AR=1
decode dominate. The agent loop is TG-bound, so we want `-ngl 0`.

OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1/chat/completions`.
Web UI at `http://127.0.0.1:8080`.

## 4. Smoke test the server

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3.6-35b-a3b",
      "messages": [{"role":"user","content":"Say hi in one word."}],
      "max_tokens": 16
    }'
```

A response under ~3 s end-to-end means the path is healthy. Slower
than that on a Q4 35B-A3B (3B active) means a config knob is wrong —
go to `optimization.md` § Findings to check.

## What's next

Run the optimization matrix in `optimization.md`. The sweep produces
CSVs in `results/csv/daily_driver_*.csv`; the headline table at the
top of `optimization.md` is updated as rows land.
