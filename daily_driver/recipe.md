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
`README.md` § Primary use case). After the Phase 1-4 sweeps
(2026-04-27) the canonical serve config landed simpler than expected:

- `-c 131072` — 128k ctx, the realistic operating point
- **No `-ctk q8_0 -ctv q8_0`** — GQA-2 architecture makes f16 KV at
  131k cost only ~5 GB; quantizing it doesn't buy memory headroom we
  need, AND f16+no-FA is **faster than q8+FA** (Phase 4: 15.27 vs
  14.16 t/s at d=32k).
- **No `-fa 1`** — Flash Attention with f16 KV livelocks on this
  llama.cpp commit (`f53577432`) for both CPU and Vulkan paths.
  Without FA, llama.cpp uses the regular attention path which works
  cleanly. FA is only required when KV ≠ f16 (and the FA-on path's
  performance penalty exceeds the KV-bandwidth savings at our depth).
- **`--spec-type ngram-mod` is intentionally NOT recommended** here.
  Phase 9b made it look like a +11-15% short-ctx win, but Phase 10's
  multi-turn delta-prefill bench found ngram_mod adds ~200 tokens of
  spurious re-prefill per turn when `cache_prompt=true` (the agent-
  loop default) — net **−60% UX per turn at d=16k+**. Avoid until
  that interaction is fixed upstream. See `optimization.md` § Phase 10.

**Two canonical configs** (2026-04-27 update from Phase 8):

- **Vulkan + MXFP4_MOE** for long-ctx agent loops (d ≥ 16k or so).
  Wins TG by +10.6% over CPU at d=32k and the gap should grow at
  longer ctx (Vulkan's TG slope vs ctx is gentler than CPU's).
  Pays a cold-prefill cost (~3.5× slower than CPU) which the agent
  loop only experiences once per session.
- **CPU + Q4_K_M** for short-ctx work (chat, single-shot, d ≤ 8k)
  and as the fallback when Vulkan is unavailable. Wins TG at d=8k
  by +19% over Vulkan.

```bash
# Vulkan server — daily-driver default for the agent loop (Phase 10c)
# --flash-attn off: FA + f16 KV livelocks on both CPU and Vulkan in
#   this llama.cpp commit (Phase 4); default --flash-attn auto enables
#   it during warmup and the server hangs. Will revisit once upstream
#   fixes FA — could give Vulkan extra TG headroom.
# --no-warmup: defers Vulkan shader-compilation to the first real
#   request (otherwise it blocks startup beyond any reasonable health
#   timeout). The first request pays a ~5 min one-time cost on top of
#   its prefill; subsequent requests are normal.
# Spec-decode flags (--spec-type / --lookup-cache-*) are NOT wired
# into the Vulkan build at this commit and aren't recommended on CPU
# either (Phase 10 found a slot-cache penalty).
GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
    llama.cpp/build-vulkan/bin/llama-server.exe \
        -m models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
        -ngl 99 \
        -c 131072 \
        --flash-attn off --no-warmup \
        --host 127.0.0.1 --port 8080

# CPU fallback — only when Vulkan is unavailable, or for trivially
# short single-turn chat. Phase 10c measured the agent-loop UX as
# 21.8 s/turn here vs 13.2 s/turn on Vulkan at d=16k+.
llama.cpp/build-cpu/bin/llama-server.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -t 8 \
    -c 131072 \
    --host 127.0.0.1 --port 8080
```

Both run with default KV (f16) and no `-fa`. KV stays f16 because:

- GQA-2 architecture makes the KV small (5.2 GB at 131k) — no
  memory pressure to quantize.
- Phase 4 found f16 + no-FA is faster than q8 + FA on CPU.
- FA + f16 KV livelocks on this llama.cpp commit on both CPU and
  Vulkan.

**Memory at 131k ctx**: model 22 GB + f16 KV 5.2 GB ≈ 27 GB resident.
Comfortable on 48 GB system.

**OpenCL is not the recommended GPU path** despite winning Phase 1
PP (180 t/s). At long ctx its TG drops to half of Vulkan's
(8.58 vs 16.89 at d=32k) — Adreno OpenCL's per-token kernel-launch
overhead dominates AR=1 decode. Use OpenCL only if Vulkan's
~3.5× slower cold-prefill is unacceptable for your workload.

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
