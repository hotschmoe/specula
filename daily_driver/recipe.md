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
- `--lookup-cache-dynamic` — n-gram lookup decoding for the
  repetitive tool-call output (still recommended; see
  `optimization.md` § N-gram).

CPU is the only viable backend at long ctx (Phase 2 attempt #2 +
Phase 3 ruled out OpenCL and Vulkan for FA paths; Phase 4 confirmed
CPU TG=15.27 at d=32k).

```bash
# CPU server — daily-driver canonical config (2026-04-27)
llama.cpp/build-cpu/bin/llama-server.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -t 8 \
    -c 131072 \
    --lookup-cache-dynamic .cache/llama-lookup-dynamic.bin \
    --host 127.0.0.1 --port 8080
```

**Memory at 131k ctx**: model 22 GB + f16 KV 5.2 GB ≈ 27 GB resident.
Comfortable on 48 GB system. The model is GQA-2, so KV is small
compared to most 35B-class models.

**Why GPU server isn't documented**: as of llama.cpp `f53577432`,
the OpenCL backend doesn't support FA's SET_ROWS op (so KV must
stay f16 + no-FA), and Vulkan's f16-codepath silently falls into a
slow scalar fallback for quantized weights (so MXFP4_MOE needs the
F16-off env knob — which then breaks FA paths anyway). Phase 1
short-ctx Vulkan numbers (TG=22.7 at d=128) are below CPU's
single-stream TG even at d=32k. Re-evaluate when llama.cpp gets
better Adreno kernel coverage; until then, CPU.

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
