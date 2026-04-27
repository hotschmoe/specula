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
# CPU / CPU+KleidiAI path (Q4_K_M — K-quant, best ~Q4 quality on CPU)
.venv/Scripts/hf.exe download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --local-dir models/

# GPU OpenCL / Vulkan path (Q4_0 — Adreno fast-path quant)
.venv/Scripts/hf.exe download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-Q4_0.gguf \
    --local-dir models/
```

Notes:

- **Source: Unsloth.** `unsloth/Qwen3.6-35B-A3B-GGUF` is the canonical
  community quant for this model — Unsloth has become the default
  publisher for fresh Qwen GGUFs. Bartowski may also ship one later;
  prefer Unsloth for now.
- Expected sizes: **Q4_K_M ≈ 21 GB**, **Q4_0 ≈ 19-20 GB**. If
  downloads land smaller, the file is probably the placeholder
  `.gguf` symlink — re-run with `--local-dir-use-symlinks False`.
- For multi-part GGUFs (some 35B+ quants ship as `*.gguf-00001-of-N`,
  `*.gguf-00002-of-N`), download the whole shard set instead:
  `hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "*Q4_K_M*" --local-dir models/`.
- **Optional — DFlash draft model.** z-lab has trained a DFlash
  speculative-decoding draft specifically for this target:
  `z-lab/Qwen3.6-35B-A3B-DFlash`. Pull it now if you want to bench
  spec decoding (see `optimization.md` § DFlash):
  `hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir models/dflash-draft/`.
  Note: DFlash drafts run on z-lab's own runtime, not llama.cpp —
  benching it means standing up that runtime separately.

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

# GPU OpenCL (Q4_0 model)
llama.cpp/build-opencl/bin/llama-bench.exe \
    -m models/Qwen3.6-35B-A3B-Q4_0.gguf \
    -p 128 -n 32 -ngl 99 -r 1

# GPU Vulkan (Q4_0 model + the F16-off knobs known to fix Adreno PP)
GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
    llama.cpp/build-vulkan/bin/llama-bench.exe \
        -m models/Qwen3.6-35B-A3B-Q4_0.gguf \
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

Once a backend passes sanity, you can serve it. CPU + Q4_K_M is the
simplest path; switch to `build-vulkan` + Q4_0 once the optimization
matrix confirms the GPU win at 35B (the 7B doc says Vulkan-Q4_0
beats CPU on TG single-stream; confirm at 35B before defaulting).

```bash
# CPU server, default port 8080
llama.cpp/build-cpu/bin/llama-server.exe \
    -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -t 8 -c 8192 --host 127.0.0.1 --port 8080

# GPU Vulkan server (once confirmed)
GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
    llama.cpp/build-vulkan/bin/llama-server.exe \
        -m models/Qwen3.6-35B-A3B-Q4_0.gguf \
        -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

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
