# Qwen3-4B all-backends baseline — testing methodology

Companion to `docs/qwen3_4b_baseline_all_backends.md`. This doc is the
**recipe**; that doc is the **results table**.

## Goal

Put one comparable Qwen3-4B throughput number on every compute island
available on the X2 Elite Extreme, using each island's *current* blessed
runtime. Two metrics per backend — **prompt processing (PP)** and
**token generation (TG)**. Knowing where each island is strong or weak
informs the heterogeneous-pipeline decisions in `docs/roadmap.md` W1
(prefill placement) and W4 (3-phase × 3-island matrix) — and tells us
whether to keep investing in w4a16 draft-model quantization or pivot.

This pauses `docs/w4a16_investigation_continued.md` until the baseline
matrix is filled. Once filled, the matrix tells us which workstream to
prioritize next.

## Matrix shape

Rows = backend, columns = metric. Each backend runs the same weight
footprint (~w4 / Q4_K_M ≈ 2.5 GB) so we're comparing hardware on a
roughly like-for-like quantization, not comparing quant schemes.

| backend | runtime / build                           | PP512 (t/s) | TG128 (t/s) |
|---|---|---|---|
| NPU           | genie-t2t-run (QAIRT 2.45)                 | — | — |
| CPU           | llama.cpp build-cpu (ARM64 NEON)           | — | — |
| CPU+KleidiAI  | llama.cpp build-cpu-kleidiai (NEON + i8mm) | — | — |
| GPU (OpenCL)  | llama.cpp build-opencl (Adreno kernels)    | — | — |
| GPU (Vulkan)  | llama.cpp build-vulkan                     | — | — |

Secondary cells to fill if the primary row is ambiguous: PP128 (short
prompt TTFT regime), TG512 (long decode / thermal soak). Don't gate the
headline on them.

## Model inputs

Two artifacts, both already on disk:

1. **NPU**: `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`
   — Qualcomm's shipping bundle. 4 × `qwen3_4b_part_N_of_4.bin` (QNN
   context binaries, w4a16, weight-shared), `tokenizer.json`,
   `genie_config.json`, `htp_backend_ext_config.json`. Ten graphs per
   partition: five ctx tiers (512 / 1024 / 2048 / 3072 / 4096) ×
   two AR sizes (ar1 decode, ar128 prefill). Runtime auto-selects.
2. **CPU / GPU**: `models/Qwen3-4B-Q4_K_M.gguf` — unsloth's Q4_K_M
   quant from `https://huggingface.co/unsloth/Qwen3-4B-GGUF`. 2.5 GB.
   Downloaded via `hf_hub_download(repo_id="unsloth/Qwen3-4B-GGUF",
   filename="Qwen3-4B-Q4_K_M.gguf", local_dir="models")` with the
   project venv's huggingface_hub.

### Quant parity caveat

The two artifacts are *not* bit-identical quants:

- **Qualcomm w4a16** = 4-bit weights, 16-bit activations. Per-channel /
  per-row weight scales; attention projections likely mixed precision.
  IO: uint16 hidden states, uint8 past_kv, int32 input_ids.
- **GGUF Q4_K_M** = mixed 4-bit / 6-bit weights (Q4_K for attn/FFN, Q6_K
  for embed / output / gate). Activations run at whatever the compute
  kernel dictates (fp16 on most paths). Typically ~4.5 bits/weight
  average.

Net effect: weight footprint within ~10% across both. Accuracy curves
are close but not identical. This is the best apples-to-apples we get
without owning the Qualcomm quant pipeline end-to-end — good enough for
an order-of-magnitude compute-island comparison, not fine enough to
settle tie-breakers within a factor of 1.2×.

## Reporting conventions

Match the Phase 5.5 AC/battery discipline already baked into
`docs/roadmap.md`:

1. **Power state is a first-class axis.** Every number reports `AC` or
   `BAT`. Default is AC (wall power, plugged in >5 minutes, idle). Add
   a separate row for battery runs if we take them; don't average the
   two.
2. **Warmup.** Discard the first run of any timed loop — NPU HMX
   context init and GPU shader compile both add multi-second fixed
   cost to the first call.
3. **Iterations.** Report median over 3–5 measured runs (llama-bench's
   `-r 3` default is fine). Note min/max in the results doc if
   variance exceeds ±5% of median.
4. **Thermal state.** Close Chrome, Teams, etc. Wait 60 s of CPU-idle
   before benchmarking. For any backend where TG128 is < 10× TG512 of
   the same backend, repeat TG512 on a cold laptop to quantify thermal
   throttling — flag in the results doc, don't bury it in the number.
5. **Context size.** Use ctx=2048 across all backends. Genie bundle
   ships `{512,1024,2048,3072,4096}` — 2048 comfortably fits PP512 +
   TG128 with headroom. llama.cpp defaults are fine at that ctx.
6. **Prompt content.** Use the same synthetic 512-token prompt for all
   PP512 runs — llama-bench generates one internally; for genie-t2t-run
   save a 512-token chunk of Wikipedia plain text to
   `results/qwen3_4b_baseline/pp512_prompt.txt` and point both runtimes
   at the same token count (not the same literal text — llama.cpp is
   GGUF-tokenized, Genie uses the bundle's tokenizer).

## Environment setup — NPU (Genie)

`genie-t2t-run.exe` needs QAIRT DLLs on `PATH`. On our machine the
SDK lives at `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406`. Add its
`lib/aarch64-windows-msvc` to `PATH` and run from **within** the bundle
directory so the relative paths in `genie_config.json` (for the .bin
files, tokenizer.json, htp_backend_ext_config.json) all resolve:

```bash
# One-shot, per-session — add the QAIRT ARM64 libs to PATH
export QAIRT_ROOT="/c/Qualcomm/AIStack/QAIRT/2.45.40.260406"
export PATH="$QAIRT_ROOT/lib/aarch64-windows-msvc:$PATH"

# Work from inside the bundle directory (genie_config.json uses
# relative paths to tokenizer.json + the 4 .bin files)
cd "models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
```

**Version caveat.** The bundle was compiled with QAIRT 2.42; our SDK is
2.45. Same major series, expected to load. If Genie refuses, install
2.42 alongside and switch `QAIRT_ROOT`.

**QAIRT 2.45 bundles QAIRT 2.42-compiled binaries OK for ORT-QNN** per
`results/qwen3_4b_genie_w4a16_probe.md` (the side-quest already loaded
these same .bin files via ORT-QNN 1.24.4 + QAIRT 2.42). Genie 2.45's
forward-compat with 2.42 binaries is expected but not yet empirically
confirmed by us — if it fails, that's the first troubleshooting step.

## Backend recipes

### NPU (Genie) — genie-t2t-run

Primary tool. Ships with QAIRT 2.45. Consumes `genie_config.json` +
the 4 × `.bin` partitions + `tokenizer.json` directly.

```bash
# PP + TG in one call. --profile emits per-graph timings we can
# parse for PP rate and TG rate separately.
"$QAIRT_ROOT/bin/aarch64-windows-msvc/genie-t2t-run.exe" \
    --config genie_config.json \
    --prompt_file ../../../results/qwen3_4b_baseline/pp512_prompt.txt \
    --profile ../../../results/qwen3_4b_baseline/genie_profile.json \
    --log info
```

**Parsing the result.** `genie-t2t-run` prints PP rate (tokens/s) +
TG rate (tokens/s) in the `--log info` output; `--profile` dumps
per-graph call counts + durations in JSON. Source of truth is the
profile JSON — the stdout summary rounds. Compute:

- `PP t/s = prefill_tokens / prefill_time_sec` (prefill batch
  is ar=128 per Qualcomm's convention — 4 prefill calls cover 512
  tokens).
- `TG t/s = generated_tokens / decode_time_sec` (128 ar=1 calls).

If the profile JSON doesn't split prefill vs decode cleanly, fall
back to the stdout summary and record both sources in the results
doc so we can retroactively re-derive.

**Bounding generation.** Genie's `genie_config.json` currently has
no explicit `n-predict` cap — it generates until EOS or context
fills. For deterministic TG128, either:
- Inject a `"dialog.stop": {"max_tokens": 128}` field into a copy
  of the config (schema subject to change across QAIRT versions;
  verify with Qualcomm docs) **or**
- Use a prompt that reliably elicits ≥ 128 tokens and post-filter
  the profile JSON to the first 128 decode calls.

Option 2 is more robust across SDK versions. Script
`scripts/bench_genie_pp_tg.py` (to be written in the first
measurement session) handles the parse + slice.

### CPU (ARM64 NEON) — llama.cpp `build-cpu`

Canonical baseline. Built with `scripts/build_llama_cpp.ps1 -Preset cpu`
(KleidiAI OFF, no ARM SME). `llama-bench` reports PP512 and TG128 in
one command.

```bash
llama.cpp/build-cpu/bin/llama-bench.exe \
    -m models/Qwen3-4B-Q4_K_M.gguf \
    -p 512 -n 128 \
    -c 2048 \
    -t 8 \
    -r 3 \
    -o md
```

**Thread count.** X2 Elite has 12 Oryon cores in two 6-core clusters
(Prime + Perf). Benchmark at `-t 8` first (sticks to one cluster,
avoids cross-cluster L2 overhead); if the result saturates, scan
`-t {6, 8, 10, 12}` and keep the max.

Genie's bundle config uses `n-threads: 3` + `cpu-mask: 0xe0` (cores
5–7, the Prime cluster's lower half). That's a very specific tuning
for the **CPU sampling / de-quant work that runs alongside NPU
decode**; it is **not** the right recipe for a pure-CPU target run.
Use `-t 8` or higher for this cell.

### CPU+KleidiAI — llama.cpp `build-cpu-kleidiai`

Same model, same bench flags, different build. KleidiAI adds i8mm /
DOT ukernels (better Q4_K_M PP). SME2 runtime-fenced by
`scripts/patch_kleidiai_detect.py` (see `docs/SME_investigation.md`;
SME2 would need `GGML_KLEIDIAI_SME=<n>` env to enable — leave unset
for this baseline).

```bash
llama.cpp/build-cpu-kleidiai/bin/llama-bench.exe \
    -m models/Qwen3-4B-Q4_K_M.gguf \
    -p 512 -n 128 \
    -c 2048 \
    -t 8 \
    -r 3 \
    -o md
```

This row tells us whether the KleidiAI kernels are doing useful work
at Qwen3-4B scale. On Qwen3-0.6B it was a small win; on a 4B model
PP should benefit more because matmul tiles get bigger. If PP512
lifts <10% over the plain-CPU row, something is broken or KleidiAI
isn't dispatching — record stderr for `_NATIVE_CHECK` hits.

### GPU (Adreno via OpenCL) — llama.cpp `build-opencl`

Adreno X2-90 on X2E. llama.cpp's OpenCL backend has Adreno-specific
tuned kernels (enabled via `-DGGML_OPENCL_USE_ADRENO_KERNELS=ON` at
build time — already set in our build-opencl preset).

```bash
llama.cpp/build-opencl/bin/llama-bench.exe \
    -m models/Qwen3-4B-Q4_K_M.gguf \
    -p 512 -n 128 \
    -c 2048 \
    -ngl 99 \
    -r 3 \
    -o md
```

`-ngl 99` offloads all layers to GPU. If it OOMs (unlikely at 2.5 GB
on unified 48 GB LPDDR5X but possible under driver limits), scan
downward. Expected TG is likely lower than CPU per Phase 2's Qwen3
experience (kernel-launch overhead on small per-token ops); PP
should be dramatically higher since prefill is compute-bound on big
matmuls that OpenCL handles well. Phase 1 measured 2674 t/s PP512 on
Qwen3-0.6B Q8_0 — 4B will be proportionally slower (~450–600 t/s
expected) but still far above CPU.

### GPU (Vulkan) — llama.cpp `build-vulkan`

Same Adreno, different backend. llama.cpp's Vulkan path is more
mature than OpenCL in some respects (broader op coverage) but less
Adreno-specific than the OpenCL Adreno-tuned kernels. Worth a
separate row because Vulkan is the cross-vendor future-proof GPU
path.

```bash
llama.cpp/build-vulkan/bin/llama-bench.exe \
    -m models/Qwen3-4B-Q4_K_M.gguf \
    -p 512 -n 128 \
    -c 2048 \
    -ngl 99 \
    -r 3 \
    -o md
```

Device selection: Vulkan on X2E enumerates Adreno as `device 0` by
default. If a CPU device also shows, pass `-dev <adreno_uuid>` or use
`-sm none` to force single-device on the GPU.

## Fallback: NPU via ORT-QNN (chained 4-partition probe)

If `genie-t2t-run` refuses to load the 2.42-compiled binaries on
QAIRT 2.45 (or if we want to double-check the number against a
non-Genie runtime), the existing
`npu_engine/probe_qualcomm_qwen3_4b.py` already validated parts 1 and 2.
Extend it to parts 3 and 4 (wrappers for all four already live next
to the bundle — `oracle_part3.wrapper.onnx`, `oracle_part4.wrapper.onnx`)
and chain them in Python:

```
input_ids -> part1 (embed)
          -> part2 (layers 0..11)   -> part3 (layers 12..23) -> part4 (layers 24..35 + head)
                              ^-- past_kv propagates per layer, host-side stitch
```

Per-step wall: the side-quest measured part 2 at 7.22 ms / 12 layers
(0.60 ms/layer). Projecting linearly: full decode step ≈
`embed (0.04 ms) + 3 × 12-layer parts × 7.22 ms + head overhead` ≈
**~22 ms / step** → **~45 t/s TG** for a standalone Qwen3-4B decode.
PP at ar=128 prefill batch is batched and much faster — expect ~400+
t/s PP. Both numbers are unverified projections from single-partition
measurements; Genie is the shorter path to a verified number.

This fallback matters for two reasons beyond "Genie refuses":
1. Our own spec-decode sidecar speaks ORT-QNN, not Genie. If we ever
   *use* Qwen3-4B as a draft inside a larger pipeline, we need to
   know what we'd actually get through our runtime — not Genie's
   best-case number.
2. Genie is vendor-closed; ORT-QNN exposes lower-level knobs
   (async-exec, custom KV strategies) that future workstreams
   (W2.d tree drafts, W4 async orchestration) need.

Both numbers on the same row in the results table, annotated as
`Genie` vs `ORT-QNN chained`.

## Run discipline

One session ordering that keeps the numbers reproducible:

1. Boot fresh. Plug in. Close background apps. Wait 60 s idle.
2. Warm run of every backend (1 quick invocation each, discarded).
3. Measure in order: NPU → CPU → CPU+KleidiAI → GPU-OpenCL →
   GPU-Vulkan. NPU first because it has the longest session-init
   cost — stacking it after a GPU run means HMX context replays while
   residual thermals are still elevated.
4. Between each measured backend: 30 s idle, check CPU temp is back
   to baseline.
5. Dump raw stdout to `results/qwen3_4b_baseline/<backend>.log` and
   parsed summary to the results doc.

## Command → cell map

In practice the driver does all of this — `scripts/bench_qwen3_4b_all_backends.py`
orchestrates the five backends and emits one CSV row per backend plus a
markdown summary table to stdout.

```bash
# AC (plugged in, ~5 min)
.venv/Scripts/python.exe scripts/bench_qwen3_4b_all_backends.py \
    --power-state ac --tag YYYY-MM-DD_ac

# Battery (unplug first, ~15 min including broken Vulkan)
.venv/Scripts/python.exe scripts/bench_qwen3_4b_all_backends.py \
    --power-state bat --tag YYYY-MM-DD_bat
```

## Output layout (post-hygiene)

Per `docs/repo_hygiene.md` the runner writes:

- `results/csv/qwen3_4b_baseline_<tag>.csv` — permanent measurement
  record. Commit.
- `marked_for_deletion/qwen3_4b_baseline_<tag>/*.log` — raw
  stdout/stderr per backend. Gitignored, stays local for the soak,
  `rm -rf` once the numbers are in the doc and nothing is missing.
- The markdown summary table prints to stdout; paste it into
  `docs/qwen3_4b_baseline_all_backends.md` update log when promoting
  the run.

The prompt scaffolding at `results/qwen3_4b_baseline/pp512_prompt.txt`
+ `pp512_prompt_tokens.txt` is the pinned input (reproducible via
`scripts/gen_pp512_prompt.py`) that every rerun uses.

## Close-out for each rerun

1. Commit the CSV under `results/csv/` + update log row in
   `docs/qwen3_4b_baseline_all_backends.md`.
2. Update the post-mortem block in the results doc only if numbers
   shift materially (>10% on any headline metric) or a new backend
   result changes a workstream priority.
3. Link this doc + the results doc from `docs/roadmap.md` as the
   concrete data behind the "category × backend matrix" deliverable.

## Companion docs

- `docs/qwen3_4b_baseline_all_backends.md` — the results doc.
- `docs/roadmap.md` — the category × backend matrix this feeds into
  (§"Category × backend matrix — what we need to fill in").
- `results/qwen3_4b_genie_w4a16_probe.md` — single-partition NPU
  timing already measured; consistency check for the Genie row.
- `docs/w4a16_investigation_continued.md` — paused pending this
  matrix; resume decision depends on where NPU draft fits.
