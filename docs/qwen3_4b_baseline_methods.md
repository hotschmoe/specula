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

| backend | runtime / build                           | PP (t/s) | TG128 (t/s) |
|---|---|---|---|
| NPU (Genie)         | genie-t2t-run (QAIRT 2.45)                  | PP512 — | — |
| NPU (npu_engine)    | our ORT-QNN stack (AR128 swap + AR1 decode) | PP256 — | — |
| CPU                 | llama.cpp build-cpu (ARM64 NEON)            | PP512 — | — |
| CPU+KleidiAI        | llama.cpp build-cpu-kleidiai (NEON + i8mm)  | PP512 — | — |
| GPU (OpenCL)        | llama.cpp build-opencl (Adreno kernels)     | PP512 — | — |
| GPU (Vulkan)        | llama.cpp build-vulkan                      | PP512 — | — |

**PP cell is *not* uniform across backends.** llama.cpp backends and
Genie use PP512 (full 512-token prompt). The `npu_engine` row uses
PP256 because the Qualcomm bundle's CL=512 graphs cap prefill+decode at
511 KV slots — `pp + tg ≤ 511`, so PP=256 + TG=128 is the largest
AR128 configuration that fits. PP rate is throughput
(`tokens / compute_time`), so the 256 vs 512 difference is amortization
of fixed per-call overhead, not a fundamental geometry change. To get
PP512 on `npu_engine` we'd need wrappers for the bundle's `cl1024` (or
larger) AR128 graphs — workstream tracked separately.

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

## NPU via `npu_engine` (our stack — promoted from "fallback")

Originally a probe to verify the .bin parses outside Genie's vendor
runtime. Now the **second NPU row in the headline matrix** in its own
right: as of 2026-04-25 it beats Genie on the same .bin at PP (+27%)
and TG (+17%). Two reasons it earned the promotion:

1. Our spec-decode sidecar speaks ORT-QNN, not Genie. The "what we'd
   get if we use Qwen3-4B as a draft / verifier through *our* runtime"
   number is the load-bearing one for W4 (heterogeneous orchestration);
   Genie is the silicon ceiling for context only.
2. Genie is vendor-closed. ORT-QNN exposes lower-level knobs
   (multi-AR graph routing, IOBinding for zero-copy outputs, async exec)
   that future workstreams (W2.d tree drafts, W4 async orchestration)
   need direct access to.

### Architecture (commits `ac17196`..`3447862`)

```
input_ids -> part1 (embed) -> part2 (layers 0..11)
                           -> part3 (layers 12..23)
                           -> part4 (layers 24..35 + LM head)
                                   ^-- past_kv propagates per layer; host-side numpy stitch
```

Two graph families per partition, both already in the Qualcomm bundle:

- `ar1_cl512_part_*_of_4` — single-token decode (also usable for AR1
  prefill and any AR128 tail not divisible by 128).
- `ar128_cl512_part_*_of_4` — 128-wide batched prefill. Same hardware
  path as Genie's AR128 prefill; only the host-side dispatch differs.

### AR128 swap mode

The HTP context has a hard cap of ~7 live ORT-QNN sessions on this
bundle (`reference_ortqnn_session_limit.md`). Loading 4 AR128 + 4 AR1 = 8
sessions concurrently exhausts memory and the 8th `CreateSession` fails
with QNN error 1002. Workaround is **phase-batched swap**:

1. **Phase A (prefill).** Load 4 AR128 sessions; run prefill in 128-wide
   batches. AR128 path keeps a parallel AR128-shaped KV mirror buffer to
   avoid `np.ascontiguousarray`-copying a slice of the 511-slot master
   on every batched call (~28 MB / call saved).
2. **Phase A.5 (teardown).** Drop AR128 session refs +
   `gc.collect()` to release HTP context before phase B's loads.
3. **Phase B (decode).** Load 4 AR1 sessions; greedy-argmax decode loop
   reads KV from the master buffer.

Total wall on a one-shot 256+128 run is ~41 s, of which ~36 s is session
load/teardown. The compute itself is 4.8 s — apples-to-apples vs a warm
runtime, the swap tax disappears (see sidecar below).

### Routing (vLLM-style request-size threshold)

`--ar128-min-tokens` (default 512) decides whether a given prompt is
worth the AR128 swap. Below the threshold, prefill stays on AR1 — at
prompt sizes < ~559 tokens the 36 s of session-load amortizes to a
worse end-to-end latency than just running AR1 prefill. The bench
script logs the routing decision in one line so it's visible in any
log.

### IOBinding for zero-copy output reuse

Each session's output tensors are pre-allocated once and bound to ORT
via `io_binding.bind_output(buffer_ptr=...)`. Subsequent
`run_with_iobinding` calls reuse the same buffer — eliminating the
per-step output-allocation that vanilla `sess.run()` does on every
call. Inputs use `bind_cpu_input()` per call (zero-copy from numpy);
KV stitch happens *before* the next step reuses the bound output
buffer.

### Sidecar (long-lived process, mode state machine)

`npu_engine/sidecar.py` (commit `68c87f1`) keeps the engine alive
across requests, holding either the AR128 chain or the AR1 chain in
memory and swapping on demand. Net effect: the 36 s cold-start tax is
paid once at process boot, then amortized over every subsequent
request. Phase-batched execution (commit `564d330`) lets the sidecar
process a queue of prefill requests against the AR128 chain before
swapping to AR1 for their decodes — `prefill_all → decode_all` instead
of one-prompt-at-a-time.

### Bench command (used for the matrix)

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    npu_engine/bench_qwen3_4b_ortqnn.py \
    --power-state {ac,bat} \
    --tag YYYY-MM-DD_<state> \
    --ar128-min-tokens 128
```

Defaults: `--pp-tokens 256 --tg-tokens 128`. CL=512 caps total at 511 KV
slots, so the maximum we can drive through the bundle in one prompt is
`pp + tg ≤ 511`. To match the headline matrix's PP=256 + TG=128
geometry, the defaults are sufficient. `--ar128-min-tokens 128` forces
the AR128 swap (default 512 would route below-threshold prompts to
AR1-only; for a baseline measurement we want the AR128 path
exercised).

### Outputs

- `results/csv/qwen3_4b_ortqnn_<tag>.csv` — one row per run.
  Columns include per-phase walls (`pp_ar128_wall_s`, `pp_ar1_wall_s`,
  `tg_wall_s`, `compute_wall_s`, `swap_wall_s`, `total_wall_s`),
  per-call medians, per-partition load profile (`ar128_per_part_s`,
  `ar1_per_part_s`), and battery J/tok where applicable.
- `marked_for_deletion/qwen3_4b_ortqnn_<tag>/stdout.log` — verbose
  per-step trace; useful for root-causing variance, deletable once
  the CSV row is in the doc.

### Comparing to Genie

Apples-to-apples: same .bin, same prompt token IDs, both run AR128
prefill. PP rate is `tokens_processed / compute_time` (not including
session load — Genie pays its own one-time init that we don't
separate out, so neither does this script). TG rate is `gen_tokens /
decode_time` over 128 greedy steps. J/tok in the BAT column uses the
WMI `DischargeRate` sampler (2 s interval) over the **compute-only**
window — swap-mode session loads burn CPU, but their energy is not
load-bearing for steady-state efficiency, so we exclude them.

Genie is still the right reference for **idle power between calls** —
the 2026-04-23 J/tok delta (0.96 vs 0.54 ours) was driven by per-step
Python+numpy keeping the CPU busy at ~23 W between NPU graph calls
where Genie's C++ runtime drops to single-digit W. Whether the AR128
swap mode improves or worsens that delta is the open question for the
2026-04-25 BAT run.

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
