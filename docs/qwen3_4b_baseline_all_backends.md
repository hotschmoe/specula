# Qwen3-4B — all-backends baseline matrix

One comparable PP / TG / J/tok number per compute island on the
Zenbook A16 (Snapdragon X2 Elite Extreme, Hexagon v81, Adreno X2-90,
48 GB LPDDR5X @ 228 GB/s unified). Measurement recipe in
`docs/qwen3_4b_baseline_methods.md`. Runner:
`scripts/bench_qwen3_4b_all_backends.py`.

**First run: 2026-04-23** (tag `2026-04-23_{ac,bat}`). Intended cadence:
rerun every ~2-4 weeks so we can track how each backend matures
(llama.cpp commits, Vulkan driver updates, QAIRT releases).

## Models under test

| backend family | artifact | weight footprint |
|---|---|---|
| NPU | `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/*.bin` (4 parts, Genie) | ~w4a16, 4 × ~600 MB |
| CPU / GPU | `models/Qwen3-4B-Q4_K_M.gguf` (unsloth, HF) | 2.38 GiB (Q4_K_M) |

Quant parity is close but not identical (w4a16 vs Q4_K_M mixed
4/6-bit). See methods doc §"Quant parity caveat" before using any
tie-breaker within ~20% of the leader.

## Headline — AC (wall power, idle system)

Commit: `e365e658f` (cpu, vulkan) / `fd6ae4ca1` (opencl) / `cf8b0dbda` (cpu-kleidiai),
QAIRT 2.45.40, Genie 1.17.0, bundle compiled QAIRT 2.42. Context=2048.

| backend | runtime / build | PP (t/s) | TG (t/s) | TG tokens | notes |
|---|---|---:|---:|---:|---|
| **NPU (Genie)**       | genie-t2t-run (QAIRT 2.45, AR128 prefill)   | **1566.23** (AR128) | 23.30 | 3582 | temp=0.8 sampler; ran until ctx-fill |
| NPU (ORT-QNN chained) | our stack, chained 4-part, AR1-only         | 25.76 (AR1)        | **25.78** | 128 | same .bin as Genie; "rolling our own" ceiling |
| CPU                | llama.cpp build-cpu (-t 8 ARM64 NEON)   | 188.30 | **39.50** | 128 | |
| CPU + KleidiAI     | llama.cpp build-cpu-kleidiai (-t 8, i8mm)| 185.78 | 38.51 | 128 | -1.3% PP, -2.5% TG vs plain CPU |
| GPU (OpenCL)       | llama.cpp build-opencl -ngl 99 (Adreno) | 367.38 | 22.92 | 128 | |
| GPU (Vulkan)       | llama.cpp build-vulkan -ngl 99          | 3.91 | 31.43 | 128 | ⚠️ PP broken — see post-mortem |

## Headline — battery (DC, Q4_K_M model, same seed)

J/tok is computed as `mean(DischargeRate_W) × wall_s / (pp_tokens + tg_tokens)`.
Mean W is the average of WMI `DischargeRate` samples polled at 2 s
intervals throughout each backend's wall-clock window — stable within
±5% for all backends except Vulkan (which spent most of its wall time
shader-compiling, not steady-state).

| backend | PP (t/s) | TG (t/s) | mean W | J/tok | J / gen tok | wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| **NPU (Genie)**       | 1598.50 (AR128) | 23.33 | **13.1** | **0.537** | ~0.614 | 168.6 |
| NPU (ORT-QNN chained) | 24.57 (AR1)     | 24.32 | 23.5     | 0.959    | —     | 15.7 |
| CPU                | 191.30 | 38.52 | 25.5 | 0.899 | ~3.96 | 22.5 |
| CPU + KleidiAI     | 180.43 | 37.33 | 32.1 | 1.182 | ~5.92 | 23.6 |
| GPU (OpenCL)       | 355.79 | 18.58 | 44.6 | 2.690 | ~13.0 | 38.6 |
| GPU (Vulkan)       | — | — | 23.3 | — | — | 970 (timeout) |

"J / gen tok" separates out the generation energy by subtracting an
estimate of prefill energy: `(mean_W × wall_s − mean_W × pp_time_s) /
tg_tokens`. NPU runs 3582 gen tokens, everyone else runs 128 — NPU's
decode-throughput efficiency is dramatically larger than its PP row
even suggests.

Battery drain over the full matrix: 70160 → 58369 mWh = 11791 mWh
(~17% of a full charge) to run one pass of all 5 backends including
the 970 s Vulkan timeout. NPU's 168 s run cost 926 mWh; CPU's 22 s run
cost 841 mWh — per-second NPU draws 2× less than CPU under the same
model.

## AC vs battery consistency

Same backend, same test, different power state. Throughput should be
within ~5% at the power envelopes of this laptop; larger deltas are
the AC→thermal-boost vs battery-throttle gap documented in the
roadmap.

| backend        | PP AC | PP BAT | Δ    | TG AC | TG BAT | Δ    |
|---|---:|---:|---:|---:|---:|---:|
| NPU (Genie)    | 1566.2 | 1598.5 | +2.1% | 23.30 | 23.33 | +0.1% |
| CPU            |  188.3 |  191.3 | +1.6% | 39.50 | 38.52 | -2.5% |
| CPU + KleidiAI |  185.8 |  180.4 | -2.9% | 38.51 | 37.33 | -3.1% |
| GPU (OpenCL)   |  367.4 |  355.8 | -3.2% | 22.92 | 18.58 | -19% |
| GPU (Vulkan)   |    3.9 |   — (timeout) |       | 31.43 | — |       |

NPU is the only backend where battery performance ≈ AC performance —
**the silicon runs well under its thermal ceiling on either power
state**. OpenCL TG took the biggest hit going to battery (−19%) — the
Adreno path is power-constrained in ways the other backends aren't.

## Per-backend detail

### NPU (Genie)

```
cmd:    genie-t2t-run --config genie_config.json --prompt_file pp512_prompt.txt --log info
build:  QAIRT 2.45.40.260406 / genie-t2t-run / bundle compiled QAIRT 2.42
AC   : PP 1566.23 t/s  TG 23.30 t/s  (3582 TG tokens, 165.1 s wall)
BAT  : PP 1598.50 t/s  TG 23.33 t/s  (3582 TG tokens, 168.6 s wall, mean 13.1 W, 0.537 J/tok)
notes: Bundle uses temp=0.8 sampling. Gen runs until ctx-fill (4096 − 512 prompt = 3584 tokens avail).
       genie-t2t-run exits with code 1 after the generation because `--profile` JSON
       teardown fails on 2.42-compiled bundles under the 2.45 runtime ("Wrong array type"
       warnings) — the generation itself runs cleanly, so timing is valid. Parser
       tolerates exit=1.
```

### NPU (ORT-QNN chained 4-partition, AR1 only)

```
cmd:    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe npu_engine/bench_qwen3_4b_ortqnn.py --power-state {ac,bat}
runner: npu_engine/bench_qwen3_4b_ortqnn.py (reuses qualcomm_qwen3_4b_oracle.py machinery)
binary: same models/qualcomm-qwen3-4b-ref/*.bin as Genie
AC   : PP-AR1 25.76 t/s  TG-AR1 25.78 t/s  (256 prefill + 128 decode, 14.9 s wall)
BAT  : PP-AR1 24.57 t/s  TG-AR1 24.32 t/s  (15.7 s wall, mean 23.5 W, 0.959 J/tok)
notes: Chained 4 ORT-QNN sessions (one per partition) with host-side
       KV stitch and argmax in Python. ctx=CL512, so prompt+gen capped
       at 511 KV slots; we run 256+128=384 to stay under the cap.
       AR1-only — prefill uses the same single-token graphs as decode,
       which is why PP t/s ≈ TG t/s here. To match Genie's AR128
       batched prefill (1598 t/s) we'd need wrapper ONNXs for the
       prompt_ar128_* graphs and plumbing to feed [1, 128, ...]-shaped
       hidden states through the chain. Deferred workstream; out of
       scope for this baseline.
```

**The interesting comparison is TG + J/tok** (AR1, apples-to-apples vs Genie):

| | TG AC t/s | TG BAT t/s | mean W | J/tok |
|---|---:|---:|---:|---:|
| Genie                | 23.30 | 23.33 | 13.1 | 0.537 |
| ORT-QNN chained AR1  | 25.78 | 24.32 | **23.5** | **0.959** |
| delta                | +11% / +4% | — | +79% | +79% |

**Per-step NPU work is not the bottleneck.** Our chain is actually
~4-11% *faster* per step than Genie's dispatch. But we burn 79% more
power doing it — Python + per-step ONNX session dispatch + KV copy-and-
stitch in numpy keeps the CPU busy between NPU graph calls, where
Genie's C++ runtime sits idle and drops power rails.

**Implication for rolling our own inference engine.** Matching Genie's
throughput is tractable (we already match or beat it). Matching
Genie's *efficiency* is the real engineering challenge — it means
moving the scaffolding out of Python (C++ sidecar), keeping KV cache
in-place across partitions (no host-side copy), and aggressive
low-power idle between NPU calls. ~10 W headroom between our
baseline and Genie's — that's the envelope our W4 async
orchestration work needs to fit under.

### CPU (ARM64 NEON)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -r 3 -t 8
build:  llama.cpp build-cpu @ e365e658f (2026-04-19)
AC   : PP 188.30 t/s  TG 39.50 t/s  (22.1 s wall)
BAT  : PP 191.30 t/s  TG 38.52 t/s  (22.5 s wall, mean 25.5 W, 0.899 J/tok)
```

### CPU + KleidiAI

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -r 3 -t 8
build:  llama.cpp build-cpu-kleidiai @ cf8b0dbda (2026-04-20)
AC   : PP 185.78 t/s  TG 38.51 t/s  (22.7 s wall)
BAT  : PP 180.43 t/s  TG 37.33 t/s  (23.6 s wall, mean 32.1 W, 1.182 J/tok)
```

**KleidiAI is a regression on this model/power state.** -1.3% / -2.5%
t/s on AC, -2.9% / -3.1% on battery, and *uses more power* (25.5 W →
32.1 W, a 26% increase) for slightly less throughput. SME2 is
runtime-fenced by `scripts/patch_kleidiai_detect.py` so the i8mm
ukernels are what's active; on Q4_K_M at 4B scale they are not
pulling their weight here. Separate investigation — documented in
the post-mortem.

### GPU (Adreno / OpenCL)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-opencl @ fd6ae4ca1 (2026-04-20, GGML_OPENCL_USE_ADRENO_KERNELS=ON)
AC   : PP 367.38 t/s  TG 22.92 t/s  (31.5 s wall)
BAT  : PP 355.79 t/s  TG 18.58 t/s  (38.6 s wall, mean 44.6 W, 2.690 J/tok)
```

PP is 2× CPU but 4.3× slower than NPU. TG is worse than CPU at both
power states — consistent with Phase 2's "GPU decode is a regression
on small per-token ops" finding (kernel-launch overhead dominates at
AR=1). Adreno draws the most power of any backend here — 44.6 W mean,
vs 13 W for NPU — and is the only row where battery TG throughput
drops materially (−19%).

### GPU (Vulkan)

```
cmd:    llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-vulkan @ e365e658f (2026-04-19, Vulkan SDK / Adreno ICD)
AC   : PP 3.91 t/s (!!)  TG 31.43 t/s  (539 s wall)
BAT  : TIMED OUT after 600 s (partial; PP never finished in the subprocess timeout)
```

**Vulkan PP is broken.** 3.9 t/s on a model that OpenCL does at 367
t/s on the same GPU — that's a 94× gap within the same silicon. The
wall time (~9 min AC, timed out on BAT) suggests shader recompilation
on every prefill tile. TG at 31 t/s is plausible (between CPU's 39 and
OpenCL's 23) so the decode kernels work, but the prefill path
collapses. Flag for the roadmap, don't use Vulkan for PP-heavy
workloads until investigated.

## Post-mortem

### Which island wins each workload?

**PP (prompt processing).** NPU wins decisively — 1566 t/s vs the
next-best 367 t/s on OpenCL, a **4.3× margin**. Prefill is big-matmul,
bandwidth-friendly work; HMX's batched AR=128 graphs dominate. The
hypothesis going in was "GPU should win PP"; that was wrong on this
silicon with this bundle. Confirmed: when we have a model compiled to
NPU, prefill stays on NPU.

**TG (token generation).** CPU wins at both power states — **39.50
t/s on AC, 38.52 t/s on battery**. NPU is 60% of CPU's throughput
(23 t/s) but at less than half the power. Ranking at AC:
CPU (39.5) > CPU+KleidiAI (38.5) > GPU-Vulkan (31.4) > NPU (23.3) >
GPU-OpenCL (22.9). **GPU via OpenCL is the worst TG path** — kernel-
launch overhead at AR=1 on Adreno is the known issue (Phase 2 saw the
same on Qwen3-8B). TG on OpenCL is a non-option at the X2E + Adreno
combo until per-token kernel dispatch gets fixed.

**J/tok (energy efficiency).** **NPU wins by a massive margin** —
0.537 J/tok vs CPU's 0.899 and OpenCL's 2.690. Per *generated* token
(subtracting prefill energy) the gap widens: NPU ≈0.61 J/gen-tok,
CPU ≈4.0, OpenCL ≈13.0. **NPU is 6.7× more efficient than CPU and 22×
more efficient than OpenCL per generated token.** That's the headline.

### Qualitative UX axis not captured in t/s

NPU at 100% load is **invisible**:
- **Silent.** Zero fan noise, zero coil whine — acoustically identical
  to idle. CPU and GPU both make audible fan+coil noise at sustained
  100%.
- **Doesn't interfere with daily use.** Browsing, screen rendering,
  and input latency stay normal during NPU load. CPU and GPU at 100%
  both noticeably degrade UI responsiveness — scroll lag, input lag,
  visible compositor stutter.

For a UX-sensitive product (opencode sessions, background agentic
loops, "can I use my laptop while the LLM works"), **NPU's value over
CPU is much larger than its 60% TG throughput suggests**. Pegging CPU
at 39 t/s makes the machine unusable; pegging NPU at 23 t/s leaves the
user's workflow untouched. This is the single biggest non-obvious
finding of this run.

### Decisions this matrix unblocks

1. **W1 — NPU/GPU prefill** (`docs/roadmap.md` §W1) promoted to high
   priority. The 4.3× NPU-over-OpenCL PP gap is enormous and was not
   reflected in our prior data. W1.a (GPU prefill of 8B target) was
   gated on expected >10× speedup over CPU-prefill at 164 t/s — the
   OpenCL route here hits 2× CPU, which is below the W1.a gate. But
   **W1.b (NPU prefill of 8B target) just became the obvious play**:
   if 4B prefills at 1566 t/s, 8B should land at ~700-900 t/s
   (half the speed for twice the weights assuming bandwidth-bound),
   still 4-5× CPU's 164 t/s on 8B.
2. **w4a16 investigation continues to matter** (not pause) —
   `docs/w4a16_investigation_continued.md` Phase 5.5.1 stays
   unfrozen. The NPU's dominance in PP + power efficiency confirms
   that investment in w4a16 draft models for spec decode is the
   right bet; the 0.6B draft question is a recipe problem, not a
   silicon problem.
3. **W4.e 3-phase × 3-island matrix** has a clearer shape now. The
   "winning pipeline" given this data: **prefill NPU + draft NPU +
   verify CPU**, because (a) NPU is >4× faster at prefill, (b) NPU
   is cooler/quieter for any sustained draft, (c) CPU has the TG
   edge and that's what verify uses on an 8B target. The matrix
   we were going to fill empirically just became "run the
   winning config and see how close to the sum-of-parts it gets."
4. **OpenCL is the blessed Adreno path on X2E; Vulkan stays off.**
   Qualcomm recommends the OpenCL backend for llama.cpp on X2E.
   Vulkan's PP is broken here (3.9 t/s vs OpenCL's 367), confirming
   we should not use it. Not a workstream — documented stance.
5. **KleidiAI regression is a known-parked item, not a new bug.**
   -3% throughput, +26% power on Q4_K_M at 4B because the CPU has
   SME2 silicon but Windows (or the driver stack) is not letting us
   access it. KleidiAI falls back to NEON-only i8mm kernels which
   underperform plain NEON here. Plan is to revisit for a workaround
   or wait for a driver update to expose SME2 (tracked in roadmap
   backlog as B8). Don't use the cpu-kleidiai build for 4B+ models
   until SME2 lands.

### What to re-measure in 2-4 weeks

- llama.cpp commit will advance (faster kernels, Vulkan fixes).
- QAIRT 2.45.x may get a point release that fixes the
  `--profile` teardown bug we hit on 2.42-compiled bundles.
- Driver: Adreno OpenCL build number was 863.0 at this run; newer
  drivers may flip the PP-dominance conclusion.
- Set up `scripts/bench_qwen3_4b_all_backends.py` as a scheduled job
  (cron or manual) so the matrix fills in over time.

## Artifacts

Layout follows `docs/repo_hygiene.md`:

- **Permanent** (committed, never delete):
  - `results/csv/qwen3_4b_baseline_2026-04-23_ac.csv`
  - `results/csv/qwen3_4b_baseline_2026-04-23_ac_rerun.csv` — OpenCL
    rerun after the JSON parser fix landed (367.38 / 22.92 t/s is the
    canonical AC OpenCL cell; supersedes the parse-failed row in the
    main `_ac.csv`).
  - `results/csv/qwen3_4b_baseline_2026-04-23_bat.csv`
  - `results/qwen3_4b_baseline/pp512_prompt.txt` +
    `pp512_prompt_tokens.txt` — pinned input; reproducible via
    `scripts/gen_pp512_prompt.py`.
  - This doc.
- **Staged for deletion** (gitignored,
  `marked_for_deletion/qwen3_4b_baseline_2026-04-23/`):
  - Per-backend `.log` files (raw llama-bench JSON + Genie per-graph
    timestamp traces) — findings already captured in CSV + this doc.
  - Auto-generated `bench_2026-04-23_*.md` from the runner — redundant
    with this consolidated doc.
- **Runner**: `scripts/bench_qwen3_4b_all_backends.py`. Rerun with:
  ```bash
  .venv/Scripts/python.exe scripts/bench_qwen3_4b_all_backends.py \
      --power-state {ac,bat} --tag YYYY-MM-DD_<state>
  ```
  CSV lands in `results/csv/` automatically; logs go to
  `marked_for_deletion/qwen3_4b_baseline_<tag>/`. Markdown summary
  table prints to stdout — paste it into this doc's Update log when
  promoting a new run.

## Update log

- 2026-04-23: first full run. AC + battery matrices landed. Vulkan
  PP broken; KleidiAI regression; NPU-via-Genie dominates J/tok.
- 2026-04-23 (follow-up, same day): NPU-via-ORT-QNN (chained 4-part,
  AR1) added. TG throughput matches/beats Genie (25.8 vs 23.3 t/s on
  AC), but J/tok is 78% worse (0.96 vs 0.54) because Python + numpy
  KV stitching between NPU calls burns ~10 W the C++ Genie runtime
  doesn't. The 10 W gap is the engineering target for our
  heterogeneous sidecar (W4).
