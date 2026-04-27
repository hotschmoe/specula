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

Commit: `e365e658f` (cpu, vulkan) / `fd6ae4ca1` (opencl) / `cf8b0dbda`
(cpu-kleidiai). NPU (ORT-QNN) row refreshed 2026-04-25 with our
`npu_engine` (AR128 swap-mode prefill + AR1 decode + IOBinding); CSV
`results/csv/qwen3_4b_ortqnn_2026-04-25_ac.csv`. **GPU rows refreshed
2026-04-26** at llama.cpp `f53577432` against pure-Q4_0 model
(`Qwen3-4B-Q4_0.gguf` from unsloth) per upstream Adreno guidance —
Q4_K is officially unsupported on the OpenCL backend. CSV
`results/csv/qwen3_4b_baseline_2026-04-26_ac_knob_sweep.csv`.
QAIRT 2.45.40, Genie 1.17.0, bundle compiled QAIRT 2.42. Context=2048
(llama.cpp), CL=512 (NPU bundle).

| backend | runtime / build | PP (t/s) | TG (t/s) | TG tokens | notes |
|---|---|---:|---:|---:|---|
| NPU (Genie)           | genie-t2t-run (QAIRT 2.45, AR128 prefill)   | 1566.23 (AR128) | 23.30 | 3582 | temp=0.8 sampler; ran until ctx-fill |
| **NPU (ORT-QNN, npu_engine)** | our stack, chained 4-part, AR128 swap + AR1 decode | **1985.46** (AR128) | **27.25** | 128 | 256-tok prefill (CL512 cap); beats Genie on PP (+27%) and TG (+17%) |
| CPU                | llama.cpp build-cpu (-t 8 ARM64 NEON), Q4_K_M | 188.30 | **39.50** | 128 | |
| CPU + KleidiAI     | llama.cpp build-cpu-kleidiai (-t 8, i8mm), Q4_K_M | 185.78 | 38.51 | 128 | -1.3% PP, -2.5% TG vs plain CPU |
| **GPU (OpenCL)**   | llama.cpp build-opencl -ngl 99 (Adreno), **Q4_0** | **569.12 ± 1.89** | 26.22 ± 0.04 | 128 | refreshed 2026-04-26; +55% PP / +14% TG vs old Q4_K_M (367/23) — Q4_0 is the official Adreno path |
| **GPU (Vulkan)**   | llama.cpp build-vulkan -ngl 99, **Q4_0**, env `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` | **115.04 ± 0.12** | **38.51 ± 0.11** | 128 | refreshed 2026-04-26; PP fixed (+29× vs broken default 3.91); TG ties CPU and beats OpenCL by 47% |

## Headline — battery (DC, same seed)

J/tok is computed as `mean(DischargeRate_W) × wall_s / (pp_tokens + tg_tokens)`.
Mean W is the average of WMI `DischargeRate` samples polled at 1-2 s
intervals throughout each backend's wall-clock window — stable within
±5% for all backends. **GPU rows refreshed 2026-04-26** at llama.cpp
`f53577432`: Q4_0 model for the canonical OpenCL/Vulkan rows, with
Vulkan running `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1`
per the AC sweep. Old Q4_K_M rows kept as comparators. CSV
`results/csv/qwen3_4b_gpu_knobs_2026-04-26_bat.csv`.

| backend | model | PP (t/s) | TG (t/s) | mean W | J/tok | J / gen tok | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| **NPU (Genie)**                       | w4a16   | 1598.50 (AR128) | 23.33 | **13.1** | **0.537** | ~0.614 | 168.6 |
| NPU (ORT-QNN, npu_engine, AR128 swap) | w4a16   | 2118.83 (AR128) | 25.26 | 9.4† | 1.17† / 0.128‡ | — | 47.9 (5.2 compute) |
| NPU (ORT-QNN chained, AR1, baseline)  | w4a16   |   24.57 (AR1)   | 24.32 | 23.5 | 0.959 | — | 15.7 |
| CPU                                   | Q4_K_M  |  191.30 | 38.52 | 25.5 | 0.899 | ~3.96 | 22.5 |
| CPU + KleidiAI                        | Q4_K_M  |  180.43 | 37.33 | 32.1 | 1.182 | ~5.92 | 23.6 |
| **GPU (OpenCL, refreshed)**           | **Q4_0**  | **539.46** | 24.94 | 27.52 | **1.291** | ~1.10§ | 30.0 |
| **GPU (Vulkan, F16-off+host-mem)**    | **Q4_0**  | 114.57 | **34.69** | 28.25 | 1.425 | **~0.81§** | 32.3 |
| GPU (OpenCL, refreshed)               | Q4_K_M  |  350.18 | 22.21 | 28.43 | 1.489 | — | 33.5 |
| GPU (Vulkan, F16-off+host-mem)        | Q4_K_M  |   83.55 | 31.57 | 38.14 | 2.356 | — | 39.5 |
| GPU (OpenCL, **2026-04-23 baseline, superseded**) | Q4_K_M | 355.79 | 18.58 | 44.6 | 2.690 | ~13.0 | 38.6 |
| GPU (Vulkan, **2026-04-23 default, broken**) | Q4_K_M |    — |    — | 23.3 |    — |    — | 970 (timeout) |

† Mean W (9.4) on the npu_engine row is averaged over the full 47.9 s
wall, ~36 s of which is AR128/AR1 session-load I/O at low CPU draw.
The 1.17 J/tok number amortizes the *full one-shot* energy
(`mean_W × wall_s / tokens`) over all 384 tokens — comparable to "what
does it cost to ask one prompt from a cold process." For a long-lived
sidecar that pays the load tax once at boot and amortizes over many
requests, the steady-state J/tok is in between this number and
Genie's 0.537 — needs separate measurement with the sidecar warm.

‡ The script's "compute-only" J/tok (0.128) multiplies the *whole-run*
mean W by *compute-only* wall — silently assumes compute ran at the
global average, which is wrong when ~76% of the wall is mostly-idle
session loading. Number is in the CSV for completeness; do not cite
without the caveat. To get an honest steady-state J/tok we'd need a
sampler that brackets only the compute window (or a sidecar warm-keep
that erases the imbalance) — open methodology TODO.

"J / gen tok" separates out the generation energy by subtracting an
estimate of prefill energy: `(mean_W × wall_s − mean_W × pp_time_s) /
tg_tokens`. NPU runs 3582 gen tokens, everyone else runs 128 — NPU's
decode-throughput efficiency is dramatically larger than its PP row
even suggests.

§ "J / gen tok" for the 2026-04-26 GPU refresh rows is computed
`mean_W × (128 / TG_t/s) / 128` ⇒ `mean_W / TG_t/s`. **Vulkan-Q4_0 at
0.81 J/gen-tok is the best non-NPU per-token efficiency in the matrix**
— better than CPU's ~3.96 by a factor of 4.9× and OpenCL-Q4_0's 1.10
by 26%, despite Vulkan's mean W being slightly higher than OpenCL's.
The win comes from Vulkan's 39% faster TG (34.69 vs 24.94) — energy
× time integrates more cheaply when each token is faster. Caveat: this
ignores the prefill window's contribution to the average. For
TG-heavy workloads (long output, agentic decode loops) the headline
J/gen-tok number is honest; for one-shot short prompts the J/tok
column is the right axis.

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

| backend        | model / knobs | PP AC | PP BAT | Δ    | TG AC | TG BAT | Δ    |
|---|---|---:|---:|---:|---:|---:|---:|
| NPU (Genie)                  | w4a16 | 1566.2 | 1598.5 | +2.1% | 23.30 | 23.33 | +0.1% |
| NPU (ORT-QNN, ours)          | w4a16 | 1985.5 | 2118.8 | +6.7%* | 27.25 | 25.26 | -7.3% |
| CPU                          | Q4_K_M | 188.3 |  191.3 | +1.6% | 39.50 | 38.52 | -2.5% |
| CPU + KleidiAI               | Q4_K_M | 185.8 |  180.4 | -2.9% | 38.51 | 37.33 | -3.1% |
| **GPU (OpenCL, refreshed)**  | **Q4_0** (default) | **569.12** | **539.46** | **-5.2%** | **26.22** | **24.94** | **-4.9%** |
| **GPU (Vulkan, refreshed)**  | **Q4_0** + `DISABLE_F16+PREFER_HOST` | **115.04** | **114.57** | **-0.4%** | **38.51** | **34.69** | **-9.9%** |
| GPU (OpenCL, refreshed)      | Q4_K_M (default) |  378.31 |  350.18 |  -7.4% | 23.43 | 22.21 |  -5.2% |
| GPU (Vulkan, refreshed)      | Q4_K_M + `DISABLE_F16+PREFER_HOST` |   84.19 |   83.55 |  -0.8% | 33.67 | 31.57 |  -6.2% |
| GPU (OpenCL, 2026-04-23, superseded) | Q4_K_M | 367.4 |  355.8 |  -3.2% | 22.92 | 18.58 |  -19% |
| GPU (Vulkan, 2026-04-23 default, broken) | Q4_K_M |    3.9 |   — (timeout) | | 31.43 | — | |

NPU is the only backend where battery performance ≈ AC performance —
**the silicon runs well under its thermal ceiling on either power
state**. OpenCL TG took the biggest hit going to battery (−19%) — the
Adreno path is power-constrained in ways the other backends aren't.

\* The npu_engine PP +6.7% on BAT-vs-AC is real but driven by per-call
variance on a tiny sample (only 2 AR128 calls per run, median 60 ms vs
64 ms). The AR128 in-call rate hits ~2000-2270 t/s in both runs —
within hardware-ceiling noise. Headline for cross-backend ranking is
"on the order of 2000 t/s on either power state." The TG drop on BAT
(−7.3%) is the real signal: AR1 decode steps are 39.3 ms BAT vs 37.6
ms AC, consistent with mild power-throttling under DC.

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

### NPU (ORT-QNN chained 4-partition, AR128 swap + AR1 decode)

```
cmd:    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe npu_engine/bench_qwen3_4b_ortqnn.py \
            --power-state ac --tag 2026-04-25_ac --ar128-min-tokens 128
runner: npu_engine/bench_qwen3_4b_ortqnn.py (reuses qualcomm_qwen3_4b_oracle.py machinery)
binary: same models/qualcomm-qwen3-4b-ref/*.bin as Genie
AC   : PP 1985.46 t/s (AR128, 2 calls × 128)  TG 27.25 t/s (AR1, 128 steps)
       compute-only wall 4.8 s; total wall incl. swap 40.9 s
       per-call AR128 latency median 64 ms (1985 t/s in-call); AR1 step median 37.6 ms
       per-partition load (s): AR128 [1.6, 2.4, 4.5, 6.4]; AR1 [1.6, 2.3, 4.4, 6.4]
BAT  : PP 2118.83 t/s  TG 25.26 t/s
       compute-only wall 5.2 s; total wall incl. swap 47.9 s
       mean W 9.4 over full wall; J/tok 1.17 (one-shot, including 42.7 s
       of cold-start session swap); see methodology footnote on the BAT
       headline table for why the script's 0.128 J/tok number is wrong.
       per-call AR128 latency median 60 ms (2120 t/s in-call); AR1 step median 39.3 ms
       per-partition load (s): AR128 [1.8, 2.8, 5.3, 8.1]; AR1 [1.9, 2.8, 5.3, 8.2]
       Sessions load ~20% slower on BAT than AC (likely cold disk cache
       between the two runs, not power-state-driven).
notes: Chained 4 ORT-QNN sessions per phase. **Phase A** loads the 4
       AR128 graphs (`ar128_cl512_*_of_4`) and runs prefill in 128-wide
       batches; **phase A.5** tears them down to free HTP context; **phase B**
       loads the 4 AR1 graphs (`ar1_cl512_*_of_4`) for decode. The swap
       is necessary because the bundle exhausts HTP memory at ~7 live
       sessions (B7 cap; see `reference_ortqnn_session_limit.md`).
       IOBinding pre-allocates output buffers once per session, eliminating
       per-step output-tensor allocation. KV stitch is host-side numpy
       copy from the per-step output buffers into a persistent KVStore;
       AR128 path keeps a parallel AR128-shaped mirror buffer to avoid
       slicing the 511-slot master on every batched call.
       ctx=CL512, so prompt+gen capped at 511 KV slots; defaults are
       PP=256 + TG=128 (AR128 path uses 2 prefill calls). Larger prompts
       are valid up to (511 − tg) — the script auto-falls-back to AR1 for
       any tail tokens not divisible by 128, and an `--ar128-min-tokens`
       threshold (default 512) makes routing vLLM-style: small prompts
       skip the swap because the 36 s of session load dominates end-to-end
       latency below ~559 tokens.
```

**Headline: we beat Genie on both PP and TG, on the same .bin.**

| | PP t/s (AC) | TG AC t/s | TG step ms |
|---|---:|---:|---:|
| Genie                              | 1566.23 (PP512) | 23.30 | ~43 |
| ORT-QNN, npu_engine (AR128 swap)   | **1985.46** (PP256) | **27.25** | 37.6 |
| delta                              | **+27%**       | **+17%** | −13% |

**Caveats on the comparison.** Our PP cell is over 256 tokens (2 AR128
calls), Genie's is over 512 — same in-call hardware throughput, but ours
amortizes the per-call overhead differently. Both numbers report the
same `tokens / compute_time` definition; the AR128 in-call rate is
~1985-2039 t/s (close to the per-graph hardware ceiling). PP cell does
*not* include session load — see total wall below.

**Total wall (for honest end-to-end latency).** Compute is 4.8 s, but
the swap-mode session loads cost 36 s (AR128 load 14.9 + teardown 6.3
+ AR1 load 14.8). For one-shot benches this is the dominant cost; for a
long-lived sidecar (warm sessions) it's amortized to zero. The current
sidecar (`npu_engine/sidecar.py`) keeps the engine alive across requests,
so the 36 s only hits at process start.

**Per-partition load profile.** AR128 [1.6, 2.4, 4.5, 6.4]s, AR1
[1.6, 2.3, 4.4, 6.4]s — load time is *not* I/O-bound: part 1 (742 MB)
loads in 1.6 s = 457 MB/s, part 4 (1021 MB) takes 6.4 s = 160 MB/s.
HTP context init scales with partition depth (LM head + later layers),
not raw .bin size. Implication: a sidecar with warm sessions sidesteps
this entirely; nothing else makes the 14.9 s cold-start go away.

**J/tok under swap mode (BAT result, with caveats).** Per-prompt
amortized: **1.17 J/tok** including the 42.7 s of session swap — *worse*
than Genie's 0.537 because the cold-start tax dominates a one-shot
bench (76% of the wall-clock window). Per-prompt amortized over only
the active compute is harder to measure cleanly: the WMI sampler
averages 9.4 W over the full 47.9 s wall, but most of that is
mostly-idle session-load I/O. Multiplying that average by just the
5.2 s compute window (the script's `0.128 J/tok` number) understates
real per-token energy because compute-time CPU draw is higher than the
load-window draw. **The right experiment is a sidecar warm-keep run**:
load sessions once, run many prompts, measure W across only the
steady-state decode window. The current bench can't separate those
windows; the sidecar (`npu_engine/sidecar.py`) is the test vehicle —
extending it with a per-window sampler is the next step. Until then:
**1.17 J/tok is the conservative ceiling**, **Genie's 0.537 is the
floor** for this hardware on this bundle, and our true steady-state
J/tok with warm sessions sits somewhere in between.

A separate note on Watts: the BAT-run mean of 9.4 W is *lower* than
Genie's 13.1 W because the average is dragged down by session-load
idle time, not because we're more efficient. The prior AR1-only
2026-04-23 BAT run (no session swap, all wall is steady-state Python +
NPU dispatch) measured 23.5 W mean — that is the apples-to-apples
floor for our chain's steady-state draw, and is the right number to
compare against Genie's 13.1 W (gap = ~10 W, unchanged from the prior
analysis).

**Implication for the inference engine workstream.** The headline
inverts the 2026-04-23 reading: matching Genie's *throughput* is now
*beaten* — PP +27% AC / +33% BAT, TG +17% AC / +8% BAT — on the same
silicon and same .bin, using ORT-QNN + Python + numpy. The remaining
gaps are **(a)** the 36-43 s swap-mode session-load tax (gone with
sidecar warm-keep), and **(b)** the steady-state J/tok delta vs Genie,
which we *cannot cleanly measure* under one-shot swap mode because the
sampler averages compute-time and load-time draws together. The
2026-04-23 AR1-only baseline (no swap) measured 23.5 W steady-state vs
Genie's 13.1 W → ~10 W gap, and there's no reason to believe AR128
swap fundamentally changes that steady-state delta. Engineering
targets shift from "catch up to Genie" to **two specific items**:
**(1)** stand the sidecar up as the long-lived measurement vehicle so
we can sample only the active-compute window (closes the methodology
gap on J/tok); **(2)** close the steady-state ~10 W delta vs Genie via
C++ rewrite of the per-step Python+numpy hot path or moving KV stitch
into a pinned-buffer arrangement that lets the CPU drop into
low-power idle between NPU calls (W4 async orchestration is the
right home for this).

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

**Canonical AC numbers (Q4_0, 2026-04-26):**

```
cmd:    llama-bench -m Qwen3-4B-Q4_0.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-opencl @ f53577432 (2026-04-26, GGML_OPENCL_USE_ADRENO_KERNELS=ON,
        GGML_OPENCL_EMBED_KERNELS=ON), Adreno OpenCL 3.0 driver build 863.0
AC   : PP 569.12 ± 1.89 t/s   TG 26.22 ± 0.04 t/s   (Q4_0 pure)
```

**Quant matters more than any runtime knob** (2026-04-26 finding).
Upstream's Adreno OpenCL backend has optimized kernels for Q4_0,
Q8_0, MXFP4 only — Q4_K is officially unsupported (see
`llama.cpp/docs/backend/OPENCL.md` §DataType Supports). On Q4_K_M the
backend dispatches a non-optimized fallback for the q4_K tensors; on
Q4_0 it hits the fast path. Numbers from a knob sweep at
llama.cpp `f53577432`, `-r 3`, AC, same Adreno X2-90 silicon:

| model | knob | PP512 t/s | TG128 t/s |
|---|---|---:|---:|
| Q4_K_M (old baseline) | default | 367.38 | 22.92 |
| Q4_K_M (rebuild only) | default | 378.31 ± 1.61 | 23.43 ± 0.02 |
| **Q4_0** | default | **569.12 ± 1.89** | **26.22 ± 0.04** |
| Q4_K_M | `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1` | 367.51 ± 0.93 | 23.05 ± 0.03 |
| Q4_0 | `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1` | 528.14 ± 1.63 | 25.85 ± 0.11 |

`GGML_OPENCL_ADRENO_USE_LARGE_BUFFER` reads as a small regression at
PP=512 with `-r 3` confidence intervals (the +2.8% it showed at
`-p 128 -r 1` was within run-to-run noise). All other env vars
(`GGML_OPENCL_DISABLE_FUSION`, `--ubatch-size {256,1024,2048}`) are
flat within noise. **Default settings on a Q4_0 model is the best
config; runtime knobs add nothing.**

**BAT (Q4_0, 2026-04-26):** PP 539.46 / TG 24.94 / mean 27.52 W /
1.291 J/tok / 30.0 s wall. AC→BAT delta is now -5.2% PP / -4.9% TG —
much tighter than the 2026-04-23 OpenCL Q4_K_M reading (-19% TG). The
mean W also dropped substantially: **44.6 W → 27.52 W** on the new
build, even on the same Q4_K_M model (28.43 W in the comparator row).
Most likely upstream OpenCL-kernel improvements between
`fd6ae4ca1`→`f53577432` reduced GPU active duty cycle for the same
work; secondarily, the old reading may have included transient driver
state we don't see in the new run.

PP at 569 (AC) / 539 (BAT) t/s is now **2.9× CPU** (188) and
**2.8-3.7× slower than NPU** (1985 AC / 2118 BAT) — the gap closes
versus the old 4.3×. TG (26.22 / 24.94) is still worse than CPU's
39.5 — kernel-launch overhead dominates at AR=1 on Adreno — but is
markedly better than the old Q4_K_M number (24.94 vs 18.58 BAT,
+34%) and J/tok improves correspondingly (1.291 vs 2.690, **-52%**).
OpenCL is no longer the most battery-throttled backend.

### GPU (Vulkan)

**Canonical AC numbers (Q4_0 + env knobs, 2026-04-26):**

```
cmd:    GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1 \
            llama-bench -m Qwen3-4B-Q4_0.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-vulkan @ f53577432 (2026-04-26, Vulkan SDK / Adreno ICD)
device: Qualcomm(R) Adreno(TM) X2-90 GPU (Adreno Vulkan Driver)
        uma=1, fp16=1 (force-disabled), bf16=0, warp=64, int dot=1, matrix cores=KHR_coopmat
AC   : PP 115.04 ± 0.12 t/s   TG 38.51 ± 0.11 t/s   (Q4_0 pure)
```

**The 2026-04-23 "Vulkan PP is broken (3.91 t/s)" finding has been
resolved as a runtime configuration issue, not an upstream bug.**
Two independent fixes compose:

1. **`GGML_VK_DISABLE_F16=1` is the breakthrough.** On Adreno's
   Vulkan ICD (driver build 863.0), the FP16 matmul codepath silently
   falls into a slow scalar fallback for Q4_K and to a lesser degree
   Q4_0 — disabling FP16 forces the FP32 path which the driver
   handles cleanly. Effect on Q4_K_M @ PP=512: **3.91 → 84.19 t/s
   (+21.5×)**. This is independent of `GGML_VK_DISABLE_COOPMAT` —
   the issue is in the FP16 dispatch logic, not the matrix-core path.
2. **Pure Q4_0 model adds another tier.** Same fix on Q4_0 instead of
   Q4_K_M: **84.19 → 115.04 t/s (+37%)**. The combined improvement
   from the original broken default is **+29.4× PP**.
3. **`GGML_VK_PREFER_HOST_MEMORY=1`** adds a small but consistent
   gain in TG (Adreno reports `uma=1`; biasing allocations toward
   host-visible memory avoids redundant copies). Effect: ~+1-2% PP,
   ~+8% TG.
4. **Counter-knob (don't do this):** `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`
   is a 3.2× *regression* (PP 6.17 → 1.93 at PP=128). The integer
   dot product path is critical for Q4 decode and works correctly.

Knob sweep at `f53577432`, AC, PP=128 (cheap to scan), and PP=512
(authoritative `-r 3`):

| knobs | quant | PP128 t/s | TG32 t/s | PP512 t/s | TG128 t/s |
|---|---|---:|---:|---:|---:|
| (default) | Q4_K_M | 6.17 | 31.30 | 3.91 (old) | 31.43 (old) |
| `DISABLE_COOPMAT=1` | Q4_K_M | 6.16 | 31.35 | — | — |
| `DISABLE_COOPMAT2=1` | Q4_K_M | 6.15 | 31.95 | — | — |
| `DISABLE_FUSION=1` | Q4_K_M | 6.12 | 33.14 | — | — |
| `DISABLE_GRAPH_OPTIMIZE=1` | Q4_K_M | 6.16 | 32.20 | — | — |
| `FORCE_MMVQ=1` | Q4_K_M | 6.23 | 32.19 | — | — |
| `PREFER_HOST_MEMORY=1` | Q4_K_M | 6.25 | 33.09 | — | — |
| `DISABLE_INTEGER_DOT_PRODUCT=1` | Q4_K_M | **1.93** | 29.69 | — | — |
| **`DISABLE_F16=1`** | Q4_K_M | **79.31** | 31.14 | 82.88 ± — | 31.15 ± — |
| `DISABLE_F16=1 + DISABLE_FUSION=1` | Q4_K_M | — | — | 83.03 | 33.30 |
| `DISABLE_F16=1 + DISABLE_BFLOAT16=1` | Q4_K_M | — | — | 82.98 | 31.67 |
| `DISABLE_F16=1 + DISABLE_COOPMAT=1` | Q4_K_M | — | — | 82.81 | 31.64 |
| **`DISABLE_F16=1 + PREFER_HOST=1`** | **Q4_K_M** | — | — | **84.19 ± 0.22** | **33.67 ± 0.03** |
| **`DISABLE_F16=1 + PREFER_HOST=1`** | **Q4_0** | — | — | **115.04 ± 0.12** | **38.51 ± 0.11** |

**Implication for the all-backends ranking.** On Q4_0, Vulkan TG
(38.5 AC / 34.7 BAT) ≈ CPU TG (39.5 / 38.5) within a few percent, and
**beats OpenCL TG (26.2 / 24.9) by 47% AC / 39% BAT**. Vulkan PP (115)
is still 5× slower than OpenCL PP (569), so OpenCL stays the GPU PP
path of choice; but for TG-heavy workloads (chat, agentic-decode),
Vulkan is now a viable GPU path that wasn't before.

**BAT (Q4_0 + knobs, 2026-04-26):** PP 114.57 / TG 34.69 /
mean 28.25 W / 1.425 J/tok / 32.3 s wall. AC→BAT PP delta is **-0.4%**
(within noise — Vulkan is the only non-NPU backend that doesn't
degrade meaningfully on battery). TG drops -9.9% — the largest TG
hit in the matrix on a non-NPU backend. **J/gen-tok = 0.81** (mean_W
× TG_time / 128, see headline footnote §) — best non-NPU per-token
energy in the matrix; beats CPU's ~3.96 by 4.9× and OpenCL-Q4_0's
1.10 by 26%, despite mean W being slightly higher than OpenCL's,
because Vulkan TG is 39% faster so the integration window is shorter.
The old "Vulkan timed out, mean 23 W during shader compile" cell is
no longer representative.

**Why this wasn't found before.** The 2026-04-23 hypothesis was
"shader recompilation on every prefill tile" based on the wall-time
shape (9 min AC). Actual root cause is per-call FP16 codepath
fallback inside the Adreno Vulkan ICD — wall time scales with
prefill *length* because every per-token matmul takes the slow path,
not because each ubatch triggers a compile. The 2026-04-26 sweep
caught it because we tested `GGML_VK_DISABLE_F16=1` and saw 6.17 →
79.31 t/s at PP=128, which is way outside any "rebuild changed
something" envelope.

## Concurrency = 4 (agentic workload)

Same SoC, same Qwen3-4B, but each backend runs **multiple simultaneous
decode streams**. Tests aggregate throughput when several agentic
clients hit the model at once. CPU/KleidiAI/OpenCL data via
`scripts/bench_concurrency4_all_backends.py` driving `llama-batched-bench
-np 4 -npp 512 -ntg 128 -npl 4`. NPU data via
`npu_engine/bench_concurrency4_npu_ortqnn.py` (spawns N independent
`bench_qwen3_4b_ortqnn.py` processes, each with its own 4 ORT-QNN
sessions; QNN HTP `weight_sharing_enabled=true` shares weight pages
across contexts so memory cost stays bounded).

### Headline — AC

GPU rows refreshed 2026-04-26 with the canonical knob+quant combos
from the single-stream sweep. **The headline reorders: Vulkan-Q4_0
with the F16-off + host-mem knobs is now the aggregate-TG champion
at N=4** — beats CPU by 25%, beats the NPU sidecar by 3.8×.

| backend | N | S_TG agg (t/s) | per-stream user rate | step median (ms) | wall (s) | notes |
|---|---:|---:|---:|---:|---:|---|
| **GPU (Vulkan, Q4_0, F16-off+host-mem)** | **4** | **102.33** | **25.58** | — | 25.5 | refreshed 2026-04-26; coopmat tile fits 4-way batched decode much better than AR=1; **2.66× scaling vs N=1** |
| CPU                              | 4 | 82.01 | 20.50 | — | 24.4 | clean run; all 4 streams completed |
| CPU + KleidiAI                   | 4 | 79.73 | 19.93 | — | 25.5 | small regression vs plain CPU at N=4 |
| GPU (Vulkan, Q4_K_M, F16-off+host-mem) | 4 | 79.33 | 19.83 | — | 33.2 | refreshed 2026-04-26; ties CPU but Q4_0 dominates |
| **NPU (sidecar single-process)** | **4** | **26.95** | 6.74 | 38.0 | 19.0 (decode only) | clean run; aggregate flat = NPU's single-stream rate divided across streams |
| NPU (sidecar single-process)     | 8 | 26.75 | 3.34 | 38.0 | 38.3 (decode only) | aggregate still ~27 t/s; NPU concurrency is "free" up to the hardware ceiling |
| GPU (OpenCL, Q4_0)               | 4 | 19.95 | 4.99 | — | 41.8 | refreshed 2026-04-26 (Q4_0 default); +27% vs Q4_K_M but still 5× behind Vulkan |
| GPU (OpenCL, Q4_K_M, refreshed)  | 4 | 15.61 | 3.90 | — | 50.9 | re-baselined 2026-04-26; matches the 2026-04-23 reading (15.68) within noise |
| GPU (OpenCL, Q4_K_M, **2026-04-23 baseline**) | 4 | 15.68 | 3.92 | — | 50.0 | **superseded** — Q4_0 + Vulkan are the canonical GPU concurrency rows now |
| NPU (subprocess fan-out)         | 2 | 30.59 | 15.30 | ~65 | 44.3 | 1.12× "scaling" is Python-overhead overlap across processes, not NPU work parallelism |
| NPU (subprocess fan-out)         | 4 | _crashes_ | — | — | 62.6 | 2 of 4 streams die with QNN 1003; 16 sessions exceeds HTP scheduler resources |

CSVs: `concurrency4_qwen3_4b_2026-04-25_ac.csv` (CPU/KleidiAI/OpenCL,
2026-04-25 baseline), `concurrency4_gpu_knobs_2026-04-26_ac.csv` (the
2026-04-26 GPU refresh: Vulkan Q4_0+knobs is the new canonical),
`qwen3_4b_ortqnn_npuconc{2,4}_stream*_2026-04-25*_ac.csv` (NPU per-stream).

### Concurrency scaling factor (TG agg / TG single-stream)

Single-stream TG baselines (2026-04-26 canonical for GPU rows):
CPU 39.50 (Q4_K_M), KleidiAI 38.51, OpenCL-Q4_0 26.22,
**Vulkan-Q4_0 38.51** (with F16-off + host-mem), NPU (npu_engine) 27.25.

| backend | scaling factor | per-stream latency hit |
|---|---:|---:|
| **GPU (Vulkan, Q4_0, knobs)** | **2.66× (best in class)** | **-34% per stream (38.51 → 25.58)** |
| GPU (Vulkan, Q4_K_M, knobs)  | 2.36× | -41% per stream (33.67 → 19.83) |
| CPU                          | 2.08× | -48% (39.5 → 20.5 t/s) |
| CPU + KleidiAI               | 2.07× | -48% |
| GPU (OpenCL, Q4_0)           | 0.76× (still worse than N=1) | -81% per stream |
| GPU (OpenCL, Q4_K_M, refreshed) | 0.67× | -83% |
| NPU (subprocess, N=2)        | 1.12× (artifact — see detail) | -44% (27.25 → 15.30) |
| NPU (subprocess, N=4)        | _crashes_ | n/a |
| **NPU (sidecar, N=4)**       | 0.99× (NPU is one device, no parallelism gain) | -75% per stream (27.25 → 6.74) |
| NPU (sidecar, N=8)           | 0.98× | -88% per stream |

**The 2026-04-23 ranking inverts: Vulkan now leads, CPU second, NPU
third.** Three structural facts:
- **CPU has 12 ARM cores → real parallelism →** 2.08× aggregate at
  N=4, but a per-core ceiling capped by L1/L2 bandwidth. Adding
  more streams can't push past ~82 t/s on this SoC.
- **Adreno X2-90 is one GPU but its `KHR_coopmat` matrix cores like
  bigger tiles.** Single-stream AR=1 decode under-utilizes them
  (matmul shape `[1, hidden] × [hidden, vocab]` is too tall-skinny);
  N=4 batched decode amortizes the same matmul over 4 stream rows
  (`[4, hidden]`), which fits the coopmat tile much better.
  Result: **2.66× aggregate scaling** with no per-token latency
  penalty (per-stream wall ≈ N=1 wall — see CSV).
- **NPU is one Hexagon engine with no internal parallelism →**
  aggregate flat at the single-stream ceiling regardless of N.

**At N=4, the new aggregate ranking is Vulkan (102.3) > CPU (82.0) >
Vulkan-Q4_K_M (79.3) ≈ KleidiAI (79.7) > NPU sidecar (27.0) >
OpenCL-Q4_0 (20.0).** For pure-throughput multi-agent serving on
Q4_0, **GPU-Vulkan wins by 25% over CPU**. For *energy-per-token*
multi-agent serving (NPU at ~13 W vs CPU at ~25 W vs Vulkan at ~28
W), NPU's flat aggregate at half the power is still the right
tradeoff if your latency budget tolerates 1/N per-stream rate;
otherwise Vulkan is the new default.

### NPU concurrency detail — two architectures, very different results

**Subprocess-fan-out driver (`bench_concurrency4_npu_ortqnn.py`).**
N independent OS processes, each loading its own 4 ORT-QNN sessions.
Three independent attempts (v1, v2, v3 — the v3 done after the
IOBinding refactor in commit `ac17196`) all hit the same failure mode
at N=4: load + warmup complete fine, then early in the prefill loop
2 of 4 streams die with `QNN graph execute error. Error code: 1003`
while the other 2 finish. **Root cause: 4 streams × 4 partitions = 16
simultaneous QNN sessions on one Hexagon engine** — well past the
empirical single-process ~7-session ceiling
(`reference_ortqnn_session_limit.md`). `weight_sharing_enabled=true`
only dedupes within one QNN backend instance; subprocesses are
independent backends, so memory cost is 4×. The IOBinding refactor
**did not help** — the binding constraint is HTP scheduler / VTCM
allocation at execute time, not host-side ORT dispatch. The 1.12×
aggregate "scaling" the v3 N=2 run measured (30.59 vs single-stream
27.25) is *Python-overhead overlap across processes*, not real NPU
concurrency: while process A's CPU does numpy KV-stitch for one
stream, process B's NPU call runs another stream. The NPU itself is
sequential.

**Sidecar single-process bench (`bench_concurrency_sidecar.py`,
2026-04-25 architectural fix).** ONE process, ONE chain of 4 AR1
sessions (well under the 7-session ceiling), N logical `Stream`
objects each owning a `KVStore`, decode round-robin interleaved
(step 0 of every stream → step 1 of every stream …). All session
loads paid once, weight-sharing within the single QNN backend
instance, no cross-process scheduler ping-pong. **Stable at every N
tested (1, 2, 4, 8) — no crashes.**

| N | decode agg (t/s) | per-stream user rate (t/s) | per-stream step (ms) |
|---:|---:|---:|---:|
| 1 | 27.25 | 27.25 | 37.6 |
| 2 | 27.20 | 13.60 | 37.5 |
| 4 | 26.95 |  6.74 | 38.0 |
| 8 | 26.75 |  3.34 | 38.0 |

**Aggregate is flat at ~27 t/s across all N.** This is the *honest*
NPU concurrency picture. The NPU is one Hexagon engine with no
internal parallelism — adding streams divides the same throughput
budget across agents. Per-stream NPU step time stays at ~38 ms (no
contention overhead in the round-robin pattern); per-stream
user-perceived latency scales linearly with N. Aggregate-TG ceiling
is the NPU's hardware single-stream rate; you cannot exceed it
without doing the work in fewer NPU calls (i.e., AR128 batched
prefill across streams — follow-on work).

The subprocess driver's apparent 1.12× scaling came from *accidental
parallelism on host CPU*. The sidecar bench doesn't get that for
free; recapturing it would need threading (one Python thread per
stream calling its own session), since ORT-QNN sessions release the
GIL during NPU calls. Sessions are single-threaded per-call so
NPU work still serializes — but Python overhead would overlap.
Worth ~10-15% aggregate improvement; deferred as separate workstream.

CSVs: `qwen3_4b_ortqnn_sidecar_conc{2,4,8}_2026-04-25_ac.csv`
(N=1 row uses the standalone bench's `_2026-04-25_ac.csv`).

### GPU concurrency detail — Vulkan unlocks via batched matmul

**Vulkan Q4_0 + F16-off + host-mem at N=4** (2026-04-26):

| | PP_agg t/s | TG_agg t/s | total_agg t/s | wall (s) |
|---|---:|---:|---:|---:|
| N=1 (single-stream, reference) | 115.04 | 38.51 | — | — |
| **N=4** | **115.01** | **102.33** | **112.23** | **25.5** |

**Two surprises and a why.**

1. **PP doesn't change** at all going N=1→N=4 (115.04 → 115.01).
   Vulkan PP is single-graph saturated; pushing more streams
   through the prefill phase yields nothing because the prefill
   matmul is already wide enough to fill the GPU.
2. **TG scales 2.66×** (38.51 → 102.33) — better than CPU's
   2.08× and dramatically better than OpenCL's 0.76× scaling.
   Per-stream user rate at N=4 is 25.6 t/s, only 34% slower than
   N=1's 38.5; for comparison, the CPU at N=4 drops 48% per stream
   and the NPU sidecar drops 75% per stream.
3. **Why Vulkan scales and OpenCL doesn't** despite running on the
   same Adreno X2-90 silicon: at AR=1 single-stream decode the
   matmul shape is `[1, hidden]·[hidden, vocab]` — tall-skinny,
   doesn't fit Adreno's `KHR_coopmat` 16×16 tile. The shader
   spends most of its time padding. **At N=4 batched decode, the
   shape becomes `[4, hidden]·[hidden, vocab]`, four times more
   tiles per matmul that actually fit the coopmat layout**, and
   per-token kernel-launch overhead is amortized across 4 streams.
   OpenCL's Adreno kernel doesn't use coopmat (it predates Vulkan's
   matrix-cores API on this device), so it gets none of that
   benefit and only pays the launch-overhead serialization cost.

**Cross-quant story for Vulkan concurrency:** Q4_K_M with the same
knobs is 79.33 TG_agg vs Q4_0's 102.33 (-22%) — Q4_0's optimized
Adreno path matters at concurrency too, just like in single-stream.

CSV: `concurrency4_gpu_knobs_2026-04-26_ac.csv`. Driver:
`scripts/bench_concurrency4_gpu_knobs.py` (companion to
`bench_concurrency4_all_backends.py`, adds GGUF + env passthrough so
the script can target Q4_0 and the Vulkan knob env vars without
forking the original).

### Headline implication for agentic workloads (2026-04-26 inversion)

**At concurrency = 4 the ranking inverts: GPU-Vulkan-Q4_0 (102.3) >
CPU (82.0) > NPU sidecar (27.0) > OpenCL-Q4_0 (20.0).** Vulkan beats
CPU by 25% on aggregate **and** by 25% on per-stream user rate
(25.58 vs 20.50 t/s). The 2026-04-25 reading "for throughput-oriented
multi-agent serving, CPU is the right backend" is **superseded** —
Vulkan-Q4_0 with F16-off + host-mem is the new default for
agentic workloads on this SoC.

Why this didn't show up in the prior bench: the old run used Q4_K_M
(slow Adreno path) with default Vulkan settings (broken FP16 path).
Both fixes are required to unmask the concurrency win — Q4_0 alone
or knobs-on-Q4_K_M each give about 80 t/s aggregate, comparable
to CPU. Together they hit 102 t/s.

**The NPU's value at high N is still energy efficiency at a flat
ceiling.** It serves N agents at the same aggregate ~27 t/s and at
roughly half the CPU's and Vulkan's power draw. If your workload is
latency-tolerant (e.g., background drafts, low-priority agentic
loops, ambient summarization), 8 agents × 3.3 t/s on the NPU at ~13 W
still beats 8 agents on Vulkan at ~28 W on energy/token even though
Vulkan now "wins" on raw throughput. The W4 sidecar design (one
in-process multi-context runtime, never spawn N processes) remains
the architectural prerequisite for any NPU multi-tenant work.

**OpenCL is still the wrong path for concurrent workload.** Aggregate
N=4 on Q4_0 is 19.95, *worse* than single-stream (26.22); per-stream
collapses to 5.0 t/s. Adreno's per-token kernel-launch overhead × 4
interleaved streams serializes — same finding as 2026-04-23, just
shifted by the Q4_0 quant uplift (15.68 → 19.95). Q4_0 doesn't
rescue OpenCL at concurrency; the Vulkan path with coopmat does.

**KleidiAI tracks plain CPU at concurrency** (79.7 vs 82.0 t/s = -3%).
KleidiAI's per-call setup cost gets paid per stream rather than
amortized — under concurrency the small-tile-disadvantage at 4B that
showed up single-stream gets slightly worse. Plain CPU is the safer
default for CPU-routed batched/agentic loads.

## Post-mortem

### Which island wins each workload?

**PP (prompt processing).** NPU still wins decisively — Genie at 1566
t/s, **npu_engine at 1985 t/s** vs the next-best 569 t/s on OpenCL
(Q4_0, refreshed 2026-04-26). NPU vs OpenCL margin: **3.5× for our
stack**, 2.8× for Genie. Prefill is big-matmul, bandwidth-friendly
work; HMX's batched AR=128 graphs dominate. The 2026-04-26 GPU
refresh closed the gap a notch — OpenCL went from 367 → 569 PP just
by switching to the recommended Q4_0 quant — but NPU's lead remains
structural. Vulkan PP (115) is now functional but ~5× behind OpenCL,
so it's not the GPU PP backend.

**TG (token generation).** CPU still wins at AC (**39.50 t/s**), but
the 2026-04-26 GPU refresh produced a near-tie: **Vulkan-Q4_0 with
F16-off + host-mem hits 38.51 t/s** — within 1 t/s of CPU. Ranking
at AC (canonical Q4_0 for GPU rows, Q4_K_M for CPU): CPU (39.5) ≈
**Vulkan (38.5)** > CPU+KleidiAI (38.5) > GPU-OpenCL (26.2) > NPU
(23.3). Two changes from the prior reading: (a) **Vulkan moved from
broken/unranked to second-place TG**; (b) **GPU-OpenCL is no longer
last on TG** thanks to Q4_0 (26.2 vs old 22.9, +14%). Kernel-launch
overhead at AR=1 still hurts OpenCL, but the optimized Q4_0 decode
path narrows the gap. **Vulkan-Q4_0 is now the right GPU backend
for TG-heavy workloads** (chat, agentic decode); OpenCL stays best
for PP-heavy workloads.

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
4. **OpenCL is the blessed Adreno path for PP; Vulkan is now the
   blessed path for TG** (revised 2026-04-26). Qualcomm still
   recommends OpenCL for Adreno on X2E. The 2026-04-26 sweep found
   the prior "Vulkan PP broken" diagnosis was a runtime config bug
   in the FP16 codepath, not a true upstream failure. With
   `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` and the
   pure-Q4_0 model, Vulkan delivers 115 PP / **38.5 TG**, and the TG
   number ties CPU and beats OpenCL by 47% — making Vulkan the right
   GPU backend whenever decode dominates. Not a workstream change;
   documented stance: route PP→OpenCL, TG→Vulkan when GPU is the
   chosen island.
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
  - `results/csv/qwen3_4b_ortqnn_2026-04-23_ac.csv` +
    `_2026-04-23_bat.csv` — original AR1-only ORT-QNN baseline.
  - `results/csv/qwen3_4b_ortqnn_2026-04-25_ac.csv` +
    `_2026-04-25_bat.csv` — AR128 swap-mode refresh (current canonical
    AC + BAT NPU/ORT-QNN cells).
  - `results/csv/qwen3_4b_ortqnn_npuconc2_stream{0,1}_2026-04-25_ac.csv`
    — clean 2-stream NPU concurrency data (per-stream ~14.5 t/s
    PP / ~15.3 t/s TG; aggregate ~30.6 t/s = 1.12× single-stream).
  - `results/csv/qwen3_4b_ortqnn_npuconc4_stream{2,3}_2026-04-25_v3_ac.csv`
    — surviving streams from the v3 4-way attempt (streams 0+1 crashed
    QNN 1003 mid-prefill; per-survivor ~14.4 t/s effective 2-way
    contention).
  - `results/csv/qwen3_4b_ortqnn_sidecar_conc{2,4,8}_2026-04-25_ac.csv`
    — single-process N-stream sidecar concurrency bench (architectural
    fix for the subprocess fan-out's N=4 crash); aggregate ~27 t/s
    flat, per-stream rate scales 1/N. N=1 baseline is the standalone
    bench `qwen3_4b_ortqnn_2026-04-25_ac.csv`.
  - `results/csv/qwen3_4b_baseline_2026-04-26_ac_knob_sweep.csv` —
    OpenCL + Vulkan knob/quant sweep at llama.cpp `f53577432`. Canonical
    rows: OpenCL Q4_0 default (569 PP / 26 TG) and Vulkan Q4_0 with
    `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1` (115 PP /
    38.5 TG). Supersedes the 2026-04-23 GPU baselines for AC.
  - `results/csv/qwen3_4b_gpu_knobs_2026-04-26_bat.csv` — battery
    refresh of the same matrix with WMI DischargeRate sampling for
    J/tok. Canonical BAT rows: OpenCL Q4_0 default (539 / 24.9 / 1.29
    J/tok) and Vulkan Q4_0 with knobs (114.6 / 34.7 / 1.43 J/tok).
    Supersedes the 2026-04-23 GPU BAT baselines.
  - `results/csv/concurrency4_gpu_knobs_2026-04-26_ac.csv` —
    concurrency-4 (`-np 4 -npp 512 -ntg 128 -npl 4`) matrix for
    OpenCL/Vulkan × Q4_0/Q4_K_M with the canonical knob combos.
    Canonical row: Vulkan Q4_0 + F16-off + host-mem at TG_agg
    102.33 t/s — the new aggregate-TG champion at N=4. Supersedes
    the 2026-04-23 GPU concurrency rows.
  - `models/Qwen3-4B-Q4_0.gguf` — pure-Q4_0 quant from
    `unsloth/Qwen3-4B-GGUF` (HF). 2.21 GiB. Recommended Adreno OpenCL
    quant per `llama.cpp/docs/backend/OPENCL.md`.
  - `scripts/bench_qwen3_4b_gpu_knobs_bat.py` — wrapper around
    `bench_qwen3_4b_all_backends.py`'s `PowerSampler` that lets us
    pick GGUF + arbitrary env (the original driver hard-codes
    Q4_K_M and has no env passthrough). Used for the 2026-04-26 BAT
    refresh; reusable for future quant/knob experiments without
    forking the AC driver.
  - `scripts/bench_concurrency4_gpu_knobs.py` — companion to
    `bench_concurrency4_all_backends.py` (drives `llama-batched-bench
    -np 4 -npp 512 -ntg 128 -npl 4`) with the same GGUF + env
    overrides; adds Vulkan preset support that the original
    concurrency driver lacked.
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
- 2026-04-25 (AC): NPU-via-`npu_engine` rerun with **AR128 swap-mode
  prefill + AR1 decode + IOBinding** (commits `ac17196`..`3447862`).
  PP **1985 t/s** (+27% vs Genie 1566), TG **27.25 t/s** (+17% vs
  Genie 23.3) on the same Qualcomm w4a16 .bin. Compute-only wall 4.8 s
  for 256+128 tokens; one-shot total wall 40.9 s including 36 s of
  AR128/AR1 session swap. Per-partition load profile shows HTP context
  init scales with depth (1.6→6.4 s across parts 1→4) not raw .bin
  size — implication: a warm-session sidecar (now landed,
  `npu_engine/sidecar.py`) closes the swap-mode wall-clock gap, and
  any further gains on the cold-start side are HTP-init bound, not
  I/O bound. BAT pending — J/tok delta vs Genie under AR128 swap is
  the next data point.
- 2026-04-25 (BAT): NPU-via-`npu_engine` BAT rerun. PP **2118 t/s**
  (+33% vs Genie 1598), TG **25.26 t/s** (+8% vs Genie 23.33). Compute
  wall 5.2 s; total wall 47.9 s including 42.7 s session swap. Mean
  9.4 W over the full wall; one-shot J/tok = 1.17 (worse than Genie's
  0.537 because of the cold-start tax). Steady-state J/tok under swap
  mode is methodologically tricky to extract from a one-shot bench
  (sampler averages over both compute and load windows); the right
  vehicle is the sidecar warm-keep run, deferred. Throughput
  consistency across power states is excellent — both PP and TG within
  ±7% of AC, NPU is still the only backend that doesn't degrade
  meaningfully on battery. CSV
  `results/csv/qwen3_4b_ortqnn_2026-04-25_bat.csv`.
- 2026-04-26 (GPU refresh @ llama.cpp `f53577432`): pulled all 4
  llama.cpp build trees from `cf8b0dbda` to `f53577432` (81 commits),
  rebuilt cpu/cpu-kleidiai/opencl/vulkan, and ran a knob+quant sweep
  on OpenCL and Vulkan against the prior Q4_K_M default. Two big
  findings, both runtime-only (no source change):
  **(1) Vulkan PP unbroken** — `GGML_VK_DISABLE_F16=1 GGML_VK_PREFER_HOST_MEMORY=1`
  takes Vulkan from `3.91 → 84.19 t/s` on Q4_K_M (+21.5×); the prior
  "shader recompile per tile" hypothesis was wrong, the actual cause
  is the Adreno Vulkan ICD's FP16 codepath silently falling into a
  slow scalar path on Q4 matmuls. `DISABLE_F16` forces FP32 which the
  driver runs cleanly. `PREFER_HOST_MEMORY` adds ~2% PP / ~8% TG by
  biasing allocation away from VRAM-style buffers (Adreno reports
  `uma=1`). Net: Vulkan is now the best GPU backend for TG-heavy
  workloads. **(2) OpenCL is fastest on pure Q4_0**, not Q4_K_M —
  upstream's Adreno OpenCL backend has optimized kernels only for
  Q4_0/Q8_0/MXFP4; Q4_K is officially unsupported. Switching to
  `unsloth/Qwen3-4B-Q4_0.gguf` (2.21 GiB) takes OpenCL from
  `367 → 569 PP` (+55%) and `23 → 26 TG` (+14%). Vulkan with the
  fix above on Q4_0: `115.04 PP / 38.51 TG` — TG matches CPU's 39.5
  within a percentage point and beats OpenCL TG by 47%. **Anti-knobs
  found:** `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1` is a small
  regression at PP=512 with `-r 3` confidence; `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`
  is a 3.2× regression (int dot is critical). **CPU/SME unchanged**:
  KleidiAI is still pinned to v1.22.0 upstream (no commits to
  `kleidiai/`), our local `detect_num_smcus → 0` patch
  (issue [#22182](https://github.com/ggml-org/llama.cpp/issues/22182))
  remains required. Build commits stamped in
  `llama.cpp/build-{cpu,cpu-kleidiai,opencl,vulkan}/SPECULA_BUILD.txt`.
  CSV `results/csv/qwen3_4b_baseline_2026-04-26_ac_knob_sweep.csv`.
- 2026-04-26 (GPU BAT refresh): closed the BAT cells for the 2026-04-26
  GPU sweep using `scripts/bench_qwen3_4b_gpu_knobs_bat.py` (new
  wrapper around the AC driver's WMI DischargeRate sampler that
  supports per-run GGUF + env overrides). **OpenCL Q4_0 default**:
  PP 539.46, TG 24.94, mean 27.52 W, **J/tok 1.291**, AC→BAT delta
  -5.2% PP / -4.9% TG (much tighter than the 2026-04-23 reading of
  -19% TG). **Vulkan Q4_0 + `DISABLE_F16+PREFER_HOST`**: PP 114.57,
  TG 34.69, mean 28.25 W, **J/tok 1.425**, AC→BAT delta -0.4% PP
  (Vulkan is the only non-NPU backend that doesn't degrade
  meaningfully on battery). **J/gen-tok ≈ 0.81 for Vulkan-Q4_0** —
  best non-NPU per-token energy in the matrix, beats CPU's ~3.96 by
  4.9× and OpenCL-Q4_0's 1.10 by 26%. Two surprises worth noting:
  **(a)** OpenCL Q4_K_M's mean W dropped from 44.6 W (2026-04-23) to
  28.43 W (2026-04-26) on the same model, suggesting upstream OpenCL
  kernel improvements between `fd6ae4ca1`→`f53577432` reduced GPU
  active duty cycle (or 2026-04-23 had transient driver state); no
  perfect attribution but the new number is reproducible across the
  Q4_0 and Q4_K_M rows. **(b)** OpenCL J/tok went from 2.690 to
  **1.291 (-52%)** — combined effect of Q4_0 quant + power drop.
  CSV `results/csv/qwen3_4b_gpu_knobs_2026-04-26_bat.csv`. The full
  GPU column of the matrix is now populated for AC and BAT; remaining
  TODOs are CPU/CPU-KleidiAI/NPU re-baseline (lower priority — those
  rows haven't moved much).
- 2026-04-26 (concurrency-4 GPU refresh, AC): ran the new GPU canonical
  configs through `llama-batched-bench -np 4 -npp 512 -ntg 128 -npl 4`
  via new `scripts/bench_concurrency4_gpu_knobs.py` (companion to the
  original concurrency driver, adds GGUF + env passthrough + Vulkan
  preset support). **The 2026-04-25 ranking inverts**: previously
  "CPU 82 > NPU 27 > OpenCL 16" with CPU declared the right
  multi-agent backend; now **Vulkan-Q4_0 + `DISABLE_F16+PREFER_HOST`
  hits TG_agg 102.33 t/s at N=4 (+25% vs CPU)**. Scaling factor
  **2.66×** from single-stream's 38.51 — best in the matrix. Why:
  AR=1 single-stream Vulkan decode wastes Adreno's `KHR_coopmat`
  16×16 matrix-core tiles (matmul shape is tall-skinny `[1, hidden]`);
  N=4 batched decode lifts the shape to `[4, hidden]` which fits the
  coopmat layout much better, and per-token kernel-launch overhead
  is amortized across 4 streams. OpenCL doesn't get the same
  benefit — its Adreno kernel predates the coopmat path — and stays
  at 19.95 TG_agg even on Q4_0 (vs 15.68 on Q4_K_M, +27% from quant
  alone). PP_agg for Vulkan is **flat** (115.01 ≈ N=1's 115.04) —
  Vulkan PP is single-graph saturated, no benefit from concurrency
  on the prefill side. **NPU sidecar still has the energy edge**
  (~13 W vs Vulkan's ~28 W) — for latency-tolerant agentic work
  the J/tok argument still favors NPU; for throughput-or-latency-
  sensitive multi-agent serving, Vulkan-Q4_0 is the new default.
  CSV `results/csv/concurrency4_gpu_knobs_2026-04-26_ac.csv`. BAT
  concurrency is the next data point (J/tok at N=4 will likely
  reorder again — Vulkan-Q4_0 has the best non-NPU J/gen-tok at
  N=1, and concurrency only helps).
- 2026-04-25 (concurrency): NPU-via-`npu_engine` concurrency-N matrix
  added in two architectures. **(a) Subprocess fan-out** (existing
  `bench_concurrency4_npu_ortqnn.py`): N=2 aggregate 30.59 t/s
  (1.12× — Python-overhead overlap, not NPU parallelism), N=4
  crashes (third independent confirmation; IOBinding refactor doesn't
  fix it; root cause is 4×4=16 QNN sessions exceeding HTP scheduler
  resources). **(b) Sidecar single-process** (new
  `bench_concurrency_sidecar.py`, the architectural fix): one set of 4
  sessions shared across N logical streams, decode round-robin
  interleaved. Stable at N=1, 2, 4, 8 — no crashes. Aggregate flat at
  ~27 t/s = the NPU's single-stream ceiling. Per-stream user-perceived
  rate divides cleanly by N. **At N=4, CPU outperforms NPU on
  aggregate by 3.0×** (82.0 vs 26.95 t/s) — the gap is structural (CPU
  has 12 cores of true parallelism; NPU is one Hexagon engine), not
  fixable by more engineering on the NPU side. NPU at high N is a
  power-efficiency play, not a throughput play.
