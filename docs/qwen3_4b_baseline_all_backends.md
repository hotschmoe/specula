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
QAIRT 2.45.40, Genie 1.17.0, bundle compiled QAIRT 2.42. Context=2048
(llama.cpp), CL=512 (NPU bundle). NPU (ORT-QNN) row refreshed
2026-04-25 with our `npu_engine` (AR128 swap-mode prefill + AR1
decode + IOBinding); CSV `results/csv/qwen3_4b_ortqnn_2026-04-25_ac.csv`.

| backend | runtime / build | PP (t/s) | TG (t/s) | TG tokens | notes |
|---|---|---:|---:|---:|---|
| NPU (Genie)           | genie-t2t-run (QAIRT 2.45, AR128 prefill)   | 1566.23 (AR128) | 23.30 | 3582 | temp=0.8 sampler; ran until ctx-fill |
| **NPU (ORT-QNN, npu_engine)** | our stack, chained 4-part, AR128 swap + AR1 decode | **1985.46** (AR128) | **27.25** | 128 | 256-tok prefill (CL512 cap); beats Genie on PP (+27%) and TG (+17%) |
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
| NPU (ORT-QNN, npu_engine, AR128 swap) | 2118.83 (AR128) | 25.26 | 9.4† | 1.17† / 0.128‡ | — | 47.9 (5.2 compute) |
| NPU (ORT-QNN chained, AR1, 2026-04-23 baseline) | 24.57 (AR1) | 24.32 | 23.5 | 0.959 | — | 15.7 |

† Mean W (9.4) is averaged over the full 47.9 s wall, ~36 s of which
is AR128/AR1 session-load I/O at low CPU draw. The 1.17 J/tok number
amortizes the *full one-shot* energy (`mean_W × wall_s / tokens`) over
all 384 tokens — comparable to "what does it cost to ask one prompt
from a cold process." For a long-lived sidecar that pays the load tax
once at boot and amortizes over many requests, the steady-state
J/tok is in between this number and Genie's 0.537 — needs separate
measurement with the sidecar warm.

‡ The script's "compute-only" J/tok (0.128) multiplies the *whole-run*
mean W by *compute-only* wall — silently assumes compute ran at the
global average, which is wrong when ~76% of the wall is mostly-idle
session loading. Number is in the CSV for completeness; do not cite
without the caveat. To get an honest steady-state J/tok we'd need a
sampler that brackets only the compute window (or a sidecar warm-keep
that erases the imbalance) — open methodology TODO.
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
| NPU (Genie)         | 1566.2 | 1598.5 | +2.1% | 23.30 | 23.33 | +0.1% |
| NPU (ORT-QNN, ours) | 1985.5 | 2118.8 | +6.7%* | 27.25 | 25.26 | -7.3% |
| CPU                 |  188.3 |  191.3 | +1.6% | 39.50 | 38.52 | -2.5% |
| CPU + KleidiAI      |  185.8 |  180.4 | -2.9% | 38.51 | 37.33 | -3.1% |
| GPU (OpenCL)        |  367.4 |  355.8 | -3.2% | 22.92 | 18.58 | -19% |
| GPU (Vulkan)        |    3.9 |   — (timeout) |       | 31.43 | — |       |

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

**PP (prompt processing).** NPU wins decisively — Genie at 1566 t/s,
**npu_engine at 1985 t/s** vs the next-best 367 t/s on OpenCL, a
**5.4× margin** for our stack and 4.3× for Genie. Prefill is big-matmul,
bandwidth-friendly work; HMX's batched AR=128 graphs dominate. The
hypothesis going in was "GPU should win PP"; that was wrong on this
silicon with this bundle. Confirmed: when we have a model compiled to
NPU, prefill stays on NPU. The 2026-04-25 npu_engine refresh further
shows our ORT-QNN dispatch path can *beat* Genie's by 27% on PP at the
same hardware ceiling — vendor runtime is not Pareto-optimal here.

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
  - `results/csv/qwen3_4b_ortqnn_2026-04-23_ac.csv` +
    `_2026-04-23_bat.csv` — original AR1-only ORT-QNN baseline.
  - `results/csv/qwen3_4b_ortqnn_2026-04-25_ac.csv` +
    `_2026-04-25_bat.csv` — AR128 swap-mode refresh (current canonical
    AC + BAT NPU/ORT-QNN cells).
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
