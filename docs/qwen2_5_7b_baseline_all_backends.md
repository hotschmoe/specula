# Qwen2.5-7B-Instruct — all-backends side-quest matrix

Side-quest companion to `docs/qwen3_4b_baseline_all_backends.md`. Same
SoC (Snapdragon X2 Elite Extreme, Hexagon v81, Adreno X2-90, 48 GB
LPDDR5X @ 228 GB/s unified), same protocol, repointed at a 7B-class
model to measure per-parameter scaling on each compute island. Driven
by the W1.b roadmap question — "what does NPU prefill look like at the
8B target scale?"

**First run: 2026-04-25** (tag `2026-04-25_{ac,bat}`).

## Why this side-quest, why this model

The 4B baseline left an open question: how do PP/TG/J curves *scale*
with parameter count on this silicon, and does the partition count
grow in a way that affects the W4 spec-decode handoff cost? Qwen2.5-7B
is the closest arch-parity neighbour to Qwen3-4B that AI Hub Workbench
will compile for X2 Elite end-to-end (Llama-3.1-8B is the alternative
but its license blocks Qualcomm from publishing a pre-quantized
intermediate, forcing the heavy local FP16 export path — punted to
Scenario A in `docs/rent_cloud_compute.md`).

**Partition count growth, confirmed**: 4B compiles to **4 partitions**;
7B compiles to **6 partitions**. That's a real cost increase for the
host-side KV-stitch loop in any "rolling our own" runtime — each
additional partition adds one extra ORT-QNN dispatch per decode step.

**Quantization difference**: the AI Hub-shipped 4B bundle is `w4a16`
uniform; the 7B Workbench export produced `w8a16` (the AIMET
intermediate quantization grade exposed by `qai-hub-models`). Per-param
weight footprint is therefore ~2× heavier in bytes than naive scaling
predicts — bundle size 4.7 GB at 7B vs 3.1 GB at 4B (factor 1.5× for
1.75× params). See "Per-backend detail → NPU" for what that means at
runtime.

## Models under test

| backend family | artifact | weight footprint |
|---|---|---|
| NPU | `models/qualcomm-qwen2_5-7b-ref/qwen2_5_7b_instruct-genie-w8a16-qualcomm_snapdragon_x2_elite/*.bin` (6 parts, Workbench-compiled) | ~w8a16, parts: 1090/711/711/711/711/1024 MB ≈ 4.7 GB |
| CPU / GPU | `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (bartowski) | 4.36 GiB (Q4_K_M) |

The NPU bundle was produced via:

```bash
.venv-qairt/Scripts/python.exe -m qai_hub_models.models.qwen2_5_7b_instruct.export \
    --chipset qualcomm-snapdragon-x2-elite --device-os 11 \
    --context-length 4096 --skip-profiling --skip-inferencing \
    --model-cache-mode enable \
    --output-dir models/qualcomm-qwen2_5-7b-ref/qwen2_5_7b_instruct-genie-w8a16-qualcomm_snapdragon_x2_elite
```

The Workbench export only emits the `.bin` partitions + `tool-versions.yaml`.
The `genie_config.json`, `htp_backend_ext_config.json`, and `tokenizer.json`
were hand-scaffolded — the htp config was copied from the 4B bundle (same
SoC v81/88), the tokenizer from `Qwen/Qwen2.5-7B-Instruct/tokenizer.json` on
HF, and the genie config adapted from the 4B's with `n-vocab=152064`,
`ctx-bins` listing all 6 partitions, otherwise identical knobs.

## Headline — AC (wall power, idle system)

Commit: `e365e658f` (cpu, vulkan) / `fd6ae4ca1` (opencl) / `cf8b0dbda`
(cpu-kleidiai), QAIRT 2.45.40, Genie 1.17.0, bundle compiled QAIRT
2.45.0.260326154327. Context=4096 (Workbench export's only allowed value).

| backend | runtime / build | PP (t/s) | TG (t/s) | TG tokens | notes |
|---|---|---:|---:|---:|---|
| **NPU (Genie)**     | genie-t2t-run (QAIRT 2.45, AR128 prefill) | **1219.05** | 22.91 | 254 | EOS at 254 (vs ctx-fill on 4B); see notes |
| CPU                 | llama.cpp build-cpu (-t 8 ARM64 NEON)     | 122.98     | 24.17 | 128 | |
| CPU + KleidiAI      | llama.cpp build-cpu-kleidiai (-t 8, i8mm) | 125.57     | **25.42** | 128 | +2% PP, +5% TG vs plain CPU |
| GPU (OpenCL)        | llama.cpp build-opencl -ngl 99 (Adreno)   | 237.02     | 10.72 | 128 | TG halved vs 4B |
| GPU (Vulkan)        | llama.cpp build-vulkan -ngl 99            | —          | —     | 0 | timed out at 600 s — same PP-broken pattern as 4B |

The NPU PP row in `results/csv/qwen2_5_7b_baseline_2026-04-25_ac.csv`
initially failed to parse (the 4B's parser hardcoded `_4_of_4` to detect
last-partition decode completions, missed 7B's `_6_of_6`). The fixed
parser landed in `scripts/bench_qwen2_5_7b_all_backends.py` and the rerun
canonical numbers are in
`results/csv/qwen2_5_7b_baseline_2026-04-25_ac_npu_rerun.csv` —
**that file supersedes the NPU row in the main `_ac.csv`** (mirrors the
4B's OpenCL JSON-parse rerun pattern).

## Headline — battery (DC, Q4_K_M model, same seed)

Same J/tok formula as 4B (`mean(DischargeRate_W) × wall_s /
(pp_tokens + tg_tokens)`). 2 s polling interval.

| backend | PP (t/s) | TG (t/s) | mean W | J/tok | wall (s) |
|---|---:|---:|---:|---:|---:|
| **NPU (Genie)**     | 1189.59 | 22.39 | **13.7** | **0.328** | 18.4 |
| CPU                 | 132.81  | 24.87 | 18.2     | 0.953    | 33.4 |
| CPU + KleidiAI      | 120.99  | 24.35 | 38.7     | 2.144    | 35.5 |
| GPU (OpenCL)        | 224.09  | 10.74 | 58.3     | 5.330    | 58.5 |
| GPU (Vulkan)        | —       | —     | 10.8     | —        | 633 (timeout) |

**The NPU's J/tok of 0.328 is a denominator illusion.** 4B Genie ran
3582 decode tokens (until ctx-fill); 7B Genie EOS'd at 254 decode
tokens, so the 766-token denominator is much smaller than 4B's 4094.
PP energy is mostly amortized over decode; with fewer decode tokens the
ratio looks artificially good. The honest comparison is **per-generated-
token** energy, subtracting prefill:

| | total J/tok | J / generated token |
|---|---:|---:|
| 4B NPU (BAT) | 0.537 | ~0.615 |
| **7B NPU (BAT)** | **0.328** | **~0.967** |
| 4B CPU | 0.899 | ~3.96 |
| **7B CPU** | **0.953** | **~4.16** |
| 4B OpenCL | 2.690 | ~13.0 |
| **7B OpenCL** | **5.330** | **~25.5** |

Per generated token, NPU went from 0.615 J/gen-tok at 4B to 0.967 at
7B — a **57% increase**. CPU went from ~3.96 to ~4.16 (+5%). OpenCL
roughly doubled. So: NPU is still 4.3× more efficient than CPU per
generated token at 7B (vs 6.4× at 4B), and 26× more efficient than
OpenCL (vs 21× at 4B). NPU's efficiency advantage *narrows* at 7B vs
CPU but *widens* vs OpenCL.

Battery drain over the BAT matrix: 71170 → 68453 mWh = **2717 mWh**
(~4% of full charge). Much smaller than 4B's 11791 mWh because the
7B NPU run EOS'd at 254 tokens vs the 4B running 3582 to ctx-fill.

## AC vs battery consistency

| backend        | PP AC | PP BAT | Δ    | TG AC | TG BAT | Δ    |
|---|---:|---:|---:|---:|---:|---:|
| NPU (Genie)    | 1219.1 | 1189.6 | -2.4% | 22.91 | 22.39 | -2.3% |
| CPU            |  123.0 |  132.8 | +8.0% | 24.17 | 24.87 | +2.9% |
| CPU + KleidiAI |  125.6 |  121.0 | -3.7% | 25.42 | 24.35 | -4.2% |
| GPU (OpenCL)   |  237.0 |  224.1 | -5.5% | 10.72 | 10.74 | +0.2% |
| GPU (Vulkan)   |    —   |   —    |       |   —   |   —   |       |

Same pattern as 4B: **NPU is the only backend where battery ≈ AC** (±2.4%).
CPU's +8% PP on battery is noise (single-run variance). The OpenCL TG
*not* dropping on battery (vs the −19% it dropped at 4B) is interesting
— Adreno is so poorly utilized at 7B that the AC↔BAT delta is below
the noise floor.

## Per-backend detail

### NPU (Genie)

```
cmd:    genie-t2t-run --config genie_config.json --prompt_file pp512_prompt.txt --log info
build:  QAIRT 2.45.40.260406 / genie-t2t-run / bundle compiled QAIRT 2.45.0.260326154327
AC   : PP 1219.05 t/s  TG 22.91 t/s  (254 TG tokens, 17.6 s wall)
BAT  : PP 1189.59 t/s  TG 22.39 t/s  (254 TG tokens, 18.4 s wall, mean 13.7 W, 0.328 J/tok)
notes: Bundle uses the default temp=0.8 sampler. Gen EOS'd at 254 tokens
       (Qwen2.5-7B-Instruct hits <|im_end|> reliably on the synthetic
       technical prompt — 4B-Instruct rambles through ctx-fill).
       Run exits cleanly (exit code 0) — no `--profile` teardown bug
       like the 4B's 2.42-compiled bundle had on 2.45 runtime.
```

The TG step trace shows ~7-11 ms per partition × 6 partitions ≈ 50 ms
per token = 20.8 t/s steady-state, matching the measured ~22 t/s
(the residual is host-side framing overhead).

**Per-step decode work scales sublinearly with parameter count.** At
4B (4 partitions, ~600 MB each) NPU TG was 23.30 t/s; at 7B (6
partitions, ~700-1000 MB each) it's 22.91 t/s — only −1.7%. AR=1
decode is dispatch-bound, not weight-bandwidth-bound — each token
spends most of its time in ORT/Genie dispatch + KV slot updates rather
than weight matmul. This is the most important non-obvious finding of
this run.

### CPU (ARM64 NEON)

```
cmd:    llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -t 8
build:  llama.cpp build-cpu @ e365e658f
AC   : PP 122.98 t/s  TG 24.17 t/s  (36.7 s wall)
BAT  : PP 132.81 t/s  TG 24.87 t/s  (33.4 s wall, mean 18.2 W, 0.953 J/tok)
```

CPU PP scaled from 188.30 → 122.98 t/s = -35% for 1.75× more
parameters. Roughly 1/parameter scaling, consistent with bandwidth-
bound prefill. Mean W *dropped* from 25.5 → 18.2 W on BAT — at 7B the
CPU spends more time waiting on memory cache misses, less time burning
through ALUs. Net effect: J/gen-tok rose only +5% (4.16 vs 3.96 at 4B)
even though throughput dropped −39%.

### CPU + KleidiAI

```
cmd:    llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -t 8
build:  llama.cpp build-cpu-kleidiai @ cf8b0dbda
AC   : PP 125.57 t/s  TG 25.42 t/s  (34.0 s wall)
BAT  : PP 120.99 t/s  TG 24.35 t/s  (35.5 s wall, mean 38.7 W, 2.144 J/tok)
```

**KleidiAI flipped from regression at 4B to win at 7B** — +2.1% PP,
+5.2% TG vs plain CPU on AC. The i8mm ukernels' fixed setup cost
amortizes better at bigger matmul tiles. Same CPU silicon, same
kernel set (SME2 still runtime-fenced by
`scripts/patch_kleidiai_detect.py`); the difference is purely matmul
dimensions.

That said, KleidiAI burns +112% more power than plain CPU on BAT
(38.7 W vs 18.2 W) for that ~5% throughput gain — **J/tok is 2.25×
worse** (2.144 vs 0.953). On battery, plain CPU wins decisively. Net
recommendation: pick the build by power state, not by silicon.

### GPU (Adreno / OpenCL)

```
cmd:    llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-opencl @ fd6ae4ca1 (GGML_OPENCL_USE_ADRENO_KERNELS=ON)
AC   : PP 237.02 t/s  TG 10.72 t/s  (55.9 s wall)
BAT  : PP 224.09 t/s  TG 10.74 t/s  (58.5 s wall, mean 58.3 W, 5.330 J/tok)
```

**OpenCL TG halved 4B → 7B** (22.92 → 10.72 t/s). Adreno's per-token
kernel-launch overhead at AR=1 is now the binding constraint:
launching the same number of kernels per token on a model with bigger
matmuls means each kernel runs longer in absolute wall time, so the
launch overhead's *fraction* of step time stays roughly fixed, but
the absolute step time grows faster than 1/throughput. **OpenCL TG
falls below half of CPU TG at 7B** — even less viable than at 4B.

OpenCL also draws 58.3 W mean — 4× the NPU, 3× the CPU. PP at
237 t/s is only 19% of NPU's 1219 t/s. There is no workload at 7B on
this silicon where OpenCL is the right pick.

### GPU (Vulkan)

```
cmd:    llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
build:  llama.cpp build-vulkan
AC   : timed out at 600 s — PP collapsed (same pattern as 4B)
BAT  : timed out at 633 s
```

Same shader-recompilation-per-prefill-tile failure mode as 4B. Don't
use Vulkan for any prefill workload on X2E + Adreno until investigated.

## Concurrency = 4 (agentic workload)

Same SoC, same models, but each backend runs **4 simultaneous decode
streams** via `llama-batched-bench -np 4 -npp 512 -ntg 128 -npl 4`.
Tests aggregate throughput when 4 agentic clients hit the model at
once. NPU absent — Genie is single-stream and the multi-stream ORT-QNN
sidecar work is in flight (see "NPU concurrency experiment" below).

Runner: `scripts/bench_concurrency4_all_backends.py`.

### Headline — AC, both models

| model | backend | S_PP agg (t/s) | S_TG agg (t/s) | per-stream TG | wall (s) |
|---|---|---:|---:|---:|---:|
| 4B | CPU          | 126.61 | **82.01** | 20.50 | 24.4 |
| 4B | KleidiAI     | 116.63 | 79.73     | 19.93 | 25.5 |
| 4B | OpenCL       | 251.08 | 15.68     | 3.92  | 50.0 |
| 7B | CPU          | 104.64 | **62.78** | 15.70 | 30.4 |
| 7B | KleidiAI     |  99.65 | 62.25     | 15.56 | 31.4 |
| 7B | OpenCL       | 196.35 | 13.18     | 3.29  | 60.7 |

Stored in `results/csv/concurrency4_qwen3_4b_2026-04-25_ac.csv` and
`results/csv/concurrency4_qwen2_5_7b_2026-04-25_ac.csv`.

### Concurrency scaling factor (TG aggregate / TG single-stream)

| model | CPU | KleidiAI | OpenCL |
|---|---:|---:|---:|
| 4B | 2.08× | 2.07× | **0.68× (worse than single-stream)** |
| 7B | **2.60×** | 2.45× | 1.23× |

**CPU scaling factor *grows* with model size** — at 7B, 4-way batching
recovers 2.60× of the single-stream throughput vs 2.08× at 4B. Bigger
matmul per step amortizes batched dispatch better. Per-stream TG drops
from 24 → 16 t/s, a ~35% latency hit per agent for ~2.6× aggregate.
Tradeoff curve agentic deployments would actually accept.

**OpenCL is the wrong path for any concurrent workload.** At 4B the
aggregate at concurrency=4 is *worse* than concurrency=1 (15.68 < 22.92
t/s). At 7B it's barely positive (1.23×) but absolute throughput is
13 t/s aggregate — **CPU beats OpenCL by 4.8× at concurrency=4 / 7B**.
Adreno's per-token kernel-launch overhead × 4 interleaved streams
serializes badly.

**KleidiAI's small 7B win evaporates under concurrency.** The +5% TG
single-stream advantage becomes −0.8% at concurrency=4 — vector-kernel
setup cost gets paid once per stream rather than amortized. At
concurrency, plain CPU is the safer default.

### NPU concurrency experiment (ORT-QNN-chained, spawn-N-procs)

We have an existing chained-4-partition ORT-QNN runtime for the 4B
(`scripts/bench_qwen3_4b_ortqnn.py`) that drives the same .bin files
as Genie through ORT-QNN sessions in Python. Spawning N simultaneous
instances of that script lets us probe what NPU concurrency would
look like under our own runtime — independently of Genie, which has
no concurrency knob.

Driver: `scripts/bench_concurrency4_npu_ortqnn.py`. Each of N spawned
Python processes loads its own 4 ORT-QNN sessions; QNN HTP's
`weight_sharing_enabled=true` lets the NPU share weight pages across
contexts so memory cost stays bounded. Results assembled across two
attempts at N=4 (the second after a wrapper-build race fix in the
canonical 4B bench) and the first attempt's surviving 3-way subset:

| concurrency N | per-stream PP (t/s) | per-stream TG (t/s) | step median (ms) | aggregate TG (t/s) | scaling vs N=1 |
|---:|---:|---:|---:|---:|---:|
| 1  | 25.76 | 25.78 | ~39 | 25.78 | 1.00× |
| 2† | ~14.05 | ~14.76 | ~64 | ~29.5 | 1.14× |
| 3‡ | ~10.20 | ~10.45 | ~90 | ~31.4 | **1.22×** |
| 4  | — | — | — | — | **unstable — see below** |

† 2-way data is from streams 1 and 2 of a 4-way attempt where streams
0 and 3 crashed early (within the first PP steps). Those two streams
ran most of their wall under effective 2-way contention.

‡ 3-way data is from a separate 4-way attempt where stream 1 failed
on a wrapper-build write race (since fixed); the other three completed.

**4-way concurrency is unstable on this stack.** Two attempts both
saw 2 of 4 streams crash mid-execution with `QNN graph execute error.
Error code: 1003`. Different streams crashed each time (failures are
not deterministic per stream slot). The pattern: load and warmup
complete fine across all 4 streams, then early into the prefill loop
(steps 0–32), 2 streams hit the QNN runtime error and exit while the
remaining 2 finish their work uninterrupted.

Hypothesis: **HTP backend resource ceiling.** 4 streams × 4 partitions
= 16 simultaneous ORT-QNN sessions on one Hexagon engine. Likely
candidates for the binding limit: VTCM (TCM) regions, DSP RPC slots,
or the QNN context cache. We didn't dig in further because the
finding ("don't push past 3 concurrent ORT-QNN contexts on this
stack") is already actionable.

**Aggregate decode throughput plateaus at ~1.22× single-stream.** The
Hexagon engine is the binding constraint on aggregate decode; adding
contexts past 2 just spreads the same pie. **Per-stream tail latency
scales roughly linearly with N**: 39 ms → 64 ms → 90 ms.

**Headline for agentic workloads:** at concurrency=4, **CPU outperforms
NPU on aggregate** (CPU 82 t/s on 4B vs NPU's 31 t/s plateau and
crash-prone behavior). NPU is single-tenant or low-tenant (≤3
concurrent agents); CPU owns ≥4 concurrency. The W4 sidecar's
heterogeneous orchestration becomes especially important here — to
*serve* multiple agents from the NPU at all, we'd need a single
in-process multi-context runtime that can interleave agent steps
without spawning N independent QNN-context sets.

Why 4B not 7B for this experiment: only the 4B has the wrapper ONNXs
we need for ORT-QNN chaining. The 7B Workbench bundle ships only raw
context binaries — the oracle wrapper pipeline that produced
`oracle_part1.wrapper.onnx` … `oracle_part4.wrapper.onnx` for the 4B
has not been adapted yet for the 7B's 6-partition split + w8a16 IO
(a follow-on workstream).

### Genie 4× async (item #1 — feasibility analysis, not run)

The simpler "fake concurrency" path is spawning 4 `genie-t2t-run.exe`
processes simultaneously against the same bundle. **Yes, this works
mechanically**, but it has worse properties than the ORT-QNN approach:

- Each Genie process loads its own context (no inter-process weight
  sharing — `weight_sharing_enabled` only deduplicates *within* a
  process). Memory cost ~4× the bundle size.
- Genie's CLI gives no per-step timing hooks — we'd only get aggregate
  wall time + Genie's reported PP/TG, no contention diagnostics.
- HTP scheduler interleaves contexts but with significant context-
  switch cost vs the in-process multi-context model ORT-QNN can drive.

Skipped because the ORT-QNN-chained run gives us strictly more data
for the same wall time. If we ever want a "naive 4-agents-each-shells-
out-to-genie" datapoint for vendor-default-experience purposes, it's
trivial to add — but unlikely to be informative beyond "much worse
than the in-process path."

## Post-mortem

### Per-parameter scaling 4B → 7B

The headline question this side-quest set out to answer:

| metric | 4B | 7B | Δ | implication |
|---|---:|---:|---:|---|
| NPU partition count | 4 | 6 | +50% | spec-decode handoff cost grows; W4 sidecar planning needs to budget 6 ORT dispatches per AR1 step |
| NPU bundle size | 3.1 GB | 4.7 GB | +52% | mostly w4a16 → w8a16 (2× weight bytes) compensating for 1.75× params |
| NPU PP (AC) | 1566 | 1219 | -22% | sublinear in 1/params — bandwidth-bound prefill scales gently |
| NPU TG (AC) | 23.30 | 22.91 | -1.7% | dispatch-bound; per-step weight bandwidth is *not* the bottleneck |
| NPU J/gen-tok (BAT) | 0.615 | 0.967 | +57% | per-token dispatch cost roughly doubles with 1.5× partitions; this is the cost of partitioning |
| CPU PP | 188 | 123 | -35% | proportional to params (memory-bound) |
| CPU TG | 39.5 | 24.2 | -39% | same |
| CPU mean W (BAT) | 25.5 | 18.2 | -29% | cache miss dominance lowers utilization → lower draw |
| OpenCL PP | 367 | 237 | -35% | proportional to params |
| OpenCL TG | 22.9 | 10.7 | -53% | catastrophic — kernel-launch overhead cliff |
| OpenCL mean W (BAT) | 44.6 | 58.3 | +31% | higher absolute draw on a bigger model |

**Projection to 8B target (W1.b)**: extrapolating NPU PP linearly from
1219 t/s @ 7B with another ~14% parameter increase → expect **NPU PP
≈ 1050-1100 t/s at 8B**. That's **6-7× CPU's projected 8B PP** (~150
t/s) and **5× OpenCL's projected 200 t/s**. The W1.b roadmap entry
("if 4B prefills at 1566, 8B should land 700-900 t/s") was *too
pessimistic* — the actual scaling is gentler than the original 1/param
guess.

### Which island wins each workload at 7B?

**PP**: NPU dominates by a wider margin than at 4B — 1219 t/s vs
OpenCL's 237 (5.1×) and CPU's 123 (9.9×). Take prefill to the NPU
whenever a bundle exists.

**TG single-stream**: KleidiAI > CPU > Vulkan > NPU > OpenCL.
KleidiAI flipped from a regression at 4B to a small win at 7B
(25.4 vs 24.2 plain CPU). NPU at 22.9 is still useful for the silent/
cool/UX-friendly use case (see 4B doc § Qualitative UX axis), just
not the throughput champion.

**TG concurrency=4**: CPU > KleidiAI > OpenCL. CPU's 2.6× scaling at 7B
is the headline — agentic loads land here, not on GPU.

**J/gen-tok**: NPU still dominates by 4.3× over CPU and 26× over
OpenCL, but the gap to CPU narrowed from 6.4× at 4B. NPU's efficiency
advantage will continue to narrow at larger models because TG is
dispatch-bound (per-token cost grows with partition count) while CPU
TG is memory-bound (per-token cost grows with weight bytes, which is
fundamental).

### Decisions this matrix unblocks

1. **W1.b — NPU prefill of 8B target** stays the obvious play. The
   1219 t/s @ 7B + sublinear scaling means 8B should land near 1000+
   t/s — close to 4B's prefill rate, dramatically better than CPU.
2. **Spec-decode partition handoff cost grows with target size.** 4B
   target → 4 host-side stitches per decode step. 8B target → likely
   6-8 stitches. The ORT-QNN baseline's +79% J/tok overhead vs Genie
   on 4B (per the 4B doc) will widen at 8B unless the W4 sidecar
   moves the chain into C++ with in-process KV.
3. **w8a16 vs w4a16 — pick recipe per goal.** The Workbench-default
   w8a16 export got us a bench-able bundle with no quality fuss but a
   2× weight footprint vs Qualcomm's shipping w4a16 4B bundle. Actual
   bench numbers (PP scaled 1.6× better than naive 1/param) suggest
   the w8a16 didn't hurt prefill noticeably. For W1.b proper (8B
   target with quality matching), the AIMET SEQ_MSE/AdaScale path
   (Scenario B in `docs/rent_cloud_compute.md`) should produce a
   smaller w4a16 bundle if quality tracking matters.
4. **OpenCL ruled out at 7B+.** Too slow (TG halved), too power-hungry
   (mean W ↑31%), worse than CPU at concurrency. Don't include OpenCL
   in any 7B+ deployment. Document and move on.
5. **KleidiAI re-introduced as a build option for AC-only large-model
   workloads.** The per-power-state pick (KleidiAI on AC, plain CPU
   on BAT) is a mode the runner doesn't currently express; either
   parameterize via flag or document the rule of thumb in
   `docs/build_picks.md` (TBD).
6. **CPU as the agentic-workload backend.** 7B at concurrency=4
   delivers 63 agg t/s on CPU — viable for serving a handful of
   concurrent agents per laptop. NPU goes single-tenant until W4
   sidecar is built; until then, CPU owns concurrency.

### What to re-measure in 2-4 weeks

- llama.cpp commit advances (faster Q4_K_M kernels, Vulkan PP fix
  candidates).
- QAIRT minor releases — 2.45.x point updates may improve HTP
  scheduling for multi-context workloads, which would change the
  concurrency picture.
- Drop a Llama-3.1-8B run once the cloud-Linux export path
  (`docs/rent_cloud_compute.md` Scenario A) lands — gives us the
  actual W1.b 8B point with cross-arch confound for free.
- Adreno OpenCL driver build number was 863.0 here; newer drivers may
  flip the OpenCL conclusions (unlikely but cheap to retest).

## Artifacts

Layout follows `docs/repo_hygiene.md`:

- **Permanent** (committed, never delete):
  - `results/csv/qwen2_5_7b_baseline_2026-04-25_ac.csv`
  - `results/csv/qwen2_5_7b_baseline_2026-04-25_ac_npu_rerun.csv` —
    NPU rerun with the parser fix; supersedes the broken NPU row in
    `_ac.csv`.
  - `results/csv/qwen2_5_7b_baseline_2026-04-25_bat.csv`
  - `results/csv/concurrency4_qwen3_4b_2026-04-25_ac.csv`
  - `results/csv/concurrency4_qwen2_5_7b_2026-04-25_ac.csv`
  - `results/csv/qwen3_4b_ortqnn_npuconc4_stream{0,2,3}_2026-04-25_ac.csv`
    — three successful streams from the 3-way subset of the first 4-way
    attempt (stream 1 lost the wrapper-build race; aggregate plateau
    ~31 t/s, per-stream ~10 t/s).
  - `results/csv/qwen3_4b_ortqnn_npuconc4_stream{1,2}_2026-04-25_ac_v2.csv`
    — surviving streams from the second 4-way attempt; streams 0 and 3
    crashed with QNN error 1003 mid-prefill. Streams 1 and 2 effectively
    measure 2-way contention (~14 t/s per stream, ~29 t/s aggregate).
  - `results/qwen2_5_7b_baseline/pp512_prompt.txt` +
    `pp512_prompt_tokens.txt` — pinned prompt; reproducible via
    `scripts/gen_pp512_prompt_qwen2_5_7b.py`.
  - `results/qwen2_5_7b_baseline/tokenizer_upstream.json` — cached
    upstream Qwen2.5 tokenizer (byte-identical to bundle copy).
  - This doc.
- **Staged for deletion** (gitignored,
  `marked_for_deletion/qwen2_5_7b_baseline_2026-04-25_*/` and
  `marked_for_deletion/concurrency4_*_2026-04-25_*/`):
  - Per-backend `.log` files (raw llama-batched-bench output + Genie
    per-graph timestamp traces).
- **Bundle** (in `models/qualcomm-qwen2_5-7b-ref/`, gitignored, kept
  on local disk indefinitely):
  - 6 × `qwen2_5_7b_instruct_w8a16_part_K_of_6.bin` (~4.7 GB total)
  - hand-scaffolded `genie_config.json` (committed values are encoded
    in this doc's "Models under test" section if regen is needed),
    `htp_backend_ext_config.json` (copied from 4B), `tokenizer.json`
    (copied from upstream).
- **Runners**:
  - `scripts/bench_qwen2_5_7b_all_backends.py` — single-stream matrix.
  - `scripts/bench_concurrency4_all_backends.py` — concurrency=4 (CPU,
    KleidiAI, OpenCL).
  - `scripts/bench_concurrency4_npu_ortqnn.py` — NPU concurrency=4 via
    spawning 4 ORT-QNN-chained processes (4B only).
  - `scripts/gen_pp512_prompt_qwen2_5_7b.py` — prompt scaffolding.

## Update log

- 2026-04-25: first run. NPU partition count 6 (vs 4B's 4) confirmed.
  All AC + BAT cells filled (Vulkan timed out as expected). Concurrency
  4 added on the same day (CPU/KleidiAI/OpenCL on both 4B and 7B).
  NPU concurrency-4 via ORT-QNN spawn-4-procs landed alongside.
  Genie 4× async path documented as feasibility-only, not run.
