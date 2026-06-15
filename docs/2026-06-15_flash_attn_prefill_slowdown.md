# 2026-06-15 — Flash attention slows prefill on Snapdragon X2E (CPU + Adreno)

Session 36. Spin-off from the A1 bisect (see
`results/csv/backend_refresh_2026-06-12.md` §UPDATE): the session-35
"dense-prefill regression" was llama-bench commit `aa46bda89` flipping
the `-fa` default OFF→AUTO, where AUTO enables flash attention on these
backends. That is not a regression — but it surfaced a real,
characterizable effect worth reporting upstream.

## TL;DR

- **Flash attention is slower for prefill on every X2E backend tested**,
  by up to **~2.2×** on the dense 4B model. The penalty is largest on
  dense models and on the Adreno OpenCL path.
- **Flash attention's effect on decode (TG) is small and mixed** — it
  *helps* CPU decode (+11%) and 35B-MoE `-ngl 0` decode (+9%), but is a
  wash-to-negative on the GPU-offload paths.
- **The cleanest upstream bug: Adreno OpenCL FA prefill is ~2.2× slower
  than non-FA with ~zero decode benefit** (588→270 pp512, 30.5→29.4
  tg128 on 4B). That is a pure loss on that backend.
- Practical rule: **pin `-fa 0` for prefill-heavy / long-prompt work.**
  Leave FA on only for decode-heavy CPU workloads where the +11% TG
  matters and prompts are short.

## Data

`llama-bench`, build `e37abd6b5`, AC / Balanced, `-fa 0` vs `-fa 1`.
4B Q4_0 r=5; 35B-A3B MXFP4 r=3. Raw: `output/fa_sweep_2026-06-15.md`.

| model / backend | metric | -fa 0 | -fa 1 | FA-off speedup |
|---|---|---:|---:|---:|
| 4B Q4_0  CPU -ngl0 -t16          | pp512 | 375.97 | 188.71 | **1.99×** |
| 4B Q4_0  CPU -ngl0 -t16          | tg128 | 49.00 | 54.52 | 0.90× (FA on +11%) |
| 4B Q4_0  OpenCL -ngl99 -ub512    | pp512 | 587.92 | 270.19 | **2.18×** |
| 4B Q4_0  OpenCL -ngl99 -ub512    | tg128 | 30.53 | 29.41 | 1.04× (tie) |
| 35B-A3B  OpenCL -ngl0 -t18       | pp512 | 183.71 | 159.69 | 1.15× |
| 35B-A3B  OpenCL -ngl0 -t18       | tg128 | 28.94 | 31.66 | 0.91× (FA on +9%) |
| 35B-A3B  OpenCL -ngl99 -t18      | pp512 | 272.57 | 196.22 | 1.39× |
| 35B-A3B  OpenCL -ngl99 -t18      | tg128 | 26.87 | 24.14 | 1.11× |

## Reading

1. **Prefill always favors FA-off.** Dense 4B takes the biggest hit
   (~2×) because attention is a larger share of its prefill compute; the
   35B MoE dilutes it across expert FFNs (+15% to +39%). The FA prefill
   kernel on both the ARM CPU and the Adreno OpenCL backend is simply
   less optimized than the default batched attention path for the
   `pp512` shape.
2. **Decode is a different story.** FA's whole point — streaming the KV
   without materializing the full scores matrix — pays off most for the
   AR1 token shape. We see it help CPU decode (+11%) and 35B `-ngl 0`
   decode (+9%). On GPU-offload paths the Adreno FA decode kernel gives
   nothing back (and costs 11% on 35B `-ngl 99`).
3. **`-fa auto` currently resolves to "enabled" here**, so the new
   llama-bench default penalizes every prefill measurement vs the
   historical FA-off records. Any cross-version perf comparison must pin
   `-fa` explicitly.

## Recommendation matrix (X2E)

| workload | backend | `-fa` |
|---|---|---|
| long prompt / RAG / summarize (prefill-bound) | any | **0** |
| short prompt, long generation (decode-bound) | CPU | 1 |
| short prompt, long generation (decode-bound) | OpenCL offload | 0 (FA gives nothing) |
| benchmarking across llama.cpp versions | any | pin explicitly |

## Upstream / contribution framing

- **Strongest report (A2/E4):** "Adreno OpenCL flash-attention prefill
  kernel ~2.2× slower than non-FA with no decode benefit (Qwen3-4B
  Q4_0, Snapdragon X2 Elite, Adreno X2-90)." Clean, reproducible, single
  backend, no compensating upside → a genuine perf bug, not a tradeoff.
- **Secondary datapoint:** ARM-CPU FA is a prefill-vs-decode tradeoff
  (−2× pp, +11% tg), so the right default is workload-dependent; a note,
  not a bug.
- Reproduce on more dense models before filing to show it is not
  Qwen-specific.

## Corrections to prior docs

- Session-35 `backend_refresh_2026-06-12.md` "keep build-opencl-old for
  prefill" and the 35B "PP 190→151 regression" note are both this `-fa`
  artifact (the §UPDATE section there already records the fix).
- Memory `reference_llamacpp_prefill_regression_e37abd6b5` corrected.
- Companion: `reference_qwen3_4b_q4_0_beats_q4km` (C1 perplexity),
  `reference_opencl_ngl0_coprocessor`.
