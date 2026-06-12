# Backend refresh perf — llama.cpp 856c3adac → e37abd6b5 (session 35)

Date: 2026-06-12, AC, Balanced power plan (only scheme available; the
"Best Performance" overlay is not exposable via powercfg on this build).
All `llama-bench`, r=5 unless noted. New build = `e37abd6b5`.

## Qwen3-4B Q4_0 — all backends (pp512 + tg128)

| backend | config | PP512 t/s | TG128 t/s | vs session-26 record |
|---|---|---:|---:|---|
| CPU        | -ngl0 -t16        | 215.97 | 55.20 | TG new high |
| CPU-kleidiai | -t16            | 194.40 | 48.30 | kleidiai now slower than plain CPU |
| OpenCL     | -ngl0 -t16        | 201.80 | **55.65** | PP 379→202 (−47%); TG 50.80→55.65 (+9.5%, NEW RECORD) |
| OpenCL     | -ngl99 -ub512 -t16 | 259.81 | 30.09 | PP 586→260 (−56%) |
| Vulkan     | -ngl99 -t16       | **6.36** | 38.42 | prefill STILL broken (p8 smoke 145 was misleading) |

TG is up across the board; dense prefill is down hard on OpenCL.
Ruled out as causes: thermal (re-measure after idle reproduced;
σ tight), ubatch (flat 197-201 across ub 128..2048).

## A/B — same machine, same session, OLD vs NEW (pp512, r=5)

DECISIVE control. build-opencl-old = `856c3adac` built today.

| 4B Q4_0 pp512 | OLD 856c3adac | NEW e37abd6b5 | Δ |
|---|---:|---:|---|
| ngl0 -t16        | 369.15 ± 10.4 | 194.53 ± 3.1 | **−47%** |
| ngl99 -ub512 -t16 | 544.01 ± 5.1 | 231.68 ± 4.9 | **−57%** |

OLD reproduces the session-26 records today (369≈379, 544≈586) → the
prefill drop is a GENUINE REGRESSION introduced in the 489-commit
window, not power-state/thermal. Both ngl0 (CPU matmul) and ngl99
(OpenCL matmul) regress ~equally → shared upstream cause (batching /
graph-build / a common op), not a single kernel. Bisection across
[856c3adac .. e37abd6b5] is the follow-up.

## Qwen3.6-35B-A3B MXFP4 — OpenCL offload (pp512 + tg128, r=3)

| config | PP512 t/s | TG128 t/s | note |
|---|---:|---:|---|
| ngl0 -t18              | 151.52 | 27.49 | PP ~190→151 (same prefill regression); best TG |
| **ngl99 -t18**         | **191.72** | 22.84 | **GPU OFFLOAD NOW WORKS** (was clCreateImage -40 in session 27); best PP |
| ngl99 -t18 +LARGE_BUFFER | 180.91 | 21.35 | LB is overhead (model 20.2 GB fits 24 GB view) |

WIN: new Adreno MoE kernels (#23303/#23449) + OP_GATED_DELTA_NET
(#23312) fixed the SSM-tensor clCreateImage -40 → 35B MoE now offloads
to Adreno. ngl99 best prefill (191), ngl0 best decode (27.5).

## Qwen3.6-27B-MTP Q4_0 — MTP on MAINLINE (llama-server, r=1, 256 tok)

build-opencl (mainline e37abd6b5), -ngl0 -t18. Confirms build-opencl-mtp
(PR build) is RETIRED — MTP merged (#23643/#24025/#23287).

| n_max | PP t/s | TG t/s | accept % | session-27 (PR build) |
|---|---:|---:|---:|---|
| 0 (no-MTP) | 47.44 | 7.75 | — | baseline 8.39 |
| 4 | 52.93 | **12.40** | 56.9 | — |
| 8 | 45.39 | 10.09 | 37.1 | TG 12.17, **95.8% accept** |
| 12 | 47.99 | 7.46 | 25.7 | — |

WIN: MTP loads + runs on mainline (+60% TG at n4). CAVEAT: acceptance
is far lower than the PR build measured (37% vs 95.8% at n8) → sweet
spot shifted n8→n4; the mainline MTP rework and/or the PR-era unsloth
GGUF's MTP head may not fully match mainline's expected graph.
Note r=1 (high variance) — re-run with more samples before trusting
absolute MTP numbers.

## Bottom line

The update is a MIXED BAG on this hardware:
- GAINS: 4B TG new high (~55.6); 35B-A3B GPU-offload now works on OpenCL;
  MTP on mainline (retire the PR build); new self-spec ngram-* types.
- LOSSES: dense prefill ~halved on OpenCL (A/B-confirmed regression);
  Vulkan prefill still broken; mainline MTP acceptance much lower than PR.

Recommendation: keep build-opencl-old (856c3adac) as the known-good
prefill build for prefill-heavy/long-context work until the regression
is bisected and (ideally) fixed upstream. New build wins for decode-
heavy and for 35B GPU offload.
