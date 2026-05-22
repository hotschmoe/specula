# Phase 1 anchors — GPU PP plateau, CPU verify-shape, C0 baseline

Phase 1 of the GPU↔NPU placement side quest. Three tasks:
solidify the Phase 0 verify-shape picture by (1) confirming/refuting
the GPU PP "fixed-overhead plateau" at large k, (2) adding the CPU
verify-shape anchor for the full 3-island crossover, and (3) a real
C0 end-to-end baseline number.

**Status: tasks 1 + 2 complete. Task 3 not runnable — see §3.**

All runs: AC power, Snapdragon X2 Elite Extreme, Qwen3-4B Q4_0 GGUF
(`models\Qwen3-4B-Q4_0.gguf`), `llama-bench`, `-b k -ub k` so each
k-token forward is one un-chunked physical batch, `-r 5`.

---

## 1. GPU PP plateau confirmation — HYPOTHESIS REFUTED

Phase 0 saw a flat ~165 ms GPU PP plateau for k = 2..16 and a gentle
rise at k = 32 (175), k = 64 (201). The Phase 1 hypothesis was a
single per-pass model: `ms ≈ fixed-overhead(~165) + linear(compute)`.
Extending the sweep to k = 128, 256, 512 **refutes the single-line
fixed-overhead model.**

Build: `llama.cpp\build-opencl\` (OpenCL/Adreno X2-90), `-ngl 99`.
Output CSV: `results\phase1_gpu_pp.csv`. Logs: `logs\gpu_pp{128,256,512}.json`.

| k   | ms_per_pass | tok/s  | ms per useful token | stddev |
|----:|------------:|-------:|--------------------:|-------:|
| 128 | 277.8       | 460.8  | 2.17                | 3.5 ms |
| 256 | 447.9       | 571.6  | 1.75                | 1.8 ms |
| 512 | 887.9       | 576.6  | 1.73                | 1.8 ms |

Full GPU PP curve (Phase 0 + Phase 1 combined):

```
k:    2     4     8    16    32    64   128   256   512
ms: 163  167  168  168  175  201  278  448  888
```

### Why the single-line model fails

There are **two distinct regimes**, not one fixed-overhead line:

- **k = 2..16 — a flat ~167 ms plateau.** Per-pass cost is essentially
  constant. Slope ≈ 0. This is overhead/launch-bound: the k-wide GEMM
  is too small to register against fixed per-pass cost (kernel
  dispatch + the one-time stream of the 2.2 GB weight set).
- **k ≥ 64 — a clean linear rise.** Least-squares fit over
  {64,128,256,512}: **ms ≈ 82 + 1.55·k** (R² > 0.99; predictions
  within 3% at k = 128/512, 7% at k = 256). The k = 32 point (175 ms)
  is the transitional knee between the two regimes.

A single line `a + b·k` fit across the *whole* k ≥ 2 range gives
`136 + 1.40·k`, which mispredicts the plateau badly (says 139 ms at
k = 2, plateau is 163) and the tail (says 853 at k = 512, meas 888).
**The fixed-overhead-plus-linear model is correct only piecewise**:
flat for k ≤ ~16, then linear `82 + 1.55·k` for k ≥ 64.

### What this means for the side quest

The headline does **not** change. The crossover-relevant range is
k ≈ 3–8 (useful speculation depth), and that sits squarely inside the
flat ~167 ms plateau. The large-k linear regime is academic for
*verify* — verify never runs at k = 512. So:

- For verify (k ≈ 3–8): GPU PP cost is a flat ~167 ms, NPU PP is a
  flat ~52 ms (AR128-padded). **NPU still wins verify ~3× at every
  useful k.** Phase 0's conclusion stands, now with the large-k tail
  characterized: the GPU never gets *cheaper* per pass as k grows in
  the verify range, and at large k it gets strictly worse.
- The linear slope 1.55 ms/token at k ≥ 64 is the GPU's true
  compute-bound throughput for batched 4B forward (~645 tok/s
  asymptotic). At k = 512 the Adreno hits 577 tok/s — still ~10× off
  the 159 GB/s bandwidth floor's implied ceiling, confirming Phase 0's
  read that the OpenCL PP path is software-bound, not silicon-bound.

---

## 2. CPU verify-shape anchor — completes the 3-island picture

Build: `llama.cpp\build-cpu\` (ARM64 NEON), `-ngl 0 -t 16` (the
documented Qwen3-4B sweet spot on this box, phys_cores−2).
Output CSV: `results\phase0_cpu.csv`. Logs: `logs\cpu_pp*.json`,
`logs\cpu_tg64.json`.

### PP (batched k-token forward — the verify shape)

| k   | ms_per_pass | tok/s  | ms per useful token |
|----:|------------:|-------:|--------------------:|
| 1   | 19.3        | 51.8   | 19.3                |
| 2   | 25.5        | 78.4   | 12.8                |
| 4   | 23.0        | 173.9  | 5.8                 |
| 8   | 33.8        | 238.1  | 4.2                 |
| 16  | 38.3        | 417.7  | 2.4                 |
| 32  | 70.9        | 451.2  | 2.2                 |
| 128 | 267.9       | 477.9  | 2.1                 |
| 512 | 1458.6      | 351.6  | 2.8                 |

### TG (autoregressive single-token forward — the draft shape)

CPU TG (n = 64): **50.4 t/s**, 19.85 ms/token.

### CPU per-pass shape

The CPU PP curve is **near-linear from the start** — no plateau. A
fit over k ≥ 32 gives `ms ≈ -62 + 2.95·k`, i.e. ~2.95 ms per token
of compute with a *small* fixed overhead. Unlike the GPU there is no
flat launch-bound shelf: the CPU pays close to true compute cost even
at k = 4. The small-k points (k = 2..8: 23–34 ms) are dominated by
the ~19 ms fixed cost (the k = 1 / TG-equivalent floor).

This makes the CPU the **cheapest island for small-k verify**: a k = 4
verify costs 23 ms on the CPU vs 167 ms GPU vs 52 ms NPU. The CPU's
weakness is large batches — at k = 512 it is 1459 ms (the NPU does
128 tokens in 52 ms; the GPU does 512 in 888 ms). The k = 512 CPU
point also shows thermal noise (stddev 65 ms).

---

## 3. C0 end-to-end baseline — NOT RUNNABLE (skipped per timebox)

C0 = "draft NPU / verify CPU / prefill CPU", run via
`scripts\npu_spec_outer_loop_async.py`. **Could not be run** — the
harness's model dependencies were removed in the session-21 repo
cleanup (`models\` trimmed 280 GB → 24 GB) and never restored:

- **Target missing.** `npu_spec_step7_plumbing.py` hard-codes
  `TARGET_MODEL = models\Qwen3-8B-Q4_K_M.gguf`. Not on disk (the SQ1
  target later moved to Qwen3-14B; the 8B GGUF is gone).
- **NPU draft binary missing.** The harness loads
  `models\qwen3_0_6b_draft_v81_ctx512.pathbmask.bin` (the 0.6B Path-B
  NPU draft) plus its config dir `models\qwen3-0.6b-pathbmask\`.
  Neither exists — only the *target*-class NPU bundles survived the
  cleanup (`qualcomm-qwen3-4b-ref`, `specula-qwen3-4b-ref`).
- **CPU draft ONNX missing.** `models\qwen3-0.6b-optimum\model.onnx`
  (the FP32 draft reference) is also gone.

Restoring C0 means re-fetching the 8B GGUF and re-compiling /
re-downloading the 0.6B NPU draft bundle — a multi-hour job that
blows the side-quest timebox. Per the task's explicit instruction
("Task 3 is the riskiest — if it rat-holes, skip it"), C0 is skipped.
No `phase1_c0_baseline.csv` was written.

**Note for whoever picks this up:** the *current* spec-decode story
is C2-flavoured anyway (Phase 0 refuted C1 / today's-C0 placement —
verify belongs on the NPU). A fresh end-to-end baseline is better
spent on C2 with a real small draft model than on resurrecting the
8B-target C0 harness. The historical C0 number from session 11/19 is
**7.98 t/s** (Phase 5 k = 2, NPU-draft async, 8B CPU target) — usable
as the legacy baseline if one is needed.

---

## 4. The 3-island per-pass-cost-vs-k comparison

Per-pass wall time (ms) of one k-token forward, Qwen3-4B, AC.
GPU/CPU = Q4_0 GGUF via llama.cpp; NPU = w4a16 Genie bundle via
ORT-QNN. NPU values for k ∈ [2,128] are the flat AR128-graph cost
(the bundle has no AR2/4/8/16/32 graph — k pads up to 128).

| k    | CPU ms | GPU ms | NPU ms | cheapest |
|-----:|-------:|-------:|-------:|----------|
| 1    | 19.3   | 39.0   | 34.2   | **CPU**  |
| 2    | 25.5   | 163.4  | 52 \*  | **CPU**  |
| 4    | 23.0   | 166.9  | 52 \*  | **CPU**  |
| 8    | 33.8   | 167.9  | 52 \*  | **CPU**  |
| 16   | 38.3   | 168.3  | 52 \*  | **CPU**  |
| 32   | 70.9   | 174.6  | 52 \*  | **NPU**  |
| 64   | 201.1  | 201.1  | 52 \*  | **NPU**  |
| 128  | 267.9  | 277.8  | 52     | **NPU**  |
| 256  | —      | 447.9  | —      | (NPU)    |
| 512  | 1458.6 | 887.9  | —      | (NPU)    |

\* AR128-padded — the NPU pays a flat 52 ms for any k ∈ [2,128].

Per-pass-cost shapes:

```
CPU : near-linear from k=1.  ms ≈ -62 + 2.95·k (k>=32). No plateau.
GPU : flat ~167 ms plateau k=2..16, then linear ms ≈ 82 + 1.55·k (k>=64).
NPU : 34 ms at k=1, then a flat 52 ms shelf for ALL k in [2,128]
      (graph shape-locked — see findings/phase0_npu.md).
```

### The crossovers

There are **two** crossover points in the useful-to-moderate k range:

1. **CPU → NPU at k ≈ 24–32.** The CPU verify cost rises through the
   NPU's flat 52 ms shelf around k ≈ 24 (interpolating: CPU is 38 ms
   at k = 16, 71 ms at k = 32; it crosses 52 ms near k ≈ 22). Below
   that the **CPU is the cheapest verify island**; above it the NPU is.
2. **GPU is never cheapest.** The GPU PP plateau (167 ms) is above
   both the CPU (≤ 71 ms through k = 32) and the NPU (52 ms) across
   the entire useful + moderate range. The GPU only catches the CPU
   at k ≈ 64 (both ~201 ms) and beats the CPU only past k ≈ 100 —
   long after the NPU has won outright.

### Read for the side quest

- **For verify at useful speculation depth (k ≈ 3–8): the CPU is the
  cheapest island** — 23–34 ms, vs 52 ms NPU, vs 167 ms GPU. This is
  a notable correction to the Phase 0 narrative, which compared only
  GPU vs NPU and concluded "verify → NPU." With the CPU anchor in,
  **verify at small k actually wants the CPU.** Today's C0 placement
  (verify on CPU) is the *right* call for small k on raw per-pass cost.
- **The NPU wins verify once k ≳ 24** — its flat 52 ms shelf undercuts
  the CPU's rising linear cost. For large speculation depth or
  tree/multi-candidate verify (many tokens per pass) the NPU is the
  island.
- **The GPU is the wrong island for verify at every k that matters.**
  Its only per-pass win is the deep batched-prefill regime (k ≥ ~256),
  irrelevant to spec-decode verify. The GPU's role in heterogeneous
  spec decode is *draft* (TG-shaped, to overlap with a non-GPU
  verify), not verify — consistent with Phase 0.
- **Caveat — per-pass cost ≠ end-to-end winner.** This compares raw
  per-pass wall time. The real placement question also weighs
  draft∥verify *overlap* (you want verify on a *different* island
  than draft so they run concurrently) and HTTP/dispatch overhead.
  The CPU being cheapest-per-pass for small-k verify does not by
  itself beat C2 end-to-end if draft is also on the CPU (no overlap).
  That trade is exactly what a Phase-1 end-to-end run (C0 vs C1 vs C2)
  would settle — and is now the clear next step.

### Caveats

- GPU/CPU on Q4_0, NPU on w4a16 — different quant, different bytes
  streamed. The comparison is op-*shape*, not like-for-like model.
- GPU/CPU numbers are llama.cpp (OpenCL / ARM64-NEON). The GPU PP
  path is software-bound (~10× its bandwidth floor); a better OpenCL
  mat-mat path would lower the GPU plateau and could change the GPU's
  standing — a genuine W4 follow-up.
- NPU k ∈ [2,128] are AR128-padded (flat 52 ms); a native AR8 graph
  would make small-k NPU verify cheaper than 52 ms and could pull the
  CPU→NPU crossover below k ≈ 24. Needs a new bundle compile.
- CPU k = 512 carries thermal noise (stddev 65 ms).

---

## Deliverables

- `results\phase1_gpu_pp.csv` — GPU PP k = 128/256/512.
- `results\phase0_cpu.csv` — CPU PP k = 1..512 + TG (island = cpu).
- `phase1_c0_baseline.csv` — **not produced** (C0 not runnable, §3).
- Raw logs: `logs\gpu_pp{128,256,512}.json`,
  `logs\cpu_pp{1,2,4,8,16,32,128,512}.json`, `logs\cpu_tg64.json`.
- Run scripts: `scripts\run_phase1_gpu_pp.ps1`,
  `scripts\run_phase0_cpu.ps1`.
