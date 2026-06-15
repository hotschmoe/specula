# 2026-06-14 — contribution & research opportunities (WoA + Snapdragon X2E)

Captured at the close of session 35's perf review. This is the
"where can we contribute / research / explore" map, written down so we
can act on it. Tomorrow: repo cleanup, then start working these items.

The thesis: **almost nobody is doing serious LLM-inference
characterization on Windows-on-ARM + Snapdragon X2 Elite + Adreno X2-90
+ Hexagon v81.** That neglected surface is exactly the contribution
moat. Ordered A→E by leverage (impact ÷ effort).

## Context: the two ceilings (why this is worth doing)

Two separate ceilings, and we're nowhere near the soft one:

- **Hard ceiling (physics).** 228 GB/s SoC bandwidth is *less* than
  Strix Halo (~256) and far less than M4 Max (~410). Single-stream TG
  is bandwidth-bound, so we will **never** beat an M4 Max on raw
  decode of one stream — that's silicon. Against Strix Halo / DGX
  Spark (256–273 GB/s) we're only ~15–20% behind on bandwidth, so
  *that* gap is closeable.
- **Soft ceiling (software maturity).** Our best decode path realizes
  only **~120–130 GB/s effective ≈ 53% of the 228 GB/s SoC peak**
  (see `docs/memory_bandwidth_ceiling.md`). The headroom to the soft
  ceiling is large and itemizable — that is what A–E below attack.

Unique upside vs Strix / Spark / Mac: we have a **third compute island
(Hexagon NPU)** and **unified memory where a cross-island handoff is a
cache fence, not a DMA.** The realistic win is not "beat M4 Max on
t/s" — it's "match Strix/Spark on throughput once the WoA stack matures
**and win decisively on energy/token**" (NPU ~13 W vs CPU ~25 W vs
Vulkan ~28 W, and silent/lag-free per `project_npu_silent_operation`).

Reference latest numbers (build `e37abd6b5`, 2026-06-12,
`results/csv/backend_refresh_2026-06-12.md`):
- Qwen3-4B Q4_0: best TG 55.65 t/s (OpenCL `-ngl 0`); dense prefill
  A/B-confirmed regressed −47% vs `856c3adac`.
- Qwen3.6-35B-A3B MXFP4: TG 27.49 (`-ngl 0`) / PP 191.72 (`-ngl 99`,
  GPU offload now works).
- Qwen3.6-27B-MTP Q4_0: ~12 t/s with MTP n4 (Q4_0 is the right quant —
  Q8_0 is ~⅔ the speed, bandwidth-bound).

---

## A. Upstream llama.cpp / Adreno fixes — highest impact, real OSS reach

Concrete, reproducible bugs the broader Adreno/WoA community shares.

- **A1 — Bisect the dense-prefill regression.** A/B-confirmed −47%
  (`-ngl 0`) / −57% (`-ngl 99`) across the 489-commit window
  `856c3adac..e37abd6b5`. Both CPU-matmul and OpenCL-matmul paths
  regress ~equally → shared upstream cause (batching / graph-build /
  a common op), not a single kernel. This is a clean, high-value first
  PR and recovers ~2.3× on PP. **Highest-leverage concrete next step.**
  Harness: `git bisect` across the window, one canonical
  `llama-bench -m Qwen3-4B-Q4_0.gguf -p 512 -n 128 -r 5 -ngl 0 -t 16`.
  See `results/csv/backend_refresh_2026-06-12.md` §A/B.
- **A2 — Broken Vulkan F16 path on the Adreno ICD.** Vulkan prefill is
  6.36 t/s (unusable) while Vulkan TG (38.4) and concurrency already
  win. File with a minimal repro; likely a driver or llama.cpp
  FP16-dispatch fix. If fixed, Vulkan becomes a strong general path.
- **A3 — Characterize the "OpenCL `-ngl 0` coprocessor" effect.** It is
  8–10% faster than a pure-CPU build and nobody knows why (open
  question, `docs/2026-05-13_overnight_perf_results.md`). Profiler
  trace → a generally useful finding / possible PR. See
  `reference_opencl_ngl0_coprocessor`.
- **A4 — Upstream our existing local build fixes.** clang-via-
  vcvarsarm64 and the KleidiAI `.S` patch already work locally; strip
  local-only bits and PR them (roadmap W5.c).

## B. First-class ggml QNN backend — project-defining

- **B1 — Hexagon NPU backend for llama.cpp** (roadmap B4). No one has
  landed a first-class ggml QNN backend. It would let the whole
  community drive the NPU from stock llama.cpp instead of our custom
  ORT-QNN sidecar, collapsing specula's stack to "a few llama.cpp
  flags." Multi-session and structural, but this is the contribution
  that puts the project on the map. The NPU already does ~2167 t/s
  prefill; it is barely tapped for general inference today. Gate on
  A4 / the ARM-Windows compile path being solid first.

## C. Quantization-on-HTP research

- **C1 — Q4_0 vs Q4_K_M perplexity validation.** We have the speed win
  across every model size but **not** the quality confirmation — still
  an open TODO (`reference_qwen3_4b_q4_0_beats_q4km`). Run
  `llama-perplexity` on wikitext before declaring Q4_0 the production
  default.
- **C2 — Format A/B on the actual silicon** (roadmap W9): w4a16 /
  w8a16 / w8a8 / MX formats, per-row vs per-group vs per-tensor,
  mixed-precision (V/O at w8). Rank by throughput × cos × size.
- **C3 — Calibration-free PTQ / TurboQuant** (roadmap W3) and the
  existing novel AIMET MoE-adapter work
  (`reference_aimet_moe_adapter_pattern`). Publishable as a "Snapdragon
  X2E quantization cheat-sheet."

## D. Intra-SoC heterogeneous orchestration — the publishable research

- **D1 — Layer-wise KV streaming across CPU/GPU/NPU** (roadmap W4.d/e).
  exolabs did this across *machines* over a slow link; doing it across
  three islands *inside one SoC*, where the handoff is a fence not a
  copy, is unexplored. This is where the 53%→~100% effective-bandwidth
  gap gets closed — and it is a paper. The same layer-streaming
  primitive is also the exact pattern Phase 4 DFlash/DDTree needs, so
  it pays off twice.
- **D2 — 3-phase × 3-island assignment policy.** Empirically fill the
  {prefill, draft, verify} × {CPU, Adreno, Hexagon} matrix and emit a
  context-sensitive decision tree (prompt-length, power-state,
  session-count) → island assignment.

## E. Characterization artifacts — highest visibility-per-effort

There is essentially **no good public data** on WoA LLM performance.
We are sitting on data nobody else has.

- **E1 — Published category × backend matrix** (the roadmap's headline
  deliverable) with AC + battery, solo + concurrent.
- **E2 — Cross-platform laptop bake-off** vs M-series (MLX/ANE), Strix
  Halo (ROCm/Vulkan), DGX Spark (CUDA). "Which laptop is best for
  local LLM" has no clean public answer today; one published table is
  high-visibility (roadmap B13). Requires borrow/rent access.
- **E3 — Energy/token + sustained thermal curves** (roadmap B1/B2).
  Per-rail (CPU/GPU/NPU) power during decode; 30-min throttle curves.
  This is the axis where Snapdragon actually leads — measure it
  rigorously.
- **E4 — Negative-results writeups.** Adreno OpenCL regression, AI Hub
  preserve-list bug, V-projection quant collapse — each is a concrete
  datapoint on an understudied platform. Document and upstream.

---

## Suggested starting order

1. **A1** (bisect the prefill regression) — recoverable 2.3× on PP,
   already A/B-isolated, clean first upstream PR, establishes
   credibility for the bigger B1 NPU-backend work.
2. **C1** (Q4_0 perplexity) — cheap, unblocks the production-quant
   decision for Qwen3.6.
3. **A2 / A3** (Vulkan PP + ngl0 profiler trace) — parallelizable
   investigation sessions.
4. **E1/E3** characterization — runs in the background of everything.
5. **B1 / D1** — the big structural bets, after the ARM-Windows
   compile path and the single-island cells are solid.

Companion docs: `docs/roadmap.md` (W4/W5/W9, B4/B13), 
`docs/memory_bandwidth_ceiling.md` (the two-ceilings analysis),
`results/csv/backend_refresh_2026-06-12.md` (latest numbers + the
A/B regression control).
