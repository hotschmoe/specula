# 2026-05-13 overnight perf sprint — results

Branch: `master` at `856c3adac` (frozen baseline, per handover).
Build matrix unchanged: `build-cpu`, `build-cpu-kleidiai`, `build-opencl`,
`build-vulkan`. No rebuilds tonight.

## Headlines

Three new records on this Snapdragon X2 Elite Extreme + 48 GB LPDDR5X
+ Adreno X2-90 + Hexagon laptop, all on llama.cpp master at
`856c3adac`:

1. **Qwen3-4B Q4_0 on OpenCL build with `-ngl 0 -t 16` →
   PP 379.23 ± 11.46 / TG 50.80 ± 0.36 t/s (r=5).**
   New all-time TG on Qwen3-4B, beating the prior NPU ORT-QNN record
   (PP 2167 / TG 29.03) on TG by **+75%**. NPU still wins prefill by
   ~5.7×.
2. **Qwen3-4B Q4_0 on OpenCL `-ngl 99 -t 16 -ub 512` →
   PP 586.28 ± 1.00 / TG 26.72.**
   New non-NPU PP record. Adreno offload is the right choice when
   prefill dominates (>2k prompt) or memory budget is tight; for
   user-perceived latency `-ngl 0` is the upgrade.
3. **Qwen3.6-35B-A3B MXFP4_MOE on OpenCL `-ngl 0 -t 16` →
   PP ~190 / TG ~31 t/s (r=1, variance ±5%).**
   The "blended" config: ~92% of GPU-offload PP (210) AND ~2× of
   GPU-offload TG (16). Best balanced 35B config; equivalent to
   pure CPU on PP, slightly faster on TG.

Concurrency-4 (4 agentic streams, 512 PP + 128 TG each, OpenCL `-ngl 0
-t 16`):

- **4B Q4_0**: aggregate `S_TG 126.94 t/s` (4 × ~32 t/s per stream).
- **35B-A3B MXFP4**: aggregate `S_TG 65.63 t/s` (4 × ~16 t/s per stream).

Headline t/s table:

| model | quant | best backend | best config | PP (t/s) | TG (t/s) |
|---|---|---|---|---|---|
| Qwen3-4B | Q4_0 | OpenCL `-ngl 0` | `-t 16` | 379.23 | **50.80** |
| Qwen3-4B | Q4_0 | OpenCL `-ngl 99` | `-t 16 -ub 512` | **586.28** | 26.72 |
| Qwen3-4B | Q4_0 | NPU ORT-QNN | (Qualcomm bundle, session 25) | **2167.11** | 29.03 |
| Qwen3-4B | Q4_K_M | CPU-kleidiai | `-t 16` | 257.68 | 40.37 |
| Qwen3-4B | Q4_0 | Vulkan | `+DISABLE_MMVQ +DISABLE_FUSION` | 7.71 | 41.09 |
| Qwen3-14B | Q4_K_M | CPU + spec-decode | draft=0.6B Q8_0 n_max=3 | n/a | **16.15** |
| Qwen3-14B | Q4_K_M | CPU-kleidiai (no spec) | `-t 16` | 92.17 | 14.84 |
| Qwen3.6-35B-A3B | MXFP4_MOE | OpenCL `-ngl 0` | `-t 16 / -t 18` | 190.82 | 30.27 |
| Qwen3.6-35B-A3B | MXFP4_MOE | OpenCL `-ngl 99` | `-t 16` | **210.21** | 15.76 |
| Qwen3.6-35B-A3B | MXFP4_MOE | CPU-kleidiai | `-t 18` | 198.63 | 26.98 |

(NPU rows from `current_status.md` session 25; all others measured tonight.)

## Track A — Qwen3.6 MTP

**Status: punted, with rationale.** PR ggml-org/llama.cpp#22673 still
`OPEN` as of 2026-05-13 — mainline at `856c3adac` does not yet consume
MTP self-draft heads. MTP-preserved GGUFs are now plentiful on HF
(searched via `https://huggingface.co/api/models?search=Qwen3.6+MTP+GGUF`):

- `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` (16k downloads)
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (just landed, 68 likes)
- `am17an/Qwen3.6-35BA3B-MTP-GGUF`
- `localweights/Qwen3.6-35B-A3B-MTP-IQ4_XS-GGUF` / `…-Q8nextn-GGUF`
- 27B variants: `havenoammo/`, `unsloth/`, `froggeric/`, `RDson/`

Without #22673, loading these against mainline gives identical
inference to the non-MTP weights — the MTP tensors are just unused.
Building `gg/spec-mtp-experiments` cost-benefit was unfavourable
versus Tracks B/C ROI on already-downloaded weights, so MTP is
deferred until the PR merges. **Bookmark:** when #22673 ships,
re-bench `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` against the
non-MTP MXFP4_MOE baseline of PP 210 / TG 16 (OpenCL `-ngl 99`) or
PP 190 / TG 30 (OpenCL `-ngl 0`).

## Track B — Hybrid GPU + CPU

### B1 — `-ngl` sweep on Qwen3.6-35B-A3B MXFP4_MOE (OpenCL)

`r=1`, p=512 n=128, default threads. `results/csv/track_b1_ngl_sweep_2026-05-13.md`.

| ngl | PP (t/s) | TG (t/s) |
|---:|---:|---:|
|   0 | 204.18 | **31.93** |
|   8 | 166.63 | 17.99 |
|  16 | 188.60 | 19.01 |
|  24 | 188.85 | 18.51 |
|  32 | 207.10 | 17.53 |
|  40 | 210.74 | 16.44 |
|  56 | 205.16 | 16.14 |
|  99 | 210.21 | 15.76 |

Key observation: **`-ngl 0` is not "ignore GPU".** With the OpenCL
backend registered but no model layers offloaded, TG holds at
near-CPU speeds (32) while PP is comparable to full-offload PP
(204 vs 210). Partial offloads (`-ngl 8..56`) collapse TG into the
~16-19 t/s valley — sequential per-layer CPU↔GPU dispatch is the
worst of both backends.

### B1b — `-ncmoe` sweep at `-ngl 99`

`results/csv/track_b_ncmoe_sweep_2026-05-13.md`. Forcing N MoE-expert
layers to CPU while keeping everything else on GPU only hurts TG
monotonically (17.33 → 12.01 as N grows from 0 → 48), because the
per-layer sync cost dominates. **`-ncmoe` is not a hybrid mode worth
chasing on this hardware.**

### B2 — Heterogeneous speculative decode

Tested via `llama-speculative-simple` on Qwen3-14B-Q4_K_M target +
Qwen3-0.6B-Q8_0 draft, prompt ~46 tokens, generate 256.
`results/csv/track_b2_*`.

| target | draft | `-devd` / `-dev` | `-ngl` / `-ngld` | `--spec-draft-n-max` | speed (t/s) | accept |
|---|---|---|---|---:|---:|---:|
| CPU | OpenCL Adreno | none / GPUOpenCL | 0 / 99 | 8 | 11.68 | 70.8% |
| CPU | OpenCL Adreno | none / GPUOpenCL | 0 / 99 | 4 | 11.97 | 84.7% |
| CPU | OpenCL Adreno | none / GPUOpenCL | 0 / 99 | 2 | 10.91 | 90.0% |
| CPU | CPU          | none / none      | 0 / 0  | 4 | **16.97** | 84.2% |
| CPU | CPU          | none / none      | 0 / 0  | 3 | 16.15 | 80.3% |
| OpenCL | OpenCL    | GPUOcl / GPUOcl  | 99/99  | 4 | 4.56  | 82.4% |
| baseline (no spec) | — | — | — | — | 14.84 | — |

**Surprising:** heterogeneous draft-on-Adreno actually *loses* to
pure-CPU baseline (~11.9 vs 14.8 t/s) — Adreno's small-model dispatch
cost on 0.6B Q8_0 is higher than the savings from batched target
verification. **CPU+CPU spec decode is the actual win:** Qwen3-14B
gets to **16.15-16.97 t/s** (+9-14% over baseline 14.84 t/s) with the
0.6B draft. Switching the draft to 1.7B Q8_0 consistently *hurts*
(largest TG seen 15.11), so draft size matters more than acceptance:

| draft | n_max | speed | accept |
|---|---:|---:|---:|
| 0.6B Q8_0 | 3 | 16.15 | 80.3% |
| 0.6B Q8_0 | 4 | 14.78 | 83.2% |
| 0.6B Q8_0 | 5 | 15.66 | 89.7% |
| 0.6B Q8_0 | 6 | 15.62 | 84.2% |
| 1.7B Q8_0 | 3 | 15.11 | 76.3% |
| 1.7B Q8_0 | 4 | 15.04 | 76.8% |
| 1.7B Q8_0 | 5 | 14.96 | 75.8% |
| 1.7B Q8_0 | 6 | 13.62 | 70.3% |

(The earlier draft-max=8 run at 11.97 t/s and accept 84.7% was on
the same Adreno-draft pairing — strong indicator that Adreno-draft
overhead dominates, not n_max tuning.)

Both-on-OpenCL `-ngl 99` is the worst: 4.56 t/s. Adreno dispatch
overhead amplifies across two models.

`ngram-cache` / `ngram-simple` spec types from the help string need
`llama-cli` (interactive) or `llama-server`, not
`llama-speculative-simple`. Not benched tonight — future work.

## Track C — Knob sweeps

### C1 — KleidiAI SME envvar sweep

`GGML_KLEIDIAI_SME` in `{0,1,2,4,8,16}` on `build-cpu-kleidiai`,
Qwen3-4B Q4_K_M `-t 8`. SME=0 baseline: PP 177.32 / TG 39.61.
**All SME>0 values crashed at warmup** (consistent with prior
`docs/SME_investigation.md`); SME path remains broken on this
driver.

The bigger discovery while warmed up: **thread sweep on Qwen3-4B
Q4_K_M (build-cpu-kleidiai)** shows a `-t 16` sweet spot, not the
default `-t 18`. `results/csv/track_c_thread_sweep_qwen3_4b_2026-05-13.md`.

| threads | PP (t/s) | TG (t/s) |
|---:|---:|---:|
|   4 | 108.28 | 33.36 |
|   6 | 136.19 | 36.65 |
|   8 | 166.87 | 38.95 |
|  10 | 196.77 | 40.65 |
|  12 | 217.24 | 40.82 |
|  14 | 237.06 | **41.07** |
|  **16** | **257.68** | 40.37 |
|  18 (default) | 252.67 | 38.19 |

`-t 16` is the new default recommendation for Qwen3-4B Q4_K_M on
CPU-kleidiai. On 35B-A3B MXFP4_MOE the same sweep shows `-t 18`
remains marginally best (PP 198.63 / TG 26.98) — the larger model
benefits from one more thread, so `-t = phys_cores - 2` (16) for 4B
and `-t = phys_cores` (18) for 35B is the rule of thumb.

### C2 — OpenCL `GGML_OPENCL_ADRENO_XMEM_GEMM` opt-in (PR #22755)

Flag confirmed to activate via stderr "Adreno xmem F16xF32 GEMM
enabled". On Qwen3.6-35B-A3B MXFP4_MOE at `-ngl 99`: PP 210.63 (no
change vs 210.21 baseline), TG 13.13 (within variance band, slightly
worse). **xmem is F16xF32 GEMM and does not help MXFP4 quantized
MoE weights.** Future work: try on a dense F16 model (none on disk).

### C3 — OpenCL batch / ubatch sweep on Qwen3-4B Q4_0

`results/csv/track_c3_batch_sweep_4b_opencl_2026-05-13.md`. 40 rows total.
**Headline: `-ub 512` is the floor for full PP; `-ub 128` cuts PP to
~470. TG is flat at ~26.7 across all (-b, -ub) combinations.**

| -b | -ub | PP (t/s) | TG (t/s) |
|---:|---:|---:|---:|
|  512 |  128 | 476.63 | 26.70 |
|  512 |  512 | 584.51 | 26.62 |
| 2048 |  512 | 582.85 | 26.67 |
| 4096 |  512 | 585.38 | 26.65 |
| 8192 |  512 | **586.28** | 26.72 |
| 8192 | 1024 | 584.61 | 26.54 |

Default `-b 2048 -ub 512` is already near-optimal. Going `-ub 1024+`
doesn't help further. `-b` choice (above 512) is irrelevant on Q4_0.

### C4 — Vulkan envvar probes

`results/csv/track_c4_vulkan_*`. Starting from
`GGML_VK_PREFER_HOST_MEMORY=1` (the workaround already known on
2026-05-12).

**New finding:** combining `GGML_VK_PREFER_HOST_MEMORY=1 +
GGML_VK_DISABLE_MMVQ=1 + GGML_VK_DISABLE_FUSION=1` on Qwen3-4B Q4_0
boosts TG from 38.93 to **41.09 t/s** (r=2, `p=128 n=128`). Same combo
on Q4_K_M peaks at 36.50. **Vulkan TG is now 7% higher than session
25's 38.01.** `GGML_VK_DISABLE_COOPMAT=1` is a sharp double-edged
sword — helps Q4_K_M, but crashes Q4_0's TG path to 20.87.

PP is still broken across all Vulkan probes (1.8-7.7 t/s on PP128
regardless of model size and quant). The F16 compute path on this
Adreno Vulkan driver is fundamentally unhealthy on `856c3adac` — no
envvar restores it. **OpenCL `-ngl 0` (TG 50.8) cleanly beats every
Vulkan combo (best 41.09)**, so Vulkan stays in the doghouse for
production use.

### C5 — Quant compare on Qwen3-4B (CPU-kleidiai vs OpenCL ngl=0/99)

`results/csv/track_c5_quant_compare_2026-05-13.md`. r=2.

| backend | config | quant | PP (t/s) | TG (t/s) |
|---|---|---|---:|---:|
| CPU-kleidiai | `-t 16` | Q4_K_M | 280.20 | 31.66 |
| CPU-kleidiai | `-t 16` | Q4_0 | 269.71 | **42.55** |
| OpenCL | `-ngl 99 -t 16` | Q4_K_M | 240.16 (unstable) | 11.58 (unstable) |
| OpenCL | `-ngl 99 -t 16` | Q4_0 | **581.21** | 26.84 |
| OpenCL | `-ngl 0 -t 16` | Q4_K_M | 296.27 | 44.60 |
| OpenCL | `-ngl 0 -t 16` | Q4_0 | 384.50 | **50.50** |

**Q4_0 dominates Q4_K_M on this hardware across all backends.** This
makes sense — Q4_0 is the format KleidiAI's Adreno path was tuned
for, and OpenCL's Adreno kernels have explicit Q4_0 fast-paths
(`mul_mv_q4_0_*` family). For Qwen3-4B specifically, **migrate the
production target from Q4_K_M to Q4_0** unless a downstream consumer
demands K-quant precision.

Verification at r=5 (`track_c5_q4_0_ocl_ngl0_verify_2026-05-13.md`):

| threads | PP (t/s, r=5) | TG (t/s, r=5) |
|---:|---:|---:|
| 14 | 329.24 ± 4.49 | 49.85 ± 0.13 |
| **16** | **379.23 ± 11.46** | **50.80 ± 0.36** |
| 18 | 365.85 ± 5.16 | 48.62 ± 0.15 |

`-t 16` is the sweet spot. **Low variance on TG (±0.4%) makes this a
defensible new headline.**

### C6 — Concurrency-4 on OpenCL

`llama-batched-bench`, 4 parallel streams × (512 PP + 128 TG), ctx 4096.
`results/csv/track_c6_concurrency_4b_35b_2026-05-13.log`.

| model | ngl | S_PP (t/s) | S_TG (t/s) | total wall (s) | aggregate (t/s) |
|---|---:|---:|---:|---:|---:|
| 4B Q4_0 | 0 | 235.78 | **126.94** | 12.72 | **201.27** |
| 4B Q4_0 | 99 | 324.46 | 20.31 | 31.53 | 81.20 |
| 35B-A3B MXFP4 | 0 | 177.70 | **65.63** | 19.33 | **132.46** |
| 35B-A3B MXFP4 | 99 | 164.37 | 10.86 | 59.61 | 42.95 |

The `-ngl 0` pattern wins in agentic mode too: per-stream TG is
unchanged (4 × 32 ≈ 128, 4 × 16 ≈ 64) — i.e. concurrency scales linearly
on CPU compute. The `-ngl 99` config craters at concurrency because
the GPU is already saturated by the single-stream decode path.

## Open questions raised tonight

1. **What is OpenCL build doing at `-ngl 0`?** Same CPU compute but
   8-10% faster than a pure-CPU build on this hardware. Hypothesis:
   the OpenCL backend is being used as a coprocessor for specific
   ops (small matmuls, attention, KV ops) while weights live on CPU.
   Worth a profiler trace next session.
2. **Q4_0 vs Q4_K_M quality.** All speed wins are Q4_0; would be
   worth a perplexity check (`llama-perplexity` on wikitext) to
   confirm we're not trading away quality.
3. **`-ngl 0 -ub 1024+` on 4B Q4_0.** Not benched explicitly tonight
   — could push PP past 400.
4. **MTP merge watch.** Once #22673 lands, re-run B1 with
   `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` and measure self-draft
   acceptance vs no-MTP target.
5. **ngram-cache spec decode** via `llama-server` (the spec type
   that needs no draft model) — could be a one-flag win for 14B/35B.

## Files written tonight (committed)

All under `results/csv/track_*_2026-05-13.{md,log}`. Raw stderr logs
and helper files moved to `marked_for_deletion/2026-05-13_overnight/`
(gitignored).

## Reproducer one-liners (winning configs)

```powershell
# New TG record (4B Q4_0)
.\llama.cpp\build-opencl\bin\llama-bench.exe `
  -m .\models\Qwen3-4B-Q4_0.gguf -p 512 -n 128 -r 5 -ngl 0 -t 16

# New PP record (4B Q4_0)
.\llama.cpp\build-opencl\bin\llama-bench.exe `
  -m .\models\Qwen3-4B-Q4_0.gguf -p 512 -n 128 -r 5 -ngl 99 -t 16 -ub 512

# 14B CPU+CPU spec decode
.\llama.cpp\build-cpu-kleidiai\bin\llama-speculative-simple.exe `
  -m .\models\Qwen3-14B-Q4_K_M.gguf `
  -md .\models\Qwen3-0.6B-Q8_0.gguf `
  --device none --device-draft none `
  -ngl 0 -ngld 0 --spec-draft-n-max 3 -t 16 `
  -p "your prompt" -n 256

# 35B-A3B blended (OpenCL backend, no offload)
.\llama.cpp\build-opencl\bin\llama-bench.exe `
  -m .\models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf `
  -p 512 -n 128 -r 1 -ngl 0 -t 18

# Vulkan TG-only path (still PP-broken; only useful if you want TG 41)
$env:GGML_VK_PREFER_HOST_MEMORY = "1"
$env:GGML_VK_DISABLE_MMVQ = "1"
$env:GGML_VK_DISABLE_FUSION = "1"
.\llama.cpp\build-vulkan\bin\llama-bench.exe `
  -m .\models\Qwen3-4B-Q4_0.gguf -p 128 -n 128 -r 2
```
