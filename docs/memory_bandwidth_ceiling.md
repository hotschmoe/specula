# Memory bandwidth ceiling — per-island bandwidth vs the 228 GB/s SoC fabric

Reference doc. Underpins the W4 heterogeneous-orchestration motivation
in `roadmap.md`: LLM token generation is bandwidth-bound, and the X2
Elite's headline 228 GB/s is a *whole-SoC* number — no single compute
island (CPU / GPU / NPU) can reach it alone. All three single-island
anchors are now measured; see the summary table under "Implication".

## Measured: CPU memory bandwidth (STREAM, 2026-05-22)

STREAM benchmark (`jeffhammond/STREAM`, patched for MSVC/ARM64),
80M-element arrays = 1.8 GiB working set, native ARM64 build with
`/openmp`. Thread-count sweep, Triad kernel (`a = b + s*c`):

| Threads | Triad     |
|---------|-----------|
| 1       | 66 GB/s   |
| 2       | 103 GB/s  |
| 6       | 110 GB/s  |
| 8       | 115 GB/s  |
| 12      | 117 GB/s  |
| 16      | 117 GB/s  |
| 18      | 114 GB/s  |

Bandwidth **saturates at ~117 GB/s by 8–12 threads** — adding cores
past that does nothing. A single thread already pulls 66 GB/s.

Two corrections to read this number correctly:

1. **STREAM under-counts real DRAM traffic by ~33%.** Triad touches 3
   arrays in the program's view (24 B/elem), but writing the
   destination triggers a read-for-ownership of that cache line first
   — real DRAM traffic is 4 arrays (32 B/elem). So true CPU-cluster
   bandwidth ≈ 117 × 32/24 ≈ **156 GB/s**.
2. The reported 117 is what you'd compare against other STREAM
   numbers; 156 is what's actually crossing the memory bus.

## Measured: GPU memory bandwidth (Adreno X2-90, 2026-05-22)

Custom STREAM-style OpenCL benchmark against the native Adreno OpenCL
3.0 driver, 1 GB buffers (>> the 1 MB GPU L2 cache), best of 20 timed
samples. Device confirmed as `Qualcomm(R) Adreno(TM) X2-90 GPU` — not a
CPU fallback. Results stable across 256 MB / 512 MB / 1 GB buffers.

| Kernel | Bandwidth |
|--------|-----------|
| Copy   | 174 GB/s  |
| Scale  | 176 GB/s  |
| Add    | 157 GB/s  |
| Triad  | 159 GB/s  |

Same addressed-bytes accounting as STREAM (write-allocate not counted),
so the Triad figure is directly comparable to the CPU's 117 GB/s. The
GPU streams meaningfully faster than the CPU cluster — its wider path
into the fabric is the reason GPU offload exists. True DRAM traffic is
higher again after write-allocate, same caveat as the CPU number.

## Estimated: NPU effective bandwidth (Hexagon HTP, 2026-05-22)

No synthetic STREAM-equivalent exists for the Hexagon NPU, so this is
*derived* from LLM decode, which is bandwidth-bound: each token streams
the full quantized weight set from LPDDR5X once.

    effective_BW ≈ (weight bytes per token) × (decode tokens/sec)

For Qwen3-4B w4a16 (Qualcomm bundle): ~2.41 GB of transformer-block +
lm_head weights stream per token — the 778 MB embedding table is a
single-row gather, so it is excluded. At the measured ORT-QNN decode
rate of 27.25 t/s:

    2.41 GB × 27.25 /s ≈ 66 GB/s

Across the spread of measured decode rates (20–29 t/s) this lands at
**~50–70 GB/s**. This is a **lower bound**, not a peak: the decode step
also spends time on compute (w4 dequant, INT matmul) and per-op
dispatch overhead, both of which deflate the derived figure. The NPU's
true DRAM port is higher than ~66 GB/s by that compute+overhead tax — a
synthetic HTP copy benchmark would be needed to measure the peak
directly (deferred; see W4.a below).

## Why no island reaches 228 GB/s

The 228 GB/s is the SoC fabric peak, shared across CPU + GPU + NPU —
not a CPU-only number. The CPU cluster has its own narrower port into
the memory fabric. This is the same story as Apple Silicon: M1 Max
advertises 400 GB/s, but CPU-only STREAM tops out around 240 — you
only approach the advertised figure when the GPU/NPU are pulling
concurrently too.

156 GB/s of real CPU traffic against a 228 GB/s aggregate is ~68% —
completely normal for this class of unified-memory SoC. The 228 figure
isn't wrong; it's just not reachable by a CPU memcpy benchmark alone.
The way you'd actually exercise it is a mixed workload — e.g.
llama.cpp with GPU offload, where CPU and Adreno hammer memory at the
same time.

Bottom line: 228 GB/s = peak the whole chip can do; ~117 GB/s STREAM
(~156 real) = what the CPU cores alone can pull. Both are true.

## Implication for LLM performance (the W4 thesis)

Token generation is bandwidth-bound: each decoded token streams the
full weight set from LPDDR5X once. Decode throughput is therefore
capped by *achievable* bandwidth, not FLOPs.

- A CPU-only decode path is hard-capped near the ~156 GB/s CPU-cluster
  ceiling, regardless of how many threads or how fast the cores are.
- **MAX POTENTIAL throughput requires saturating the full 228 GB/s,
  and that can only happen by running more than one island at once** —
  CPU + Adreno + Hexagon pulling from the shared fabric concurrently.
- This is the quantitative case for W4 heterogeneous orchestration:
  the win isn't just hiding latency, it's that the bandwidth budget a
  single island can claim is structurally less than the chip's total.
  Concurrency is the only way to spend the remaining headroom.

### Per-island anchors (single island, nothing else running)

| Island | STREAM-style Triad | Kind of number |
|--------|--------------------|----------------|
| CPU cluster        | 117 GB/s  | measured; ~156 GB/s real DRAM after write-allocate |
| GPU (Adreno X2-90) | 159 GB/s  | measured; true DRAM higher (write-allocate uncounted) |
| NPU (Hexagon HTP)  | ~66 GB/s  | decode-derived **lower bound**, not a synthetic peak |
| SoC fabric peak    | 228 GB/s  | datasheet (whole chip) |

Two takeaways for W4:

- **No single island reaches 228 — but the gap is smaller than the
  CPU-only view suggested.** The GPU is the strongest single consumer
  (~159 GB/s Triad, true DRAM higher still), the CPU mid (~117/156),
  the NPU lowest *as measured* (~66, but that is a decode lower bound
  carrying compute+dispatch overhead, not the HTP's real port width).
- **The per-island demands do not sum freely.** CPU real traffic
  (~156) alone is ~68% of the fabric; the NPU's effective ~66 nearly
  exactly fills the rest (156 + 66 ≈ 222, just under 228). So to
  first order CPU + NPU concurrent decode lands right at the ceiling —
  a mostly-idle fabric does take a second consumer almost for free —
  but a *third* concurrent consumer gains little, and the GPU alone is
  already large enough that GPU + anything will contend. Heterogeneous
  concurrency is a real win, but the fabric saturates after roughly
  two busy islands.

Still open for W4.a: the actual **all-concurrent** measurement — run
CPU + GPU + NPU memory-bound workloads simultaneously and confirm
whether aggregate throughput reaches ~228 or whether memory-controller
contention claws back a chunk before then. The single-island anchors
above predict near-saturation at two islands; W4.a is the experiment
that confirms where contention actually bites. A synthetic HTP copy
benchmark (to replace the NPU lower bound with a true peak) is the
other open measurement.

## Reproduce

- **CPU** — STREAM at `C:\Users\hotschmoe\Documents\GitHub\STREAM`
  (MSVC ARM64, `build_msvc.bat`). Re-run: `$env:OMP_NUM_THREADS=12;
  .\stream.exe`.
- **GPU** — custom OpenCL benchmark at
  `C:\Users\hotschmoe\Documents\GitHub\adreno-bw-bench`
  (`build.bat`, then `bwbench.exe 1024`).
- **NPU** — derived, not a standalone benchmark: effective bandwidth =
  weight-bytes-per-token × decode t/s, using the ORT-QNN Qwen3-4B
  numbers in `results/csv/` and
  `docs/qwen3_4b_baseline_all_backends.md`.
