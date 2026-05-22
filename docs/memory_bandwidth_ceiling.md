# Memory bandwidth ceiling — why CPU-only LLM perf can't touch 228 GB/s

Reference doc. Underpins the W4 heterogeneous-orchestration motivation
in `roadmap.md`: LLM token generation is bandwidth-bound, and the X2
Elite's headline 228 GB/s is a *whole-SoC* number — no single compute
island can reach it alone.

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

## Why 156 GB/s, not 228 GB/s

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
  Concurrency is the only way to spend the remaining ~72 GB/s.

Open question for W4.a: when CPU + GPU + NPU contend for the same
fabric, does aggregate throughput actually approach 228, or does
controller contention claw back most of the headroom? The sweep above
is the CPU-alone anchor; W4.a needs the equivalent GPU-alone and
NPU-alone numbers, then the all-three-concurrent number, to know how
much of the 72 GB/s gap is real and recoverable.

## Reproduce

Build + sweep live at `C:\Users\hotschmoe\Documents\GitHub\STREAM`
(MSVC ARM64, `build_msvc.bat`). Re-run: `$env:OMP_NUM_THREADS=12;
.\stream.exe`.
