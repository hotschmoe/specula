# Phase 1 — native spec decode (llama.cpp built-in)

End-to-end speculative decoding using llama.cpp's **built-in** `llama-speculative`
binary — no custom harness. This measures the two single-accelerator
spec-decode anchors directly:

- **CPU-only spec decode** — draft 0.6B + target 4B both on CPU.
- **GPU-only spec decode (config C3)** — draft 0.6B + target 4B both on
  the Adreno X2-90 (OpenCL build, `-ngl 99 -ngld 99`).

This is the first *real end-to-end* spec-decode number in this side
quest. Phase 0 and the phase1_anchors/phase1_c2 work measured per-pass
op-shape costs and feasibility; nothing had run the full draft→verify→
accept loop. `llama-speculative` does, with no orchestration code to write.

## Setup

- Hardware: Snapdragon X2 Elite Extreme, Windows 11 ARM64, **AC power**
  (battery status 2 = AC).
- Draft: `models/Qwen3-0.6B-Q8_0.gguf` (633 MB). Target:
  `models/Qwen3-4B-Q4_0.gguf` (2.37 GB).
- Builds: CPU = `llama.cpp/build-cpu/` (ARM64 NEON); GPU =
  `llama.cpp/build-opencl/` (Adreno OpenCL). Both build 9128, commit
  `856c3adac`. GPU device confirmed `Qualcomm(R) Adreno(TM) X2-90 GPU`,
  OpenCL 3.0 driver 863.0 — not a CPU fallback.
- Fixed prompt (one technical paragraph, 31 tokens), `-n 128`,
  `--temp 0` (greedy), `-c 2048`. CPU `-t 16`.
- `--spec-draft-n-max` sweep ∈ {2,4,8} (`--draft-max` renamed in this
  llama.cpp; see "Flag notes" below). `--spec-draft-n-min 0`.
- Spec runs 1× each (back-to-back, GPU timeboxed). Baselines via
  `llama-bench -r 5`.

## Results — the headline numbers

### Target-4B-alone TG (the speedup baseline)

| island | target-alone tg (t/s) | ms/token |
|--------|----------------------:|---------:|
| CPU    | **50.72**             | 19.7     |
| GPU    | **27.03**             | 37.0     |

(`llama-bench` tg128, r5. CPU stddev 0.67 t/s, GPU 0.16 t/s.)

### Spec decode end-to-end (`llama-speculative`)

| island | draft_max | spec tok/s | accept rate | accept/round | vs target-alone |
|--------|----------:|-----------:|------------:|-------------:|----------------:|
| CPU    | 2         | 46.74      | 65.2%       | 1.30         | **0.92×**       |
| CPU    | 4         | 54.59      | 54.9%       | 2.20         | **1.08×**       |
| CPU    | 8         | 39.20      | 33.6%       | 2.69         | **0.77×**       |
| GPU    | 2         | 9.73       | 54.0%       | 1.08         | **0.36×**       |
| GPU    | 4         | 11.00      | 39.5%       | 1.58         | **0.41×**       |
| GPU    | 8         | 9.76       | 21.8%       | 1.64         | **0.36×**       |

"vs target-alone" = spec tok/s ÷ that island's target-4B-alone tg.
accept/round = n_accepted ÷ rounds (rounds = n_drafted ÷ draft_max).

### Draft 0.6B per-step cost

| island | draft-alone tg (t/s) | ms/token |
|--------|---------------------:|---------:|
| CPU    | 156.93               | 6.37     |
| GPU    | 111.77               | 8.95     |

## Does native spec decode actually win?

### CPU — marginal win, and only at draft_max=4.

The best CPU config (draft_max=4) reaches **54.6 t/s vs 50.7 t/s
target-alone — a 1.08× speedup.** That is real but thin. draft_max=2
*loses* (0.92×) and draft_max=8 loses badly (0.77×).

Why so thin: the CPU target-4B itself is fast (50.7 t/s, 19.7 ms/token),
so the verify mini-prefill is cheap and there is little slow autoregressive
target work for the draft to hide. Meanwhile the 0.6B draft is *not free*
on the CPU — it contends for the same 16 threads as the target, so every
drafted token costs ~6.4 ms of CPU time whether accepted or not. At
draft_max=8 the accept rate collapses to 33.6% (5 of 8 drafted tokens
rejected on average): the wasted draft compute plus the wider, more
expensive verify batch overwhelm the gain. The k=4 sweet spot is the
only place the accepted-token yield (2.2/round) outpaces the draft+verify
overhead.

**Verdict (CPU): native spec decode barely wins — 1.08× at best, and
fragile.** It is the right call only if draft_max is tuned to 4; mis-set
it to 2 or 8 and you are slower than plain target decode. On this box,
CPU-only native spec decode is not a compelling lever.

### GPU (C3) — a clear, large loss.

Every GPU spec-decode config is **~0.36–0.41× of GPU target-alone tg** —
i.e. spec decode makes the GPU **2.5–2.8× slower** than just running the
4B target directly. C3 is a decisive loss.

The cause is the llama.cpp OpenCL backend's batched-forward (PP) path,
already documented in Phase 0 / phase1_anchors as software-bound: a
k-token verify forward costs a flat ~165 ms on the Adreno regardless of
k∈[2,16], ~10× its bandwidth floor. Spec decode's verify step *is* that
PP-shaped forward. So every spec round pays ~165 ms for verify — versus
the ~37 ms it costs to just decode one token autoregressively. With
accept/round only 1.1–1.6, each ~165 ms verify yields barely 1–2 useful
tokens: ~100+ ms per token, against 37 ms for plain TG. Spec decode on
the GPU trades the GPU's *healthy* mat-vec (TG) path for its *broken*
mat-mat (PP) path, and loses every time.

Note GPU spec tok/s rises slightly k2→k4 (9.7→11.0) then falls at k8 —
the verify-pass cost is nearly k-independent on the OpenCL plateau, so
more drafted tokens per fixed-cost verify helps until the accept rate
craters (21.8% at k=8).

**Verdict (GPU/C3): native spec decode loses hard — 2.5–2.8× slower than
target-alone. Do not use GPU-only spec decode with this OpenCL backend.**

## Cross-island picture

- **CPU is the better island for native spec decode by a wide margin** —
  not because its spec decode is great (1.08×) but because its baseline
  target decode (50.7 t/s) and its verify path are both healthy. The GPU
  is hamstrung by the OpenCL PP plateau.
- Best end-to-end number in this whole measurement is **CPU spec decode
  draft_max=4 at 54.6 t/s**. Plain CPU target-alone (50.7 t/s) is a close
  second and far simpler. The GPU configs (9.7–11.0 t/s) are not in
  contention.
- This is fully consistent with the Phase 0 / phase1_anchors finding that
  the GPU is the *wrong* island for the verify (PP) op. Native GPU-only
  spec decode forces verify onto the GPU and pays exactly the predicted
  ~165 ms-per-pass penalty. Phase 0's recommendation — keep verify off
  the GPU — is confirmed end-to-end here.
- The accept-rate trend is the expected one: deeper drafting raises
  accepted-tokens-per-round (CPU 1.30→2.20→2.69) but lowers the per-token
  accept *rate* (65%→55%→34%) as the draft runs further past where it
  diverges from the target. The optimum draft depth is shallow (k≈4) for
  this 0.6B/4B pair on greedy decoding.

## Caveats / honesty notes

- Spec-decode runs are **n=1 each** (timeboxed; another agent needs the
  GPU). The CPU k4 1.08× win is within plausible run-to-run noise of
  break-even — treat "CPU native spec decode ≈ break-even, best case
  ~1.1×" as the honest read, not a hard 1.08×. The GPU 2.5×+ loss is far
  too large to be noise.
- This is single-island spec decode (draft and verify on the *same*
  accelerator) — the CPU-only and GPU-only/C3 anchors. It does **not**
  measure heterogeneous configs (C1/C2: draft and verify on different
  islands with overlap). `llama-speculative` cannot place draft and
  verify on different backends, so C1/C2 still need the custom async
  harness described in `phase1_c2.md`. What this run *does* establish is
  the C3 anchor cleanly: GPU-only spec decode is a loss, so any
  GPU-involving win must come from heterogeneous overlap, not C3.
- Target 4B is Q4_0 (only Qwen3-4B GGUF on disk); draft is Q8_0. Greedy
  decoding (`--temp 0`), so accept rates reflect exact-match speculation.
- `--spec-draft-p-min` left at its default 0.75 — early-exit on
  low-probability drafts is active and partly explains why accept/round
  stays below draft_max.

## Flag notes (this llama.cpp build)

`--draft-max` / `--draft-min` were **removed** in build 9128. The current
flags are `--spec-draft-n-max` / `--spec-draft-n-min`. `--no-warmup` is
not accepted by `llama-speculative` (only `llama-cli`/bench). Draft GPU
offload uses `-ngld` / `--gpu-layers-draft`.

## Artifacts

- CSV: `gpu_npu_sidequest/results/phase1_native_specdecode.csv`
- Raw logs: `gpu_npu_sidequest/logs/specdecode_{cpu,gpu}_k{2,4,8}.log`,
  `baseline_{cpu,gpu}_target_tg.json`, `draft_{cpu,gpu}_tg.json`
- Run script: `gpu_npu_sidequest/scripts/run_phase1_native_specdecode.ps1`
