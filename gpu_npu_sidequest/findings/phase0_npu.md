# Phase 0 — NPU track: verify-shape crossover microbench

NPU half of the GPU↔NPU placement side quest. Measures the Hexagon NPU's
per-pass cost for a k-token batched forward of the Qwen3-4B bundle, to
locate where `verify` (PP-shaped, batched) flips from wanting-GPU to
wanting-NPU.

## TL;DR

- **The NPU bundle has exactly TWO pinned input-batch shapes: k=1 (AR1)
  and k=128 (AR128). There is no AR2/AR4/AR8/AR16/AR32 graph.** The
  requested k ∈ {1,2,4,8,16,32} sweep is **not reachable**. The
  reachable points are **k=1 and k=128** — the two endpoints of the
  curve.
- **k=1 (AR1):** 34.2 ms/pass, 29.2 tok/s.
- **k=128 (AR128):** 52.0 ms/pass steady-state → 2459 tok/s.
- **Per-pass cost is almost flat in k.** Going from k=1 to k=128 — a
  128× larger batch — costs only +52% wall time (34→52 ms). Cost per
  token collapses from 34.2 ms/tok to 0.41 ms/tok (**~84× cheaper per
  token**). The NPU is *strongly* compute-efficient at batch and is the
  right island for `verify` at large k. The crossover is not k-limited;
  it is shape-limited — see below.

## Bundle / venv / script

| item | value |
|---|---|
| bundle | `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/` (Qualcomm Genie w4a16, 4 parts, ~3.1 GB) |
| venv | `specula\.venv` — the only venv with `QNNExecutionProvider`; ORT-QNN **1.24.4** (matches QAIRT 2.42). `.venv-qairt` (ORT 1.22) and `.venv-ort21` (ORT 1.25) lack the QNN EP. |
| script | `npu_engine/bench_qwen3_4b_ortqnn.py` (chained 4-partition, AR128 swap-mode prefill + AR1 decode, IOBinding) |
| runs | `--pp-tokens 256 --tg-tokens 64` and `--pp-tokens 384 --tg-tokens 96`, both `--ar128-min-tokens 128`, AC, ctx tier cl512 |
| logs | `gpu_npu_sidequest/logs/bench_ortqnn_ar128_ar1.log`, `bench_ortqnn_pp384.log` |

No new QNN bundle was built; the existing harness was reused unchanged.

## The shape constraint (critical — this gates the side quest)

The NPU runs Qwen3-4B as **QNN context binaries with pinned input
shapes**. The bundle's `metadata.yaml` lists the available graphs per
4-part split. For each context tier `cl{512,1024,2048,3072,4096}` there
are exactly **two** autoregressive-width graphs:

- `ar1_cl{N}_*_of_4`  — input_ids shape `[1, 1]`   (single-token decode)
- `ar128_cl{N}_*_of_4` — input_ids shape `[1, 128]` (128-token prefill batch)

`AR128_BATCH = 128` is hard-coded in `qualcomm_qwen3_4b_oracle.py`; the
KV mask/rope helpers (`*_quantized_ar128`) all build `[1,1,128,ctx]`
tensors. **There is no graph for k = 2, 4, 8, 16, or 32.** A QNN context
binary cannot accept a batch dimension other than the one it was
compiled for — the shape is frozen at compile time on the HTP.

How specula reports an NPU "pp" number: it does **not** feed an
arbitrary k. It runs the **AR128 prefill graph** in 128-wide tiles
(`pp_tokens // 128` calls) and falls back to the AR1 graph for any tail
not divisible by 128. So the bundle's prefill is a *fixed 128-token
window*, not a variable-k forward.

**Consequence for the side quest:**

- The verify-shape microbench's intended k ∈ {1,2,4,8,16,32} sweep
  **cannot run on this NPU bundle as-is.** Only k=1 and k=128 are
  measurable.
- A `verify` of k≈3–8 speculative tokens (the useful spec-decode range)
  has **no native NPU graph**. Two ways to run it on the NPU, both with
  caveats:
  1. **Pad k up to 128** — feed the 3–8 verify tokens plus padding into
     the AR128 graph. Costs the full 52 ms AR128 pass regardless of k,
     i.e. you pay the k=128 price for k=3. Still only 52 ms, which is
     ~1.5× the k=1 cost — cheap in absolute terms, but wasteful.
  2. **Run k AR1 steps** — k sequential 34 ms decode passes. For k=4
     that is 4×34 = 137 ms, worse than one padded AR128 pass (52 ms).
- **So on this bundle, any NPU verify of k ≥ 2 should pad to the AR128
  graph** and costs a flat ~52 ms. AR1-per-token only wins at k=1.
- A *native* small-k verify graph (e.g. AR8) would need a **new QNN
  bundle compile** (qairt-converter + context-binary-generator with the
  batch dim pinned to 8). That is explicitly out of scope here
  ("do NOT build a new QNN bundle"), and is the gating item if the side
  quest wants a true NPU verify-cost-vs-k curve in the 3–8 range.

## Numbers (AC power, Snapdragon X2 Elite Extreme, Hexagon v81)

| k (op_shape) | graph | ms_per_pass | tok/s | ms per token | source |
|---:|---|---:|---:|---:|---|
| 1 (TG / AR1) | `ar1_cl512` | 34.2 | 29.2 | 34.2 | median of 160 decode steps |
| 128 (PP / AR128) | `ar128_cl512` | 52.0 | 2459 | 0.41 | steady-state median, 4 calls |

Raw AR128 per-pass samples: 72.5 (cold, 1st call run 1), 51.6, 52.7,
50.5, 61.8 ms. Excluding the single cold call, steady-state is ~50–62 ms,
median 52 ms. Raw AR1 decode: median 34.2 ms across 160 steps (warmup
step discarded), min ~25 ms, max ~42 ms.

These match the project's documented baselines (npu_engine AR128 PP
~2000–2300 t/s, AR1 TG ~27–30 t/s) — the harness is behaving as expected.

## Per-pass cost vs k

Only two points exist, but they are the curve's endpoints and the shape
is unambiguous:

```
ms/pass:   k=1 -> 34.2 ms        k=128 -> 52.0 ms      (+52% for 128x the work)
ms/token:  k=1 -> 34.2 ms/tok    k=128 -> 0.41 ms/tok  (84x cheaper per token)
```

**The NPU per-pass cost amortizes extremely well as k grows.** The pass
is dominated by fixed per-call cost (HTP dispatch + KV stream + the 4
inter-partition handoffs) — the actual matmul work for 128 tokens vs 1
token adds only ~18 ms. This is the signature of a **compute-bound,
batch-friendly** engine: at large k the NPU's quantized-matmul throughput
dominates and per-token cost craters. Exactly the behaviour the side
quest predicted for "large k → verify wants the NPU."

For the C1-vs-C2 crossover: the NPU verify cost is **flat ~52 ms for any
k from 2 to 128** (because k<128 must pad up to the AR128 graph). So the
NPU verify curve is a step function: 34 ms at k=1, then a flat 52 ms
shelf for all k∈[2,128]. Whether the GPU beats that depends entirely on
the GPU's verify-cost-vs-k curve (the GPU track of Phase 0) — but note
the NPU shelf is *very* low: 52 ms to verify up to 128 tokens. If the
GPU's k-token forward exceeds 52 ms anywhere in k∈[2,128], the NPU wins
verify there outright.

## Blockers / caveats

- **k ∈ {2,4,8,16,32} are not natively measurable** — no QNN graph for
  those batch widths. This is the headline finding and it gates the
  microbench's design: the NPU side of the verify curve is two points
  (k=1, k=128) plus a flat 52 ms shelf in between (via AR128 padding),
  not a smooth sweep. A native small-k curve needs a new bundle compile.
- cl512 context tier: total KV slots capped at 511, so prefill+decode
  must fit in 511. Did not constrain these short runs. Larger ctx tiers
  (1024–4096) exist with the same AR1/AR128-only shape story.
- Power: WMI `PowerOnline`/`DischargeRate` returned unavailable on this
  box (`import wmi` failed in `.venv`); ran with `--skip-power-check` on
  confirmed AC. No J/tok measured — out of scope for this microbench.
- The first AR128 call carries ~20 ms residual warmup even after the
  harness's 1-call warmup; steady-state numbers exclude it.
- Session swap (AR128 load → teardown → AR1 load) costs ~17–18 s
  one-shot — irrelevant to per-pass cost (a warm sidecar pays it once),
  excluded from all per-pass numbers above.
