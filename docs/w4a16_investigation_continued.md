# W4A16 investigation — continued

Phase 5.5 Lever C **has not actually closed**. `docs/w4a16_investigation.md`
declared it NEGATIVE as a product after the w8a16-local / w4a16-local-pr
AC sweeps, but the exhaust path missed several leads that *individually*
could flip the outcome and that *collectively* tell us everything we need
to know about w4a16 on Hexagon v81 before graduating to Qwen3.5.

This doc is the "chase every remaining lead" plan. Goal: either land a
variant that beats Lever B's **18.12 t/s fp16 AC baseline**, or close each
door permanently with evidence so Qwen3.5 / Qwen3.6 graduation starts
from a clean slate instead of a question mark.

Target hardware is Snapdragon X2 Elite (Hexagon v81 HMX). ARM + on-device
LLM is the frontier for edge inference; every lead we close here retains
value for future models on the same silicon family.

## What we know now (inputs to this plan)

### The side-quest DID run and ripped — `results/qwen3_4b_genie_w4a16_probe.md`

Qualcomm's shipping Qwen3-4B w4a16 Genie bundle, loaded via **our**
ORT-QNN 1.24.4 runtime (not Genie):

| stage | measured |
|---|---:|
| load (session init) | ~1 s |
| embedding lookup (part 1) | 0.04 ms median (20 iters, AC) |
| 12 attention layers (part 2) | **7.22 ms median** (7.12–7.42 range) |
| per-layer cost | **0.60 ms/layer** |

Extrapolation to our 28-layer Qwen3-0.6B: **~17 ms/step projected** if we
match Qualcomm's IO convention. Our current w4a16-local lands at 21-24
ms/step with fp32 preserved IO. **The ~6 ms gap is memory-bandwidth on
the past_kv boundary** — Qualcomm ships uint8 past_kv (1 B/elem), we
ship fp32 (4 B/elem). Per step we shove ~4× more bytes across the HTP-
DDR boundary than Qualcomm does.

### Critical takeaways

1. **ORT-QNN 1.24.4 accepts `TensorProto.UINT8` + `TensorProto.UINT16`
   wrapper inputs, no flag gymnastics.** The whole premise that drove us
   to `--preserve_io_datatype` was wrong — ORT-QNN does not require fp32
   IO at EPContext boundaries.
2. **Qualcomm's IO dtype convention, validated end-to-end on X2E:**
   input_ids int32, hidden/mask/cos/sin/logits uint16 per-tensor
   asymmetric, past_key_N / past_value_N uint8 per-layer (offset=-128
   symmetric-shifted), scale varies per layer.
3. **Speculative-decode projection from the probe** (step=17 ms,
   verify=157 ms, accept≈0.80):

   | k | committed | round wall | projected t/s |
   |---|---:|---:|---:|
   | 2 | 2.6 | 157 ms | 16.6 |
   | 4 | 4.2 | 157 ms | 26.8 |
   | 6 | 5.8 | 157 ms | 36.9 |
   | 8 | 7.4 | 157 ms | 47.1 |
   | 10 | 9.0 | **170 ms** (NPU-bound) | 52.9 |

   **w4a16's real leverage isn't per-call speed at k=2.** At k=2 we're
   verify-bound regardless. The value appears at k=4-8 where fp16
   (~40-65 ms/step) would become NPU-bound but w4a16 (~17 ms/step) stays
   hidden under verify.

### The bottleneck diagnosis — NPU 48% / CPU 100%

Observed during the w8a16-local AC sweep. Round wall = `max(k × step,
verify)`. At k=2 that's `max(48, 157) = 157 ms` → NPU is idle ~70% of
each round. Making NPU faster (w4a16) at k=2 buys ~nothing; PTQ accept-
rate noise buys straight losses. This is why w4a16-local-pr (21 ms/step)
underperformed w8a16-local (24 ms/step) in the sweep.

**Consequences:**
- At k=2, w4a16 is a non-lever on 0.6B unless we also attack verify.
- At k=6-8, w4a16 becomes the *enabler* — fp16 would be NPU-bound,
  w4a16 stays hidden, larger draft surfaces amortize target verify.
- Raising draft accept rate without a giant per-step cost increase is
  the other active lever (bigger draft, GPTQ-grade quant, QAT).

### What's already closed (don't re-litigate)

| lever | result | evidence |
|---|---|---|
| AI Hub w4a16 cloud compile | BLOCKED by preserve_io_datatype bug | `results/aihub-compile-jg93r1jqg-*` |
| local QAIRT 2.42 `tf` / `tfe` / `mse` act-cal | cos 0.33–0.36 (V-proj collapse) | shotgun session 18 |
| local QAIRT per-row w4 (`-pr`) | cos 0.888, soft pass, loses AC sweep −33% | shotgun + AC sweep |
| local QAIRT w8a16 | cos 0.963, full pass, loses AC sweep −29% at k=2 | shotgun + AC sweep |
| local QAIRT CLE | byte-identical to baseline (no-op on MatMul) | shotgun md5 |
| fp16-local rebuild | cos 0.9999, 49.8 ms/step (2× slower, no payoff) | session 16 |

**Root-cause diagnosed:** layer-1+ V-projection weights collapse under
w4 per-tensor/per-row PTQ (session-17 differential probe, fp16 oracle vs
w4a16 binary: V-tensor cos 0.957 → 0.130 from layer 0 → layer 1, stays
<0.2 through layer 27; K-tensor degrades gracefully via rotary smoothing).

## The remaining lead map

Three orthogonal axes. Each lead lives on one axis; chase every lead on
every axis or document why it's killed.

### Axis A — More w4a16 quantization methods (fix V-proj collapse)

Goal: get cos ≥ 0.95 at w4 precision, then measure.

#### A.1 Per-tensor quantization overrides (V/O pinned to w8)

**Mechanism.** `qairt-quantizer --quantization_overrides <json>` pins
specific tensors to a chosen bitwidth. We know exactly which tensors fail
(28 × V-projection weights, probably also 28 × O-projection after the
attention output — session-17 probe didn't separate those). Pin them to
w8; everything else stays w4.

**Cost.** ~2 h author the JSON (script that emits the override list from
the Qwen3-0.6B layer pattern) + ~80 s local QAIRT compile per iteration.
~0.5 session total.

**Expected outcome.** `w4a16-local-pr` hit cos 0.888 with just per-row
granularity on all weights. Pinning V+O to w8 should close the remaining
gap to ≥ 0.95 (they're the exact layers per the differential probe).

**Transfer.** `--quantization_overrides` JSON format is QAIRT-generic,
works on Qwen3.5 / Qwen3.6 / any transformer. The **diagnostic pattern**
(differential probe vs fp16 oracle → localize to V/O → pin those layers)
is the methodology that transfers even more broadly.

**Kill criteria.** cos < 0.95 → advance to A.2. cos ≥ 0.95 → run AC sweep
to measure; if sweep beats 18.12 t/s → lever C closes POSITIVE. If sweep
loses, we've verified the 0.6B-draft ceiling on PTQ and ruled out "the
weight collapse was the lever."

#### A.2 Full-quant IO (match Qualcomm's convention)

**Mechanism.** Drop `--preserve_io_datatype` entirely. Let qairt-quantizer
pick uint8 past_kv + uint16 everything-else, matching Qualcomm's
reference bundle. Our existing x86 pipeline already passes encodings.json
alongside the .bin; we just need runtime to quant/dequant at session
boundaries using those scales (we already have the `QuantSpec` layer
from session 15).

**Cost.** ~0.5 session. Remove the preserve flag from the x86 recipe;
runtime-side plumbing (`quant_to_uint8`, `dequant_from_uint8` for past_kv
alongside existing uint16 helpers) is ~30 LOC on top of what's already in
`npu_load_qwen3_bin.py`. Recompile ~80 s.

**Expected outcome.** Per-step drops from 24 ms (fp32 preserved IO) to
~17 ms (probe projection). At k=2 verify-bound, this changes nothing
(still 18 t/s max). At k=4-8, the NPU stays hidden longer — real
throughput unlock if combined with A.1 or A.3.

**Transfer.** Huge. This is Qualcomm's blessed IO convention and the
same one future AI Hub bundles will ship. The QuantSpec / encodings-
sidecar plumbing we've already built is the exact thing every future
w4a16 or w8a16 binary will use.

**Kill criteria.** If per-step doesn't drop to ≤19 ms, something else is
bandwidth-bound (logits at uint16 still 4-byte vocab × 151936 = not
small). Profile, investigate, report.

#### A.3 AIMET pre-quantization with SmoothQuant / AWQ on V/O

**Mechanism.** AIMET `QuantizationSimModel` emits QDQ-annotated ONNX
with explicit per-tensor scales. Crucially it supports **SmoothQuant**
(shift activation magnitude into weight) and **AWQ** (weight-only scale
search) — proven techniques for LLM PTQ that standard qairt-quantizer
doesn't do. Feed the AIMET-annotated ONNX to local QAIRT;
qairt-quantizer treats the encodings as fixed instead of re-calibrating.

**Cost.** ~1-2 sessions. AIMET install + learning curve. Then a pipeline
step to consume our calibration bundle and emit QDQ ONNX. x86 side.

**Expected outcome.** SmoothQuant specifically targets the activation-
magnitude explosion that causes V-projection collapse in small LLMs
(it's literally the canonical failure mode AWQ/SmoothQuant papers
address). Expect cos ≥ 0.95 at pure w4 without per-row or V/O pinning.
If it works, we keep the 32% smaller binary that `-pr` gave us *and*
gain the accept-rate recovery.

**Transfer.** Maximum. AIMET is model-agnostic; the same pipeline runs
on Qwen3.5 / Qwen3.6 / any HF-exportable transformer. SmoothQuant/AWQ
are the two most cited PTQ techniques in the LLM quantization
literature — having them in-house is a durable capability.

**Kill criteria.** Cos < 0.95 after SmoothQuant + AWQ both tried →
V-projection precision is a fundamental 0.6B property, not a method
gap. Close A.3 negative, document in Qwen3.5 graduation notes.

#### A.4 `qairt-accuracy-debugger` (localize the exact failing op)

**Mechanism.** Linux-only QAIRT tool (use WSL2 on the x86 box). Runs
the DLC op-by-op and compares against a reference ONNX runtime, flags
the first op where numerical drift exceeds threshold. Authoritative
localization instead of "differential probe points at V-projection."

**Cost.** ~0.5 session to get WSL2 + QAIRT 2.42 Linux SDK running, plus
one debugger run per variant.

**Expected outcome.** Names the first failing op. If it's not in the
V-projection chain, our session-17 diagnosis was wrong and A.1 will
miss. If it confirms V-projection, A.1's override list is provably
complete.

**Transfer.** High — the tool works on any QAIRT DLC. Becomes the
default triage step for any future PTQ correctness failure.

**Kill criteria.** n/a. This is a diagnostic; its output flows into A.1
or A.3. If neither A.1 nor A.3 has been run yet, do A.4 *first* so we
over-specify the override list correctly on the first try.

#### A.5 GPTQ via AutoGPTQ → ONNX → QAIRT

**Mechanism.** AutoGPTQ is a popular PyTorch-side GPTQ implementation
(second-order weight quantization, widely used on LLaMA/Qwen/Mistral).
Run it on the Qwen3-0.6B PyTorch checkpoint → export GPTQ-quantized
weights → re-embed as ONNX with frozen scales → feed to QAIRT
converter. Alternative to AIMET for the same "pre-quantized ONNX"
shape.

**Cost.** ~1 session. GPTQ runs in ~10 min on a single GPU. ONNX
export of a GPTQ-quantized PyTorch model needs a custom path (GPTQ's
packed int4 weights aren't native ONNX).

**Expected outcome.** Similar ceiling to AIMET-AWQ. Often slightly
better (GPTQ's second-order approach captures inter-weight correlation
that AWQ doesn't). Widely published numbers for Qwen models so we have
external reference.

**Transfer.** High — GPTQ is a dominant technique in the open LLM
community. Skills and pipeline transfer to Qwen3.5 unchanged.

**Kill criteria.** Combined with A.3. If AIMET-AWQ already lands ≥0.95
we can skip GPTQ; if AIMET-AWQ fails, GPTQ is the sibling attempt.

#### A.6 QAT (quantization-aware training) — nuclear, last resort

**Mechanism.** Fine-tune Qwen3-0.6B with simulated quantization
(QAT-aware forward pass, straight-through-estimator backward) for a
short run (1-10k steps on a small Qwen-flavored code+chat corpus).
Produces weights that *know* they'll be quantized and route error
accordingly.

**Cost.** ~2-3 sessions + single-GPU compute. Dataset curation
non-trivial. Output is a *new* 0.6B checkpoint, not a quant of the
stock release.

**Expected outcome.** The literature says QAT recovers the
PTQ-correctness gap reliably. For a 0.6B model it's tractable on
consumer GPU.

**Transfer.** The *pipeline* transfers; the *output* (a fine-tuned
QAT checkpoint) is model-specific. Would have to re-run for Qwen3.5.

**Kill criteria.** Only attempt if A.1–A.5 all fail. QAT is expensive
and specific.

### Axis B — Bigger draft (make the NPU the bottleneck)

Goal: move the `max(NPU, verify)` equation so NPU draft time approaches
or equals verify time. Then w4a16 / w8a16 per-step savings translate
directly to throughput.

#### B.1 Qwen3-1.7B fp16 drafter

**Mechanism.** Compile Qwen3-1.7B through our existing fp16 pathbmask
pipeline (no new quant risk). 1.7B / 0.6B = ~2.8× weight bandwidth →
expect ~120-150 ms/step NPU draft. Ratio vs Qwen3-8B target = 4.7×,
well above the 2× spec-decode viability floor.

**Cost.** ~1 session. One AI Hub fp16 compile (Qwen3-1.7B export
exists in Phase 5 scripts, just not yet compiled). One AC sweep.

**Expected outcome.** Accept rate should climb from ~71% (w8a16 0.6B)
to ~85-90% (Qualcomm's published Qwen accept table, 1.7B draft).
Round wall `max(2 × 130 ms, 157 ms) = 260 ms`. Committed per round
`1 + 2 × 0.87 = 2.74`. Throughput ≈ 10.5 t/s. **Worse than Lever B at
k=2 by itself, but it opens k=4+** where committed grows without
verify cost growing much.

**Transfer.** Directly to Qwen3.5 — drafter-size selection is the
same question on a new model family.

**Kill criteria.** If accept rate doesn't climb ≥10 pp vs 0.6B, 1.7B
as drafter is a loss and B.2 / B.3 become moot.

#### B.2 Qwen3-1.7B w4a16 (the point where w4a16 matters)

**Mechanism.** After B.1 lands correctness, repeat the local QAIRT
pipeline for Qwen3-1.7B. Apply whichever w4a16 recipe won Axis A.
Expected per-step: ~55-65 ms (w4a16's ~2.3× speedup on 1.7B).

**Cost.** ~1 session. Toolchain reuses end-to-end.

**Expected outcome.** Round wall = `max(2 × 60 ms, 157 ms) = 157 ms`
still. Committed = `1 + 2 × 0.87 = 2.74`. Throughput ≈ 17.4 t/s.
Matches Lever B at k=2. The win stacks at k=4: committed = `1 + 4 ×
0.80 = 4.2`, round wall `max(4 × 60, 4 × 37 + 157) = max(240, 305) =
305`. Throughput ≈ 13.7 t/s. Hmm, that's worse than k=2 because
verify scales with batch. Model needs refinement once we have real
numbers — the point is this configuration *activates w4a16 as a
lever* in a way 0.6B never did.

**Transfer.** Full. Proves or disproves the "bigger draft + w4a16
together" thesis for ARM edge LLM inference.

**Kill criteria.** If the k-sweep never beats Lever B's 18.12, close
B.2 negative and the 0.6B/fp16/k=2 operating point stands as the X2E
Qwen3 spec-decode high-water mark.

#### B.3 Qwen3-4B via Genie runtime (side-quest extension)

**Mechanism.** Qualcomm's Qwen3-4B Genie bundle already runs at what
we projected to ~19 t/s standalone. Not viable as a draft of Qwen3-8B
(4B / 8B = 2×, right at the no-go boundary). Viable as a draft of
**Qwen3-30B** or **Qwen3.5-32B** — target ratio 7.5×, accept rate
should be >90%. Requires swapping our outer loop's verify target
from Qwen3-8B.gguf to a larger GGUF on CPU *or* on OpenCL.

**Cost.** ~2-3 sessions. Download/bench Qwen3-30B GGUF on this rig.
Rebuild outer loop to drive 4B draft via Genie SDK (new sidecar,
not ORT-QNN). Genie has a Python binding but it's distinct from
ORT's API.

**Expected outcome.** If Qwen3-30B runs at ~8-12 t/s alone on our
CPU (plausible at Q4_K_M given 8B runs ~26 t/s), the verify budget
per round is ~120-150 ms per token × `k+1`. Draft at ~22 ms/step
via Genie w4a16 stays hidden under verify for k up to ~6. Committed
per round at k=4 with 90% accept = 4.6. Throughput ≈ 4.6 /
(max(88, 600-800)) ≈ 6-7 t/s — worse than our current number
because verify dominates. Only worth it if we pair with a faster
30B target path.

**Transfer.** Lower. Genie is a different runtime; the sidecar we'd
build here doesn't map 1:1 to ORT-QNN work elsewhere. But it's the
vendor-blessed path, so a one-time investment to have a Genie
sidecar in the toolbox is durable.

**Kill criteria.** If we don't have a Qwen3-30B target baseline
landed first, B.3 is out of scope. Needs Phase 6 target-model work
as a prereq.

### Axis C — Attack the verify side (so draft-side speedups register)

Goal: if verify is 157 ms and draft is 24 ms, even a 10× draft speedup
leaves throughput at k=2 unchanged. Trim verify, and every Axis A/B
gain materializes.

#### C.1 Target on OpenCL (revisit Phase 2 negative)

**Mechanism.** Phase 2 found tgt=OpenCL drafting was slower due to
kernel-launch overhead at tiny verify batches. That was a
*draft-side* problem. For large verify batches (k=8, 9-token
verify), OpenCL's PP throughput (2674 t/s PP512 on 0.6B Q8_0 from
Phase 1) might actually flip. Haven't measured target-side
specifically on an 8B target with k-scaled verify batches.

**Cost.** ~0.5 session. Same sweep harness, flip `--target-backend
opencl`. Phase 2 CSVs cover a narrow config; we have the data to
extend.

**Expected outcome.** Unknown. Worth checking even if just to
document that OpenCL target is (or isn't) verify-faster than CPU at
k=8 on this rig.

**Transfer.** Direct. Every X2E deployment will have this question.

**Kill criteria.** If k=8 OpenCL target is >20% slower than CPU
target → OpenCL target stays off → C.1 closes. If faster → we have a
new combined baseline and every Axis A/B result should be re-run
against it.

#### C.2 ORT-QNN target (8B on NPU)

**Mechanism.** Compile Qwen3-8B to NPU fp16/w8a16. We haven't
attempted this — Phase 5 focused on drafter compile. Qwen3-8B at ~8×
the 0.6B memory would probably need the 4-partition split Qualcomm
uses for Qwen3-4B. Non-trivial.

**Cost.** ~3-4 sessions. Partition scheme + chained runtime session
management + AI Hub compile (which will almost certainly hit the
same bugs that blocked w4a16).

**Expected outcome.** If it lands at ~80 ms/step (Qualcomm-ish
scaling from 4B/22ms), verify per round shrinks dramatically.

**Transfer.** This is Phase-6-scale work. Listed here for
completeness; not in the Phase-5.5 close-out scope.

**Kill criteria.** Out of scope unless Phase 6 starts.

#### C.3 Tree verify (already listed as R3, not Phase 5.5)

Skip. Requires custom verifier, Phase 6+.

### Axis D — Architectural alternatives (different drafter, not different quant)

Goal: when PTQ ceiling is real and bigger-draft math doesn't flip, change
the drafter *architecture* so accept rate climbs independently of quant
choices. Compounds cleanly with any Axis-A w4a16 win.

#### D.1 EAGLE-3 drafter on NPU

**Mechanism.** EAGLE-3's drafter is a tiny learned head (2-3 transformer
layers + LM head, ~100-200M params) trained to predict the target's next
token *from the target's own hidden states*. Published accept rates for
LLaMA-2/3 are 85-92% — 10-20 pp above vanilla draft-model spec. For a
Qwen3-8B target the corresponding head would be ~150M params.

**w4a16 compounding.** Direct and strong:
- The accept-rate floor rises from ~71% (our w8a16-local) / ~55%
  (w4a16-local-pr) to ~85-90% regardless of drafter quant noise. **PTQ
  accept-rate tax gets absorbed into the trained head's margin.**
- Drafter is ~3-4× smaller than Qwen3-0.6B, so per-step NPU cost is
  lower even at fp16. w4a16 on the EAGLE head compounds weight-bandwidth
  savings on top of the tiny base — but the absolute savings are small
  in ms (you can't shave much from already-fast).
- **The real question is accept rate, not ms/step.** EAGLE's edge is
  that it trains against the target, so the V/O-projection precision
  sensitivity we saw on Qwen3-0.6B may not translate — the EAGLE head
  is co-adapted to the target's distribution.

**Cost.** HIGH — ~4-6 sessions.
- Target hidden-state exposure: llama-server `/completion` returns only
  tokens, not residuals. Shared prerequisite with **DFlash+DDTree
  (Phase 4)** and **tree-verify (W2.d)** — the custom multipath-capable
  verifier is tracked as B20 in `docs/roadmap.md`. All three levers pay
  its cost once; it's not EAGLE-specific.
- EAGLE-3 head training: needs a Qwen3-8B-specific head, not
  off-the-shelf. Single-GPU training run on a code+chat corpus ~1-3
  days. Or: check HuggingFace for pre-trained EAGLE-3 Qwen3-8B heads
  (unlikely to exist; Qwen3 is recent).
- ONNX export of the EAGLE head + compile for NPU.
- Outer-loop rewrite: the hand-off from target to drafter changes
  shape (pass hidden states, not just tokens).

**Transfer.** EAGLE-3 method is target-agnostic but head-specific per
target. Qwen3.5-8B cutover means re-training the head. Pipeline
transfers; artifact does not.

**Kill criteria.** Phase 3 of the original plan (a viability probe, not
an anchor). Reopen after Phase 5.5 w4a16 close; treat as Phase 6
material. Listed here for the *compounding* question: yes, it
compounds, and it's the single biggest-potential-win lever remaining
because it attacks the accept-rate axis w4a16 can't move.

#### D.2 Medusa / speculative-heads variants

Similar to EAGLE-3 but multiple parallel heads predicting positions
2..k+1 instead of sequential drafting. Lower quality per head, larger
draft surface. Phase 6+ consideration; sibling to D.1. Same cost
ballpark, same "doesn't help us answer w4a16 today" verdict.

#### D.3 Self-speculation / LayerSkip

Target skips layers to self-draft. Zero-size drafter, zero quant
question. Accept rates ~60-70% in published numbers. Phase 6+;
listed for completeness since it sidesteps the entire drafter-quant
problem but requires target-side modifications we haven't explored.

## Ordered investigation plan

Constraints: each step ends with a commit + doc update (per
`feedback_milestone_discipline`). Battery→AC swings dominate signal —
every measurement is AC + idle.

### Phase 5.5.1 — Validate the Qualcomm-IO runtime path (1 session)

**Lead A.2 is the cheapest, highest-certainty lead remaining.** Execute
first because its result makes every subsequent lead faster to measure.

1. Remove `--preserve_io_datatype` from local QAIRT recipe. Recompile
   pathb w4a16 with `--apply_algorithms pr` (the variant closest to
   Qualcomm's convention). ~80 s.
2. Extend `QuantSpec` + `quant_to_uint8` / `dequant_from_uint8` to cover
   past_kv. ~50 LOC.
3. Wire runtime probe to dequant logits from uint16 on return.
4. Preflight + correctness gate (expected same cos as w4a16-local-pr:
   0.888; the quant math doesn't care where the dequant happens).
5. Steady-state latency probe — expect ~17 ms/step (Qualcomm extrap).
6. AC sweep if per-step confirms. **If sweep still loses to Lever B,
   A.2 kills the "memory-bandwidth was the hidden lever" hypothesis
   permanently.**

**Exit gate.** Sweep ≥ 18.12 t/s → lever C closes positive. Otherwise:
per-step win confirms the runtime path; correctness still capped by
V-projection collapse. Advance to A.4.

### Phase 5.5.2 — Definitive correctness localization (0.5 session)

7. Install WSL2 + QAIRT 2.42 Linux SDK on x86 box. Run
   `qairt-accuracy-debugger` on our pathb.w4a16-local DLC.
8. Confirms V/O as the failing ops (or names different ops).

**Exit gate.** List of failing op names → input to A.1.

### Phase 5.5.3 — Per-tensor override w4a16 (0.5 session)

9. Emit override JSON pinning V-proj + O-proj weights to w8 (or
   whichever subset A.4 named). Recompile with full-quant IO (carry
   A.2's recipe forward). ~80 s.
10. Correctness probe. Expected cos ≥ 0.95 at w4 precision.
11. Steady-state latency. ~18 ms/step projection (slightly heavier than
    pure w4 due to w8 V/O, still well under verify at k=2).
12. AC sweep.

**Exit gate.** AC sweep ≥ 18.12 t/s → lever C closes positive. Below
→ advance to A.3 only if we believe the SmoothQuant angle still has
headroom, else accept 0.6B-draft PTQ ceiling is real and pivot to
Axis B.

### Phase 5.5.4 — AIMET / GPTQ (1-2 sessions; parallelizable)

13. AIMET install on x86. SmoothQuant + AWQ pass on pathb ONNX.
    Feed annotated ONNX to local QAIRT.
14. AutoGPTQ pass on Qwen3-0.6B PyTorch → GPTQ ONNX → local QAIRT.

Done in parallel if two compile sessions available. Either one landing
at cos ≥ 0.95 without V/O pinning closes the "PTQ method was the
lever" question positively.

**Exit gate.** Both fail → V-projection precision is a fundamental
0.6B property. Close w4a16-on-0.6B negative with full confidence;
this result transfers to Qwen3.5 as "try A.3 first because 0.6B-class
drafters need QAT-grade quant techniques."

### Phase 5.5.5 — Qwen3-1.7B drafter (1-2 sessions)

15. Compile Qwen3-1.7B fp16 via existing pathb/pathbmask pipeline.
    Probe + sweep. Measure accept-rate lift.
16. If accept lifts ≥10 pp, apply whichever w4a16 recipe won Axis A
    (or its lack of winner → try A.2 + A.1 on 1.7B since larger
    models are more forgiving of PTQ noise per published
    literature).
17. k-sweep on the 1.7B variant. Report k ∈ {2, 3, 4, 6, 8}.

**Exit gate.** Beats 18.12 t/s at any k → new high-water mark. Loses →
on this silicon at this target size, 0.6B fp16 k=2 pathbmask async-
pipelined is the ceiling; document and move on.

### Phase 5.5.6 — Target-side probe (0.5 session)

18. Flip `--target-backend opencl` in sweep harness for k ∈ {4, 8} on
    the winning drafter config. Measure.

**Exit gate.** If OpenCL target shaves verify at large k, re-run top
3 configs with the new target. Else close C.1.

### Phase 5.5.7 — Writeup + Qwen3.5 graduation notes

19. Final combined table: every lever, every axis, every kill criterion
    result. This becomes the Qwen3.5 graduation prior.

**Budget total:** ~6-8 sessions to exhaustion. Compare to the
"1 session close negative" path — the extra work buys certainty and a
complete toolbox for the *next* model, which is where the real
deployment target lives.

## Related research — inspiration, not directly actionable

- **dflash** (`https://github.com/z-lab/dflash`, lucebox-hub RTX 3090
  paper cited in `docs/reference-projects.md`). Current state-of-the-art
  spec decoder: **AL ≈ 8.9, 3.43× speedup** with block-diffusion draft
  into tree verify. *Not directly usable on NPU:*
  - CUDA-only implementation (custom verify kernels + KV management).
  - Draft is block-diffusion (non-autoregressive, predicts a K-wide
    block at once). No Qwen3-family non-AR drafter exists; we'd have
    to train one.
  - Tree verify requires multi-path verification in a single target
    call, which llama-server `/completion` doesn't expose.
  - **Inspiration value is high:** Phase 4 in the original plan
    ("DFlash+DDTree") cites this exact work. When Phase 4 opens, the
    research starts from dflash's ideas — port block-diffusion to ORT
    / OpenCL for the draft, build a custom verifier for tree-decode.
  - **Zero interaction with the w4a16 question.** Orthogonal axis; does
    not compound with Axis-A quant work, does not change Axis-D
    drafter architecture decisions beyond "if we ever port it, we'd
    still quantize the block-diffusion drafter."
- **SpecInfer / Sequoia / Triforce** — other tree-verify spec decoders.
  Same CUDA / custom-verifier constraint as dflash. Inspiration tier.
- **Mamba / state-space drafters** — confirmed not-deployable on X2E
  (`gated_delta_net` / `ssm_conv` absent from QNN op library per
  `docs/npu_scoping.md`). Dead end for Hexagon v81.

## Cross-references / anti-duplication

- Background: `docs/w4a16_investigation.md` (sessions 12-19)
- Side-quest results: `results/qwen3_4b_genie_w4a16_probe.md` (do not
  re-run; the probe script `npu_engine/probe_qualcomm_qwen3_4b.py` is
  there if needed)
- Local QAIRT pipeline: `docs/phase5_local_qairt_compile.md`,
  `docs/phase5_local_qairt_compile_findings.md`
- Upstream lever plan: `docs/qwen3_perf_levers_investigation.md`
- Bug receipts (reference memories):
  - `reference_ai_hub_preserve_io_bug.md`
  - `reference_ai_hub_ptq_order.md`
  - `reference_rotary_emb_hoist.md`
  - `reference_ort_qnn_qairt_match.md`

## Success criteria for Phase 5.5 final close

- **Minimum.** Every lead in the map above has a documented
  pass/fail with evidence. No "we didn't try X" items remaining.
- **Target.** One Axis-A or Axis-B config beats Lever B's 18.12 t/s
  AC baseline. Documented + sweep CSV committed.
- **Stretch.** 30+ t/s combined, matching the probe's k=4-8
  projection. Would be publishable.

If minimum passes without target, Qwen3.5 graduation inherits a
complete map of what fails on X2E at 0.6B scale. That's a productive
close, not a loss.

## Update log

- 2026-04-22 (this doc created) — continuation plan written after
  confirming the side-quest succeeded (ran at 7.22 ms/12 layers on
  Qwen3-4B, ORT-QNN loads full-quant IO cleanly). Reopens Lever C
  with a specific axis map instead of re-treading the closed leads.
- 2026-04-22 (same session) — added Axis D (EAGLE-3 / Medusa /
  LayerSkip architectural drafter alternatives) and a "related
  research" section covering dflash and its CUDA-based cousins as
  Phase 4+ inspiration rather than directly-actionable leads.
- 2026-04-23 (session 20) — **Phase 5.5.1 executing.** A.2 + A.1
  approved; ARM-side runtime plumbing landed, x86 compile ask
  shipped in `docs/phase5_lever_c_x86_ask.md` Update 3. See
  "Phase 5.5.1 progress log" section below for details.

## Phase 5.5.1 progress log

### ARM-side plumbing — LANDED (2026-04-23)

All four preparations needed to consume a Qualcomm-IO-convention
(uint8 past_kv + uint16 rest) binary are in place. Existing variants
are unaffected — the new code path is explicit-whitelist-gated.

1. **`scripts/npu_load_qwen3_bin.py`**:
   - new `IS_LOCAL_FULL_QUANT_IO` flag, `True` only for
     `VARIANT in {"w4a16-local-fqio", "w4a16-local-mixed"}`.
   - new `quant_to_uint8` / `dequant_from_uint8` helpers (mirror
     uint16 ones, clip to [0, 255]).
   - new `quant_tensor` / `dequant_tensor` per-tensor dispatchers
     keyed on `spec.bitwidth` — callers no longer need to know the
     target dtype.
   - `_describe_inputs_pathb_local` / `_describe_outputs_pathb_local`
     accept optional `past_kv_dtype` / `present_kv_dtype` params
     (default = unified `dtype`). When `IS_LOCAL_FULL_QUANT_IO`,
     past_kv/present_kv get UINT8, everything else UINT16.
   - `quantized_zero` is bitwidth-agnostic (reads `spec.qmax`); no
     changes needed.
   - `main()` preflight switched from `VARIANT == "w4a16-local"`
     literal to `IS_LOCAL_W4A16` so every PTQ local variant loads
     encodings correctly.

2. **Probe updates** (all four call sites of the uint16-specific
   helpers migrated to the bitwidth-aware dispatcher):
   - `scripts/npu_short_prompt_probe.py` — `_quantize_feed` and
     per-layer present dequant now route through `quant_tensor` /
     `dequant_tensor`.
   - `scripts/probe_npu_steady_state_latency.py` — VARIANTS list
     extended with `w4a16-local-fqio` + `w4a16-local-mixed`; feed
     build uses dispatcher.
   - `scripts/probe_w4a16_quant_roundtrip.py` — `_roundtrip_stats`
     uses dispatcher, `at_max` indexed by `spec.qmax` instead of
     hard-coded 65535, bitwidth added to reported row.
   - `scripts/probe_w4a16_vs_fp16_differential.py` — same.

3. **AST parse-check green** across all five modified files. Import
   test deferred until binary arrives (avoid triggering QNN EP load
   without a target binary).

### x86 compile asks — SHIPPED (2026-04-23)

`docs/phase5_lever_c_x86_ask.md` Update 3 contains:

- **A.2 recipe**: `--weights_bitwidth 4 --act_bitwidth 16
  --quantization_overrides quant_overrides_fqio.json`. The 112-entry
  override file pins every past_kv + present_kv tensor to 8-bit
  symmetric (offset=-128 to match Qualcomm's reference). Committed
  at `models/calibration/quant_overrides_fqio.json`.
- **A.1 recipe**: same flags, override JSON at
  `models/calibration/quant_overrides_mixed.json` — 168 entries
  (112 activation + 56 param) pinning past_kv to 8-bit AND
  V-projection + O-projection weights to 8-bit.
- NAS drop paths:
  - A.2 → `Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_fqio\`
  - A.1 → `Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_mixed\`
- Schema caveat: the JSON format is our best guess at
  qairt-quantizer 2.42's expected `--quantization_overrides` shape
  (AIMET-style `activation_encodings` / `param_encodings` with
  per-tensor `{bitwidth, dtype, is_symmetric, offset}` list). The
  x86 team confirms/fixes per `qairt-quantizer --help` and commits
  the working schema to `HANDOFF_{fqio,mixed}.md` so future Qwen3.5
  runs inherit the canonical reference.

### Pending — binaries from x86 + ARM-side measurement

The moment A.2 + A.1 binaries land in NAS, the protocol is:
1. MD5 verify + copy to `models/`.
2. `scripts/probe_w4a16_quant_roundtrip.py` for both variants
   (quant layer smoke test; expect RMS ≈ 0.001% per the bitwidth
   math, now shown per-tensor with bitwidth column).
3. `scripts/npu_short_prompt_probe.py --path pathb` with each
   variant, `SPECULA_NPU_VARIANT` set (correctness gate vs CPU fp32;
   A.2 expected cos ≈ 0.33 if PTQ V-collapse still dominates; A.1
   expected cos ≥ 0.95).
4. `scripts/probe_npu_steady_state_latency.py` — full 9-variant
   table (A.2 expected 17-18 ms/step, A.1 expected 18-19 ms).
5. If any variant clears cos ≥ 0.95, AC sweep via
   `scripts/sweep_npu_spec.py --mode async-pipelined -n 200`.
6. Findings commit + this doc's progress log updated.

### Decision tree after measurement

- **A.2 beats Lever B's 18.12 t/s at k=2**: lever C closes
  **POSITIVE**. Memory bandwidth was the hidden lever. Investigation
  complete for Qwen3-0.6B; same recipe transfers to Qwen3.5.
- **A.1 clears cos ≥ 0.95 AND beats Lever B**: lever C closes
  **POSITIVE** on mixed precision; first fully-numerically-clean w4
  regime. Document as canonical Qwen3.5 PTQ starting point.
- **Both compile but neither beats Lever B**: memory-bandwidth
  thesis empirically disproven at this target/draft ratio. Lever C
  stays negative as a product; pivot to Axis B (Qwen3-1.7B draft)
  or stop w4a16 work for Qwen3-0.6B permanently and move to W1.a
  (GPU prefill target-side attack).
- **A.1 fails correctness** (cos < 0.95 even with V/O at w8): the
  V-projection collapse isn't purely a weight-precision issue —
  advance to Axis A.3 (AIMET SmoothQuant/AWQ) or A.4 (WSL2
  qairt-accuracy-debugger).
- **A.2 compiles but binary fails to load on ORT-QNN 1.24.4**: some
  IO shape/dtype property we haven't anticipated. Revisit wrapper
  construction + compare with Qualcomm's Qwen3-4B EPContext wrapper
  from the side-quest.
