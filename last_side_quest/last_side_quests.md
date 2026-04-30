# last_side_quests — final realignment plan

Created 2026-04-27. Workspace: `last_side_quest/` at repo root, with
per-side-quest subdirectories created as work lands. This doc is the
umbrella plan; per-SQ findings get their own writeups under their
subdirectory.

## Why this set, why now

We're approaching the natural close of the Qwen3 generation of work
before graduating to Qwen3.5/3.6 (per `CLAUDE.md` and feedback memory
"close out Qwen3 before graduating to Qwen3.5"). The roadmap
(`docs/roadmap.md`) has a long workstream list (W1–W9 + B-backlog) but
the project has *never* shown a single end-to-end heterogeneous demo,
never run AIMET locally, never run a Qwen MoE on NPU, and the NPU
engine is still capped at ctx=512. These are the deliverables that
turn specula from "characterized the silicon" into "demonstrated the
thesis" — and once they land, Qwen3.5/3.6 graduation becomes a
pipeline re-run rather than a fresh investigation.

Six side quests, mostly independent. Plan to ship a writeup + at
minimum one CSV per SQ, regardless of positive/negative outcome.

## Side quest inventory

### SQ1 — Heterogeneous LLM inference: NPU 4B draft + CPU target

**Goal.** One demo run where Qwen3-4B drafts on NPU (via
`npu_engine/sidecar.py` infrastructure that already beats Genie at PP
+39% / TG +19%) and a larger model on CPU verifies, end-to-end. Even
naive throughput, even slower than CPU-only target — the win is
**proving the architectural pattern works on this silicon**, which no
other public project has demonstrated.

**Recommended target model.** Default: **Qwen3-14B-Q4_K_M** (~9 GB,
dense, not on disk yet). Memory math: 4B NPU bundle (~3.1 GB) + 14B
CPU target (~9 GB) + KV at 4K ctx (~2 GB) ≈ 14 GB resident.
Comfortable fit in 48 GB. Architecturally clean: same family as the
draft, predictable accept rate.

Stretch options if 14B works:
- **Qwen3-32B-Q4_K_M** (~19 GB dense). Bigger draft/target ratio
  (8×) maximizes spec-decode payoff; total ~25 GB still fits.
- **Qwen3.6-35B-A3B-Q4_K_M** (~18 GB MoE, *already on disk*). Two
  storylines in one demo — heterogeneous AND MoE target. But mixing
  Qwen3-family draft with Qwen3.6 target risks a vocab mismatch;
  must verify tokenizer compatibility before committing.

**Reusable assets.**
- `npu_engine/sidecar.py` — long-lived NPU engine with phase-batched
  prefill_only / decode_only primitives. The "decode_only with N=K"
  pattern is exactly draft-K-tokens-and-let-target-verify.
- `scripts/serve_daily_driver.ps1` — the existing CPU/Vulkan
  llama-server harness (currently driving Qwen3.6-35B-A3B).
- `scripts/npu_draft_sidecar.py` — older spec-decode prototype; mine
  for the loop shape, replace the model with NPU 4B.

**The hard part.** `llama-server`'s `/completion` endpoint returns
tokens only — no hidden states, no batched alternate-candidate
scoring. The "external draft sends K tokens to llama-server target"
pattern that the user proposed is naively implementable as:

> For each drafted token candidate, POST `/completion` with
> `prompt = full_context_so_far`, `n_predict = 1`, `temperature = 0`.
> Compare the returned token to our draft. If match → accept; else
> reject and rewind.

This is **not real spec-decode** (it's K independent forward passes
on the target, no batched verify, identical wall-time to CPU-only).
It still demonstrates the NPU draft path end-to-end.

**Real spec-decode** requires either (a) the in-process llama.cpp
Python bindings approach (B20 path 1 in roadmap), or (b) writing a
custom multipath verifier. (a) is ~2-3 sessions; out of scope for
this side-quest unless we want to keep going.

**Decision gate.** Land the naive "demo" first (SQ1.a) — proves the
plumbing. If TTFT and per-token latency feel acceptable, escalate to
the real-spec-decode path (SQ1.b, becomes B20 promotion). If the
naive path proves nothing useful, document that and close.

**Open questions to answer in execution.**
- Does NPU 4B's tokenizer match Qwen3-8B/14B's? (Yes — same family,
  same vocab — but verify with `tokenizer.json` MD5.)
- For the 35B-A3B stretch target: does the Qwen3 ↔ Qwen3.6 vocab
  match? (`models/Qwen3.6-35B-A3B-Q4_K_M.gguf` tokenizer vs
  `qualcomm-qwen3-4b-ref/tokenizer.json`.)
- What ctx tier do we run the NPU at — cl=512 (already wired) or
  cl=2048+ (needs SQ5)? Demo can ship at cl=512 if SQ5 is incomplete.

**Cost.** 1 session for the naive demo, +1 session if we escalate to
real spec-decode via B20 path 1.

**Workspace.** `last_side_quest/sq1_heterogeneous/` — driver script,
demo CSVs, writeup.

---

### SQ2 — AIMET local venv survey

**Goal.** Find out what `quic/aimet` (currently 2.26.0) actually does
when installed locally — not the Qualcomm-published rent-cloud pipeline,
the bare AIMET surface. Then run **basic PTQ on Qwen3-0.6B**, our existing
pathb ONNX, and see what it emits. Output: one writeup
(`aimet_local_survey.md`) covering:

- What installs cleanly, what doesn't, on Windows-on-ARM Python.
- What installs via WSL2 ARM64 Linux (CPU only, no CUDA).
- What AIMET techniques are accessible without cloud GPU
  (basic PTQ vs SEQ_MSE / AdaScale / SmoothQuant / AWQ / QAT).
- What MoE-specific support actually exists (the user noted "it now
  supports MoE??" — verify against AIMET 2.26 release notes).
- How AIMET's encodings.json format compares to QAIRT's
  `--quantization_overrides` JSON (open question P2 in
  `docs/one_pipeline_cloud_gpu.md`).

**Why isolated venv.** Our existing `.venv-qairt` and project venv
have 50+ pinned packages including specific torch / onnx / numpy
versions. AIMET 2.26 wants `torch==2.4` and `onnxruntime-gpu==1.23.2`;
forcing into the existing venv risks breaking what already works.
A `last_side_quest/sq2_aimet_local/.venv-aimet/` is throwaway.

**Reality check (likely findings, to confirm by trying).**
- `aimet_onnx` is published as `cu121-cp310-cp310-manylinux_2_34_x86_64.whl`
  per `docs/rent_cloud_compute.md`. **No Windows wheel, no ARM wheel.**
  Likely nothing installable directly on Windows-on-ARM.
- `aimet_torch` may have CPU wheels with broader platform coverage —
  Linux x86_64 yes; Linux ARM64 (WSL2) maybe; Windows ARM almost
  certainly no.
- A pure-CPU basic-PTQ run on Qwen3-0.6B might be feasible in WSL2
  ARM64 Linux with the torch CPU wheel. ~1-3 hours wall time
  estimate.
- AIMET MoE support: AIMET 2.x added MoE quantization-sim support in
  2025 — verify exact version + scope (per-expert quantization?
  shared-expert handling?) by reading their docs.

**Decision gate.** If basic PTQ runs locally on Qwen3-0.6B and emits
encodings.json, this side quest succeeds: we've gained an in-house
PTQ tool that doesn't depend on QAIRT's quantizer. If it can't be
made to run locally at all, the verdict is "everything AIMET-related
is cloud-only" — feeds directly into SQ4 cloud sizing.

**Cost.** 1 session for install + survey + Qwen3-0.6B basic PTQ
attempt. Possibly 0.5 session of follow-up if a non-obvious workaround
exists (e.g. running aimet_torch under WSL2 emulation).

**Workspace.** `last_side_quest/sq2_aimet_local/` — venv, install
notes, the survey doc.

---

### SQ3 — Smallest Qwen MoE for AIMET → NPU

**Goal.** Identify the smallest Qwen MoE that is realistic to
quantize via AIMET and target onto NPU. Today the smallest options
seem to be:

| candidate | active / total params | typical Q4_K_M size | notes |
|---|---|---:|---|
| **Qwen3-30B-A3B** | 3B / 30B | ~18 GB | Smallest Qwen3-family MoE. Not on disk yet. |
| **Qwen3.6-35B-A3B** | 3B / 35B | ~18 GB Q4_K_M | **On disk already** as `models/Qwen3.6-35B-A3B-Q4_K_M.gguf` and as MXFP4_MOE GGUF. Currently the daily-driver target. |
| **Qwen3.5-14B-A3B** (if exists) | 3B / 14B | ~9 GB | If shipped — would be ideal; needs verification. |

**Open question for the user.** Do we (a) pick whichever has the
**most up-to-date AIMET adapter ready**, (b) pick the smallest
on-disk one (35B-A3B Qwen3.6 already here, but a Qwen3.6 model is
ahead of our Qwen3 baseline schedule), or (c) wait until SQ2 + SQ4
tell us realistic budget before committing?

Recommended: **survey first, commit later.** Read AIMET's MoE-
quant docs in SQ2 → confirm whether Qwen3-30B-A3B has a published
arch adapter in `qai_hub_models` or whether we'd be the first
external attempt. Pick the candidate with the cheapest path to
"first w4a16 binary on HTP."

**Reusable assets.**
- The `one_pipeline_cloud_gpu.md` design — directly extensible to
  the MoE case via per-expert AIMET adapter (work needed: verify
  per-expert quantization is what AIMET emits, and that
  `qairt-quantizer` consumes it without per-expert routing
  surgery).
- The Qwen3-4B 4-part bundle structure as a reference shape for
  what an MoE bundle would look like. MoE adds an expert-routing
  layer per attention block; partition seam choice changes.

**Hard parts known in advance.**
- MoE routing on HTP — has Qualcomm published any MoE-on-NPU
  reference? Quick search needed (`qaihub-public-assets` for any
  `*moe*` model).
- Active vs full memory: 30B-A3B is 3B active per token but 30B
  resident weight footprint. NPU bundle size will be ~30B × 0.5 B/param
  (w4a16) ≈ 15 GB across however many partitions — likely 8-12
  partitions vs our current 4 for Qwen3-4B. Past the HTP session
  ceiling we already hit.

**Decision gate.** This SQ probably ends "we have a candidate, here's
the per-step cost projection, here's what cloud rental we'd need" —
not "binary on disk" unless local AIMET (SQ2) supports MoE PTQ
without a CUDA card. Feeds straight into SQ4.

**Cost.** 0.5 session of literature scan + memory math, no compile.

**Workspace.** `last_side_quest/sq3_qwen_moe_npu/` — candidate doc,
seam-map sketch.

---

### SQ4 — Cloud-compute sizing decision

**Blocked by:** SQ2, SQ3.

**Goal.** A one-page verdict that decides the next cloud-rental
session, with $ and VRAM target nailed:

- "If we want to ship an AIMET-quant'd Qwen3-0.6B PTQ binary, rent X."
- "If we want to ship Qwen3-4B with SEQ_MSE+AdaScale to close the
  V/O collapse, rent Y."
- "If we want to ship the smallest-Qwen-MoE bundle, rent Z."

Slot the table into `docs/one_pipeline_cloud_gpu.md` §Budget and
update its "first conversion" recommended path.

**Model roster for the cloud session(s):**

| candidate | size | arch | risk profile |
|---|---|---|---|
| **Qwen3-30B-A3B** (or Qwen3.5/3.6 MoE) | 30B / 3B act | dense-attn + Qwen3MoE | low — AIMET 2.29 ships qwen3_moe adapter; pure transformer MoE |
| **Granite-3B-A800M** (cloud retry) | 3B / 800M act | dense-attn + GraniteMoeParallelExperts | very low — locally validated at 1B; just confirms cloud pipeline matches local results |
| **NVIDIA Nemotron 3 Nano Omni** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) | 30B / 3B act | **hybrid: 23 Mamba SSM + 23 MoE (128 experts top-6) + 6 GQA** | **HIGH** — Mamba SSM has never (afaik) been deployed on Hexagon HTP; multimodal vision+audio adds graph complexity; 128 experts top-6 is denser routing than what we've handled. Plan for 3+ sessions, not 1. Native FP8/NVFP4 checkpoints exist — investigate whether we can skip the AIMET PTQ step entirely and convert FP8 directly via QAIRT |

**Cost.** 0.25 session writeup once SQ2 + SQ3 land.

**Workspace.** Edits the existing `docs/one_pipeline_cloud_gpu.md`;
no new directory needed.

---

### SQ5 — Long-context NPU (>4K) viability

**Goal.** Stop running the npu_engine at ctx=512 (the artificial cap
from `npu_engine_todos.md` TODO #1). Extend to use the
`cl{1024,2048,3072,4096}` graphs that **already exist in the
Qualcomm Qwen3-4B bundle** — no recompile required, just
generalize `build_part_cfg(metadata, ar=1, ctx=512)` to take a `ctx`
parameter and use the right `{prefix}_ar{ar}_cl{ctx}_{N}_of_4`
graph name.

Then answer the strategic question: **at 4K context (the bundle's
ceiling), is NPU-as-coding-assistant-draft viable?**

Real coding-assistant prompts run 8K–32K tokens (system prompt + a
few file reads + chat history). The NPU bundle caps at 4K. Two
paths if 4K is insufficient:

1. **Compile a longer-ctx bundle ourselves** — needs the cloud
   pipeline (`one_pipeline_cloud_gpu.md`). Adds ctx tiers up to
   whatever HTP memory + KV cache allow.
2. **Sliding-window draft** — let the draft only see the last N
   tokens of the target's context. Loses long-range coherence in
   the draft but keeps the NPU usable. Affects accept rate
   negatively but stays simple.

**Reusable assets.**
- The Qualcomm bundle's existing cl=1024..4096 graphs (already
  shipped, `genie-t2t-run.exe` selects them based on ctx-bins
  config — we need to do the same in our wrapper-ONNX path).
- `metadata.yaml` + `genie_config.json` already enumerate them.

**Hard parts known in advance.**
- HTP session ceiling shifts at larger ctx: per
  `reference_ortqnn_session_limit.md` we already hit a ~7-session
  ceiling at cl=512. Larger past-KV input shape ⇒ more HTP context
  memory ⇒ ceiling likely drops. May need one-session-per-binary
  loading (worse swap cost) at cl=4096.
- Per-tier IO scales/offsets may differ — re-extract from
  `metadata.yaml` per tier rather than reusing cl=512 values.

**Decision gate.** After cl=2048 lands:
- If t/s holds within ~10% of cl=512 numbers → extend to cl=4096,
  ship as standard.
- If t/s collapses (>30% drop) → memory-bound; document and decide
  whether SQ1 spec-decode demo runs at cl=512 (small prompts) or
  whether we need the cloud pipeline for a custom long-ctx bundle.

**Strategic answer to the user's "are we dead in the water" question.**
Tentative: **no, but with caveats.** 4K is a real ceiling for the
NPU draft; *target* on CPU has full 32K (no constraint). Spec-decode
where draft-ctx ≪ target-ctx is novel territory — accept rate
will degrade vs full-ctx draft, by some unmeasured amount. The
honest answer for coding-assistant viability needs the SQ5 cl=2048
measurement before committing.

**Cost.** 1 session per `npu_engine_todos.md` (mostly mechanical).

**Workspace.** Direct edits to `npu_engine/`; results docs to
`last_side_quest/sq5_long_context_npu/`.

---

### SQ6 — Small-model server harness for opencode

**Goal.** Stand up `Qwen3-4B-Q4_K_M` and `Qwen2.5-7B-Q4_K_M` as
their own `llama-server` endpoints (alongside the existing
Qwen3.6-35B-A3B daily-driver served by `serve_daily_driver.ps1`),
point opencode at each in turn, and capture a subjective
"is the small model usable as a coding assistant" datapoint plus
TTFT, sustained TG, and context behavior at realistic agent
prompt lengths.

**Reusable assets.**
- `scripts/serve_daily_driver.ps1` — fork to `serve_small.ps1`
  (or parameterize `-Model`), point at Q4_K_M smaller models.
- Existing baseline numbers in
  `docs/qwen3_4b_baseline_all_backends.md` and
  `docs/qwen2_5_7b_baseline_all_backends.md` — these are
  microbenchmark t/s; SQ6 adds *real-prompt agent-loop*
  numbers that microbenchmarks can't predict.

**The user's ask, distilled.** Even though we have the 35B-A3B
daily-driver running, what does the coding-assistant experience
*feel* like with just a 7B or 4B? Is it usable, or is the model
quality the binding constraint regardless of speed? We don't yet
have that datapoint; it informs whether the NPU-only future
(where the largest viable NPU model is ~7B) is a real product or
a research curiosity.

**Hard parts known in advance.**
- Tool-calling / structured output quality on smaller models is
  empirically bad. opencode relies on tool-calling. May fall over
  on the small models.
- Context window: 4B and 7B both support 32K; opencode-style use
  with 5–10 tool-call turns can blow past that quickly. Note the
  ctx fill rate.

**Decision gate.** Subjective verdict per model:
- "usable for X but not Y" — useful datapoint, ship the writeup.
- "unusable for any agent-loop work" — also a useful datapoint;
  argues the NPU-future requires bigger-than-NPU-can-host models,
  reinforcing the heterogeneous (SQ1) story.

**Cost.** 0.5 session.

**Workspace.** `last_side_quest/sq6_small_server/` — serve script,
opencode session traces, writeup.

---

## Suggested execution order

```
SQ2 ─┬─► SQ4 ─► (cloud rental decision)
SQ3 ─┘

SQ5 ─► SQ1 (real ctx for the demo)
SQ6 (independent, anytime)

SQ1.a (naive demo, cl=512) ──┐
                              ├─► End-of-Qwen3 close-out
SQ1.b (real spec-decode)  ───┘
```

Suggested sequence (each ≤ 1 session unless noted):
1. **SQ6** — fastest, gives us subjective small-model feel before
   diving into anything heavy. Drives later prioritization (if 4B
   feels useless, we don't need to ship a 4B-on-NPU spec-decode).
2. **SQ5** — long-context NPU first. Mechanical extension of
   existing engine; results gate SQ1's ctx tier choice.
3. **SQ1.a** — naive heterogeneous demo at whatever ctx SQ5 supports.
   Ships the headline "two compute islands cooperating" artifact.
4. **SQ2** — AIMET local survey. Independent of everything above;
   can run in parallel with SQ5/SQ1 if there's time.
5. **SQ3** — MoE candidate scan. After SQ2 so we know what AIMET
   does and doesn't support locally.
6. **SQ4** — cloud sizing verdict. Composes SQ2+SQ3 into a
   one-page rental recommendation.
7. **SQ1.b** (optional, if SQ1.a went well) — real spec-decode via
   in-process llama.cpp bindings (B20 promotion).

## Locked decisions (2026-04-27 user pass)

- **SQ1 target model: Qwen3-14B-Q4_K_M** (dense, ~9 GB, not on
  disk yet — download when SQ1 starts).
- **SQ2 first model: Qwen3-0.6B** — reuse existing pathb ONNX,
  fastest iteration loop.
- **Starting order: SQ5 first** — long-context NPU extension.
  Rationale: gates SQ1's ctx tier; mechanical extension of an
  engine that already exists; if it fails (HTP context-memory
  ceiling collapses past cl=512), it changes everything downstream.

## Progress

| SQ | status | one-line outcome |
|---|---|---|
| **SQ1.a** | ✅ **Path A landed 2026-04-27** | NPU 4B + CPU 14B exchange tokens; JSON 100% accept, Python K=8 6/8 accept, Qwen3.6 incompat (sep memory) |
| **SQ1.b/c** | ✅ **closed POSITIVE 2026-04-28** | Path B (multi-round naive serial verify) and Path C (batched verify + KV cache rewind via in-process llama-cpp-python) both run end-to-end. JSON 91% accept reproduced across both. Path C 3.5× more per-round-efficient than Path B at K=8. Same-arch JSON speedup 0.67× (Path C) vs 0.19× (Path B). Two engineering follow-ups gate real speedup: **NPU sidecar rewind op** (~half-session) + **ARM64 llama-cpp-python build** (clang-on-ARM toolchain). Cross-arch numerical drift: free-form Python prompts collapse to 9% accept when target is Prism x86_64 vs draft ARM64; structural prompts (JSON) resist this. |
| **SQ1 NPU rewind** | ✅ **landed 2026-04-28** | Path C STATEFUL (`demo_path_c_stateful.py`) uses the SQ6 stateful stream API. NPU re-prefill share collapses to 0; JSON K=8 r=4 runs in 5.96 s wall (vs 21.70 stateless), spec-decode 5.53 t/s (vs 1.52), **2.37× speedup vs same-host baseline** — first-ever cross of 1.0×. 91% accept byte-identical to stateless. CSV `results/csv/sq1_path_c_stateful_2026-04-28.csv`. Native ARM64 llama-cpp-python build remains the only outstanding follow-up (cross-arch drift on free-form prompts). |
| **SQ2** | ✅ **closed POSITIVE 2026-04-28** | aimet_torch v2 + SEQ_MSE/AdaScale work locally on Prism + WSL2 ARM64; aimet_onnx + qai_hub_models wrapper still cloud-only; basic-PTQ Qwen3-0.6B end-to-end demo lands negative-but-expected (cos -0.065 = V/O collapse repro) |
| **SQ3-small** | ✅ **closed POSITIVE 2026-04-28** (Granite-MoE branch + 3 follow-ups) | Granite-3.0-1B-A400M ran AIMET basic-PTQ end-to-end on Prism CPU; cos +0.656 (per-tensor experts) → +0.712 (per-channel experts, "A1" champion). SEQ_MSE-4 regressed (0.640); SEQ_MSE-16 partially recovered (0.682) — **A1 wins on Prism budget**. OLMoE-1B-7B failed end-to-end across 3 iterations — per-expert dispatch architecturally hostile to AIMET v2 (Granite's fused-experts + Qwen3-MoE's hit-filter both avoid this). AIMET 2.29 has no granitemoe adapter; written in ~80 LOC. **Qwen3-30B-A3B remains cloud-only** (corrected math: 30B FP32 = 120 GB, BF16 = 60 GB — neither fits 48 GB DRAM). |
| SQ4 | 🚀 **plan committed 2026-04-29, M1 pending** | scope expanded from "sizing writeup" to "execute the cloud pipeline." Plan: `last_side_quest/sq4_cloud_adventure/findings.md`. Five milestones M1→M5 (Qwen3-0.6B → 4B → 14B → Qwen3.6-27B dense → Qwen3.6-35B-A3B MoE). Hardware: RunPod A40 48 GB at $0.44/hr for M1-M3; A100 80 GB for M4-M5. Validation anchor: `models/qualcomm-qwen3-4b-ref/` for M2 byte-/argmax-comparison. Total budget $50-75. |
| **SQ5** | ✅ **closed POSITIVE 2026-04-27** | engine generalized cl=512..4096; coding-asst contexts viable up through 4K at 20 t/s |
| **SQ6** | ✅ **Phases A→E landed 2026-04-28..29** | OpenAI-compat NPU HTTP server (`npu_engine/http_server.py`). A stateless + A.5 SSE + B stateful KV-LCP streams (2.3-2.6× speedup) + C real-world pi A/B (NPU silent-island UX wins; throughput tied with OpenCL; Vulkan-4B chat broken at this commit) + D daily-driver Q3.6-35B-A3B CPU 32 t/s comparator + E Qwen2.5-7B-NPU sidecar generalization. Phase F follow-ups open: tool-call parser + 7B AR1↔AR128 swap fix. |

## Cross-cutting findings landed this session

- **Qwen3 ↔ Qwen3.6 tokenizer INCOMPATIBLE** (memory:
  `reference_qwen_tokenizer_generations.md`). Heterogeneous-decode
  pairings cannot cross Qwen generations without a vocab translator.
  Constrains SQ1 stretch targets to Qwen3 family until a translator
  exists or NPU bundles are recompiled for new vocabs.
- **NPU long-context viability**: 20 t/s decode at cl=4096 (sublinear
  scaling from cl=512's 27 t/s). Coding-assistant context lengths
  through 4K are usable today; >4K routes to the cloud pipeline.
- **JSON / structured output is the SQ1 sweet spot**: 100% accept
  on the demo's JSON prompt (16/16 byte-identical) means tool-call
  workloads see full benefit if Path B/C lands.
- **AIMET PyTorch is locally usable** (SQ2): `aimet_torch` 2.29 v2
  surface (basic PTQ + SEQ_MSE + AdaScale + AdaRound + CLE +
  experimental quant) runs on Prism CPU and WSL2 aarch64 — no CUDA,
  no cloud. Only `aimet_onnx` + Qualcomm's `qai_hub_models` wrapper
  remain cloud-only. AIMET 2.29 ships first-class Qwen3-MoE
  quantsim hooks. This collapses SQ4's expected rental footprint
  from "every iteration" to "production blessed bundles only."
- **AIMET extensibility to non-blessed MoE archs is cheap** (SQ3-small):
  AIMET 2.29 ships adapters for {gemma3, internvl, llama, mistral,
  phi3, qwen2/2_5/3/3_5/3_moe/3_vl}. Granite-MoE / OLMoE / DBRX /
  JetMoE need user-written adapters, but the bar is **~80 LOC** of
  mechanical `@QuantizationMixin.implements` code following the qwen3
  template (ignore RoPE, custom RMSNorm, custom expert FFN). Verified
  on granitemoe; `last_side_quest/sq3_small_moe/granite_moe_adapters.py`
  is the canonical reference.
- **MoE quantizes better than dense at the same w4a16 recipe**
  (single data point, 2026-04-28): Granite-1B-A400M (1.3B/400M act)
  cos +0.656 vs Qwen3-0.6B-dense cos -0.065, identical AIMET basic-PTQ
  recipe. Hypothesis (untested): expert specialization narrows
  per-expert weight distributions, so per-tensor quantization captures
  them better. Needs 2-3 more (model, recipe) cells to confirm; if it
  holds, it's a real publishable finding for the NPU-MoE story.
- **Silent compute island is the SQ6 punchline, not throughput**
  (Phase C live observation, 2026-04-28). NPU 4B and OpenCL 4B are
  near-tied on TG (24-27 t/s both); NPU loses 15-30% on wall. But
  while running on battery with archicad in the foreground, OpenCL
  produces audible fan + coil whine and contends with Adreno for the
  primary workflow; NPU is silent and independent. Power: NPU ~13.7 W
  vs OpenCL ~58 W (~4× draw, 26× J/gen-tok edge). For background
  coding agents during real foreground work, NPU is the only backend
  that doesn't tax the user's actual workload — a 30% wall penalty is
  cheap insurance against fan noise + GPU contention + battery drain.
- **Daily-driver Q3.6-35B-A3B on CPU is the throughput baseline to beat**
  (Phase D, 2026-04-29). 32 t/s wall avg via `build-cpu/llama-server -t 8`
  on a 35B/3B-active MoE — beats both 4B candidates per-second because
  active-only CPU memory bandwidth (228 GB/s on 3B = ~32 t/s) outpaces
  dense 4B's 33% larger weight read. OpenCL-on-big-MoE is a dead path:
  Adreno's 2 GB `max_mem_alloc_size` fragments the 20 GB Q4_K_M and
  TG collapses to 5 t/s. Vulkan-4B chat completions are also a known-
  bad config at the current llama.cpp commit (token-soup + parse error
  on `/v1/chat/completions`; `llama-bench` numbers are throughput-real
  but quality-untested). NPU-4B has to win on **quality** vs the 35B,
  not speed — and currently doesn't (W4A16 V/O collapse manifests as
  occasional duplicate-attribute output).
- **Qwen2.5 has cleaner multi-turn KV behavior than Qwen3** (Phase E,
  2026-04-29). Qwen2.5 has no thinking-mode at all, so the upstream
  chat template never injects a `<think></think>` block, and
  per-assistant-turn LCP isn't capped by template/KV divergence. On
  the 7B-NPU bundle, turn 2 of a 2-turn chat ran in 0.89 s vs turn 1
  at 1.83 s *despite a longer prompt* — pure stateful-stream LCP win.
  Phase B's `enable_thinking=False` workaround on Qwen3 is a real
  protocol-level constraint that simply doesn't exist on Qwen2.5.
- **Tool-calling protocol gap blocks real agent loops** (Phase E,
  2026-04-29). `npu_engine/http_server.py` accepts `tools=[...]` from
  OpenAI clients but ignores it, generates plain text, returns. Pi
  parses the placeholder string the model emits as a final answer; no
  multi-turn loop ever happens. Adding a Hermes-XML or OpenAI-JSON
  tool-call parser is the gating engineering item before the SQ6
  http_server is a real coding-agent backend.

## Outstanding follow-ups (post 2026-04-29)

Six SQs are closed positive on their primary thesis; SQ4 is the only
fully-pending umbrella deliverable. The follow-ups below are the
"would-be-nice" tail per SQ — not blocking the close-out narrative but
worth one more session each if budget allows.

**SQ1 — heterogeneous spec-decode**
- Native ARM64 `llama-cpp-python` build (clang-on-ARM toolchain). Free-
  form prompts collapse to 9% accept on Prism x86_64 target via cross-
  arch logit drift (`reference_cross_arch_logit_drift.md`); JSON
  resists. Same-arch target would unblock Python/prose workloads.

**SQ2 — AIMET local**
- Run SEQ_MSE / AdaScale / AdaRound / CLE on **real Qwen3** (only
  TinyMLP and Granite verified at scale to date).
- Qwen3-4B basic PTQ end-to-end (~3h on Prism CPU est.).
- Confirm which `encodings.json` schema QAIRT's `--quantization_overrides`
  actually consumes (the P2 question in `one_pipeline_cloud_gpu.md`,
  still open after SQ2).
- `aimet_torch.onnx.export` with `use_external_data_format=True` for
  ≥0.6B (the v1 `sim.export` trips a 2 GB protobuf cap; v2 export API
  exists but untested).

**SQ3 — small-MoE**
- Per-tensor vs per-channel comparison on a **dense** Qwen3-1.7B
  baseline to confirm the "MoE quantizes better than dense" hypothesis
  (currently 1 data point — Granite vs Qwen3-0.6B).
- AdaScale on GraniteMoe attention head (imports work, behavior
  untested).
- Granite-3.0-3B-A800M (~80 LOC adapter predicted, mirrors 1B path).
- Per-expert axis-0-only weight quant variant as deployment middle
  ground.

**SQ4 — cloud pipeline execution** (scope expanded 2026-04-29 from
"sizing writeup" to "actually run the rentals"). Full plan in
`last_side_quest/sq4_cloud_adventure/findings.md`. Execution order:
- **M1** Qwen3-0.6B w4a16 NPU bundle (~$1, A40 48 GB) — decisive test
  of whether SEQ_MSE+AdaScale closes SQ2's V/O collapse (cos -0.065 →
  ≥0.95 target).
- **M2** Qwen3-4B w4a16 NPU bundle (~$3-5, A40 48 GB) — gold-reference
  reproduction vs `models/qualcomm-qwen3-4b-ref/`. Target: 46/46
  argmax agreement on 46-token oracle (matches Qualcomm's shipping
  bundle).
- **M3** Qwen3-14B w4a16 NPU bundle (~$3-15, A40 or A100) — first
  novel artifact, no Qualcomm reference exists.
- **M4** Qwen3.6-27B dense (~$13-15, A100 80 GB) — first cross-
  generation; tokenizer incompat per memory; may need ~80 LOC AIMET
  adapter if `qwen3_6` not blessed.
- **M5** Qwen3.6-35B-A3B MoE (~$25-35, A100 80 GB or 2×) — production
  target, headline deliverable. Validates AIMET MoE quantsim + QAIRT
  MoE compile + HTP MoE routing end-to-end.

**SQ5b — long-context NPU tail**
- Battery J/tok at cl=4096 (current numbers are AC).
- pp=2048 tg=512 at cl=2048+ (bundled prompt file caps at 512 tokens).
- AR1+AR128 simultaneous sessions at cl=2048+.

**SQ6 — small-server (Phase F+ work)**
- **AR1↔AR128 swap fails on 7B** for >512-token prompts — sidecar
  crashes mid-request (`sidecar closed stdout mid-request`). HTP teardown
  for 6 sessions doesn't clean up before AR128 reload. Workaround today:
  bump `--ar128-min-tokens` to 1500. Real fix: `gc.collect() + sleep`
  between teardown/reload, or load both chains simultaneously if HTP
  can hold 12 sessions at cl=4096 (untested; 7B is w8a16 → smaller HTP
  footprint than 4B w4a16 might fit).
- **Tool-call protocol** (Hermes-XML or OpenAI-JSON parser) — gates
  real multi-turn agent loops. Without it, pi/opencode degrade silently
  to text and the model hallucinates tool output.
- **AIMET SEQ_MSE+AdaScale on the 4B NPU bundle** to recover output
  quality (the duplicate-attribute regression observed in Phase C is
  the W4A16 V/O collapse manifesting subtly). Now locally tractable
  via SQ2's aimet_torch path.
- 7B has only one ctx tier (4096) — long-prompt prefill always pays
  full KV size. AR1 prefill ≈18 t/s for >256-token prompts is slow.
- 7B `BACKEND_PATH` hardcoded to `C:/Qualcomm/AIStack/QAIRT/2.45.40.260406/...`
  → move to env var.
- Multi-turn agent loops with tools enabled never exercised on 4B
  (would surface chat-template/KV-LCP issues; quick test recommended
  once tool-call parser lands).
- `wsl_setup_notes.md` referenced in Phase C but never written.

## Where this fits in the bigger picture

Per `CLAUDE.md` priority path, the headline goal of the next phase
is the all-backends baseline matrix + the cloud-pipeline. This
side-quest set is the **complement**: deliverables that prove the
*architectural thesis* (heterogeneous compute, NPU-MoE feasibility,
end-to-end AIMET in our hands) in addition to the matrix data. Both
land together as Qwen3's close-out narrative.

When this side-quest set closes, the project's "Qwen3" chapter ends
with: a publishable matrix, a working heterogeneous demo, a known
local AIMET surface, a sized cloud rental recipe for any future
graduation, a long-context NPU answer, and an opencode harness
across three model sizes. That's the cleanest possible handoff to
Qwen3.5/3.6.

## Update log

- **2026-04-27** — Doc created. SQ1–SQ6 scoped, no execution yet.
  User to confirm SQ1 target model + SQ2 first model + execution
  order before starting.
- **2026-04-28** — SQ2 closes positive. `aimet_torch` v2 surface +
  SEQ_MSE + AdaScale + Qwen3-MoE quantsim hooks all work on Prism
  Windows x86_64 and WSL2 aarch64. `aimet_onnx` remains
  manylinux-only. Qwen3-0.6B basic PTQ ran end-to-end (254 s cal),
  reproduced V/O collapse (cos -0.065). Deliverable:
  `last_side_quest/sq2_aimet_local/aimet_local_survey.md`.
- **2026-04-28** — SQ3 small-MoE branch closes positive on Granite-3.0-1B-A400M.
  Qwen3-30B-A3B deferred to cloud (memory math corrected: doesn't fit
  48 GB DRAM at FP32 or BF16). Granite-1B-A400M ran AIMET basic-PTQ
  end-to-end on Prism CPU in 99.8 s; cos +0.656 vs FP32. Wrote
  ~80 LOC `granite_moe_adapters.py` (3 classes) — first reference
  implementation for extending AIMET to a non-blessed MoE arch.
  Deliverable: `last_side_quest/sq3_small_moe/findings.md`.
- **2026-04-28** — SQ1.b and SQ1.c both close. Path B: multi-round
  naive serial spec-decode end-to-end, JSON 91% accept reproduced,
  0.19× speedup (NPU re-prefill dominates). Path C: real batched
  verify + KV rewind via in-process llama-cpp-python (built from
  source on Prism x86_64), JSON 91% accept reproduced again,
  0.67× same-arch speedup — batched verify is 3.5× more per-round-
  efficient than serial. Free-form Python: 9% accept due to cross-
  arch drift (Prism x86_64 vs ARM64 native at precision boundary).
  Two follow-ups would deliver real speedup: NPU rewind op (~half-
  session) and ARM64 llama-cpp-python build (clang-on-ARM toolchain).
  Deliverables: `demo_path_b.py`, `demo_path_c.py` updates to
  `findings.md`.
- **2026-04-28** — SQ6 Phases A → C land in one session.
  Phase A: `npu_engine/http_server.py` is a working OpenAI-compatible
  HTTP server backed by `sidecar.py serve_chat_request`; smoke-tested
  EOS, stop sequences, multi-turn, real HTML coding task. NPU TG
  24-27 t/s. Phase A.5: SSE streaming via `chat_stream` op + async
  generator in `_do_chat_streaming` (BPE-correct rolling decode).
  Phase B: stateful streams in sidecar (`stream_open`/`truncate`/
  `append`/`decode`/`close`) + `ConversationState` LCP machinery in
  http_server. Wins: 2.6× on strict-prefix re-send, 2.3× on realistic
  2-turn coding chat. Heuristic: stay AR1-append for deltas <1024
  tokens. The same Phase B refactor unlocks SQ1's NPU rewind op
  (Path C STATEFUL).
  Phase C: real-world pi (`@mariozechner/pi-coding-agent` v0.70.6)
  A/B between NPU and OpenCL Qwen3-4B on a coding prompt. Throughput
  near-tied (NPU 7.96 s vs OpenCL 6.91 s direct curl). Headline win
  is the UX axis: silent NPU vs audible fan + coil whine on OpenCL
  while running archicad in foreground on battery. Vulkan-4B chat
  found broken at this llama.cpp commit (token soup + parse error).
  Deliverable: `last_side_quest/sq6_small_server/findings.md`.
- **2026-04-29** — SQ6 Phase D lands. Daily-driver Q3.6-35B-A3B CPU
  comparator measured. `build-cpu/llama-server -t 8` hits 32 t/s wall
  avg (direct curl), 39 s pi-default wall — beats both 4B candidates
  per-second on the same prompt. OpenCL ruled out for big-MoE: Adreno's
  2 GB `max_mem_alloc_size` fragments the 20 GB Q4_K_M and TG drops
  to 5 t/s. Conclusion: NPU 4B has to win on quality, not speed; the
  silent-island UX axis remains its argument.
- **2026-04-29** — SQ6 Phase E lands. Qwen2.5-7B-NPU sidecar
  generalization: new `qualcomm_qwen2_5_7b_oracle.py` and
  `bench_qwen2_5_7b_ortqnn.py` for the 6-partition w8a16 bundle;
  sidecar+http_server now select model via `--model` flag /
  `NPU_MODEL` env var. Cross-version QAIRT mismatch fixed via
  explicit `backend_path=` to system-installed QAIRT 2.45.40.
  E.3 stderr drainer thread fixes a 7B startup hang (PIPE buffer
  fill at 6-session ORT-QNN load). 7B HTML prompt 6.80 s wall /
  19.9 t/s — clean output, no W4A16 dup-attribute regression.
  Multi-turn KV-LCP cleaner than 4B (Qwen2.5 has no thinking-mode
  → no template/KV divergence). pi+tools test surfaces a structural
  gap: http_server ignores `tools=[...]`, model hallucinates tool
  output. AR128 7B prefill works direct (1289 t/s) but AR1↔AR128
  swap on 7B crashes the sidecar — HTP teardown/reload bug.
  Workaround: bump `--ar128-min-tokens` to 1500 to stay in AR1.
- **2026-04-29** — Umbrella plan resync. SQ6 progress row updated
  (Phases A→E landed); cross-cutting findings augmented with
  silent-island UX, daily-driver throughput baseline, Qwen2.5
  KV-LCP, tool-call protocol gap. Outstanding follow-ups
  consolidated as a new section. SQ4 confirmed as the only fully-
  pending umbrella deliverable.
- **2026-04-29** — SQ4 scope expanded from "sizing writeup" to
  "execute the cloud pipeline." New workspace
  `last_side_quest/sq4_cloud_adventure/` with unified plan doc
  (`findings.md`). Five milestones M1→M5 (Qwen3-0.6B → 4B → 14B →
  Qwen3.6-27B dense → Qwen3.6-35B-A3B MoE). Hardware: RunPod A40
  48 GB at $0.44/hr for M1-M3; A100 80 GB for M4-M5. Validation
  anchor for M2 is `models/qualcomm-qwen3-4b-ref/` (Qualcomm's
  shipping bundle on disk; target: 46/46 argmax match). Total
  budget $50-75. M1 pre-rent checklist in plan; first rent not
  started.
