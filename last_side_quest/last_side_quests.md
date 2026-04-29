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
| **SQ2** | ✅ **closed POSITIVE 2026-04-28** | aimet_torch v2 + SEQ_MSE/AdaScale work locally on Prism + WSL2 ARM64; aimet_onnx + qai_hub_models wrapper still cloud-only; basic-PTQ Qwen3-0.6B end-to-end demo lands negative-but-expected (cos -0.065 = V/O collapse repro) |
| **SQ3-small** | ✅ **closed POSITIVE 2026-04-28** (Granite-MoE branch + 3 follow-ups) | Granite-3.0-1B-A400M ran AIMET basic-PTQ end-to-end on Prism CPU; cos +0.656 (per-tensor experts) → +0.712 (per-channel experts, "A1" champion). SEQ_MSE-4 regressed (0.640); SEQ_MSE-16 partially recovered (0.682) — **A1 wins on Prism budget**. OLMoE-1B-7B failed end-to-end across 3 iterations — per-expert dispatch architecturally hostile to AIMET v2 (Granite's fused-experts + Qwen3-MoE's hit-filter both avoid this). AIMET 2.29 has no granitemoe adapter; written in ~80 LOC. **Qwen3-30B-A3B remains cloud-only** (corrected math: 30B FP32 = 120 GB, BF16 = 60 GB — neither fits 48 GB DRAM). |
| SQ4 | ⏳ partially fed by SQ2 | new prior: rent on demand, not by default — local AIMET unblocks design iteration on ≤4B; cloud only for production blessed bundles |
| **SQ5** | ✅ **closed POSITIVE 2026-04-27** | engine generalized cl=512..4096; coding-asst contexts viable up through 4K at 20 t/s |
| SQ6 | ⏳ pending | independent, anytime |

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
