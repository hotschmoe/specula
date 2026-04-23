# specula — long-horizon roadmap (post Phase 5.5)

Last updated: 2026-04-22 (companion to `current_status.md`).

This doc is the *longer-than-current_status* vision: where the project
goes once Phase 5.5 Lever C lands (positive or negative), and what
has to be true for specula to be a real answer to "how well does
Qwen3.5/6-class spec decode on a 48 GB unified-memory X2 Elite
Extreme laptop." `current_status.md` tracks the immediate active
work; this doc is the map we consult between phases.

The other two active investigations (`w4a16_investigation.md`,
`qwen3_perf_levers_investigation.md`) continue in parallel. Nothing
here preempts them. When they close (positive or negative), the
first workstream below picks up.

## Guiding principles

1. **Production targets are Qwen3.6 → Gemma4.** Qwen3 is scaffold;
   Qwen3.5 is the graduation checkpoint; Qwen3.6 + Gemma4 are the
   real deliverables. Every investigation below is evaluated on
   "does the answer transfer to Qwen3.6/Gemma4?" before we invest.
2. **Report by category, not by backend.** Prefill (prompt
   processing), token generation, and async orchestration are
   three different hardware-mapping problems. Each gets measured
   on NPU, GPU, CPU, and any hybrid pairing. The headline output
   of this roadmap is a category × backend matrix for the Zenbook
   A16.
3. **AC vs battery is a first-class axis.** The Phase 5.5 AC
   rerun showed a +27% delta on identical binaries. Every
   reported number has a power-state label. Any ranking based on
   mixed power states is invalid.
4. **Arm/Windows contributions are compounding.** Every build fix
   we land upstream (torch, QAIRT, llama.cpp, onnxruntime) makes
   the next session cheaper and the project reproducible by
   someone outside our shell. Treat PRs as infrastructure.
5. **Negative results are publishable.** The Adreno-OpenCL
   regression, the AI Hub preserve-list bug, the V-projection
   quant collapse — each is a concrete datapoint about an
   understudied platform. Document and upstream them.

## Hardware reference

- **Zenbook A16** (Snapdragon X2 Elite Extreme): 48 GB LPDDR5X @
  228 GB/s unified across CPU / Adreno / Hexagon. Hexagon v81,
  Adreno X2-90.
- **Power envelope matters.** AC (wall) vs battery shifts HTP
  thermals enough to swamp lever-level wins. Real-world use is
  both; we report both.
- **ARM + Windows is the production surface.** Linux/ARM
  (WSL2 or native) is the compile-path surface for toolchains
  (QAIRT, AIMET) that lag on Windows. x86 is dev-box only.

## Category × backend matrix — what we need to fill in

Headline deliverable at the end of this roadmap. Today's coverage
is sparse; every workstream below adds one or more cells.

| category | CPU | Adreno (OpenCL) | Hexagon (NPU) | NPU+CPU async | GPU+CPU async | NPU+GPU async |
|---|---|---|---|---|---|---|
| **prefill (PP512)** | Qwen3-8B: 164 t/s (18t) | 0.6B: 2674 t/s ✓ | ? | ? | ? | ? |
| **token generation (TG)** | 8B: 25.9 t/s (18t) | 8B: 13.5 t/s | 0.6B draft: 42 t/s | — | — | — |
| **spec-decode throughput** | 40.2 t/s k=3 ✓ | regression (Ph2) | 18.12 t/s AC (Ph5.5-B) | ← that's 5.5 | ? | ? |
| **concurrent sessions (2+ streams)** | ? | ? | ? | ? | ? | ? |
| **tool-calling / structured output** | ? | ? | ? | ? | ? | ? |

Cells marked `?` are workstream targets. Cells marked `—` are
category-incompatible (TG is single-stream by definition).

**Heterogeneous pipelined configurations** (W4.e) are a separate
axis layered on top of this matrix: the same **phase** (prefill,
draft, or verify) can run on different **islands** across the
pipeline, with **layer-wise KV streaming** (W4.d) hiding the
inter-island handoff under unified-memory fencing. The "best"
pipeline is context-dependent (short vs long prompt, AC vs
battery, single vs concurrent sessions), so W4's deliverable is
a policy-shaped answer — a decision tree over (prompt-length,
power-state, session-count) → (prefill island, draft island,
verify island) — not a single leaderboard row.

## Workstreams

Ordered by *information-per-session-hour*. Each ends with an
artifact: a CSV, a doc, or an upstream PR.

### W1 — NPU/GPU for prefill & prompt processing (target-side)

**Question.** Can we move the 8B target's prefill off CPU onto NPU
or GPU, cleanly? Today the target is CPU-only (Phase 2's mixed-
device was a regression on decode; prefill was never the problem).
Prefill is batched and bandwidth-friendly — the exact workload
Adreno (2674 t/s PP512 on 0.6B) and Hexagon (Qualcomm's Qwen3-4B
reference uses 128-token prefill chunks) handle well.

**Sub-questions.**
- W1.a: **GPU-prefill of 8B target, CPU-decode.** Split at the
  KV boundary: OpenCL runs through the prompt, copies KV to
  CPU-visible buffers (shared LPDDR5X, no DMA), CPU takes over
  for decode. Expected: PP2500+ t/s vs today's 164. Stretch
  hypothesis — our Phase 2 mixed-device regression was decode-
  side kernel-launch overhead; PP doesn't have that profile.
- W1.b: **NPU-prefill of target (Qwen3-8B).** Requires compiling
  an 8B context binary with prefill batch = 64 or 128. Qualcomm
  does this for 4B at 512 ctx × 128 prefill in 4-part splits
  — the recipe exists, just not at 8B scale on our pipeline.
  Binary size estimate: ~5 GB per part × 4 parts = 20 GB disk
  (w4a16). Fits comfortably in 48 GB RAM.
- W1.c: **Concurrent prefill + warm draft.** While 8B target
  prefills on GPU, keep 0.6B draft warm on NPU (weights
  resident, session alive). First decode step starts with both
  models already loaded — latency-to-first-token collapses.

**Deliverable.** Prefill benchmark CSV: (model × backend ×
prompt-length × power-state) → t/s, wall-to-first-token, memory
footprint. One row per cell.

**Cost.** W1.a is the cheap cell (~1 session — existing OpenCL
binary, just plumb the handoff). W1.b is 2-3 sessions (compile
pipeline extension for 8B already partially exists; 4-part split
is new). W1.c is incremental on top of A+B.

**Gate.** W1.a must show ≥10× prefill speedup with no decode
regression to justify W1.b. If W1.a is flat (unlikely given the
raw OpenCL PP number), investigate before spending the compile
budget on 8B NPU.

**Transfer to Qwen3.6/Gemma4.** Full — prefill is
architecture-agnostic for dense models. Hybrid (SSM + attention)
needs per-layer routing; flag for Phase 4 DFlash crossover.

### W2 — Larger NPU draft models (48% util → ?)

**Question.** Phase 5.5 Lever C left ~52% of Hexagon idle at the
current 0.6B × w8a16 draft. What does scaling to 1.7B or 4B draft
buy us, given:
- accept rate climbs with draft capability (literature: +5-10 pp
  at k=2 going 0.6B → 1.7B on same target);
- per-step NPU latency scales sub-linearly with weight count when
  we're bandwidth-bound (not compute-bound) — 3× weights doesn't
  mean 3× wall;
- the silicon headroom is measured: 48% util at baseline means a
  2.1× bigger model should still fit the per-step budget.

**Sub-questions.**
- W2.a: **Empirical utilization re-measurement.** Is 48% real or
  measurement artifact? Use Snapdragon Profiler / QNN perf
  counters during a steady-state sweep. Pin the number before
  scaling.
- W2.b: **Qwen3-1.7B draft on NPU.** Same pathb hoist + local
  QAIRT pipeline as 0.6B; weights triple, activations identical.
  Predicted per-step ~40-50 ms (vs 22 ms for 0.6B w8a16 AC).
  Predicted decode t/s at k=2 ≈ (1 + 0.85 × 2) / (45 ms × 3 + 160
  ms verify) = 2.7 / 0.295 = 9.2 t/s. That's **below** current
  Lever B 18.12 — scaling draft loses unless we also fix the
  per-call dispatch cost.
- W2.c: **Qwen3-4B draft on NPU.** 6× weights; predicted decode
  ≈ 5-6 t/s — clearly worse. Draft/target compute ratio must stay
  above some threshold (empirically ~6×) for spec-decode math to
  hold. Qwen3-4B × Qwen3-8B target is ratio 2× — below threshold.
- W2.d: **Tree/batched drafts on NPU (parallel at one NPU call).**
  The real lever behind the 48% utilization: draft k=4 tokens in
  *one* NPU call instead of k sequential calls. Needs a compile-
  time batch=4 variant **and** a verify-side tree-merge
  (the target-side multipath capability is B20 — shared prerequisite
  with EAGLE-3 / DFlash). This is where the headroom actually
  converts to throughput. Compare against Qualcomm's published
  128-token prefill-batch pattern; same idea at decode-batch scale.

**Deliverable.** Draft-sizing × utilization × throughput table.
Expected outcome: scaling the draft model *alone* is negative;
**tree-batched drafts (W2.d) are the real win** if the NPU
per-call overhead is fixed. That's the headline.

**Cost.** W2.a: 0.5 session (profiler). W2.b: 1 session.
W2.d: 2-3 sessions (compile + verify-side tree decoder).

**Gate.** Only proceed to W2.b if W2.a confirms >40% idle. If
utilization is already 80%+, scaling bigger is the only lever
and W2.d goes first.

**Transfer.** Partial — tree decode transfers to any family;
draft-sizing ratios are architecture-dependent and need
re-measuring on Qwen3.6 / Gemma4.

### W3 — TurboQuant investigation (Google research paper)

**Question.** Google's TurboQuant (random-projection / rotation-
based post-training quantization, typically for LLM weights and
activations) claims near-FP16 quality at w4 without calibration
datasets. Phase 5.5 Lever C spent a session on calibration-algorithm
sweeps (tf / tf_enhanced / mse / percentile) and found weight
precision dominates activation-range choices. TurboQuant's
rotation trick specifically targets the outlier-activation problem
that forces per-tensor w4 into V/O projection collapse.

**Sub-questions.**
- W3.a: **Reference reproduction.** Find a published TurboQuant
  impl (paper companion code or HuggingFace adaptation). Run
  against Qwen3-0.6B FP16 weights; compare the rotated+quantized
  ONNX against our current PTQ w4a16-local-pr baseline (cos=0.89).
  Target: cos ≥ 0.95 without calibration data.
- W3.b: **QAIRT pipeline compatibility.** TurboQuant emits a
  rotated ONNX graph (extra Linear/MatMul for the random
  projection). qairt-converter needs to preserve the rotation
  MatMuls as first-class compute, not fold them. Unknown whether
  HTP lowering handles this efficiently — probably requires
  leaving the rotation ops unquantized (fp16) while quantizing
  the underlying weight matmuls. Use `--quantization_overrides`
  for the rotation layers.
- W3.c: **Cross-family applicability.** If W3.a + W3.b land on
  Qwen3-0.6B, rerun the recipe on Qwen3.5-dense and Gemma4 draft
  sizes without re-engineering the pipeline. Score: is this a
  *reusable* Lever C replacement, or a one-off?

**Deliverable.** `docs/turboquant_investigation.md` with (a) paper
summary + our impl delta, (b) cos / accept / throughput on
Qwen3-0.6B draft, (c) portability notes.

**Cost.** 1-2 sessions if reference code exists. 3-4 if we have to
reimplement from the paper. Gate on finding companion code first.

**Transfer.** Full if it works — the whole point is calibration-
free PTQ. Becomes the default recipe for every future draft quant.

### W4 — Heterogeneous async orchestration (exolabs-style, 3-island)

**Question.** exolabs
([blog](https://blog.exolabs.net/nvidia-dgx-spark/)) showed
heterogeneous inference across machines (DGX Spark + Mac Studio)
wins by **overlapping communication with compute at layer
granularity** — Layer N's KV starts streaming to the Mac the moment
Layer N's prefill finishes on the DGX, while Layer N+1's prefill
launches in parallel. On *one* X2 Elite with three compute islands
(CPU / Adreno / Hexagon) sharing LPDDR5X, the same pattern should
apply intra-device, with a bonus: **unified memory eliminates the
"communication" cost entirely** — the "transfer" is a cache-line
flush + visibility fence, not a DMA. That flips the math: exolabs'
gains came from *hiding* a slow link; ours come from *exploiting* a
shared link. Phase 5.5 Lever A proved async draft∥verify pays +37%
(the 2-island case); the full pattern is 3-way with layer-wise
streaming.

**Reframe.** Three *phases* of inference — (1) prefill, (2) draft
speculation, (3) target verify/generation — each with its own
hardware-mapping preferences. Three *islands* available. The
question isn't "which island wins" but "which **assignment** of
phases-to-islands maximizes throughput, and can we pipeline them
layer-wise so no island idles." Today every phase runs on a
hard-coded island (prefill CPU / draft NPU / verify CPU); no
evidence that's optimal.

**Sub-questions.**
- W4.a: **Per-island concurrent execution.** Can we run NPU draft,
  GPU prefill-refresh, and CPU target-decode simultaneously
  without GIL or driver serialization? Probe: three threads,
  three sessions, measure wall vs sum. Expected: shared LPDDR5X
  contention is real but <20% penalty if workloads are staggered.
- W4.b: **Ring buffer for KV / token handoff.** Shared-memory
  ring (ION-backed ideally, host-ptr fallback) between NPU and
  CPU for zero-copy token handoff. Lever C's R4 probe was
  negative at the per-step level; W4 revisits it at the
  cross-island level where the handoff volume is higher.
- W4.c: **exolabs-style split inference — target across
  CPU+GPU.** 8B Q4_K_M: layers 0-15 on GPU (PP-fast), layers
  16-63 on CPU (TG-fast). Inverse of Phase 2's mixed-device
  which put *whole model* on GPU. Per-layer split was not tested;
  llama.cpp supports `--override-kv` / device-per-layer.
- W4.d: **Layer-wise KV streaming across islands (the exolabs
  trick, unified-memory variant).** When GPU prefills the target,
  expose per-layer KV buffers to CPU the moment each layer
  finishes — don't wait for the full prompt to complete on GPU
  before CPU decode can start. On unified LPDDR5X this is a
  `clFinish` / `qnn_fence` + atomic handoff, not a copy. Layer 1's
  KV becomes CPU-visible while GPU is still prefilling layers
  2..N. TTFT drops by the prefill depth; end-to-end throughput
  gains whatever fraction of prefill overlaps with first-token
  decode. Requires: (a) per-layer sync primitives exposed by each
  backend (OpenCL events, QNN fences, stdatomic on CPU); (b) a
  shared buffer layout both sides agree on (probably easier to
  standardize on GGML's KV layout than invent our own).
- W4.e: **Three-phase × three-device assignment matrix.** Empirical
  grid filling the "which island should do what" question. Each of
  {prefill, draft, verify} benchmarked on each of {CPU, Adreno,
  Hexagon} at the reference model sizes, plus the 9 one-way + 27
  three-way pipelined combinations, scored on
  (throughput, TTFT, power, idle time per island):

  | phase → island ↓ | CPU | Adreno | Hexagon |
  |---|---|---|---|
  | prefill | W1 baseline | W1.a | W1.b |
  | draft (0.6B / 1.7B) | Ph2 baseline | Ph2 regression | Ph5.5 ✓ |
  | verify (8B target) | Ph2 baseline | Ph2 regression | W1.b needs 8B NPU |

  Winning candidates to bench as full pipelines (3-way layer-
  streamed via W4.d): e.g. **prefill NPU / draft NPU / verify
  CPU** (today's config), **prefill GPU / draft NPU / verify
  CPU** (expected W1.a winner), **prefill GPU / draft CPU /
  verify CPU** (fallback). The actual optimum is almost certainly
  context-dependent (short prompt → prefill doesn't matter; long
  prompt → prefill becomes the bottleneck), so the deliverable is
  a **policy**, not a single winner.

**Deliverable.** `docs/async_orchestration.md` with: (a) per-
island timing trace; (b) the W4.e phase×island matrix filled in
with AC + battery numbers; (c) 3-5 pipelined recipe configs
benched end-to-end with layer-wise streaming enabled; (d) a
context-sensitive decision tree for "which pipeline to use at
prompt-length P and battery-state S."

**Cost.** W4.a: 1 session. W4.b: 2 sessions (memory plumbing hard
on Windows ARM). W4.c: 1 session (llama.cpp flags only).
W4.d: 2-3 sessions (per-backend sync primitives + shared-layout
negotiation). W4.e: 2 sessions measurement (depends on W1 + W2
filling the single-phase cells first). Total 8-9 sessions but
parallelizable with W1/W2.

**Transfer.** Full across model families. More importantly, the
layer-streaming primitive from W4.d is the **exact pattern
DFlash+DDTree on OpenCL will need** (Phase 4 notes in
current_status.md call out GPU↔CPU per-layer handoff as the
pattern). Landing W4.d pays off twice.

**Dependency.** W4.e is downstream of W1 (prefill single-phase
cells) and W2.a (utilization measurements). W4.d is prerequisite
for W4.e's pipelined cells.

### W5 — ARM/Windows build portability + upstream contributions

**Question.** Today our compile loop requires an x86 dev box
(local QAIRT 2.42 install worked on x86; on ARM Windows it's
unvalidated). Every toolchain gap we close is a compounding return.

**Sub-questions.**
- W5.a: **QAIRT on ARM Windows.** Does QAIRT 2.42 / 2.45 install +
  run converter/quantizer natively on Snapdragon X2 + Windows 11
  ARM? Today we run it on x86 and cross-ship `.bin`s. If it runs
  locally, the compile cycle becomes minutes instead of "find
  the x86 box". Probe: install + `--check` dry-run.
- W5.b: **PyTorch on ARM Windows.** torch currently has spotty
  ARM/Windows wheel coverage. AIMET (if we go back for W3/Lever C
  alternatives) depends on torch. The ecosystem gap is real and
  pyturch's issue tracker has active threads on it. Contribution:
  reproduce the failing cases, file/update issues, land fixes
  where tractable.
- W5.c: **llama.cpp ARM/Windows.** We already landed two local
  build fixes (clang-via-vcvarsarm64, KleidiAI .S patch). Next:
  upstream them plus the Adreno-OpenCL negative-result writeup
  (`docs/adreno_debugging.md` + `docs/adreno_opencl.md`) as a
  PR or discussion thread. Also: QNN EP direct integration (no
  external-drafter sidecar) — this is a major contribution if it
  lands.
- W5.d: **onnxruntime-qnn / QAIRT compat bugs.** Session 15 of
  Lever C catalogued three ORT-QNN 2.1.0 bugs on X2E
  (`docs/npu_ort_qnn_version_match.md`). File upstream with
  repros. Low cost; high goodwill / reproducibility value.

**Deliverable.** (a) A "build specula on pristine ARM Windows
laptop" doc that walks through every install step and works.
(b) Tracked list of upstream PRs + issues with links. (c) At
least one *landed* PR in each of: llama.cpp, torch or
onnxruntime, qairt samples (if Qualcomm accepts external
contribs).

**Cost.** W5.a: 1 session (install + verify). W5.b: ongoing,
opportunistic. W5.c: 2 sessions (strip local-only bits from
patches, write PR descriptions). W5.d: 0.5 session (file bugs
with existing logs).

**Transfer.** Infrastructure — benefits every subsequent phase.

### W6 — Production model graduation (Qwen3.6 + Gemma4)

**Question.** When Qwen3.5 lands as the graduation target (Phase
4 DFlash), we stay one tick behind the actual production target.
Qwen3.6 and Gemma4 are the real goals. The question is *when* to
cut over, not *if*.

**Sub-questions.**
- W6.a: **Qwen3.6 architecture audit.** As soon as Qwen3.6
  weights are public, read the arch diff vs 3.5. Flag any ops
  that aren't on our NPU/GPU kernel coverage yet. Update
  `docs/reference-projects.md` "kernel coverage" section.
- W6.b: **Gemma4 architecture audit.** Same for Gemma4 — expected
  different RoPE variant, different norm, possibly different
  attention (sliding-window). Per-family pathb rewrite might
  need a Gemma-specific variant.
- W6.c: **Cutover trigger.** Move from Qwen3.5 to Qwen3.6 when
  (i) W1 + W2 + W4 have a baseline on Qwen3.5 and (ii) Qwen3.6
  weights + gguf conversion are available. Same for Gemma4.
  Don't fork; just graduate.

**Deliverable.** Per-target audit doc + one end-to-end
throughput number that's directly comparable to the
Qwen3-baseline reference. Same category × backend matrix, just a
different model column.

**Cost.** W6.a/b: 0.5 session each once weights are out. W6.c:
once W9.b lands, the pipeline cost collapses to "invoke
`convert_hf_to_htp.py --model-id qwen3.6-0.6b` + probe +
sweep" — estimate 1 session per new model (vs 2-3 without the
automation tool). This dependency is the whole argument for
front-loading W9.

**Transfer.** This IS the transfer.

### W8 — End-to-end model-creation reproducibility day

**Question.** If a Qualcomm engineer, a researcher at another lab,
or a future-us on a fresh laptop followed our docs, could they
recreate `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` from a
clean HuggingFace checkpoint — with no guesswork and no out-of-band
folklore? Today the answer is "no": the recipe is spread across
`phase5_export_on_x86.md`, `phase5_local_qairt_compile.md`,
`phase5_local_qairt_compile_findings.md`, `w4a16_investigation.md`,
plus five gotchas only memory knows about (dot→underscore renaming,
preserve-list bug, calibration-order ordering, ORT-basic fold,
rope_tables formula).

**Deliverable.** One day (genuinely one session), one doc
(`docs/reproduce_npu_pipeline.md`) that walks linearly from:

1. Clean Windows 11 ARM laptop (or x86 dev box; both paths covered
   side-by-side).
2. Install QAIRT 2.42 (exact URL + the gateway-403 UA trick).
3. Export Qwen3-0.6B via optimum-onnx (exact version pins).
4. Apply `prep_onnx_for_ai_hub.py` (shape pin + ORT-basic fold).
5. Apply `rewrite_qwen3_pathb.py` (rotary hoist) — with the seam
   gotcha (3D vs 4D cos/sin shape) called out.
6. Capture calibration bundle.
7. Run qairt-converter + qairt-quantizer + qnn-context-binary-
   generator with the exact flags that produced w8a16-local.
8. Load in ORT-QNN 1.24.4 on ARM; validate cos ≥ 0.95 via probe.

Two branches documented explicitly: **AI Hub cloud path**
(preserves the preserve-list bug caveat + its workaround) and
**local QAIRT path** (no orchestration bugs; requires the SDK install).

**Acceptance gate.** Hand the doc + a clean laptop to someone who
has not seen this project. They must reach `cos ≥ 0.95` on the
Lever B baseline prompt within 8 hours. Time includes downloads,
model export, compile. Measure on a real outsider — a team member,
a HuggingFace contributor, or the x86-sessions collaborator.

**Artifacts.**
- `docs/reproduce_npu_pipeline.md` — linear recipe.
- `docs/reproduce_npu_pipeline_TROUBLESHOOTING.md` — every
  failure mode we hit, one paragraph each, with fix.
- `scripts/reproduce_npu_pipeline.ps1` / `.sh` — driver that
  runs the whole thing if all pins line up.
- A 2-3 minute screencast walking the first-timer through it
  (optional, publication-ready stretch).

**Cost.** 1 focused session. Most of the content already exists
in the phase docs — the cost is curation, linear ordering, and
validating the pipeline against current tool versions (pins
drift).

**Transfer.** Full. The same structure hosts Gemma4 / Qwen3.6
recipes (one more section each) once those land.

### W9 — Quant-format landscape on Snapdragon + automated multi-model pipeline

**Question.** Two coupled sub-questions:

1. **Is w4a16 actually optimal on Snapdragon X2E HTP?** The
   Phase 5 assumption came from Qualcomm's Qwen3-4B bundle
   shipping w4a16 — observational evidence, not a controlled
   measurement. Other formats that exist on QAIRT 2.42+ and
   might be better on our specific workload:
   - **w8a8** (INT8 symmetric weights + INT8 activations) — half
     the weight BW of w8a16, same HTP fast-path class as w4a16.
   - **w4a8** — strict-symmetric INT4 + INT8 act; smaller than
     w4a16 but activation range has to fit in 256 levels.
   - **mx4 / mx6 / mx8** (OCP microscaling formats) — QAIRT
     backend adds these progressively; not all X2E drivers
     accept them.
   - **Per-group weight quantization** (group=32, 64, 128) vs
     per-row vs per-tensor. We measured per-row beats per-tensor
     on w4 (Lever C w4a16-local-pr); untested on w8.
   - **Mixed-precision overrides** (V/O projections at w8,
     everything else at w4) — the differential-probe finding
     from Lever C session 17 suggests this is the real best-of-
     both-worlds point.

2. **Does quant-friendliness correlate with architecture?** Qwen3
   V-projection collapse at w4 was our specific pain point; Llama-3,
   Gemma-3, Phi-3, Mistral-7B each have their own idiosyncratic
   weight distributions. Do some architectures (e.g. Gemma's smaller
   head-dim, Llama's grouped-query attention) quantize to w4a16
   more cleanly than others on the exact same HTP silicon?

**Sub-questions.**
- W9.a: **Format A/B on Qwen3-0.6B.** Using the local-QAIRT pipeline
  from W8, produce seven binaries for the same pathb ONNX:
  fp16, w8a16, w8a8, w4a16-per-tensor, w4a16-per-row,
  w4a8-per-row, w4a16-mixed (V/O at w8, rest at w4). Measure
  per-step latency + cos + accept on Lever B reference prompt.
  Rank formats by (throughput, cos, binary size). Table goes into
  `docs/quant_format_landscape.md`.
- W9.b: **Automated conversion pipeline (`scripts/convert_hf_to_htp.py`).**
  Wraps the W8 recipe as a library callable with `--model-id
  qwen3.5-0.5b`, `--format w4a16-per-row`, `--ctx 256`. Handles:
  download from HuggingFace, optimum-onnx export (with arch
  detection), pathb rewrite (if architecture matches Qwen-family
  rotary), calibration capture, QAIRT compile, wrapper ONNX
  generation. Per-architecture adapters in
  `scripts/arch_adapters/{qwen3,qwen35,gemma3,llama3,phi3}.py`.
- W9.c: **Model-family survey.** Pick 4-5 open-weight models at
  0.5-1B draft scale: Qwen3.5-0.5B, Gemma-2-2B, Llama-3.2-1B,
  Phi-3.5-mini (small), TinyLlama-1.1B. Run each through W9.b at
  w4a16, w8a16, and w4a16-mixed. Report: does each compile
  cleanly? What's the accuracy drop? What's the per-step
  latency on X2E? Is there a pattern (head-dim, MLP ratio,
  GQA group size) that predicts quant-friendliness?

**Deliverable.**
- `docs/quant_format_landscape.md` — the headline doc. Table per
  format × model, ranking by throughput + accuracy, with
  architectural commentary.
- `scripts/convert_hf_to_htp.py` — the one-shot conversion tool.
  MVP produces a working binary for at least 3 architectures.
- A published "Qualcomm X2E quantization cheat-sheet" (blog
  post / HuggingFace collection / gist) from the doc above.
  External-facing artifact.

**Cost.** W9.a: 1-2 sessions (seven compiles × probe gate; local
QAIRT is fast per run, ~2 min each, but calibration differs).
W9.b: 2-3 sessions (per-arch adapters are the real cost). W9.c:
1 session per extra architecture once W9.b is stable.

**Gate.** W9.a must complete before W9.b starts — no point
automating a recipe we haven't confirmed produces best-of-class
numbers. If w4a16-mixed dominates (likely, per session-17
differential evidence), the automation tool's default is w4a16-mixed,
not w8a16.

**Transfer.** Direct enabler for W6 (Qwen3.6 / Gemma4) — the
graduation workstream becomes "run `convert_hf_to_htp.py`" instead
of "redo the Lever C investigation." This is the lever that
compounds.

### W7 — Real-world harness: concurrency + tool calling via opencode

**Question.** Single-stream decode t/s is a microbenchmark. Real
coding-assistant use is: multiple concurrent sessions, tool calls
with structured output, prompts that range from 200 tokens
(conversation) to 8K (codebase context). Numbers from W1-W4
*predict* but don't *demonstrate* production performance.

**Sub-questions.**
- W7.a: **opencode as the harness.** opencode is a CLI coding
  assistant that talks to any local or remote model server. Run
  our llama-server + NPU-spec sidecar behind opencode on 5-10
  realistic development tasks (read a file, modify it, run a
  command, iterate). Log per-interaction latency, TTFT, tokens
  generated.
- W7.b: **Concurrent sessions.** 2 or 3 opencode instances
  against one server. Measure per-session degradation vs solo.
  Maps to "can one laptop serve a team of two?"
- W7.c: **Tool calling with structured output.** Tool schemas
  force JSON decode; JSON has different accept-rate profile than
  free-form code. We have `prompts/structured_json.jsonl` but
  haven't run it through an agentic tool-call loop. Compare
  draft accept rate in tool-call mode vs free-form.
- W7.d: **AC vs battery end-to-end.** Every W7 run is repeated
  on wall power and on battery. Final report's AC/battery delta
  at the *task-completion* level (not t/s) is the number
  laptop buyers actually care about.

**Deliverable.** `docs/zenbook_a16_real_world.md` — the headline
doc for laptop-class spec-decode. Target audience: anyone
deciding whether local inference on an X2E laptop is viable.
Includes tables, short task traces, and AC/battery caveats.

**Cost.** 2 sessions (harness setup + multi-run analysis). Worth
more because it's the external-facing artifact.

**Transfer.** Full — same harness covers Qwen3.5 / Qwen3.6 /
Gemma4 without reconfiguration.

## Rolling backlog — ideas captured before they're forgotten

Everything below is a "capture now, formalize when it matures"
item. Promote to a proper Wn workstream when the question
sharpens or a session slot opens up. Group-numbered (B1, B2, …)
so later cross-refs don't churn.

### Measurement & characterization

- **B1. Joules/token power-rail measurement.** Windows ETW +
  Qualcomm telemetry to expose per-rail (CPU/GPU/NPU) power
  during decode. AC-vs-battery answers *"can I sustain this?"*;
  J/token answers *"how much battery does one agentic loop
  cost?"* — the question a laptop buyer actually has. Plug into
  the W7 opencode harness so every task has a power-cost column.
- **B2. Sustained-throughput thermal curve.** Formal
  characterization over 30-minute continuous decode: t/s vs
  time, with throttle-knee identification. Separates burst
  numbers (what we report today) from real-world numbers (what
  an agentic loop actually gets). Applies to every binary we
  benchmark; one harness, many curves.
- **B3. Performance regression CI.** Nightly / per-HEAD micro-
  harness that reruns one canonical prompt at k=2 steady-state
  against llama.cpp HEAD + ORT-QNN + QAIRT. Catches upstream
  regressions before they cost us a session of "why did the
  number change." Cheap once the sweep harness exists.

### Runtime architecture (structural, expensive, compounding)

- **B4. Upstream llama.cpp QNN backend.** Community interest
  exists; no one has landed a first-class QNN ggml backend.
  Would collapse our external-drafter sidecar into stock
  `--draft` flags. Multi-session contribution but potentially
  project-defining — future specula versions could be "three
  llama.cpp flags" instead of "custom sidecar + wrapper ONNX
  + QAIRT pipeline." Gate on W5.a landing so we know the ARM
  compile path for the EP.
- **B5. Long-context compile tiers.** Our binary bakes ctx=256.
  Qwen3 supports 32K+. Qualcomm's reference ships 5 tiers
  (512/1024/2048/3072/4096) with weight-sharing. A tiered
  loader that picks the smallest tier ≥ current prompt is a
  real runtime feature — not a benchmark number. Cost:
  multiple compiles × one wrapper selector. Payoff: the project
  handles realistic conversation / codebase contexts, not just
  humaneval-scale prompts.
- **B6. Concurrent-session KV memory model.** W7.b asks "can 2
  users share one server." The deeper question: can we
  partition 48 GB so N sessions each get their own NPU KV slice
  without re-uploading weights? Touches QAIRT's
  `weight_sharing_enabled` flag (Qualcomm's reference uses it)
  + session-level KV allocation in our wrapper.
- **B7. Continuous batching × spec-decode interaction.**
  llama-server supports continuous batching. Our sidecar doesn't
  participate. When two users' prompts arrive mid-round, does
  spec-decode correctly pause / resume? Untested. Feature gap
  surfaces the moment B6 lands.

### Stalled workstreams worth un-stalling

- **B8. KleidiAI / SME2 retry.** Phase 0 deferred this because
  runtime SME2 trapped. Compiler probe already passes
  (`HAVE_SME - Success`). If SME2 lands, CPU PP + TG move
  materially and every "NPU vs CPU" ratio we cite shifts. Worth
  one focused session once the current phase winds down.
  `scripts/build_llama_cpp.ps1 -Preset cpu-kleidiai` wires the
  build; the ZA-tile user-mode state is the runtime suspicion.
- **B9. EAGLE-3 on NPU draft.** Demoted pre-Phase-5 because it
  only moves accept rate, and CPU wasn't accept-bound. Two
  reasons to reopen now:
  1. On NPU where per-step is the bottleneck at large k, EAGLE's
     smaller-per-step profile might change the calculus.
  2. **Compounding with w4a16 (new insight, 2026-04-22).** EAGLE-3
     drafters are trained against the target's hidden states and
     publish accept rates 85-92% on LLaMA — 10-20 pp above vanilla
     draft-model spec. That absorbs the PTQ accept-rate tax that
     killed w4a16-local-pr (-17 pp vs w8a16). Makes the accept-rate
     axis, which w4a16 alone can't move, attackable independently.
     The V/O-projection precision sensitivity we saw on Qwen3-0.6B
     may not transfer to a co-adapted EAGLE head.
  Cost is no longer "one cheap session" — EAGLE-3 needs target
  hidden-state exposure (llama-server `/completion` returns tokens
  only), so it shares its main cost with B20 (custom verifier).
  See `docs/w4a16_investigation_continued.md` Axis D.1 for full
  scoping. Gate on B20 landing.

### Cross-cutting prerequisites (enable whole classes of lever)

- **B20. Custom multipath-capable verifier (post-`/completion`).**
  The binding structural limitation behind three otherwise-independent
  levers. `llama-server`'s `/completion` endpoint returns tokens only
  — no hidden states, no multipath scoring, no batched alternate
  candidates. That single constraint blocks:
  - **Tree / multipath verify** (R3 in levers doc, W2.d "verify-side
    tree-merge"): can't score K parallel draft paths in one target
    call.
  - **EAGLE-3** (B9): head needs to consume target's residual stream
    per step.
  - **DFlash + DDTree** (Phase 4, see current_status.md §Phase 4):
    lucebox-hub's reference impl uses a custom tree-verify kernel;
    no `/completion`-shaped endpoint can drive it.
  Three build paths, roughly equal cost:
  1. **In-process target via llama.cpp Python bindings** — run
     `llama_cpp.Llama` directly in our sidecar, hook residual
     stream + expose a multipath `decode_tree(prompt, tree)` call.
     Cheapest; no fork. Prototype on CPU target; add OpenCL later
     once the API settles.
  2. **Fork llama-server** — add `/completion_tree` and
     `/completion_hidden` endpoints. Upstream the fork as a PR after
     it stabilizes. Most reusable for the broader llama.cpp community.
  3. **Write a minimal target runner** — ggml-based single-binary
     that loads a GGUF + exposes our exact API over a Unix socket.
     Most control, most code; valuable only if we end up needing
     custom target-side features anyway.
  Cost: ~2-3 sessions for path (1), ~4-5 for (2), ~6+ for (3). Pick
  (1) first; promote to (2) once we know which endpoint shape
  actually matters. **Transfer: full** — the resulting endpoint
  serves Qwen3.5/Qwen3.6/Gemma4 identically; this is
  infrastructure.
  **Promotion trigger: any of B9 / W2.d / Phase 4 becoming active.**
  When two of those fire, B20 goes first because it's a shared
  prerequisite.

### Distribution & reach (turn our work into artifacts others use)

- **B10. Publish HTP binaries to HuggingFace.** Upload
  `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` +
  wrapper + README as a HF model. Zero-effort reproduction
  for anyone else with an X2E laptop. Becomes a near-trivial
  release cadence once W9.b exists and we can automate.
- **B11. Snapdragon SKU-transfer survey.** X2 Elite Extreme is
  one SKU; X2 Elite (non-Extreme), X1 Elite, and phone-class
  Snapdragon 8 Gen 4 share Hexagon v75-v81. Do our
  w8a16-local binaries load on them? 4-hour probe gives us a
  "covered hardware" footprint to claim. Likely requires
  borrowing hardware or CI cloud instances.
- **B12. Cross-runtime bake-off.** MLC-LLM, direct-QNN
  (no ORT), ort-QNN, llama.cpp+QNN (once B4 lands), Genie SDK.
  Same Qwen3-0.6B model, same prompt, per-runtime t/s. Tells
  us whether ORT-QNN is actually the right runtime or we're
  sitting in a local optimum. Pairs with B4 (if QNN-in-
  llama.cpp wins, that becomes our default).
- **B13. Cross-platform laptop bake-off.** M-series ANE (via
  MLX), RTX 4070/5070 laptops, Intel Core Ultra NPU (OpenVINO).
  Same draft/target model pair, same prompts. "Which laptop is
  best for local LLM" is a real consumer question with no clean
  answer today; one published table is high-visibility.
  Requires access to each platform — partner / borrow / rent.

### Stretch features (Qwen3.6 / Gemma4 era)

- **B14. Multimodal on NPU/GPU.** Gemma4 is almost certainly
  multimodal. Vision encoder mapping on NPU (image tower →
  adapter → text decoder on CPU-spec). Whole new kernel-
  coverage audit. Flag as soon as Gemma4 weights land.
- **B15. On-device LoRA / adapter swap.** Task-specific
  drafters loaded at runtime without recompile. Needs QAIRT
  to expose weight-patch APIs; probably not today. Track QAIRT
  release notes for when it becomes possible.
- **B16. Power-aware adaptive policy.** Runtime drops from
  k=3 → k=2 → greedy based on battery-charge + prompt-length.
  Feature, not a benchmark. Layers on top of B1's power
  measurements.
- **B17. Draft distillation on-device.** Task-adaptive
  drafter: collect the user's real usage (with consent) and
  fine-tune a LoRA on the draft to match their prompt
  distribution. Research-heavy but a real differentiator for
  local inference.

### Meta

- **B18. Closing paper / blog post.** When the project arc
  settles — first open benchmarks + pipeline + upstream
  contributions for local spec-decode on Snapdragon X2 laptops
  — it's worth writing up once externally. Flag now so it
  doesn't become a "we never wrote it up" regret. Natural
  artifact after W7 + W9 + B10/B12 all land.
- **B19. Security / privacy posture.** Local inference's
  selling point. What isolation does the user actually get? Are
  HTP buffers cleared between sessions? Are weights protected
  from other processes? Audit question; relevant for any
  opencode-style deployment where tool-calling touches the
  filesystem.

### Promotion criteria

Move a B-item to a W-workstream when ANY of:
(i) a closed W-workstream answered the prerequisite;
(ii) the question sharpens enough to have a gate condition;
(iii) an external event (Gemma4 ships, upstream PR merges,
     hardware arrives) makes it the right moment.

Re-read this section at the start of every new phase. Items
stale-out: if a B-item sits untouched for 3 months AND no
promotion criterion fired, move it to a separate "parked"
sub-section. Keeps the list honest.

## Prioritization

Dependency graph (→ means "enables"):

```
 Phase 5.5 Lever C close (current_status.md)
         │
         ▼
 W5.a QAIRT-on-ARM probe  ────►  W5.c/d upstream PRs  (ongoing)
         │
         ▼
 W8 reproducibility day (linear recipe, shareable)
         │
         ├────────────────────────┐
         ▼                        ▼
 W1 prefill (GPU/NPU)   W9.a quant-format A/B on Qwen3-0.6B
         │                        │
         ▼                        ▼
 W2 utilization + tree   W9.b automated HF→HTP pipeline
   drafts                         │
         │                        ▼
         │             W9.c multi-model / multi-arch survey
         │                        │
         ▼                        ▼
 W4 async orchestration  W3 TurboQuant (layered on W9 pipeline)
         │                        │
         ▼                        ▼
 W7 opencode harness     W6 Qwen3.6 / Gemma4 graduation
         │                        │
         └──────────┬─────────────┘
                    ▼
         Category × backend matrix complete
         (AC + battery; concurrent + solo; tool + free-form)
```

Starting order, once Lever C resolves:

1. **W5.a** (QAIRT-on-ARM probe) — cheapest, unblocks compile-loop
   friction for every subsequent workstream. ~0.5 session.
2. **W8** (reproducibility day) — capture the Phase-5.5 recipe
   while it's fresh, before tool versions drift. Output is the
   base any external collaborator (or future-us) uses. ~1 session.
3. **W9.a** (quant-format A/B) — builds directly on W8's
   pipeline; answers whether w4a16 is actually the right default
   before we commit to it for Qwen3.6 / Gemma4. ~1-2 sessions.
4. **W1.a** (GPU prefill of 8B target) — biggest single-step
   throughput win that doesn't require new toolchain. ~1 session.
5. **W2.a** (utilization measurement) — blocking gate on whether
   W2.b-d are worth pursuing. ~0.5 session.
6. **W9.b** (automated HF→HTP pipeline) — parallel with W1/W2;
   pays off the instant we need a second model. ~2-3 sessions.
7. **W4.c** (CPU+GPU layer split) — cheap async probe using
   existing llama.cpp flags. ~1 session.
7b. **W4.d** (layer-wise KV streaming primitive) — build the sync/
    handoff layer that W4.e and later Phase 4 DFlash both depend
    on. Parallel with W4.c since they're independent. ~2-3 sessions.
7c. **W4.e** (3-phase × 3-island assignment matrix) — depends on
    W1.a, W2.a, and W4.d landing. Fills the heterogeneous-pipeline
    cells of the category×backend matrix and produces the
    context-sensitive policy. ~2 sessions.
8. **W7.a** (opencode harness bring-up) — parallel; one session
   to get wired, then it runs in the background of every
   subsequent lever.
9. **W3** (TurboQuant) — after W9.a establishes the PTQ baseline;
   layered as just another option in the W9.b pipeline.
10. **W9.c** (multi-arch survey) — once W9.b has 2+ arch adapters.
11. **W5.c/d** (upstream PRs) — land at the end of each workstream
    that produced a PR-worthy artifact.
12. **W6** (Qwen3.6 / Gemma4 cutover) — W9.b is the shortest path
    here; "graduation" reduces to "run `convert_hf_to_htp.py
    --model-id gemma4-...`".

## Success criteria

- **Minimum.** Category × backend matrix has ≥ 80% of cells
  filled with AC-measured numbers. At least one upstream PR
  landed. opencode harness produces a reproducible real-world
  trace.
- **Target.** NPU-spec throughput at k=2 clears 25 t/s on AC
  with Qwen3-8B target (currently 18.12 — needs W1 + W2.d or
  W3). Same measurement on Qwen3.5-dense (after Phase 4). Same
  matrix extended to Qwen3.6 once available.
- **Stretch.** Concurrent 2-session opencode throughput above
  15 t/s per session on battery. ARM-Windows clean-install build
  path works end-to-end without an x86 sidecar. Our local-QAIRT
  pathb recipe lands upstream somewhere (Qualcomm samples,
  llama.cpp discussion, HuggingFace optimum).

## Admin / cleanup — separate from research workstreams

Tracked here because they have no research value but have real
disk + repo hygiene cost. Schedule in the first 30 minutes of
any session where the main workstream is blocked.

### Large-binary cleanup (models/ — total ~60 GB on disk)

Current inventory (as of 2026-04-22):

| artifact | size | status | action |
|---|---:|---|---|
| `Qwen3-0.6B-Q8_0.gguf` | 640 MB | baseline | keep |
| `Qwen3-1.7B-Q8_0.gguf` | 1.8 GB | W2 input | keep |
| `Qwen3-8B-Q4_K_M.gguf` | 5.0 GB | target | keep |
| `qualcomm-qwen3-4b-ref/` | 6.0 GB | reference decoded | keep through W3; archive after |
| `qwen3-0.6b-optimum/` | 2.9 GB | intermediate export | **archive** (pathb supersedes) |
| `qwen3-0.6b-patha/` | 2.9 GB | closed variant | **archive** |
| `qwen3-0.6b-patha-ai-hub/` | 2.9 GB | closed staging | **delete** (regeneratable) |
| `qwen3-0.6b-pathbmask/` | 2.9 GB | closed variant | **archive** (Lever B binary kept) |
| `qwen3-0.6b-pathbmask-ai-hub/` | 2.9 GB | closed staging | **delete** |
| `qwen3-0.6b-pathbmask-ai-hub-ctx256/` | 2.9 GB | closed staging | **delete** |
| `qwen3-0.6b-pathb/` | 2.9 GB | active (Lever C input) | keep |
| `qwen3-0.6b-pathb-ai-hub-ctx256/` | 2.9 GB | active staging | keep until Lever C closes, then delete |
| `qwen3_0_6b_draft_v81_ctx512.broken.bin` | 1.5 GB | pre-Path-A attempt | **delete** |
| `qwen3_0_6b_draft_v81_ctx512.patha.bin` | 1.5 GB | Phase 5 artifact | **archive** (Phase 5 closed) |
| `qwen3_0_6b_draft_v81_ctx512.pathbmask.bin` | 1.5 GB | Phase 5 artifact | **archive** |
| `qwen3_0_6b_draft_v81_ctx256.pathbmask.bin` | 1.5 GB | Lever B reference | **keep** — 18.12 t/s baseline comes from here |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-a.bin` | 918 MB | AI Hub bug victim | **archive** |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin` | 918 MB | Lever C (tf) | **delete** (negative) |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-mse.bin` | 918 MB | Lever C (mse) | **delete** (negative) |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-tfe.bin` | 918 MB | Lever C (tfe) | **delete** (negative) |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-pr.bin` | 620 MB | Lever C soft pass | keep (sweep pending) |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` | 918 MB | Lever C full pass | keep |
| `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local-pr.bin` | 918 MB | Lever C soft pass | **delete** (dominated by w8a16) |
| `qwen3_0_6b_draft_v81_ctx256.pathb.fp16-local.bin` | 1.5 GB | reference | **archive** after Lever C close |

Archive target: NAS under `Z:\exposed\archive\specula-phase5\`.
Estimated freed: ~25 GB immediate delete + ~15 GB archive-then-
delete.

### Calibration bundles (models/calibration/ — 8 GB)

| bundle | size | status | action |
|---|---:|---|---|
| `bundle_a_ctx256.npz` | 3.5 GB | pre-pathb schema | **delete** (superseded) |
| `bundle_a_pathb_ctx256.npz` | 3.5 GB | active | keep through Lever C close |
| `bundle_b_ctx256.npz` | 1.2 GB | pre-pathb, never used | **delete** |

Regeneratable via `scripts/capture_calibration_samples.py` —
keeping the code + manifest.json is sufficient for reproducibility.

### Results directory (results/ — 82 MB across 175 files)

Not large. Cleanup is cognitive, not disk-bound:

- Consolidate per-session logs into `results/archive/phase5/`,
  `results/archive/phase5.5-leverA/`, etc. Keep the CSVs
  flat; bury the `.log`/`.stdout` under session dirs.
- Every closed phase gets a `results/archive/phaseN_closed.md`
  index pointing at the artifacts.

### Docs cleanup

Keep all `docs/phase5_*.md` — they're the session-by-session
trail. Don't consolidate; future-us will want the grain. But:

- Add a `docs/INDEX.md` listing every doc with a one-liner and
  its lifecycle status (active / closed / reference).
- Once Lever C closes, collapse
  `docs/phase5_lever_c_x86_ask.md` + `docs/phase5_local_qairt_compile.md`
  + `docs/phase5_local_qairt_compile_findings.md` into one
  `docs/phase5_lever_c_retro.md`.

### Misc

- `docs/QAIRT-Docs/` — vendored Qualcomm docs. Useful; keep.
  Confirm license/redistribution terms before any public push.
- `qualcomm_qwen3_4b_part1.wrapper.onnx` in root of `models/` —
  stray artifact from a Qualcomm-reference probe; move into
  `models/qualcomm-qwen3-4b-ref/` or delete.
- `.claude/` at repo root — keep gitignored; verify `.gitignore`
  covers it.

## Companion docs

- `current_status.md` — immediate status, session-by-session.
- `docs/qwen3_perf_levers_investigation.md` — Phase 5.5 A/B/C detail.
- `docs/w4a16_investigation.md` — Lever C drill-down (active).
- `docs/npu_results.md` — Phase 5 close writeup.
- `docs/reference-projects.md` — trident / lucebox / voice_project /
  gguf_models pointers (update with Qwen3.6 / Gemma4 refs as they land).

This roadmap is a living doc. Every workstream close-out should
update its section inline (result + delta + transfer notes) the
same way `qwen3_perf_levers_investigation.md` records per-lever
results. When a workstream graduates to Qwen3.5/6 or Gemma4,
note the pre/post delta and keep both numbers for the
laptop-class reference.
