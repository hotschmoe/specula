# specula -- current status

Last updated: 2026-04-21 (session 9 -- Phase 5 step 4 closed: HTP context binary on disk)

Living document. Update every few turns. Anyone picking this up cold should
be able to read this page, skim the README, and resume work.

## Where we are in the phase plan

**Phase 0 -- Infrastructure:** in progress.

- [x] Repo scaffolded per README layout (scripts/, prompts/, docs/, models/, results/, notebooks/)
- [x] `docs/reference-projects.md` written -- pointers to trident, voice_project, gguf_models
- [x] Models downloaded (`core` tier -- Qwen3-0.6B-Q8_0, Qwen3-1.7B-Q8_0, Qwen3-8B-Q4_K_M in `models/`)
- [x] llama.cpp sibling checkout at `llama.cpp/` (HEAD `e365e658f07b63371489570dfde597f199b26c23`)
- [x] **Preset `cpu`** built (`llama.cpp\build-cpu\bin\`), runtime DLLs copied, smoke-tested
- [x] Vulkan SDK installed (`C:\VulkanSDK\1.4.341.1\`, `VULKAN_SDK` env var set)
- [x] **Preset `vulkan`** built; device enumeration correct (Adreno X2-90, native driver, `KHR_coopmat`). **Vulkan on this driver is broken for correct inference.** Tested five env-var configs (B0 baseline, B1 DISABLE_COOPMAT, B4 DISABLE_F16+COOPMAT+COOPMAT2, B6 DISABLE_INTEGER_DOT_PRODUCT, B7 all four disabled). All five produce incorrect output on Qwen3-0.6B Q8_0 with greedy/seed=1 while CPU on same seed returns coherent Qwen3 thinking-mode text. B6/B7 additionally collapse to a single repeated token (`edlyedlyedly...`) — disabling `INTEGER_DOT_PRODUCT` makes things strictly worse. `DISABLE_F16=1` makes PP ~30× faster (20 → 600 t/s) but fast + wrong, not a rescue. Vulkan memory breakdown at shutdown also shows `unaccounted | 17592186039033` MiB — a size_t underflow in the backend's buffer accounting. Decision: **pivot primary GPU attention to OpenCL** (Qualcomm's maintained backend); keep vulkan build around for later retry after a Qualcomm driver update. Full writeup in `docs/adreno_debugging.md`.
- [x] **Preset `opencl`** built and correctness+perf validated.
  **OpenCL Adreno is the working GPU backend on this machine.**
  Qwen3-0.6B Q8_0 bench: **PP128 1926 t/s, PP512 2674 t/s, TG64 111
  t/s** (vs CPU 826 / — / 111; vs Vulkan fast-but-wrong B3
  599 / 604 / 100). Output coherence matches CPU greedy reference.
  Full writeup in `docs/adreno_opencl.md`.
- [ ] **Preset `vulkan-opencl`** -- preset is wired (same SDK flags
  as `opencl`); not yet rebuilt. Vulkan side is still broken on this
  driver per `adreno_debugging.md`, so there's no immediate reason to
  exercise this preset; keep it around for post-driver-update retest.
- [ ] **Preset `cpu-kleidiai`** -- deferred to Phase 1 SME2 retry
- [ ] Hexagon backend -- out of band (Qualcomm docker toolchain); not wired into `build_llama_cpp.ps1`
- [ ] Sweep harness validated end-to-end (scripts exist; not yet producing real CSVs)

Note (2026-04-19): initial combined `vulkan-opencl` preset split into
standalone `vulkan` and `opencl` presets so a missing OpenCL SDK doesn't
block Vulkan work. Combined preset kept for when both are satisfied.

Note (2026-04-19, session 2): the vulkan build does not include
`llama-perplexity` or `llama-completion` (our `LLAMA_BUILD_TOOLS`
subset is narrower than default). This matters because the shipped
`llama-cli` silently ignores `-no-cnv` on newer llama.cpp and always
enters conversation mode — it prints *"please use llama-completion
instead"* and falls back. For scripted correctness/perplexity assays
we need to widen the build tool set (or drive `llama-server` over
HTTP). Not blocking the OpenCL pivot; bundle with the next rebuild.

Note (2026-04-20, session 3): target list in `build_llama_cpp.ps1`
now includes `llama-completion` and `llama-perplexity`. They're in
the `build-opencl/bin/` output. **Open caveat:** `llama-completion`
on HEAD `fd6ae4c…` also defaults to conversation mode and hangs
waiting for interactive input after `-n 64` tokens are generated —
the generation itself is correct (we verified coherence), but
scripted runs still need a Ctrl-C or a stdin close. For fully
hands-off automation the safer tool is `llama-server` over HTTP.

Note (2026-04-19, session 2): **Qualcomm's own GPU-compute path for
llama.cpp is the OpenCL backend**, not Vulkan. `ggml-opencl` has
Adreno-specific optimizations landed by Qualcomm engineers (lhez,
max-krasnyansky). Vulkan remains useful as a second GPU path, but
OpenCL is the vendor-blessed one and should be unblocked soon.

**Phase 1:** CPU + OpenCL baselines landed in `results/baseline-*.csv`
(sweep covers 0.6B / 1.7B / 8B at 8/12/18 threads, PP128/512 + TG64/128).
Reference numbers the rest of this document assumes:

- 8B Q4_K_M CPU @ 18t: PP512 164 t/s, TG128 **25.91 t/s**, TG64 26.48 t/s
- 0.6B Q8_0 CPU @ 18t: TG128 **149.66 t/s** (draft ceiling)
- 8B Q4_K_M OpenCL: PP512 much stronger, TG lower than CPU -- CPU wins TG

**Phase 2 -- Stock speculative decoding (session 4, 2026-04-20):**

Six sweeps complete. Fixed rig throughout: target Qwen3-8B-Q4_K_M +
draft Qwen3-0.6B-Q8_0, greedy (temp=0), `--draft-min 0`, `-n 256`,
10-prompt fixtures. Baselines from Phase 1:

- 8B CPU TG: **25.91 t/s** -- the reference for every speedup below.
- 8B OpenCL TG: 13.50 t/s (PP is strong, TG is weak on Adreno).

### Summary table (all humaneval, best k per config)

| Config (k shown) | mean accept | mean decode t/s | vs CPU TG | vs own TG baseline |
|------------------|------------:|----------------:|----------:|-------------------:|
| **CPU spec, k=3** (winner) | 79.6% | **40.19** | **1.55×** | 1.55× |
| CPU spec, k=4  | 74.6% | 37.51 | 1.45× | 1.45× |
| CPU spec, k=2  | 82.3% | 29.93 | 1.16× | 1.16× |
| CPU spec, k=6  | 65.1% | 32.32 | 1.25× | 1.25× |
| CPU spec, k=8  | 58.4% | 36.02 | 1.39× | 1.39× |
| CPU spec, k=16 | 43.0% | 30.43 | 1.17× | 1.17× |
| CPU spec, k=32 | 27.5% | 21.23 | 0.82× | 0.82× |
| Mixed tgt=OpenCL dft=CPU, k=3 | 77.1% | 9.52 | 0.37× | 0.71× |
| Mixed tgt=OpenCL dft=CPU, k=8 | -- | 14.37 | 0.55× | 1.06× |
| Mixed tgt=OpenCL dft=CPU, k=16 | -- | 16.14 | 0.62× | 1.20× |
| OpenCL-all spec, k=3 | 77.7% | 9.14 | 0.35× | 0.68× |
| OpenCL-all spec, k=8 | 59.2% | 13.19 | 0.51× | 0.98× |

### Key findings

1. **k=3 is optimal for draft-model spec on this hardware.** Not k=4 as
   the coarse {4,8,16,32} sweep suggested. k=2 has highest accept (82%)
   but lowest decode (30 t/s) -- too little amortization per verify batch.
   k=3 hits 40.2 t/s mean (1.55×) with 79.6% accept.
2. **Draft-model spec caps near ~1.6× on this rig regardless of
   workload.** JSON accept is higher than humaneval (82.0% vs 79.6% at
   k=3) but decode barely moves (40.83 vs 40.19 t/s). Peak single run
   is JSON prompt-7 (git commits) at **44.88 t/s (1.73×)**. The ceiling
   is per-round overhead, not accept-limited. Phase-3+ techniques need
   to either drastically raise accept with the same k *or* cut the
   round-trip cost; higher accept alone will not break through 1.6×.
3. **Mixed-device placement (tgt=OpenCL, dft=CPU) is a regression.**
   Monotone improvement with k (9.5 → 14.4 → 16.1 for k ∈ {3, 8, 16})
   pinpoints per-round CPU↔OpenCL sync as the bottleneck -- larger
   verify batches amortize it but never enough to beat CPU-alone TG
   (26 t/s), let alone CPU speculative (40 t/s). Peak mixed run was
   26.6 t/s at k=16, only matching (not beating) CPU-alone. Same story
   for fully-on-OpenCL spec (9.1 t/s at k=3, 13.2 at k=8) -- converges
   to OpenCL-alone TG baseline, never wins.
4. **Per-prompt variance is large.** `binary_search` (p5) accepts at
   55--91% across all k and gave the 1.88× peak (49.0 t/s at k=8).
   `flatten` (p6) is pathological: 58% accept at k=8 collapses to 13%
   at k=32, decode drops from 31.7 to 10.8 t/s (0.41×). A single
   pathological prompt can drag a whole sweep mean noticeably.
5. **Encoded (prompt-eval) speed is invariant to k** (~115 t/s across
   all CPU runs). PP is target-only and one-time; k only affects TG.
6. **OpenCL per-call kernel-launch overhead is the killer.** Adreno
   crushes large-batch PP (0.6B Q8_0 at 2674 t/s PP512) but the tiny
   4-10-token verify batches in spec decode don't amortize kernel
   dispatch. This is the architectural lesson behind the negative
   mixed-device and OpenCL-all results.

### Research implications (feeding Phases 3-5)

The 1.6× ceiling observed here is the *draft-model* spec ceiling on
this hardware. The negative OpenCL results redirect the research plan:

- **DFlash+DDTree (Phase 4) is the primary lever.** Confirmed by the
  lucebox-hub RTX 3090 paper (see `new_spec_decode_example_to_research.md`
  and `docs/reference-projects.md`): AL ≈ 8.9 and 3.43× with block-
  diffusion draft into tree verify. It attacks both binding axes -- K
  tokens drafted in one pass AND K-fat verify batches -- exactly matching
  the two constraints session 4 measured.
- **EAGLE-3 (Phase 3) becomes a viability probe, not an anchor.** It
  touches only the accept-rate axis. Cheap to try because the PR exists,
  but unlikely to beat the overhead ceiling alone. Plan: build the PR
  on CPU + OpenCL, run one sweep, decision-gate on ≥2×.
- **NPU drafting (Phase 5) becomes more important.** Hexagon and Adreno
  share LPDDR; an NPU-drafted block can be consumed by Adreno without
  the CPU↔GPU DMA round-trip that torpedoed mixed-device here. Also,
  NPU draft in parallel with GPU verify converts the per-round sync
  into pipelined overlap. The lucebox paper's top-3 perf wins included
  exactly that class of optimisation on PCIe (D2D copy, +3.3%).
- **CPU speculative is the working baseline to contribute upstream.**
  1.55× on code, 1.58× on JSON, stable, clean. llama.cpp's Adreno
  spec story is currently worse than CPU-alone; the data above is
  probably worth a docs/discussion contribution even before new
  techniques land.
- **Unified-memory / buffer-model optimisation is a latent lane.**
  ggml-opencl uses plain `clCreateBuffer(CL_MEM_READ_WRITE)` with no
  zero-copy flags -- Snapdragon X2's shared LPDDR5X is available but
  not exploited. See `docs/reference-projects.md` ("Unified memory vs
  zero-copy") for the breakdown. Not blocking any phase yet, but may
  matter if we build a custom runtime (trident/lucebox-style) later.

### Input artefacts

Prompt fixtures (`prompts/`): `humaneval_subset.jsonl` (10 code
completions), `structured_json.jsonl` (10 JSON generations).
`prose_longform.jsonl` and `chat_multiturn.jsonl` still TODO.

CSVs + per-run logs (all under `results/`):

- `spec-cpu-...-125354`     : k ∈ {4,8,16,32} CPU humaneval
- `spec-cpu-...-131451`     : k ∈ {2,3,4,6}   CPU humaneval
- `spec-cpu-...-132358`     : k ∈ {3,4}       CPU JSON
- `spec-opencl-tgt-ocl-dft-cpu-...-132850` : k=3     mixed humaneval
- `spec-opencl-tgt-ocl-dft-cpu-...-133742` : k ∈ {8,16} mixed humaneval
- `spec-opencl-...-134935`  : k ∈ {3,8}     OpenCL-all humaneval

**Phase 3 onward:** not started.

## CPU build validated

Command:
```powershell
.\llama.cpp\build-cpu\bin\llama-cli.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -t 18 -no-cnv
```
Result: coherent generation, **PP 826 t/s / TG 111 t/s** on Qwen3-0.6B Q8_0 at 18 threads.
Consistent with `gguf_models/LOCAL_LLM_NOTES.md` scaling vs Qwen3-4B Q4_K_M (PP 248 / TG 42 at 18t).

## Build recipe on this machine (captured in `scripts/build_llama_cpp.ps1`)

llama.cpp rejects MSVC for ARM64, so the build uses clang invoked from
a `vcvarsarm64` environment:

- VS BuildTools 2022: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
- LLVM: `C:\Program Files\LLVM` (clang 22.1.3)
- clang-rt: `...\lib\clang\22\lib\windows\clang_rt.builtins-aarch64.lib`

Paths with spaces must be passed to cmake as 8.3-short, forward-slashed,
unquoted (PS 5.1 argv parsing strips embedded quotes, so the reference
`.bat` trick of inner `\"..\"` doesn't translate). See comments in the
script. `build_llama_cpp.ps1 -DryRun` prints the fully resolved cmake
invocation without executing.

Per-build metadata is recorded in `llama.cpp\build-<preset>\SPECULA_BUILD.txt`.

## Outstanding issues / known gotchas

- **OpenCL build — SDK survey done, plan locked. See `docs/adreno_opencl.md`
  for the full writeup.** Summary of what session 3 established:
  - **Runtime is already on disk.** The Adreno driver
    (`qcdx8480.inf_arm64_e11dd2e33e0b42d3` in the Windows driver
    store) ships both `OpenCL.dll` and `OpenCL_adreno.dll`. QAIRT
    bundles the same two DLLs under
    `lib\aarch64-windows-msvc\`. `C:\Windows\System32\OpenCL.dll`
    (the Khronos ICD loader) is also present.
  - **ICD registry key missing.** `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`
    does not exist. Without an entry under that key pointing at
    `OpenCL_adreno.dll` the loader sees zero platforms. Admin
    PowerShell one-liner in `docs/adreno_opencl.md` §Step 3.
  - **No `QCOclIcd.dll` on this Adreno gen.** The earlier note that
    called for `QCOclIcd.dll` was based on older-gen Adreno
    naming; on this driver the ICD DLL is `OpenCL_adreno.dll`.
  - **Headers ship with QAIRT — just in a sample-app path.**
    `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\examples\QNN\SampleApp\SampleAppGPUFencing\src\CL\`
    contains `cl.h`, `cl_ext.h`, `cl_ext_qcom.h`, `cl_platform.h`,
    `cl_version.h`. Usable in a pinch; still no `OpenCL.lib`.
  - **SDK gap = import library only.** Unblock routes ranked:
    1. `vcpkg install opencl:arm64-windows` (preferred — cleanest).
    2. Build Khronos `OpenCL-Headers` + `OpenCL-ICD-Loader` for
       ARM64 with the existing clang-via-vcvarsarm64 recipe, then
       pass `-DOpenCL_INCLUDE_DIR=... -DOpenCL_LIBRARY=...`.
    3. Generate an import lib from
       `OpenCL.dll` (`dumpbin /exports` → `.def` → `lib /def:`) and
       consume QAIRT's sample-app headers. Only if (1) and (2) fail.
  - **Gotcha caught in source:** `ggml-opencl.cpp:222`
    `get_adreno_gpu_gen()` matches only A7X (`730/740/750`), A8X
    (`830/840`), and X1E (`X1` substring). **X2-90 falls to
    `ADRENO_UNKNOWN`**. Non-fatal (init still proceeds) but may
    skip gen-specific tuning branches. Pre-emptive patch is a
    one-line `strstr(name, "X2")` add; leave it until we have a
    first correct run so we can see whether unknown-gen works
    out-of-the-box.
- **KleidiAI / SME2 crashed at runtime in prior project.** See
  `gguf_models/LOCAL_LLM_NOTES.md`. Will retry as a tracked task in
  Phase 1; `scripts/build_llama_cpp.ps1 -Preset cpu-kleidiai` is wired
  up and applies the clang-on-Windows `.S` patch automatically.
- **Good sign for the SME2 retry:** the `vulkan` configure pass showed
  `HAVE_SME - Success` for the compiler feature probe, meaning the
  toolchain *thinks* SME codegen works. The runtime-trap suspicion from
  `LOCAL_LLM_NOTES.md` (ZA-tile user-mode state not enabled) may still
  bite, but the build side is not the problem.

## Directory layout snapshot

```
specula/
├── README.md                   # phase plan, hardware assumptions, rationale
├── current_status.md           # <-- this file
├── pyproject.toml, .python-version, .gitignore, .gitattributes
├── docs/
│   └── reference-projects.md   # trident / voice_project / gguf_models pointers
├── scripts/
│   ├── build_llama_cpp.ps1     # multi-preset native ARM64 builder
│   ├── patch_kleidiai.py       # clang-on-Windows .S patch for KleidiAI
│   ├── download_models.ps1     # HF GGUF fetcher (resumable)
│   ├── sweep_baseline.ps1      # Phase 1 autoregressive matrix
│   └── sweep_speculative.ps1   # Phase 2 spec-decode matrix
├── prompts/
│   └── humaneval_subset.jsonl  # 10 code-completion prompts (other workload files TODO)
├── models/                     # GGUFs (gitignored)
├── results/                    # CSVs + logs (empty)
├── notebooks/                  # analysis (empty)
└── llama.cpp/                  # sibling checkout, gitignored
    └── build-cpu/              # built; binaries in bin/, DLLs copied
```

## Phase 5 status (session 9, 2026-04-21)

**Step 4 CLOSED.** `models/qwen3_0_6b_draft_v81_ctx512.bin` (1438 MB)
is on disk. AI Hub job `jgzx6xlz5` compiled cleanly on the first
attempt off the x86-produced `nomask` ONNX: CREATED → OPTIMIZING_MODEL
→ **SUCCESS at t=400s**. This cleared the 440-465s CTX-BIN wall that
killed the prior 8 attempts — the x86-side `nomask` variant
(onnxsim + aggressive mask-subgraph removal, commit `adbbbd4`)
eliminated `attention_mask`, `Where`, and `IsNaN` entirely, so HTP
had no BOOL tensors to reject. See `qlcom_compile_status.md` for the
full retro.

Current local state:
- `models/qwen3-0.6b-nomask/` — x86 handoff input (3 GB onnx + data)
- `models/qwen3-0.6b-nomask-ai-hub/` — staged for upload (2.87 GB)
- `models/qwen3_0_6b_draft_v81_ctx512.bin` — compiled HTP context
  binary, signed via AI Hub's QAIRT stack. Target for step 5.

Steps 5-10 (per `docs/npu_scoping.md` §7) still ahead:

```
[DONE]    1. Environment snapshot                 (commit 7230210)
[DONE]    2. ORT-QNN sidecar skeleton             (commit 282e84a)
[DONE]    3. Qwen3-0.6B ONNX sourced + CPU-valid  (commit 106c756)
[DONE]    4. AI Hub compile -> Hexagon .bin       (session 9 ★)
          5. Load .bin via NPUSession, shape-check <-- next
          6. Correctness vs CPU, single greedy prompt
          7. Pipe first drafted token through llama.cpp verify
          8. External-drafter bridge for llama.cpp spec decode
          9. First NPU-spec number on 10-prompt humaneval
         10. Sweep k values, write up, close phase
```

Step 5 risk: `NPUSession` currently loads `.onnx` files. Context
binaries need a different ORT-QNN load path (either via
`qnn_context_binary_file` provider option or via an ONNX-EPContext
wrapper that AI Hub emits alongside). Small extension expected.

## Immediate next steps (next session)

**Strategy (session 5, 2026-04-20): close out Qwen3, then graduate
fully to Qwen3.5 for all further work.**

The Qwen3 family has been our scaffolding model because it has more
public literature (EAGLE PRs, community benchmarks, llama.cpp coverage).
Production target is Qwen3.5 → Qwen3.6 -- see
`memory/project_target_model.md`. Rather than running two parallel
tracks (keep poking at Qwen3, start Qwen3.5), we treat this as a
clean graduation:

- Use the Qwen3 window to land everything *adventurous* that only
  makes sense on pure-attention + smaller models: **NPU drafting** is
  the headline item (simplest path into QAIRT), but any stashed
  Phase-2 experiments (see below) that we care about should finish
  in this window too -- they'll be orphaned after graduation.
- Once NPU drafting on Qwen3-0.6B + Qwen3-8B has a stable baseline
  (win, lose, or tie against Phase 2's 1.55× CPU-spec), declare Qwen3
  *closed* and move all subsequent phases (DFlash, NPU-on-dense-small,
  MoE) to Qwen3.5+.
- This keeps the codebase from accumulating two model-family branches
  and keeps our attention on Qwen3-era questions while the tooling is
  still fresh.

**Revised phase order: NPU-first (close out Qwen3), then Phase 4
DFlash (opens Qwen3.5 era).**
Session 5 scoped Phase 4 against lucebox-hub and then re-sequenced:

- **Phase 5 (NPU drafting on Qwen3) moves ahead of Phase 4.** The
  Phase-2 mixed-device negative result told us CPU↔OpenCL sync breaks
  small-batch heterogeneous exec on this hardware. It did NOT tell us
  whether heterogeneous exec works *at all* -- only that ggml-opencl's
  per-round launch profile loses in the specific CPU-draft +
  OpenCL-target pairing. The NPU path is the structurally different
  bet: pipelined async dispatch + ION-backed buffers + parallel-to-
  target draft. Answering "does heterogeneous work on X2 at all" is
  a prior-scope question that gates DFlash's eventual Phase-5-style
  async design. If NPU-draft also regresses, we learn that before
  committing to a DFlash-on-OpenCL port with the same sync shape.
- **Phase 4 (DFlash + DDTree on Qwen3.5) follows NPU work.** The
  scoping pass already landed (see "Phase 4" section below); the
  assets can be downloaded in parallel with NPU bring-up so Phase 4
  is ready to start the moment NPU drafting answers its core
  question. Session 5 scoping remains valid regardless of ordering.
- CPU spec peaks at **1.55× (40.2 t/s) at k=3**; ceiling ~1.6×. This
  remains the target-side baseline for both phases.
- lucebox-hub hit 3.43× on RTX 3090 with DFlash+DDTree (pure-CUDA,
  single device). Our win condition is different: we're stitching
  heterogeneous compute (NPU + CPU + GPU) via shared LPDDR5X, not
  competing on single-device throughput.
- Phase-2 stretch items (`--draft-p-min`, 14B target, prose/multi-
  turn, ngram spec, upstream writeup) stay stashed -- none break 1.6×.

### Phase intersection — why NPU-first is cheap information

The two tracks share less code than they look to at first, but the
one thing they DO share is the work that matters most.

- **Shared, high-value: zero-copy buffer model on shared LPDDR5X.**
  NPU drafting needs NPU↔CPU/GPU shared allocations via
  `cl_qcom_ion_host_ptr` or QAIRT buffer handoff so the NPU draft
  doesn't pay a cache-flush round trip to hand tokens back to the
  target. DFlash-on-OpenCL needs the *same* pattern for its per-
  round `target_feat` → draft → verify loop -- the Snapdragon
  analogue of the D2D copy that gave lucebox +3.3% on PCIe. Doing
  NPU first forces us to solve ION-backed allocation + async
  dispatch first; Phase 4's OpenCL port then inherits that
  infrastructure intact. This is the single biggest intersection.
- **Shared, low-value: benchmark harness + prompt fixtures + metric
  logging.** `sweep_speculative.ps1`, the humaneval/json JSONL
  fixtures, the AL/accept/tok-s columns in our CSVs all carry over.
  Already built in Phase 2.
- **NOT shared: kernel ports.** NPU drafting doesn't touch
  ggml-opencl (draft runs on QAIRT, target uses stock llama.cpp).
  DFlash-on-OpenCL doesn't touch Hexagon. Two independent skill
  trees; doing one doesn't accelerate the other's kernel work.
- **NOT shared: target-side code.** NPU spec decode uses llama.cpp's
  stock `--draft` pipeline -- zero target patching, just swap which
  binary runs the draft forward. DFlash needs the custom
  non-libllama target loader + 5-layer hidden capture hooks.
- **NOT shared: draft/target pairing.** NPU standard spec decode
  only needs tokenizer compatibility (Qwen3-0.6B-draft +
  Qwen3-8B-target works). DFlash requires the drafter to be trained
  on the specific target's hidden states -- Qwen3-trained drafter
  cannot verify a Qwen3.5 target. So NPU-on-Qwen3 and
  DFlash-on-Qwen3.5 are independent experiments, not a shared lane.

### Session 5.5 prep (do in parallel with NPU bring-up)

To keep Phase 4 unblocked once NPU drafting answers its core
question, stage these in the background:

- [ ] Download `unsloth/Qwen3.5-27B-GGUF` Q4_K_M (~16 GB) into
      `models/`. Verify `arch=qwen35` in the GGUF metadata.
- [ ] Download `z-lab/Qwen3.5-27B-DFlash` safetensors (~3.5 GB BF16).
- [ ] Read `delta_net_chunked.cpp` (237 LOC) and list every ggml op
      it calls. Cross-check against ggml-opencl's `supports_op`.
      This is the Phase-4 gating item.
- [ ] Confirm `ggml_rope_ext` section-mode support on OpenCL (Qwen3.5
      M-RoPE uses sections `[11,11,10,0]`; plain NEOX is fine for the
      draft and verified working).

### NPU drafting on Qwen3 — why Qwen3 (not Qwen3.5) for this phase

Two sub-questions people conflate:

1. **NPU porting difficulty.** Pure-attention (Qwen3 small variants)
   maps cleanly onto QNN's shipped op library (matmul + rmsnorm + rope
   + swiglu). Hybrid variants (Qwen3.5 with `gated_delta_net` +
   `ssm_conv`) require persistent-state handling on NPU that isn't in
   the sample ops. **Hybrid is the blocker, not Qwen3 vs Qwen3.5.**
   Pure-attention variants of either family work equivalently.
2. **Draft/target pairing** for standard speculative decoding (not
   DFlash): only needs tokenizer compatibility. Qwen3 and Qwen3.5
   share the same tokenizer (Qwen family policy), so a tokenizer-
   compatible cross-family pairing is valid. But for apples-to-apples
   baselines it's cleaner to stay one-family-per-experiment.

So the Phase 5 NPU plan is: bring-up on Qwen3-0.6B-Q8_0 draft +
Qwen3-8B-Q4_K_M target (same pair as Phase 2). Once it works, swap
to Qwen3.5-*dense-small* draft + Qwen3.5-*dense* target for the
production-target experiment. We **don't** attempt NPU + hybrid
(Qwen3.5-27B-hybrid) together yet -- layering two unknowns. That
combination lands after both Phase 4 (DFlash on hybrid) and Phase 5
(NPU on pure-attention) have baselines.

### Caveats carried into Phase 5

- **Scoping doc is canonical:** `docs/npu_scoping.md` (session 5,
  2026-04-20) has the 10-step bring-up plan, toolchain pins
  (QAIRT 2.45.40, ORT-QNN 1.24.4), known-failure-mode catalog, and
  prior-art review. The bullets below are the short version; the
  doc is the single source of truth for Phase 5 execution.
- Hedge docs already absorbed into the scoping doc:
  `voice_project/current_status.md` + trident's `postmortem.md` /
  `npu_path_back.md` / `npu_current_status.md` /
  `npu_optimizations_thoughts.md`.
- Pin the NPU bring-up draft to **Qwen3-0.6B-Q8_0** (not 1.7B).
  Smaller compile iterations + fewer custom-op surprises on the
  first pass.
- Keep target side identical to Phase 2 winning config: Qwen3-8B-
  Q4_K_M on CPU at 18 threads (25.91 t/s TG baseline, 40.2 t/s
  CPU-spec at k=3). That way the first NPU-spec number is directly
  comparable to the CPU-spec result; we can say cleanly whether
  NPU-as-draft wins, loses, or ties vs CPU-as-draft.
- Hexagon arch target is **v81** on X2E -- proven by voice_project's
  working AI Hub compile (`dspArch: 81, socModel: 88`). Session 6
  step 1 still does a QAIRT device-enum sanity check before any
  code, but v81 is the pin.
- **Primary path = AI Hub cloud compile + ORT-QNN EP 1.24.4 runtime.**
  Not raw `QnnContext_createFromBinary`. voice_project hit three
  driver-signing walls on the raw path and only escaped via ORT's
  bundled signed QAIRT stack. Our Phase 5 target/draft integration
  needs an external-drafter sidecar (no QNN backend in llama.cpp),
  tracked as blocking in npu_scoping.md §8.
- **Prior-art review integrated** from
  `npu_thoughts_previous_examples.md` (sd.npu, Mirror-SD, HeteroLLM,
  Dovetail, OpenPangu). Structural plan unchanged; key post-bring-up
  lever is sd.npu's pad/recycle trick for <8-token drafts (our k=3
  is deep in that regime). Full analysis in npu_scoping.md §10.

### Stashed -- now reframed as "Qwen3 close-out" candidates

After the session-5 re-sequencing, these items fall into the Qwen3
window that closes with Phase 5 NPU. Anything we don't do here
gets orphaned at graduation -- pick deliberately. Priority marks:
**[keep]** = worth doing before graduation, **[drop]** = fine to
skip, **[carry]** = trivially portable to Qwen3.5 so not time-
pressured.

- **[keep] Tighten `--draft-p-min`.** Default 0.75 over-drafts on
  low-confidence streaks (the `flatten` pathology). Try 0.80, 0.85,
  0.90 at k=3. Cheap data point; improves our CPU-spec baseline
  before we compare it to NPU-draft. Actively useful inside the
  Qwen3 window.
- **[carry] 14B target + 0.6B draft (or 1.7B draft).** Draft/target
  compute ratio drops ~2×; break-even k shifts higher. Needs
  `Qwen3-14B-Q4_K_M.gguf` download. Not time-pressured -- we'll do
  the equivalent exercise on Qwen3.5 dense variants after
  graduation, so skipping on Qwen3 loses no learning. Do only if
  convenient alongside Phase 5 NPU.
- **[keep] Prose + multi-turn workloads.** `prompts/prose_longform.jsonl`,
  `prompts/chat_multiturn.jsonl` stubs still empty. Worth filling
  before NPU-draft data collection so the first NPU-spec numbers
  already cover the full workload matrix we'd want for a writeup.
- **[keep] Draftless ngram spec** (`--spec-type ngram-*`). Memory-
  free; should fly on JSON. Quick A/B, closes a gap in the Qwen3
  spec-decode story and sets a useful floor for "dumbest possible
  draft" that NPU-draft and DFlash both need to beat.
- **[keep] Negative-result contribution upstream.** The Adreno-
  OpenCL spec-decode story (no win at any k or placement) is worth
  a llama.cpp docs/discussion post. Write after Phase 5 NPU so we
  can contribute "here's what does work on heterogeneous X2"
  alongside. High visibility, low effort once the NPU number is in.
- **[drop] EAGLE-3 viability probe.** Was Phase 3 anchor, demoted
  already. Lucebox paper showed chain-over-tree gives +15% on Q4_K_M
  (quantization flattens draft softmax); EAGLE-3 alone won't break
  our 1.6× ceiling. Not worth Qwen3-window time. Revisit on Qwen3.5
  only if DFlash underperforms.

### Phase 4 (DFlash + DDTree on Qwen3.5) -- queued behind Phase 5 NPU

Reference impl to mine: `C:\Users\hotschmoe\Documents\GitHub\lucebox-hub/dflash/src/`
(sibling checkout pulled 2026-04-20). See
`docs/reference-projects.md` for file-level guidance on what to read
first.

#### Session 5 (2026-04-20) scoping findings

Read: `dflash_graph.h`, `qwen3_dflash_graph.cpp` (168 LOC, draft
graph), `safetensors_draft.cpp` (407 LOC, draft weights),
`internal.h` (288 LOC, shared state/cache schema),
`qwen35_target_graph.cpp` (806 LOC, hybrid target),
`delta_net_chunked.h` (chunked delta-net entry),
`gguf_target_loader.cpp` (386 LOC, non-libllama GGUF loader),
`RESULTS.md`.

**Kernel coverage on current ggml-opencl backend**
(`llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`, HEAD
`e365e658f…`):

- `GGML_OP_FLASH_ATTN_EXT` -- IMPLEMENTED (`ggml_cl_flash_attn` at
  line 9265, dispatch at 14033). Supports_op gate at 4166-4200:
  supported `{dk,dv}` pairs include `{128,128}` which matches our
  draft head_dim; supported dtypes F32/F32, F16/F16, F32+F16KV.
  BF16 input not supported so we convert at load, same trick the
  lucebox CUDA port uses for norms.
- `GGML_OP_SSM_CONV` -- IMPLEMENTED (4085, 13934). Hybrid
  delta-net 1D causal conv already runs on OpenCL.
- `GGML_OP_GATED_DELTA_NET` -- NOT implemented on OpenCL.
  This is the Qwen3.5 SSM recurrence kernel. Options:
  (i) use lucebox's `delta_net_chunked.cpp` path, which
  re-expresses the recurrence in already-supported primitives
  (237 LOC); (ii) CPU fallback for delta-net layers only; (iii)
  write an OpenCL kernel ourselves. (i) is the only tractable
  short-term path -- if its ops are all ggml-opencl-backed, we
  get hybrid on OpenCL for free.
- M-RoPE with `rope_sections [11,11,10,0]` -- still need to verify
  OpenCL supports section-mode `ggml_rope_ext` (Qwen3.5 needs it;
  pure Qwen3 uses plain NEOX which works).

**Non-kernel blockers**
- `gguf_target_loader.cpp` is POSIX (mmap / open / fstat / munmap).
  Same ~30-LOC Windows swap as `safetensors_draft.cpp`
  (`CreateFileMapping`/`MapViewOfFile`). Do both at once.
- Target arch string is `qwen35` (not `qwen3`); loader validates
  this. GGUF we download must match.
- Token embedding stays on CPU (CUDA port notes CUDA get_rows
  can't handle k-quants; OpenCL likely same). `CpuEmbedder` in
  `internal.h` already handles this -- portable as-is.
- Target graph uses `capture_layers` mode to sink 5 specific
  layer hiddens into a 4096-slot `target_feat` ring
  (`qwen35_target_graph.cpp:694` — `CAPTURE_LAYERS[]`). The draft
  reads this as its "5*hidden" input. Keep as-is; no llama.cpp
  patching needed since we use the non-libllama loader.
- `DDTree` adds per-delta-layer `ssm_intermediate` (F16,
  `[S_v, S_v, H_v, max_verify_tokens]`) and
  `conv_input_cache`. Hybrid-only overhead. Pure-attention
  targets pay ZERO per-node memory tax (RESULTS.md, "Memory
  ceiling notes") — the published DFlash paper runs budgets up
  to 1024 on pure-attention Qwen3-8B/30B. This is the single
  biggest argument for eventually doing our own pure-attention
  drafter (option B below).

**Draft graph port cost (option A, using z-lab 27B drafter):**
~near zero. Every op (`ggml_mul_mat`, `rms_norm`, `mul`,
`reshape`, `concat`, `rope_ext` NEOX, `permute`, `cont`,
`flash_attn_ext`, `silu`, `add`) already on ggml-opencl.
Weights loader is pure I/O + a hand-rolled safetensors JSON
parser. BF16-on-disk is fine; norms get converted to F32 at
load (already in the code).

#### Path decision: A (today) → B (later). Documented below.

lucebox-hub's reference impl is glued to Qwen3.5-27B-hybrid.
z-lab only publishes DFlash drafter weights for 27B. We pick A
first to de-risk (reference impl exists, working numbers exist,
no training needed) and carry B in the backlog.

**Option A — Qwen3.5-27B hybrid target + z-lab 27B DFlash drafter (chosen for now)**

Plan:
1. Download assets:
   - `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB) into `models/`.
   - `z-lab/Qwen3.5-27B-DFlash` (safetensors, BF16, ~3.5 GB).
2. Port `safetensors_draft.cpp` + `gguf_target_loader.cpp`
   mmap→Win32 mapping. ~1 day.
3. Standalone drafter smoke: port `qwen3_dflash_graph.cpp`
   against ggml-opencl backend. No target yet, just prove the
   drafter forward runs and dims match.
4. Target forward with full-attention layers only (16 of 64,
   every 4th). Confirm logits plumbing; garbage output is
   expected because delta-net layers are stubbed.
5. Delta-net layers via `delta_net_chunked` primitives path.
   If every op has OpenCL coverage → hybrid on OpenCL directly.
   Otherwise CPU-partition delta-net layers.
6. Chain verify (q_len=16) end-to-end. Expected AL 6-7 (lucebox
   Math500/GSM8K numbers).
7. DDTree verify. Expected AL 8+, ~3× target.
8. Perf-tune: f16 intermediate, `gated_delta_net_tree_persist`
   equivalent, D2D target_feat replacement (Snapdragon shared-
   LPDDR analogue: avoid cache flush pairs).

Upsides: reference impl exists, weights exist, results are
published (3.43× HumanEval on 3090). Strong alignment with
stated production target (see user memory:
`project_target_model.md`).

Downsides: 27B Q4_K_M (~16 GB) + drafter (~3.5 GB BF16) + KV +
intermediates leaves comfortable margin in 48 GB LPDDR5X --
less than Qwen3-8B (~4.5 GB weights) but nowhere near tight.
Long-context 128K via Q4_0 KV would add ~8 GB and still fit.
Hybrid port is more unknown than pure-attention. Per-node
DDTree memory tax caps budget ~22-26 on hybrid only (a
pure-attention target would be uncapped).

**Option B — Train our own DFlash drafter for a pure-attention Qwen (revisit later)**

Plan:
1. Pick target: Qwen3-8B (scaffolding) OR Qwen3.5-8B-dense
   (closer to production, no hybrid).
2. Build training infra: 5-layer DFlash drafter per the paper
   (`docs/reference-projects.md` cites Qwen3-8B/30B-MoE numbers
   from the original DFlash paper at 4-5× on HumanEval).
3. Run distillation against target hiddens on a cloud GPU
   (BF16 training -- not feasible locally on the X2). A few
   hundred GPU-hours ballpark.
4. Port the drafter weights + dims into our (by-then-working)
   DFlash pipeline. Pure-attention target skips the delta-net
   port entirely. Per-node memory tax is zero -> budgets up to
   1024 per the paper.
5. Retest DDTree at large budgets; upper bound on speedup
   likely higher than 27B-hybrid because verify-batch memory
   isn't hybrid-capped.

Upsides: pure-attention path removes the deltanet port
entirely; matches any Qwen3.5-*dense* production target;
higher budget ceiling. Weights are ours (no z-lab license
considerations).

Downsides: training cost; no existing reference impl at our
exact dims; gap between "has a drafter" and "has a *good*
drafter" (distillation quality drives AL).

**When to pivot A→B:** after Option A lands end-to-end and we
have a clean hybrid DDTree baseline on Snapdragon. At that
point we know the per-round overhead floor on our hardware and
can quantify what training our own drafter buys. Until then,
option A is cheaper information.

#### Dependency queue (session 5 → session 6)

Before writing any port code:

- [ ] Download `unsloth/Qwen3.5-27B-GGUF` Q4_K_M (~16 GB) into
      `models/`. Bandwidth budget.
- [ ] Download `z-lab/Qwen3.5-27B-DFlash` safetensors into
      `models/` (or a sibling dir; it's not a GGUF).
- [ ] Skim `delta_net_chunked.cpp` (237 LOC) and list every
      ggml op it calls; cross-check each against ggml-opencl
      `supports_op`. This is the gating item for whether
      hybrid-on-OpenCL is tractable without writing a new
      kernel.
- [ ] Confirm `ggml_rope_ext` section-mode support on OpenCL
      (qwen3.5 M-RoPE). If unsupported, plan a CPU shim for
      rope only.
- [ ] Decide on build layout: separate `phase4/` source tree
      (lucebox-shaped standalone, non-libllama) vs. integrate
      into the llama.cpp fork. Lean standalone -- inherits
      lucebox's structure 1:1 and sidesteps llama.cpp
      graph-capture patching.

#### High-level plan (as a checklist, unchanged structure)

1. **Scoping pass** (no code). *Done this session.* ✓
2. Drafter-weight pipeline: port `safetensors_draft.cpp`
   mmap→Win32.
3. Target-weight pipeline: port `gguf_target_loader.cpp`
   mmap→Win32 and confirm `arch=qwen35` handling.
4. Minimum-viable DFlash (chain-verify, no DDTree). Expected AL
   6-7, comparable to session-4 CPU chain-spec in rate but
   higher AL.
5. Add DDTree verify. Target: clear 2× end-to-end.
6. Port any delta-net ops that aren't already on OpenCL (see
   `delta_net_chunked.cpp` survey above).
7. OpenCL perf tuning (f16 intermediate, target_feat copy,
   tree-persist kernel).

### Phase 6 -- Standalone lite harness (flagged for way later)

If Phase 5 closes with llama.cpp's backend model still leaving perf on
the table (specifically the OpenCL buffer model -- no zero-copy on
shared LPDDR5X -- and uncoalesced kernel dispatch), spin a narrow
lucebox-shaped harness (~2000 LOC, no libllama link) rather than
graduating to a full runtime like trident. See README Phase 6 for
scope. Lever: `CL_MEM_USE_HOST_PTR` / `clSVMAlloc` /
`cl_qcom_ion_host_ptr` as allocation invariants, plus fused kernels
and direct QAIRT↔OpenCL ION-buffer handoff for the NPU draft path.

### Phase 7+ -- MoE targets (Qwen3.6-35B-A3B), explicitly deferred

Holding until the fundamentals land on dense/hybrid. Rationale:

- We're doing fundamental exploration on niche hardware (X2 Elite
  Extreme is an almost unstudied platform for spec decode). Stacking
  architectural novelty (MoE expert routing) on top of platform
  novelty multiplies the unknowns without adding information.
- MoE expert routing is a **new op class on NPU**. Qualcomm's shipped
  sample ops don't cover it; custom-op development would be an
  unrelated rabbit hole that delays the platform-characterization
  results we're actually trying to get.
- Spec-decode research on MoE is still unsettled in the literature --
  expert-miss rate adds a second acceptance axis beyond token-accept,
  and the interactions with tree-verify/DFlash aren't
  well-characterized.
- Our 48 GB LPDDR5X has room for Qwen3.6-35B-A3B Q5_K_M (reference
  numbers in `gguf_models/LOCAL_LLM_NOTES.md`), so the only thing
  deferring Phase 7 costs us is research novelty -- not hardware
  reach.

What triggers unblocking Phase 7:

- DFlash+DDTree on Qwen3.5 hybrid target has a stable baseline
  (Phase 4 closed).
- NPU drafting has a stable zero-copy pipeline on pure-attention
  (Phase 5 closed, regardless of whether it wins vs CPU-spec).
- At least one of (a) Phase 6 lite harness lands, or (b) we can
  articulate a concrete bandwidth-budget model that predicts MoE
  expert routing's cost on LPDDR5X.

Then Phase 7 is additive: reuse the existing DFlash target-loader
path (qwen3.6 GGUF → extend `gguf_target_loader.cpp` with MoE tensor
naming), add expert-selection to the target graph, and treat it as
an incremental A/B vs Phase 4's 27B result.

### Deferred / answered

- ~~**Mixed-device CPU-draft + OpenCL-target.**~~ *Answered, negative.
  Sync overhead dominates; monotone decode improvement with k tops
  out at 16.1 t/s, under CPU-alone TG. Not worth more scanning.*
- ~~**Narrow k sweep + JSON workload.**~~ *Done; k=3 is the optimum.*

### Backlog (non-Phase-2)

6. **Fill out the OpenCL perf matrix** beyond 0.6B Q8_0. Run
   `sweep_baseline.ps1` for 1.7B + 8B on OpenCL at `-p 1024, 2048`
   context shapes.
7. **Optional tuning:** patch `get_adreno_gpu_gen` to recognize
   `X2-90` (currently falls to `ADRENO_UNKNOWN`); opt into
   `cl_qcom_large_buffer` via `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1`.
8. **SME2 / KleidiAI retry.** `-Preset cpu-kleidiai` already builds;
   see `docs/SME_investigation.md`. Deferred until Phase 2 settles.
9. **Vulkan.** Deferred, not abandoned. If a Qualcomm driver update
   lands, re-run the session-2 correctness matrix (B0-B7).
10. **Upstream-issue search for X2-90 OpenCL** to see if anyone else
    is ahead of us. Low priority now that the backend works.
