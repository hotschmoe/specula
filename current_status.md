# specula -- current status

Last updated: 2026-04-20 (session 4 -- Phase 2 CPU/mixed/OpenCL spec-decode, 6 sweeps)

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

## Immediate next steps (next session)

**Phase 2 thoroughly mapped on 8B+0.6B.** Session 4 (2026-04-20)
produced six spec-decode sweeps: wide k, narrow k, JSON workload,
mixed-device at 3 k values, OpenCL-all at 2 k values.
- CPU spec peaks at **1.55× (40.2 t/s) at k=3**; ceiling ~1.6×.
- OpenCL-in-any-role is a regression for stock spec decode (per-round
  kernel-launch overhead dominates small-k verify batches).
- JSON (82% accept) barely improves on humaneval (80%); ceiling is
  overhead, not accept rate.
- Next work below is about pushing outward: new shapes (14B target,
  1.7B draft), tunables (`--draft-p-min`), and non-code workloads,
  *not* more mixed-device k scanning -- that axis is answered.

### Phase 2 -- near-term

1. **Tighten `--draft-p-min`.** Default 0.75 drafts aggressively even on
   low-confidence streaks (the `flatten` pathology). Try 0.80, 0.85,
   0.90 at k=3; expect lower `n_drafted` on pathological prompts with
   minimal accept loss on easy ones. Cheap to run.
2. **14B target + 0.6B draft (or 1.7B draft).** Draft/target compute
   ratio drops ~2×; break-even k should shift higher, and the
   speculative ceiling may rise past 1.6× on a heavier target. Need to
   ingest `Qwen3-14B-Q4_K_M.gguf` first.
3. **Prose + multi-turn workloads.** `prompts/prose_longform.jsonl`
   and `prompts/chat_multiturn.jsonl` stubs. Prose is the
   low-acceptance regime; multi-turn is the realistic one.
4. **Draftless ngram spec** (`--spec-type ngram-*`). Memory-free;
   should fly on JSON (very high repetition structure). Quick A/B vs
   the draft-model numbers we already have.
5. **Negative-result contribution upstream.** The Adreno-OpenCL
   spec-decode story (no win at any k or placement) is non-obvious and
   currently undocumented in llama.cpp discussions. Draft a
   docs/discussion post with the numbers from session 4.

### Phase 3/4 preview (post-Phase-2 wrap)

6. **EAGLE-3 viability probe.** Build llama.cpp PR #18039 for CPU +
   OpenCL, run one humaneval sweep. Decision gate: ≥2× continues the
   integration; <2× archives it and hands off to Phase 4. Do not port
   the full PR before the probe.
7. **Start DFlash port planning** by reading `lucebox-hub/dflash/src/`
   (sibling checkout at `C:\Users\hotschmoe\Documents\GitHub\lucebox-hub`).
   Pin down: scope of a Qwen3-8B pure-attention DFlash drafter,
   expected CUDA→OpenCL kernel translation effort, DDTree verify graph
   as a reusable component. No code yet -- this is scoping.

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
