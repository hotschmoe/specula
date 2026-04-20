# specula -- current status

Last updated: 2026-04-20 (session 4 -- Phase 2 CPU spec-decode, first sweep)

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

**Phase 2 -- Stock speculative decoding (started 2026-04-20, session 4):**

First real spec-decode sweep is in the books. Target
Qwen3-8B-Q4_K_M + draft Qwen3-0.6B-Q8_0 on CPU (18t), greedy (temp=0),
`--draft-min 0`, 10 humaneval prompts, `-n 256`, swept `--draft-max ∈ {4, 8, 16, 32}`.
Raw CSV + per-run logs: `results/spec-cpu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-Q8_0-20260420-125354{.csv,/}`.

Mean-of-10 by `--draft-max` vs the **25.91 t/s** 8B CPU TG baseline:

| draft-max | mean accept | mean decode t/s | speedup |
|-----------|-------------|-----------------|---------|
| 4         | 74.5%       | 36.85           | **1.42×** |
| 8         | 58.4%       | 36.02           | 1.39×   |
| 16        | 43.0%       | 30.43           | 1.17×   |
| 32        | 27.5%       | 21.23           | **0.82× (slower than no-spec)** |

Takeaways, in priority order for the next round:

1. **k=4 wins on mean; k=32 is actively worse than no-spec.** Break-even on
   this hardware / prompt mix lands between k=16 and k=32. The draft cost
   (draft-model forward × k rounds) overwhelms the verification savings
   once accept% falls below ~30%.
2. **Per-prompt variance is huge.** `binary_search` (prompt 5) holds 55--91%
   accept across all k; peak single run is prompt 5 @ k=8 at **49.0 t/s
   (1.88×)**. `flatten` (prompt 6) collapses from 58% → 13% as k grows and
   at k=32 decodes at 10.8 t/s (0.41× -- less than half baseline). A single
   pathological prompt can eat an entire sweep's wall-clock.
3. **k=4 and k=8 are within 2% of each other on mean decode** (36.85 vs
   36.02 t/s) despite very different accept rates -- so draft cost at k=8
   roughly cancels the higher accept count. Worth testing k ∈ {2, 3, 6}
   once we care about tuning: the optimum may be below 4.
4. **Encoded (prompt-eval) speed is insensitive to `draft-max`** as
   expected (~115 t/s mean across all runs) -- it's the target model's
   one-time prompt ingest.

Open questions these numbers raise (not yet explored):

- Does OpenCL-as-verifier change the calculus? Adreno PP is 2-3× CPU's;
  if we offload *target verify* to OpenCL while keeping *draft* on CPU,
  we might win on prompts where small-batch verification is the
  bottleneck. CPU beat OpenCL on TG in Phase 1, but spec-decode verify
  is not pure-TG -- it's batched-PP on a draft of size k.
- 14B target + 0.6B draft: draft/target compute ratio drops ~2×, so
  break-even k shifts higher. Worth measuring once we've ingested 14B.
- `--draft-p-min` was left at default 0.75; stricter thresholds should
  cut `n_drafted` on low-confidence prompts (helping the `flatten`
  pathology) at the cost of some accepts on the easy prompts.

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

**Phase 2 is live.** Session 4 (2026-04-20) produced the first
spec-decode CSV; CPU-only, 8B-target + 0.6B-draft, humaneval.
k=4 is the sweet spot at 1.42× mean (peak 1.88× on well-drafted
prompts). k=32 regresses below baseline. Next work is
narrowing the k sweep, adding mixed-device placement, and
probing non-code workloads.

### Phase 2 -- near-term

1. **Narrow the k sweep.** k=4 won; probe `--draft-max ∈ {2, 3, 4, 6}`
   to find the true optimum, same 10-prompt humaneval fixture,
   same `-n 256`. Aim: shave wall-clock variance and confirm k=4 isn't
   just the lucky floor.
2. **Tighten `--draft-p-min`.** Default 0.75 drafts aggressively even on
   low-confidence streaks (flatten pathology). Try 0.80, 0.85, 0.90 at
   k=4 and k=8; expect lower `n_drafted` on pathological prompts with
   minimal accept loss on easy ones.
3. **Non-code workloads.** Humaneval is high-repetition, high-structure.
   Add `prompts/structured_json.jsonl` (expected: higher accept, ngram
   sweet-spot territory) and `prompts/prose_longform.jsonl` (expected:
   lower accept, where spec-decode's floor matters). Stub files exist
   in the README-described layout but are empty today.
4. **Mixed-device placement (novel).** Draft on CPU, target on OpenCL.
   Phase-1 showed CPU wins TG but OpenCL wins PP512 by 2-3×. Speculative
   *verify* is batched-PP of size k on the target, which is the regime
   where OpenCL might win. `llama-speculative --device` and `-ngl` /
   `-ngld` already support asymmetric placement; needs a build that
   has both backends linked (the `vulkan-opencl` preset is closest;
   OpenCL-only would also work if we force draft onto `-ngld 0`).
5. **Draftless ngram spec.** `--spec-type ngram-*` family. Memory-free;
   should fly on structured JSON. Quick to add once we have the JSON
   prompts.

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
