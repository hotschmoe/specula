# specula -- current status

Last updated: 2026-04-20 (session 3 -- OpenCL/Adreno pre-build survey)

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

**Phase 1 onward:** not started.

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

**OpenCL/Adreno is unblocked and validated.** Session 3
(2026-04-20) built the preset, built the Khronos SDK (siblings
`../OpenCL-Headers/`, `../OpenCL-ICD-Loader/`), passed correctness
A/B, and benched 0.6B at numbers that clear CPU by 2-3× on PP.
Remaining work is filling the perf matrix at 1.7B + 8B and
handing off to `sweep_baseline.ps1`.

1. ~~**Survey the OpenCL/Adreno landscape before building anything.**~~
   *Done in session 3 (2026-04-20). Findings captured in
   `docs/adreno_opencl.md`; the ICD-DLL name, header situation,
   runtime-DLL location, ICD registry absence, and the X2-90 gen-
   matcher gap are all documented. Remaining landscape item (not yet
   done, low priority for the build itself): upstream issue/discussion
   search for `X2-90 OpenCL` / `Snapdragon X2 Elite OpenCL` to see if
   anyone else is ahead of us.*
2. ~~Build + smoke test OpenCL preset.~~ **Done in session 3.**
   0.6B correctness + perf validated. See `docs/adreno_opencl.md`.
3. ~~Widen `LLAMA_BUILD_TOOLS`.~~ **Done.** `llama-completion`
   + `llama-perplexity` now in targets list. Tooling caveat: on
   HEAD `fd6ae4c…` `llama-completion` still auto-enters
   conversation mode; scripts either need to close stdin or use
   `llama-server` over HTTP for hands-off runs.
4. **Fill out the OpenCL perf matrix.** 0.6B Q8_0 is done. Next:
   Qwen3-1.7B Q8_0 then Qwen3-8B Q4_K_M through the same bench
   template, plus longer prompt shapes (`-p 1024,2048`) for
   Phase-2-relevant shapes.
5. **Phase 1 formal baselines.** Run `sweep_baseline.ps1` across
   CPU + OpenCL backends for Qwen3-0.6B / 1.7B / 8B. First real
   CSVs land in `results/`.
6. **Optional tuning experiments** (non-blocking, may be worth
   a session once 1 and 2 are done):
     - Patch `get_adreno_gpu_gen` to recognize `X2-90` (currently
       falls to `ADRENO_UNKNOWN`). Rebuild, re-bench, compare.
     - Opt into `cl_qcom_large_buffer` via
       `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1` if extension is
       supported; re-bench 8B.
7. **SME2 / KleidiAI retry.** `-Preset cpu-kleidiai` build, then run
   a batched-matmul workload (PP-heavy) to trigger or not trigger
   the prior `STATUS_ILLEGAL_INSTRUCTION`.
8. **Vulkan — deferred, not abandoned.** With OpenCL locked in as
   the primary GPU backend there's no urgency, but if a Qualcomm
   driver update lands, re-run the session-2 correctness matrix
   (B0-B7) to see if any shader path now returns coherent output.
