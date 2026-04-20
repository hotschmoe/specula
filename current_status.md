# specula -- current status

Last updated: 2026-04-19 (session 2 -- Adreno Vulkan fp16 triage)

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
- [ ] **Preset `opencl`** -- blocked on OpenCL headers + `OpenCL.lib` for ARM64 (see Outstanding issues)
- [ ] **Preset `vulkan-opencl`** -- depends on both of the above
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

- **OpenCL build blocked.** llama.cpp's CMake requires `find_package(OpenCL
  REQUIRED)`, which needs headers (`CL/cl.h`) and an import library
  (`OpenCL.lib`) for ARM64 on disk. Neither is installed. QAIRT has a
  stray `cl.h` buried in a sample app but no lib. Unblock options:
    1. `vcpkg install opencl:arm64-windows`
    2. Clone Khronos `OpenCL-Headers` + `OpenCL-ICD-Loader`, build the
       loader for ARM64, point cmake at the results via
       `-DOpenCL_INCLUDE_DIR=... -DOpenCL_LIBRARY=...`.
  Runtime concern (separate): Adreno OpenCL ICD is not registered in
  `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`, so device discovery will fail
  even after build until `QCOclIcd.dll` is registered. Deal with that
  when `llama-bench -d GPUOpenCL` first runs.
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

**Top priority: investigate and unblock OpenCL/Adreno.** Vulkan
exhausted — all seven env-var combos tested produced incorrect
output, and the one extension left to try (`INTEGER_DOT_PRODUCT`)
made things worse. OpenCL is Qualcomm's maintained backend and the
vendor-blessed GPU path on Adreno.

1. **Survey the OpenCL/Adreno landscape before building anything.**
   Spend 20-30 min on:
     - llama.cpp `ggml/src/ggml-opencl/` — what's supported, Adreno
       extensions used (`cl_qcom_reqd_sub_group_size`,
       `cl_qcom_image_pitch`, etc.), build flags (`GGML_OPENCL_USE_ADRENO_KERNELS`).
     - llama.cpp issues/discussions: "OpenCL Adreno", "Snapdragon
       X2", "X Elite OpenCL" — known bugs, working configs, perf
       reference points.
     - Qualcomm AIStack / QAIRT install at `C:\Program Files\Qualcomm\AIStack` --
       what ships, is there already a usable `OpenCL.lib` + headers
       bundled (previous QAIRT check only found a sample-app `cl.h`).
     - ICD registration: confirm `QCOclIcd.dll` location (System32?
       elsewhere?) and the exact registry key format Khronos ICD
       loader expects. Needs admin to write.
2. **Build + smoke test OpenCL preset.**
     a. Install OpenCL headers + ARM64 `OpenCL.lib` (vcpkg
        `opencl:arm64-windows` is the fastest route; fallback is
        building Khronos `OpenCL-Headers` + `OpenCL-ICD-Loader` for
        ARM64 and pointing cmake at the results).
     b. Build `-Preset opencl`. Confirm `GGML_OPENCL_USE_ADRENO_KERNELS`
        is ON in the preset.
     c. Register the Qualcomm ICD if not already: `REG_DWORD` entry
        for `<path-to-QCOclIcd.dll>` = 0 under
        `HKLM\SOFTWARE\Khronos\OpenCL\Vendors` (needs admin).
     d. Smoke test: `llama-bench -m Qwen3-0.6B-Q8_0.gguf` with the
        OpenCL-built binary (backend will show up as OpenCL if the
        build succeeded).
     e. Manual correctness check with the same Session-2 prompt
        (*"The Snapdragon X2 Elite Extreme is"* @ `--temp 0 --seed 1`)
        for a direct A/B against the Vulkan garble.
3. **Widen `LLAMA_BUILD_TOOLS`.** While rebuilding, add
   `llama-completion` and `llama-perplexity` so the next round of
   correctness work can be scripted instead of requiring a human at
   the terminal. `llama-cli` is interactive-only in current
   llama.cpp and ignores `-no-cnv`.
4. **Only if OpenCL also misbehaves**, revisit Vulkan via
   llama.cpp bisect or file an upstream issue with the driver
   version + the B0/B4/B6 output signatures + memory-underflow dump
   captured in `docs/adreno_debugging.md`.
4. **Phase 1 formal baselines.** Once a working GPU backend is locked
   in (Vulkan B3/B4 or OpenCL), run `sweep_baseline.ps1` across all
   built backends for the Qwen3-0.6B / 1.7B / 8B model set. First real
   CSVs land in `results/`.
5. **SME2 / KleidiAI retry.** `-Preset cpu-kleidiai` build, then run a
   batched-matmul workload (PP-heavy) to trigger or not trigger the
   prior `STATUS_ILLEGAL_INSTRUCTION`.
