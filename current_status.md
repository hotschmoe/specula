# specula -- current status

Last updated: 2026-04-19 (end of session 1)

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
- [x] **Preset `vulkan`** built; device enumeration correct (Adreno X2-90, native driver, `KHR_coopmat`). **Correctness issue open** -- see `docs/adreno_debugging.md`.
- [ ] **Preset `opencl`** -- blocked on OpenCL headers + `OpenCL.lib` for ARM64 (see Outstanding issues)
- [ ] **Preset `vulkan-opencl`** -- depends on both of the above
- [ ] **Preset `cpu-kleidiai`** -- deferred to Phase 1 SME2 retry
- [ ] Hexagon backend -- out of band (Qualcomm docker toolchain); not wired into `build_llama_cpp.ps1`
- [ ] Sweep harness validated end-to-end (scripts exist; not yet producing real CSVs)

Note (2026-04-19): initial combined `vulkan-opencl` preset split into
standalone `vulkan` and `opencl` presets so a missing OpenCL SDK doesn't
block Vulkan work. Combined preset kept for when both are satisfied.

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

1. **Work through the Adreno test matrix in `docs/adreno_debugging.md`.**
   Phase A (correctness) first: run `llama-cli` with each env-var
   combination (COOPMAT/COOPMAT2/F16 disabled in various combos) and
   record which configs produce coherent English. Phase B (performance)
   for every Phase-A-passing config: `llama-bench` across
   `pp32,128,512` x `tg16,64,128` on both Qwen3-0.6B and Qwen3-8B.
   Log everything to `results/adreno-*.log`.
2. Based on Phase A+B results: either lock in a working Adreno Vulkan
   config as the Phase 1 GPU baseline, or declare Vulkan/Adreno broken
   on this driver and pivot primary GPU attention to OpenCL/Adreno.
3. **Unblock OpenCL.** Pick vcpkg or manual Khronos checkout (see
   Outstanding issues), build `-Preset opencl`, smoke-test.
4. **Phase 1 formal baselines.** Once a working GPU backend is locked
   in, run `sweep_baseline.ps1` across all built backends for the
   Qwen3-0.6B / 1.7B / 8B model set. First real CSVs land in
   `results/`.
5. **SME2 / KleidiAI retry.** `-Preset cpu-kleidiai` build, then run a
   batched-matmul workload (PP-heavy) to trigger or not trigger the
   prior `STATUS_ILLEGAL_INSTRUCTION`.
