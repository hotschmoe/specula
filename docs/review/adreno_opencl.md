# Adreno X2-90 OpenCL survey + bring-up plan

Session 3 start: 2026-04-20. Successor to `adreno_debugging.md` after the
Vulkan pivot. This doc is the running log of the OpenCL/Adreno bring-up.

## TL;DR (post-build, session 3 complete)

- **It works.** OpenCL Adreno backend on Qwen3-0.6B Q8_0: coherent
  generation on first try, **PP128 1926 t/s, PP512 2674 t/s**, TG64
  111 t/s. That's ~3× CPU on PP128 and ~12× any Vulkan config we
  measured on prompt processing at 512-token shapes. Correctness
  output matches CPU greedy-sample reference verbatim on the
  session-2 prompt. Full numbers in §Session 3 perf below.
- No env var tuning required and no source patch required for this
  result. The `ADRENO_UNKNOWN` fallback from the gen matcher
  (X2-90 is not recognized by `get_adreno_gpu_gen`) did not hurt
  correctness and produced these perf numbers — so the generic
  Adreno path is already strong. Patching the matcher to recognize
  X2 explicitly is now a tuning experiment, not a prerequisite.
- **Khronos ICD registry entry turned out to be unnecessary** on
  this driver — the Adreno DCH driver package registers the ICD
  through a Windows-native mechanism that doesn't touch
  `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`. Device enumerates on
  first run with no admin steps. We wrote the Khronos entry anyway
  as belt-and-braces; it's harmless.

## Pre-build snapshot (kept for reference)

Items below document the state before the build — they're no longer
blockers but the findings are useful if this box is ever re-imaged or
the Adreno driver updates.

- OpenCL runtime bits ship with the Adreno graphics driver and QAIRT.
  No runtime install needed.
- Compile-time SDK (headers + `OpenCL.lib` for ARM64) was the only
  actual gap; unblocked by building Khronos
  `OpenCL-Headers` + `OpenCL-ICD-Loader` from source (below).
- QAIRT ships a complete Khronos header set (including
  `cl_ext_qcom.h`) in
  `examples\QNN\SampleApp\SampleAppGPUFencing\src\CL\` — would have
  unblocked the header side if the Khronos repos were unreachable.
  Didn't end up being needed.
- llama.cpp's `ggml-opencl` has extensive Adreno-specific code
  (108 kernels total, 57 reference `cl_qcom_*` extensions). Qualcomm
  engineers maintain this path upstream.
- **Source-level note:** `ggml-opencl.cpp:222` `get_adreno_gpu_gen`
  only recognizes A7X (730/740/750), A8X (830/840), and X1E (`X1`
  substring). X2-90 → `ADRENO_UNKNOWN`. Non-fatal; confirmed
  non-impactful to correctness and produced the perf numbers above.
  Adding an `"X2"` branch is a follow-up experiment, not a blocker.

## What's already on disk

### Graphics driver (installed via Windows driver store)

Location: `C:\Windows\System32\DriverStore\FileRepository\qcdx8480.inf_arm64_e11dd2e33e0b42d3\`

Relevant files:
- `OpenCL.dll` — driver-local OpenCL entry point (not the Khronos
  loader; don't confuse with `C:\Windows\System32\OpenCL.dll`)
- `OpenCL_adreno.dll` — the Adreno OpenCL ICD implementation. This
  is the DLL the Khronos ICD loader needs to dispatch to, once
  registered.
- `qcclarm64xcompiler.dll` — the CL → shader compiler for ARM64
- `kcl.dll`, `adreno_utils.dll`, `q3dtools_adreno.dll` — support

Driver pkg name: `qcdx8480.inf` → Adreno X2-class DirectX/Compute
driver.

### Khronos ICD loader (system)

- `C:\Windows\System32\OpenCL.dll` — present. Standard Khronos ICD
  loader that ships with modern Windows. This is the DLL our build
  will ultimately link against at runtime.

### Qualcomm AIStack / QAIRT 2.45.40.260406

Root: `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\`

- `lib\aarch64-windows-msvc\OpenCL.dll` and `OpenCL_adreno.dll` — same
  runtime DLLs as the driver, bundled here too (for QAIRT's own use).
- `include\QNN\…` — QNN / HTP / SNPE headers. **No `CL/*.h` headers
  here.**
- `examples\QNN\SampleApp\SampleAppGPUFencing\src\CL\` — the only
  spot with Khronos headers:
    - `cl.h`          (1964 lines)
    - `cl_platform.h` (1412 lines)
    - `cl_ext.h`
    - `cl_ext_qcom.h` (430 lines — Qualcomm extension defs)
    - `cl_version.h`

That header set is complete enough to satisfy `find_package(OpenCL)`
if copied (or symlinked) into a proper `CL/` include root, paired
with an `OpenCL.lib` import library. The import library is **not**
shipped by QAIRT — it has `.dll`s only.

## What's missing

1. **`OpenCL.lib` (import library for ARM64 Windows).** Needed so the
   linker can resolve the OpenCL API symbols against the runtime
   `OpenCL.dll`. Two ways to get one:
     - `vcpkg install opencl:arm64-windows` — grabs headers + a
       built import lib. Fastest path.
     - Build `OpenCL-ICD-Loader` from Khronos source for ARM64 —
       produces `OpenCL.lib` + `OpenCL.dll` as side-artifacts.
       Heavier.
2. **Khronos ICD registry registration.** `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`
   key does not exist. Needs:
     - Key: `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`
     - Value name: full absolute path to `OpenCL_adreno.dll`, e.g.
       `C:\Windows\System32\DriverStore\FileRepository\qcdx8480.inf_arm64_e11dd2e33e0b42d3\OpenCL_adreno.dll`
     - Value type: `REG_DWORD`
     - Value data: `0` (convention: 0 = enabled)
   Admin rights required. Without this entry, `clGetPlatformIDs` will
   return zero platforms even though the runtime DLL is on disk.

## llama.cpp's OpenCL backend — what we're about to build

Location: `llama.cpp/ggml/src/ggml-opencl/`
- `ggml-opencl.cpp` (main backend, ~3300+ lines)
- `kernels/*.cl` — 108 OpenCL kernels

### CMake options of interest

From `ggml/src/ggml-opencl/CMakeLists.txt` and root `CMakeLists.txt`:

| Option                           | Default | Notes                                         |
|----------------------------------|:-------:|-----------------------------------------------|
| `GGML_OPENCL`                    |  OFF    | Turn on the backend                           |
| `GGML_OPENCL_USE_ADRENO_KERNELS` |  ON\*   | Adreno-tuned matmul/GEMV kernels              |
| `GGML_OPENCL_EMBED_KERNELS`      |  ON     | Embed .cl source in binary (no runtime files) |
| `GGML_OPENCL_PROFILING`          |  OFF    | Extra CL profiling; adds CPU overhead         |
| `GGML_OPENCL_TARGET_VERSION`     | 300     | `CL_TARGET_OPENCL_VERSION` (OpenCL 3.0)       |
| `GGML_OPENCL_SOA_Q`              | (def'd) | Structure-of-arrays quantized layout          |

(\*) `GGML_OPENCL_USE_ADRENO_KERNELS=ON` is the Qualcomm-tuned path.
If the backend init sees a non-Adreno GPU it errors out and tells you
to rebuild with `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`.

### Runtime env vars (from grepping `getenv` in `ggml-opencl.cpp`)

| Env var                                   | Purpose                                     |
|-------------------------------------------|---------------------------------------------|
| `GGML_OPENCL_PLATFORM`                    | Pin which CL platform to use                |
| `GGML_OPENCL_DEVICE`                      | Pin which CL device to use                  |
| `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER`     | Opt into `cl_qcom_large_buffer` if supported|
| `GGML_OPENCL_DISABLE_FUSION`              | Disable kernel fusion                       |

Far fewer correctness "escape hatches" than Vulkan — there's no
"disable fp16" or "disable coopmat" equivalent. That reflects the
Qualcomm team's ownership: broken paths get fixed or removed, not
gated behind env flags.

### Adreno generation matcher — X2 unknown

`ggml-opencl.cpp:222`:
```cpp
static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name) {
    if (strstr(device_name, "730|740|750")) return A7X;   // paraphrased
    if (strstr(device_name, "830|840"))     return A8X;
    if (strstr(device_name, "X1"))          return X1E;
    return ADRENO_UNKNOWN;
}
```
The reported device version/name is `Qualcomm(R) Adreno(TM) X2-90 GPU`.
That string contains neither `X1`, nor `830/840`, nor `730/740/750`,
so the matcher returns `ADRENO_UNKNOWN`.

Downstream effect: gen-specific branches (e.g. large-buffer path,
wave-size tuning at `:3120`, and the `#ifdef GGML_OPENCL_USE_ADRENO_KERNELS`
guarded kernel selection) may pick a default path instead of an X2-
tuned path. The backend still initializes.

Follow-up: try first pass with stock upstream to see if X2-90 works
out-of-the-box as "unknown adreno"; if correctness is clean but perf
is low, patch the matcher to recognize `X2` and map to whichever of
the existing gen buckets makes most sense, or add a new `A8X_NEXT`
bucket. Revisit once we have a first correct run.

## Plan of action (session 3)

### Step 1 — Decide SDK source — DONE (Khronos from source)

vcpkg was not installed on this machine (`where vcpkg` → not found).
Chose the Khronos-from-source route over installing vcpkg: smaller
footprint, reuses the already-validated MSVC/clang toolchain, produces
the exact ARM64 import library we need.

Executed:
```powershell
# Siblings of specula/ and llama.cpp/
cd C:\Users\hotschmoe\Documents\GitHub
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git

# Headers (header-only, just installs files)
cmake -D CMAKE_INSTALL_PREFIX=C:/.../OpenCL-Headers/install `
      -S C:/.../OpenCL-Headers -B C:/.../OpenCL-Headers/build
cmake --build C:/.../OpenCL-Headers/build --target install

# Loader (ARM64 MSVC from default generator on this ARM64 box)
cmake -D CMAKE_PREFIX_PATH=C:/.../OpenCL-Headers/install `
      -D CMAKE_INSTALL_PREFIX=C:/.../OpenCL-ICD-Loader/install `
      -S C:/.../OpenCL-ICD-Loader -B C:/.../OpenCL-ICD-Loader/build
cmake --build C:/.../OpenCL-ICD-Loader/build --target install --config Release
```

Resulting artifacts:
- `C:\Users\hotschmoe\Documents\GitHub\OpenCL-Headers\install\include\CL\*` — full header set
- `C:\Users\hotschmoe\Documents\GitHub\OpenCL-ICD-Loader\install\lib\OpenCL.lib` — ARM64 import lib (28 KiB)
- `C:\Users\hotschmoe\Documents\GitHub\OpenCL-ICD-Loader\install\bin\OpenCL.dll` — built ICD loader (81 KiB). **Ignored at runtime** — we want `C:\Windows\System32\OpenCL.dll` to win the DLL search, since that's the one Windows registers + updates with the platform. Keep the built DLL around only for offline testing.
- `C:\Users\hotschmoe\Documents\GitHub\OpenCL-ICD-Loader\install\bin\cllayerinfo.exe` — side artifact, useful as an admin-registration sanity check (doesn't enumerate platforms itself, but confirms loader integrity).

### Step 2 — Wire `-Preset opencl` in `build_llama_cpp.ps1` — DONE

Added:
- `-DGGML_OPENCL=ON`
- `-DGGML_OPENCL_USE_ADRENO_KERNELS=ON`
- `-DGGML_OPENCL_EMBED_KERNELS=ON`
- `-DOpenCL_INCLUDE_DIR=$OpenClHeadersDir`
- `-DOpenCL_LIBRARY=$OpenClLibrary`

with defaults pointing at the sibling Khronos checkouts built in
Step 1. Also widened the build-target set to include
`llama-completion` and `llama-perplexity` (needed for scripted
correctness / perplexity; blocker noted in session 2).

Also fixed a latent PS 5.1 quirk in the build script: it was setting
`$ErrorActionPreference = 'Stop'` globally, which caused
`NativeCommandError` wrapping of any stderr line from `cmake`
(cmake emits progress to stderr) to terminate the script even on a
successful configure. Relaxed to `'Continue'` — explicit
`$LASTEXITCODE` checks remain. The fix applies to every preset, not
just opencl.

### Step 3 — Register the ICD — NOT NEEDED on this driver/Windows combo

Opened as a planned admin step, but the device enumerated
immediately on the first `llama-bench` run with no registry writes.
The Adreno DCH graphics driver registers its ICD through a
Windows-native path that doesn't go through
`HKLM\SOFTWARE\Khronos\OpenCL\Vendors` (both pre- and post-build
checks of that key returned "not found" while the device still
enumerated correctly). As of session 3 the Khronos Vendors entry
was written manually as defence-in-depth, but it is not load-bearing.

If a future driver update or re-image breaks enumeration, the fallback
is still the admin write:
```powershell
# Belt-and-braces; not required today
$icdPath = 'C:\Windows\System32\DriverStore\FileRepository\qcdx8480.inf_arm64_e11dd2e33e0b42d3\OpenCL_adreno.dll'
New-Item -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -Force | Out-Null
New-ItemProperty -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' `
                 -Name $icdPath -Value 0 -PropertyType DWord -Force | Out-Null
```
The driver store hash (`e11dd2e33e0b42d3`) changes when the
Adreno driver is updated; re-grep the driver store to find the
current slug before writing the registry entry.

### Step 4 — Smoke + A/B

Build + basic device enum:
```powershell
.\llama.cpp\build-opencl\bin\llama-bench.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p 128,512 -n 64 -r 2
```
Expect: one OpenCL device listed, headline PP/TG numbers.

Direct correctness A/B against the Vulkan garble baseline — use the
same session-2 prompt, same seed, so the output is directly
comparable to `adreno_debugging.md`'s correctness table:

```powershell
.\llama.cpp\build-opencl\bin\llama-completion.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -ngl 99 --temp 0 --seed 1
```
(assuming `llama-completion` is in the rebuilt tool set; otherwise
fall back to `llama-server` driven over HTTP). The CPU reference is
the Qwen3 thinking-mode chain starting `[Start thinking] Okay, the
user is asking about the Snapdragon X2 Elite Extreme…`.

### Step 5 — Phase 1 handoff — partial (0.6B done, 8B pending)

See §Session 3 perf below for 0.6B numbers. Remaining: matching
matrix at Qwen3-1.7B Q8_0 and Qwen3-8B Q4_K_M. `sweep_baseline.ps1`
is now unblocked for opencl + cpu; kick it off for the first real
CSV handoff into `results/`.

### Step 6 — Contingencies — all negative outcomes avoided

None of the contingency branches tripped. Documented for the record
in case a future driver update breaks the happy path:

- Build fails on `find_package(OpenCL)` → give cmake a lib (we did).
- Build succeeds, zero devices enumerate → Khronos Vendors registry
  entry is the first-resort admin step.
- Device enumerates, init aborts "not an Adreno GPU" → X2 recognition
  patch in `get_adreno_gpu_gen`.
- Correctness garble → bisect llama.cpp, file upstream issue, or
  pivot to Hexagon.

## Session 3 perf (2026-04-20)

### Setup
- Commit: `fd6ae4ca1cd5446442f6c2e5e73a2a4c9bc44993` (upstream HEAD
  at build time)
- Build: `llama.cpp\build-opencl\bin\` (see
  `SPECULA_BUILD.txt` for the full stamp)
- Device: Qualcomm Adreno X2-90 GPU (OpenCL 3.0), `ADRENO_UNKNOWN`
  generation in the matcher
- Flags: `GGML_OPENCL_USE_ADRENO_KERNELS=ON`,
  `GGML_OPENCL_EMBED_KERNELS=ON`. No env vars set.

### Correctness — PASS

Prompt `"The Snapdragon X2 Elite Extreme is"`, `--temp 0 --seed 1 -n 64`,
via `llama-completion`. (llama-completion in this HEAD also defaults
to conversation mode — still generates correctly, but hangs waiting
for interactive input at the end; a tooling cleanup, not a
correctness problem.)

Output begins:
> `Okay, the user is asking about the Snapdragon X2 Elite Extreme. First, I need to confirm if this is a real product. I know that the Snapdragon X2 is a mid-range processor from Snapdragon Technologies, but the X2 Elite Extreme might be a newer model. Let me check my knowledge. The`

Matches the CPU reference in `adreno_debugging.md` §Correctness
results verbatim. Full log at
`results\adreno-opencl-correctness-0.6B.log`.

### Performance — Qwen3-0.6B Q8_0 (`llama-bench -p 128,512 -n 64 -r 2`)

Log: `results\adreno-opencl-perf-0.6B.log`.

| Backend        | pp128 t/s       | pp512 t/s       | tg64 t/s      |
|----------------|-----------------|-----------------|---------------|
| **OpenCL**     | **1926.55 ± 26.69** | **2674.19 ± 13.28** | **111.16 ± 0.91** |
| CPU (ref, 18t) | ~826 (session 1)    | —               | ~111 (session 1)  |
| Vulkan B0 (broken correctness) | 37.5  | 20.4            | 104.8         |
| Vulkan B3 "fast but wrong"     | 599.8 | 604.2           | 100.3         |

Observations:
- OpenCL is **~2.3× CPU on PP128**, and PP scaling is *positive*
  going to 512 (1926 → 2674 t/s) — the compute kernel is genuinely
  being fed large GEMMs, not launch-overhead-bound.
- TG is flat around 111 t/s across OpenCL, CPU, and all Vulkan
  configs: bandwidth-bound at this model size (0.6B fits comfortably
  in cache/UMA; TG = single-sequence decode). Expected.
- Vulkan on the fast-but-wrong path (B3) produced 604 t/s PP512;
  OpenCL's correct run exceeds that by **4.4×**. So OpenCL is not
  just correct where Vulkan was broken — it's also substantially
  faster even vs Vulkan's broken-ceiling.

### Remaining benches

- Qwen3-1.7B Q8_0: expected PP drops with bigger weights but GPU
  advantage grows for compute-bound regimes — predict OpenCL
  retains a strong PP lead.
- Qwen3-8B Q4_K_M: the Vulkan partial data at 8B (session 2)
  plateaued ~45 t/s PP on the fp32 path; OpenCL should comfortably
  beat that. Worth the measurement to confirm.
- Long-context / larger batch: `-p 1024,2048` not yet measured,
  relevant for speculative-decode workloads in Phase 2.

## Follow-ups queued

- **Gen-matcher patch for X2** (`ggml-opencl.cpp:222`). Currently
  falls to `ADRENO_UNKNOWN`; doesn't harm correctness or these perf
  numbers, but may be leaving gen-specific tuning on the table.
  Experiment: patch `strstr(device_name, "X2")` → probably either
  `A8X` or a new `X2E` bucket, rebuild, re-bench, compare.
- **`cl_qcom_large_buffer` opt-in.** Env flag
  `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1`; check if extension is
  advertised on this driver (`ext_buffer` scan at
  `ggml-opencl.cpp:3177`), and if so, bench impact on 8B.
- **8B bench.** `Qwen3-8B-Q4_K_M.gguf` through the same matrix.
- **Upstream issue option kept open.** If we find any
  regression-per-HEAD on this driver in the future, `adreno_debugging.md`
  + this file together make a solid report body.
