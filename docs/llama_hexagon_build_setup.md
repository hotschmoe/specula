# llama.cpp Hexagon NPU build on the X2E — setup + handoff

Goal: build `ggml-org/llama.cpp` with the **Hexagon (HTP) backend** so we can
run GGUF models on the NPU with layer-splitting across HTP sessions + Adreno
GPU / CPU offload (`-ngl`) — the path past the ~10 GB ORT-QNN ceiling and the
only way a **27B dense** runs on this hardware.

## Status (auto-done by the agent)
- ✅ llama.cpp already cloned at `~/Documents/GitHub/llama.cpp` (recent master,
  has `ggml/src/ggml-hexagon/` + the snapdragon preset + docs).
- ✅ Toolchain present: LLVM clang/clang-cl, CMake, Ninja.
- ✅ **Hexagon SDK 6.6.0.0** → `C:\Qualcomm\Hexagon_SDK\6.6.0.0`
- ✅ **Adreno OpenCL SDK 2.3.2** → `C:\Qualcomm\OpenCL_SDK\2.3.2`
- ✅ NPU driver present (ORT-QNN already runs on the HTP).

## YOUR part — DSP code-signing (one-time, needs admin + reboot)

The HTP skel must be signed to load on the DSP via FastRPC. On Windows we use
test-signing + a self-signed cert. **These need an elevated shell and a
reboot — the agent cannot do them.**

1. **Enable test signing** (admin PowerShell), then **reboot**:
   ```
   bcdedit /set TESTSIGNING ON
   ```
   (After reboot you'll see a "Test Mode" watermark — that's expected.)

2. **Create a self-signed cert** (admin Developer Command Prompt; `makecert`
   ships with the Windows SDK at
   `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64\`):
   ```
   makecert -r -pe -ss PrivateCertStore -n CN=GGML.HTP.v1 -eku 1.3.6.1.5.5.7.3.3 -sv ggml-htp-v1.pvk ggml-htp-v1.cer
   pvk2pfx -pvk ggml-htp-v1.pvk -spc ggml-htp-v1.cer -pfx ggml-htp-v1.pfx
   ```
3. **Import the .pfx** (run `certlm`, or):
   - into **Trusted Root Certification Authorities**, and
   - into **Trusted Publishers**.
4. Tell the agent the path to `ggml-htp-v1.pfx`.

## AGENT part — build + run (after signing)
```
$env:OPENCL_SDK_ROOT="C:\Qualcomm\OpenCL_SDK\2.3.2"
$env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0"
$env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
$env:HEXAGON_HTP_CERT="<path>\ggml-htp-v1.pfx"
cmake --preset arm64-windows-snapdragon-release -B build-wos   # in llama.cpp/
cmake --build build-wos -j
```
Then validate, smallest → largest:
```
# 4B Q4_0 we already benchmark, all layers on one HTP session:
build-wos/bin/llama-bench -m <Qwen3-4B-Q4_0.gguf> -ngl 99 --device HTP0
# 14B: split across sessions + spill rest to GPU/CPU
GGML_HEXAGON_NDEV=4 ... --device HTP0,HTP1,HTP2,HTP3 -ngl <tuned>
# 27B: hybrid -ngl so HTP holds what fits, Adreno/CPU take the rest
```

## Why this beats ORT-QNN (recap)
ORT-QNN loads contexts fully resident; the X2E HTP caps at ~10 GB resident
weights and crashes the DSP transport during execution above ~8 GB. No
ORT-QNN runtime knob (shared-mem allocator, spill-fill) moved it (tested).
llama.cpp treats each HTP session as a GGML device and offloads overflow
layers to GPU/CPU over the unified 48 GB memory — the only approach that
scales to 27B. See `docs/htp_memory_ceiling_problem.md`.
