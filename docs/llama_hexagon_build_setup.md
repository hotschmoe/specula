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

## 🏆 WORKING (2026-06-16) — Qwen3-4B runs on the Hexagon NPU

```
ADSP_LIBRARY_PATH=<build-wos>\ggml\src\ggml-hexagon  (skel + signed .cat dir)
bin\llama-bench.exe -m Qwen3-4B-Q4_0.gguf -ngl 99 --device HTP0 -p 128 -n 32
| qwen3 4B Q4_0 | 2.21 GiB | OpenCL,HTP | 99 | HTP0 | pp128 | 101.77 t/s |
|                                                    | HTP0 | tg32  |  17.98 t/s |
```
HTP session opens: `new session: HTP0 ... handle 0x...`. The ~10 GB ORT-QNN
ceiling is bypassed — llama.cpp streams/splits across HTP sessions + GPU/CPU.

### The two things that unlocked it (both required)
1. **Signed skel catalog** (Windows code-integrity): build with
   `-DGGML_HEXAGON_HTP_CERT=<pfx>` + `WINDOWS_SDK_BIN=<…\10.0.26100.0>` (parent
   dir, so `/x86\inf2cat.exe` + `/arm64\signtool.exe` resolve). **`inf2cat`
   needs the WDK** — `winget install Microsoft.WindowsWDK.10.0.26100`. The
   cert var is a CMake CACHE entry, so pass it with `-D` (env alone won't
   override a prior cert-less configure). `cmake --build build-wos --target
   libggml-htp-cat` → `libggml-htp.cat` (~4 KB). (`makecat` produces a 1.3 KB
   catalog that does NOT satisfy the loader — must be `inf2cat`.)
2. **`ADSP_LIBRARY_PATH`** must point at the dir holding the skels **and** the
   signed `.cat` — without it, `htp_iface_open` fails `0x80000406` even though
   the code enables Unsigned PD. This was the final missing piece.

Cert: `C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx` (password-less, EKU
1.3.6.1.5.5.7.3.3), imported to LocalMachine Root + TrustedPublisher.

## 🏆🏆 14B VERIFIED on the NPU (2026-06-16) — ceiling broken

`Qwen3-14B-Q4_0.gguf` (7.92 GiB; converted HF→GGUF via
`convert_hf_to_gguf.py --outtype f16` + `llama-quantize Q4_0`) **runs on the
Hexagon NPU** — which ORT-QNN could not do (14B died at the >10 GB wall):
```
GGML_HEXAGON_NDEV=4 bin\llama-bench -m Qwen3-14B-Q4_0.gguf \
   --device HTP0,HTP1,HTP2,HTP3 -ngl 34 -p 64 -n 16
| qwen3 14B Q4_0 | 7.92 GiB | OpenCL,HTP | 34 | pp64 40.9 t/s | tg16 11.2 t/s |
```

### Hardware limit found: exactly 4 HTP sessions
HTP0–3 open (domains 3/7/11/15); a **5th session fails `error 0x200`**. At
~2 GB/session that's **~8 GB max resident on the NPU** — this IS the root of
the ~10 GB ceiling (4 process domains). **Models >~8 GB require hybrid
`-ngl`** (overflow layers to Adreno GPU/CPU over unified 48 GB). 14B (8.5 GB)
→ `-ngl 34` (34/40 on HTP, 6 on GPU/CPU). A 27B (16 GB Q4_0) → ~50/50.

## 27B status — NPU ready, blocked on llama.cpp arch support
- **Multi-session works:** the 27B attempt opened **4 HTP sessions** (HTP0–3,
  domains 3/7/11/15) cleanly — the per-session split mechanism is proven.
- **But `Qwen3.6-27B-MTP-Q4_0.gguf` won't load on ANY backend** (fails on CPU
  too): `general.architecture = qwen35` — the Qwen3.6 hybrid SSM+attention
  VLM arch is **not yet supported by llama.cpp** (build b8833). Not an NPU
  issue. To run the 27B on the NPU, llama.cpp needs **`qwen35` arch support**
  (we have deep arch knowledge from the AI-Hub SSM op-compilability work —
  candidate to contribute upstream or patch locally).
- **Next demo within reach:** convert our **Qwen3-14B** (HF, standard `qwen3`
  arch, supported) → GGUF Q4_0 (~8 GB) and run it on the NPU with
  `GGML_HEXAGON_NDEV` multi-session — proves a 14B-class model past the
  per-session ceiling today.

## BUILD DONE + blocker confirmed (2026-06-16)
- ✅ Built `llama-cli.exe` + `llama-bench.exe` and **`libggml-htp-v81.so`**
  (our arch; v68–v79 also built). `cmake --preset
  arm64-windows-snapdragon-release` → `cmake --build build-wos --target
  llama-cli llama-bench` (the 2 example targets `llama-debug`/
  `llama-eval-callback` fail on an unrelated upstream `common_debug_cb_eval`
  link error — ignore; build the targets we need explicitly).
- ✅ Runtime detects the **Adreno X2-90 GPU** (OpenCL, max alloc 2048 MB) and
  **Hexagon Arch v81**, loads `libcdsprpc.dll`.
- 🛑 **`ggml-hex: failed to open session 0 : error 0x80000406`** — the HTP
  skel is **unsigned**. This is the SAME `0x80000406` we'd blamed on a
  "broken Genie DSP transport" — it is actually the **signed-PD requirement**:
  ORT-QNN works because it uses Qualcomm's *pre-signed* QnnHtp skel; llama.cpp
  ships a custom skel that must be signed. **Fix = the signing steps below,
  then rebuild the skel with `HEXAGON_HTP_CERT` set so it gets signed.**

## Dev environment: native Windows ARM64 only (NOT WSL/Docker)
The Hexagon NPU is driven via FastRPC → `libcdsprpc.dll` → the `qcnspmcdm`
Windows kernel driver — a **Windows-native** stack. **WSL2 has no NPU
passthrough; Windows/Linux containers can't reach the cDSP.** Test-signing,
the skel cert, and the runtime are all Windows-only and every DSP test must
run natively. So all NPU build+run work happens in native Windows ARM64 here.
(Offline build steps — export/quantize/context-bin — DO run in Linux/Docker on
the Threadripper; that's x86, no NPU execution. Native-Linux-on-Snapdragon is a
separate device class, not this laptop.)

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
