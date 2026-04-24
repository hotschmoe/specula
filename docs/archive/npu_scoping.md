# Phase 5 NPU scoping -- Qwen3-0.6B-Q8_0 draft on Hexagon

Session 5 scoping pass, 2026-04-20. Inputs: voice_project (proven
QAIRT+AI-Hub pipeline for Whisper), trident (NPU postmortem on LLM
GEMM dispatch), llama.cpp spec-decode wiring, specula Phase 2 results.

Goal: decide how we get Qwen3-0.6B-Q8_0 drafting on Hexagon with the
Qwen3-8B-Q4_K_M target still on CPU @ 18 threads, so the first number
is directly comparable to Phase 2's 40.2 t/s / 1.55x CPU-spec
baseline (current_status.md:82-94).

## 1. TL;DR

- **Primary path: Qualcomm AI Hub cloud compile (ONNX -> QNN
  weight-shared context binary).** This is the only NPU path that
  has ever worked end-to-end on this exact hardware locally, per
  voice_project/current_status.md:12-30. trident's npu_path_back.md
  endorses the same choice for LLMs for the same reasons (driver
  signing, version pinning, pre-linked context binaries). User's
  prior guess is confirmed.
- **Runtime EP: ONNX Runtime + QNN EP (pip
  `onnxruntime-qnn` 1.24.4)** -- not raw `QnnContext_createFromBinary`.
  voice_project hit three driver-signing walls on the raw QNN path
  (voice_project/current_status.md:53-57) and only unblocked by
  switching to ORT's bundled, fully-signed QAIRT stack. trident's
  raw-QNN runtime works on Windows too, but incurs the per-context
  overhead that made trident a 9x regression
  (trident/npu_current_status.md:4).
- **Hexagon arch is v81, not v79, on this board.** Confirmed by
  voice_project/encoder_info_v81.json:161 (`"dspArch": 81`,
  `"socModel": 88`) and voice_project/current_status.md:8
  ("Hexagon V81 (soc_model 88)"). Same SoC as us (SC8480XP / X2E
  Extreme). The README's earlier guess of v79 is wrong; update it.
- **Fallbacks in order:** (b) local QAIRT converter -> QNN context
  binary loaded via ORT-QNN (same runtime, skip the cloud), then
  (c) direct `QnnContext_createFromBinary` via QAIRT libs only if
  ORT-QNN can't express the KV-cache IO shape. Trident-style
  per-shape runtime `graphCreate` is explicitly ruled out.
- **Win-condition framing:** even a tied NPU-spec number (~40 t/s
  vs CPU-spec 40.2 t/s) is a shipping milestone because it proves
  heterogeneous compute works on X2E at small batch. The user has
  already said a working path is valuable independent of raw speed.

## 2. Paths to NPU execution

Each path evaluated against: (a) proven on X2E, (b) effort from
zero, (c) what it unlocks for specula.

### Path A: QAIRT + AI Hub cloud compile -> QNN context binary, run via ORT-QNN EP

**Toolchain:** HuggingFace PyTorch -> ONNX export -> `qai_hub.submit_compile_job`
with `--target_runtime qnn_context_binary --compute_unit npu` targeting
`Device("Snapdragon X2 Elite CRD")` -> download `.bin` -> load via
`Microsoft.ML.OnnxRuntime.QNN` / pip `onnxruntime-qnn`.

**Proven on this hardware:** YES, for Whisper encoder (414 MB ONNX)
and decoder (721 MB ONNX). See voice_project/aihub_compile.py:66-82
for the exact submit pattern, and voice_project/current_status.md:12-18
for the end-to-end dictation proof with sample timings
(652-1276 ms per utterance).

**What it gets us:** Hexagon-resident execution of the draft
transformer, with the JIT->`_ctx.onnx` cache format making cold
starts <1 s after first run (voice_project/current_status.md:30).
QAIRT 2.45 stack is signed and loads on retail Windows ARM64
without Secure Boot tricks -- the critical win.

**What it costs:** HuggingFace -> ONNX export work (Qwen3-0.6B has
no existing ONNX export we know of; we do a PyTorch tracing pass
ourselves). AI Hub account + token already in place
(trident/npu_path_back.md:7 says we have one). Quantization needs
calibration data (W4A16 or W8A16 recipe per trident/npu_path_back.md:37).
Two-file deploy: ONNX artifact + `_ctx.onnx` cache dir per SoC
firmware version.

**Known failure modes already hit:**
- AI Hub SDK version must match runtime QAIRT version; mismatch
  yields `createFromBinary err 1009` (trident/npu_path_back.md:66).
  Pin SDK == 2.45.x at both ends.
- The 32 s `contextCreateFromBinary` overhead from trident does NOT
  apply here because ORT-QNN loads the binary once per session, not
  per-shape (trident/npu_current_status.md:60-68 is for runtime
  graphCreate churn, not for the pre-linked-binary path).
- QAIRT 2.45 converter has a broken `ErfDummyLayoutInferer` that
  kills GELU-exact decoder compiles -- exported Qwen3 uses SiLU/Swish
  so this exact bug doesn't bite, but watch for analogous
  op-lowering breaks (voice_project/current_status.md:62).
- AI Hub `job.wait()` prints non-cp1252 chars and crashes Windows
  console; use explicit polling (aihub_compile.py:46-57).

**Effort estimate:** 3-5 days of hands-on work after assets exist.
Bulk is the PyTorch->ONNX export with KV cache and the per-session
IO plumbing, not the cloud compile itself.

### Path B: Local QAIRT converter (ONNX -> QNN, no cloud)

**Toolchain:** QAIRT 2.45 `qnn-onnx-converter` + `qnn-model-lib-generator`
+ `qnn-context-binary-generator`. Must run under x64 Python
emulation on ARM64 Windows -- voice_project has this wired up in
run_qnn_converter.py:14-25 and build_qnn_models.py:14-41 (both
monkeypatch `platform.processor` to pass QAIRT's ARM-hostile init).

**Proven on X2E:** Partially. voice_project built this pipeline
and has working converter scripts, but the aihub_compile.py path
is what shipped. Local converter is on-ramp for iteration when the
cloud is slow or you need a small shape fast.

**What it gets us:** Identical output artifact to Path A (QNN
context binary consumable by ORT-QNN). Unblocks iteration when AI
Hub is slow or the account token expires.

**What it costs:** x64 Python emulation toolchain pinned at 3.10.
Calibration data curation is now local. No cloud compile farm --
expect minutes-to-tens-of-minutes per compile on a decent ARM64
Windows box. QAIRT 2.45 converter's `ErfDummyLayoutInferer` bug
bites harder here because you see it in the flesh (voice_project
handled by swapping GELU-exact -> GELU-tanh in the decoder export).

**Known failure modes:**
- Int64 tensor misread under x64 emulation -- QAIRT's C++ ops
  read int64 initializers as int32 and produce garbage.
  voice_project/fix_onnx_int64.py:22-40 is the known-good
  workaround (cast int64 -> int32, clamp MAX_INT64 sentinels in
  Slice ops). Apply this between ONNX export and converter.
- Input encoding flags mandatory for non-image IO (see
  run_qnn_converter.py:54-78: `--input_encoding ... other` and
  `--preserve_io layout ...` for every non-activation tensor,
  otherwise the converter assumes NHWC layout).

**Effort estimate:** if Path A works first, Path B is ~1 day of
plumbing to produce the same artifact locally. Skip unless we
need the iteration speed.

### Path C: QNN custom-op authoring + cross-compile to Hexagon

**Toolchain:** QAIRT `qnn-op-package-generator`, Hexagon SDK
(separate install), C++ op source compiled to Hexagon libraries,
registered into a QNN graph at compile time.

**Proven on X2E:** NO. voice_project never needed it (Whisper
ops all covered by stock QAIRT op library). trident never needed
it (its LLM GEMMs are stock matmul). No local project has this
working.

**What it gets us:** Only matters if Qwen3-0.6B uses an op QAIRT's
shipped library doesn't cover. Qwen3 architecture is
matmul + rms_norm + rope + silu + residual -- all in the standard
op set. **We almost certainly don't need this for Phase 5.**

**What it costs:** Hexagon SDK install, C++ op authoring per shape,
signing concerns for unsigned Skels on retail Windows
(voice_project/current_status.md:53-55 -- Hexagon Skels must be
signed, OEM Secure Boot blocks testsigning). If we ever hit this,
the ORT-bundled QAIRT stack won't load our custom op without an
effectpack signature we can't get.

**Known failure modes:** same Hexagon-Skel signing wall
voice_project hit during the raw-SDK attempt. Deprioritize hard.

### Path D: ExecuTorch-on-Hexagon

**Toolchain:** PyTorch ExecuTorch runtime with Qualcomm HTP
backend. Not mentioned in either voice_project or trident docs.

**Proven on X2E:** Unknown from our internal corpus. ExecuTorch is
a Meta project; Qualcomm upstreamed HTP partitioner work but the
runtime-on-Windows story is thin.

**What it gets us:** Alternative to ORT-QNN for the runtime side if
ORT's EP API changes again. Speculative; not our primary bet.

**Effort estimate:** unknown. Marking unresolved.

### Path E: Direct Hexagon SDK route (libcdsprpc + raw QNN)

**Toolchain:** `libcdsprpc.dll` + raw `QnnContext_createFromBinary`
+ manual `rpcmem_alloc` + `QnnMem_register`. trident's npu/ stack
uses this (trident/npu_current_status.md:136-139).

**Proven on X2E:** YES for compute correctness, but trident shelved
it as a 9x regression due to QNN SDK overhead on the user-driver
path (trident/npu_current_status.md:4). Also driver-signing
walls -- voice_project tried this and got `0x80000406 Unable to
load lib` on unsigned Skels before pivoting to ORT-QNN
(voice_project/current_status.md:54-57).

**What it gets us:** lowest-overhead runtime IF we can keep a single
long-lived context. trident's revisit plan
(trident/npu_current_status.md:103-121) -- one pre-linked binary,
load-once at init, `rpcmem` for IO -- would apply here and would
collapse the overhead from 32 s to milliseconds. This is actually
identical to Path A's artifact, just loaded via raw QNN instead
of ORT-QNN.

**When to use:** if ORT-QNN can't express the autoregressive IO
handoff we need (e.g. updating KV-cache tensor bindings per token
is clunky in ORT sessions). Then fall through to raw QNN with a
persistent context. **Keep as the warm fallback.**

### Path F: HTP-via-ONNX-Runtime EP (what Path A actually uses)

Not a separate path. Called out explicitly here because ORT + QNN EP
is how *both* trident's revisit plan and voice_project's shipping
build actually reach the NPU. voice_project's VoiceHotkey/Transcriber.cs
uses `OrtEnv.RegisterExecutionProviderLibrary` +
`SessionOptions.AppendExecutionProvider(env, epDevices, options)`
(voice_project/current_status.md:62-63).

**Gotcha:** ORT 1.24+ plugin EP API is **not** the old
`SessionOptions(providers=["QNNExecutionProvider"])`. That string
API silently falls back to CPU. Use the new
`GetEpDevices` + `AppendExecutionProvider` pattern.

## 3. Known failure modes catalog

Organized by "applies to specula phase 5?" because re-discovery is
expensive.

### 3.1 Driver signing walls (voice_project/current_status.md:53-57)

**What was tried:** Raw `libQnnHtp.dll` + pre-built V81 context
binaries under three QAIRT stacks:
1. QAIRT 2.45 retail user driver -- `0x80000406 Unable to load lib`;
   retail Windows refuses unsigned Hexagon Skels.
2. QAIRT 2.41 driver-signed stack (`qcnspmcdm8480.inf`) -- FastRPC
   loaded, but binary version check failed AND AI Hub doesn't
   compile for 2.41.
3. QAIRT 2.42 effectpack stack (`microsofteffectpack_extension`) --
   signed but restricted to MS system apps; a plain console .exe
   can't use it.

**Root cause:** Secure Boot is OEM-locked on consumer Snapdragon X
laptops; `bcdedit /set testsigning on` is blocked.

**Applies to specula?** YES if we go Path E. **Does not apply if we
go Path A/B via ORT-QNN** -- pip `onnxruntime-qnn` 1.24.4 bundles
a fully-signed QAIRT stack that is the single non-SDK path that
works on retail. This is the load-bearing reason for
"Primary path = Path A, runtime = ORT-QNN".

### 3.2 HTP DMA exhaustion, error 0x1774 (trident/npu_current_status.md:50-58)

**What was tried:** multiple concurrent QNN contexts for per-shape
GEMM dispatch. 252 graphs -> 0x1774. 5 graphs one-context -> 0x1774.
5 separate contexts -> 0x1774 (DMA is global, not per-context).
`graphPrepareExecutionEnvironment` -> 0x1770 (not supported on
user driver). `REGISTER_MULTI_CONTEXTS` -> 0x1392 (not supported).
`INIT_ACCELERATION` config -> "API version mismatch" on user
driver.

**Root cause:** spill/fill buffers ~187 MB per graph
(trident/npu_current_status.md:82); concurrent graphs exceed HTP
DMA reservation. User-driver path lacks the APIs that would let you
decouple graph residency from DMA reservation.

**Applies to specula?** YES as a sizing constraint, but the
exposure is different. A single weight-shared context binary for
the entire Qwen3-0.6B draft forward pass is ONE context, not 180.
We keep it loaded for the whole session -- load once, bind IO per
token, execute. The DMA wall is hit by trident's per-GEMM graph
design, not by whole-model context binaries. Expected spill/fill
for Qwen3-0.6B Q8_0 ~600 MB weights + KV cache is well within
the envelope. **Sanity-check empirically after first compile.**

### 3.3 QNN SDK per-context overhead dominates (trident/npu_current_status.md:60-78)

**What was tried:** 252 `contextCreateFromBinary` calls during
prefill, 178 ms each -> 32.2 s (25% of pp512). Actual HTP compute
only 1.4 s (1.1%).

**Root cause:** QNN SDK on Windows user-driver path spends ~178 ms
per context create; Qualcomm hasn't optimised this. Trident can't
use the acceleration config that would reduce it.

**Applies to specula?** NOT if we use Path A (one context, loaded
once, reused for every draft token). The overhead is amortized to
zero over an inference run. This is the SAME observation that drove
trident's npu_path_back.md: "pre-linked binary, loaded once" is the
whole point.

### 3.4 QAIRT 2.45 converter ErfDummyLayoutInferer breaks GELU-exact

**Source:** voice_project/current_status.md:62.

**What breaks:** `qnn-onnx-converter` crashes on ONNX decoders that
use exact GELU (the one with erf). Workaround: export with
`nn.GELU(approximate='tanh')`.

**Applies to specula?** Qwen3 uses SiLU (Swish), not GELU. So this
specific bug doesn't hit us. **But** the lesson generalises: the
QAIRT 2.45 converter has latent op-lowering bugs. If ONNX export
of Qwen3 hits an op that triggers a similar converter failure,
the first move is to rewrite that op in an equivalent form (GELU
exact -> GELU tanh in voice_project; our analogue would be rope,
rms_norm, or any custom attention mask pattern).

### 3.5 int64 tensor misread under x64 emulation (Path B only)

**Source:** voice_project/fix_onnx_int64.py:22-40.

**What breaks:** QAIRT's C++ ops under x64 Python emulation on ARM64
misread int64 initializer values as int32, producing garbage.
Affects at least Slice/Reshape/Gather shape-ints.

**Fix:** run `fix_onnx_int64.py`-equivalent between ONNX export and
the converter: cast int64 initializers to int32, clamp MAX_INT64
sentinels. Not needed for AI Hub cloud compile (cloud runs native
x64).

**Applies to specula?** YES if we use Path B. Copy the voice_project
util into specula when we go local. Not needed for Path A.

### 3.6 ORT 1.24+ EP API regression ("providers" kwarg silently falls back to CPU)

**Source:** voice_project/current_status.md:63.

**What breaks:** passing `providers=["QNNExecutionProvider"]` to
`SessionOptions` in ORT 1.24+ silently runs on CPU with no error.
Looks like it works; returns wrong-speed wrong-latency results.

**Fix:** use the plugin-EP API:
`OrtEnv.RegisterExecutionProviderLibrary` +
`GetEpDevices` + `SessionOptions.AppendExecutionProvider(env, epDevices, options)`.

**Applies to specula?** YES -- we will be writing exactly this
loader code. Use voice_project's Transcriber.cs pattern as the
reference.

### 3.7 Vulkan is broken on Adreno on this driver (orthogonal, but relevant)

Already addressed at specula/current_status.md:18 and in
trident/postmortem.md:45. Not an NPU failure mode, but a reminder
that GPU paths on X2 have ongoing driver issues and the NPU might
inherit analogous weirdness. Flag as "surprise budget" rather than
a specific failure.

### 3.8 ORT-QNN ↔ AI Hub QAIRT version mismatch (error 5000 on context load)

Hit on step 5, session 9: context binary compiled cleanly by AI Hub
(QAIRT 2.45.40) but `InferenceSession` threw
`LoadCachedQnnContextFromBuffer Failed to create context from binary.
Error code: 5000`. Root cause: `onnxruntime-qnn 1.24.4` bundles
QAIRT **2.42.0**; the serialized QNN context format is version-gated,
so a 2.45-compiled binary won't load on a 2.42 runtime.

Fix: upgrade to `onnxruntime-qnn>=2.1.0` (bundles 2.45.40 and ships
`Genie.dll` as a bonus). Full gotcha writeup + diagnostic recipe in
`docs/npu_ort_qnn_version_match.md`.

Keep this pair in sync on every recompile. A 60-second version probe
beats a 20-minute compile that produces an unloadable binary.

## 4. Toolchain pinning

| Component               | Pin                                                    | Source                                         |
|-------------------------|--------------------------------------------------------|------------------------------------------------|
| QAIRT SDK               | **2.45.40.260406**                                     | voice_project/build_qnn_models.py:18; encoder_info_v81.json:21 |
| Hexagon arch            | **v81** (`dspArch: 81`, `socModel: 88`)                | voice_project/encoder_info_v81.json:161-165   |
| ORT-QNN                 | **1.24.4** (pip `onnxruntime-qnn` or NuGet `Microsoft.ML.OnnxRuntime.QNN`) | voice_project/current_status.md:27             |
| AI Hub device string    | `"Snapdragon X2 Elite CRD"`                            | voice_project/aihub_compile.py:24              |
| AI Hub compile options  | `--target_runtime qnn_context_binary --compute_unit npu` | voice_project/aihub_compile.py:25              |
| Python (for QAIRT tools)| **3.10 x64 under emulation** (required by QAIRT pyd)  | voice_project/run_qnn_converter.py:3,15        |
| AI Hub auth             | API token already provisioned                          | trident/npu_path_back.md:7                     |
| Clang / cmake           | inherit from existing specula build env (clang 22.1.3, vcvarsarm64) | specula/current_status.md:196-204 |
| `rpcmem_alloc` lib      | `libcdsprpc.dll` (system DLL)                          | trident/npu_current_status.md:136; trident/npu_path_back.md:56 |

**Unresolved:** whether pip `onnxruntime-qnn` 1.24.4 actually bundles
QAIRT 2.45.x (voice_project says ORT bundles "a fully-signed QAIRT
stack" but doesn't name the version). Confirm during step 1 below
by inspecting the installed package's native DLLs. (non-blocking)

**Unresolved:** whether Qwen3 ONNX export needs a custom KV-cache
tracing wrapper. HuggingFace's generic `optimum.onnxruntime` export
is KV-aware but may produce shapes that don't lower cleanly on
QAIRT. (non-blocking for scoping, blocking for step 3 below)

## 5. Device enumeration sanity-check plan

Run BEFORE writing any code. All commands from Powershell on the
target machine.

### 5.1 Confirm QAIRT install + arch binaries present

- Check `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\` exists (matches
  voice_project/build_qnn_models.py:18).
- Verify ARM64 runtime libs:
  `ls C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\QnnHtp*.dll`.
- Verify the v81 Skel ships: look for `libQnnHtpV81Skel.so` or
  equivalent under the QAIRT hexagon-v81 lib dir.

### 5.2 Confirm Hexagon NPU is visible at OS level

- `Get-PnpDevice | Where-Object { $_.FriendlyName -match "Hexagon" -or $_.FriendlyName -match "NPU" }`
  should return at least one device in Started state.
- `powershell Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -match "Hexagon"`
  to see the signed driver state.

### 5.3 Confirm ORT-QNN can enumerate the NPU from our user account

Minimal Python smoke test (pattern from voice_project/ort_smoke_test.py,
referenced at voice_project/current_status.md:47):

- `pip install onnxruntime-qnn==1.24.4` in a clean venv.
- In Python: `import onnxruntime as ort; env = ort.OrtEnv(...)`,
  register the QNN EP library, call `GetEpDevices`, print the list.
- Expected: at least one device entry with NPU capability.

**If this returns no NPU devices under our user account but returns
one in an admin shell, we've hit a permission wall and need to
pursue it before writing anything else.** voice_project didn't
report this issue, so expect it to just work.

### 5.4 Confirm AI Hub token is alive and device is targetable

- `pip install qai-hub`
- `qai-hub configure --api_token <redacted>` (token from prior
  voice_project setup; if the env var is in place, this is a
  no-op).
- In Python: `import qai_hub; print(qai_hub.get_devices())`
  and confirm `"Snapdragon X2 Elite CRD"` is present.

Exit criterion for this section: all four checks pass with no error
output. Record the exact QAIRT build string, ORT version, and AI
Hub device list in `results/npu_env_snapshot.txt`.

## 6. Qwen3-0.6B-Q8_0 draft specifics -- autoregressive gaps vs Whisper

voice_project solved a **one-shot encoder + fixed-iteration decoder**
problem. Our draft is a **true autoregressive causal LM with
per-token KV-cache extension**. Gaps to be aware of:

### 6.1 KV cache handling

Whisper decoder in voice_project is essentially autoregressive
(decodes tokens one at a time), and its ONNX export does expose
`k_cache_self`, `v_cache_self`, `k_cache_cross`, `v_cache_cross`
as inputs (voice_project/aihub_compile.py:39-43). So the *pattern*
is known to work.

Differences for Qwen3-0.6B:
- **No cross-attention** (not encoder-decoder). Only self-attn KV
  cache (q_dim, kv_dim per layer).
- **KV cache shape grows per token** (Whisper's was fixed-size
  `N_TEXT_CTX = 200`). Options: pre-size to max context (e.g. 512
  or 2048) and mask, or re-export per block size. Fixed-max-size
  is the QAIRT-friendly path -- Whisper decoder does exactly that.
- **GQA (grouped-query attention):** Qwen3-0.6B has KV groups. The
  ONNX export needs to preserve the GQA shape so QAIRT lowers it
  to the right matmul sequence.

### 6.2 Quantization

- We have Qwen3-0.6B-Q8_0 as GGUF. QAIRT doesn't ingest GGUF.
  **We will need to export from the original HuggingFace
  BF16/FP16 checkpoint to ONNX, then quantize on AI Hub.**
  Target: W4A16 (per trident/npu_path_back.md:37 -- Qualcomm's
  reference Llama 3 recipe) or W8A16 if W4A16 accuracy slips.
  GGUF-side Q8_0 is irrelevant to the NPU path; we keep it for
  the CPU comparison baseline only.
- Calibration data: ~50-100 representative prompt sequences per
  trident/npu_path_back.md:68. Use specula's existing
  humaneval + structured_json fixtures.

### 6.3 Int64 fixups

Only applies to Path B (local converter). Cloud compile on Path A
runs native x64 so doesn't need `fix_onnx_int64.py`. Keep the util
handy as a Path B fallback.

### 6.4 Dynamic shapes

QAIRT wants static shapes at compile time. Whisper bakes in
`N_AUDIO_CTX=1500`, `N_TEXT_CTX=200` (aihub_compile.py:29-30).
For our draft, fix `(batch=1, seq_len=1, kv_cache_len=CONTEXT_MAX)`
at compile time and handle prompt-prefill either on CPU (easier)
or via a second compiled shape `(1, PREFILL_LEN, 0)` (faster,
more fiddly).

**Simplest first cut:** prefill on CPU (we already have this
running fast), NPU takes over for single-token draft steps only.
This also sidesteps the spec-decode `-k 3` batching question --
the NPU draft only emits single tokens anyway, and the CPU target
verifies batches of 3+1.

### 6.5 Tokenizer handling

Keep tokenization on CPU. Qwen3 uses BPE via the stock llama.cpp
tokenizer. No NPU involvement needed; the draft output is a token
ID that goes straight back to llama.cpp's `--draft` pipeline.

### 6.6 llama.cpp `--draft` integration

Open question: llama.cpp's `--draft-model-url` or
`-md <gguf>` flags assume a GGUF draft that runs through the
llama.cpp backend dispatcher. We don't have a QNN backend in
llama.cpp. Two options:

(a) **External drafter process** -- run Qwen3-0.6B via an
    ORT-QNN Python/C# sidecar, pipe drafted tokens back to
    llama.cpp over a socket/stdin, have llama.cpp use its
    existing draft-token-acceptance pipeline. Requires a thin
    bridge but doesn't touch llama.cpp internals.

(b) **llama.cpp QNN backend (new ggml backend)** -- implement
    `ggml-qnn` similar to `ggml-opencl`. This is a much bigger
    project and not what Phase 5 is scoped for.

**Choose (a).** Scope is "NPU drafting working", not "llama.cpp
has a QNN backend". Bridge is acceptable for this demo; trident
uses the same architectural pattern (main Zig runtime calls out
to QNN for GEMMs via its own dispatch).

Flag: the Phase 2 numbers we want to beat use `llama-speculative`'s
built-in two-model spec decoder. For apples-to-apples, the bridge
above needs to implement the same accept/reject logic. Simplest
is to reuse llama.cpp's draft pipeline by running the verifier via
`llama-server` HTTP with `/completion` + our sidecar feeding
pre-drafted tokens as `--draft` input. Needs verification that the
HTTP API accepts externally-sourced draft tokens. (blocking for
step 8 below; non-blocking for earlier steps)

## 7. Concrete bring-up plan (Session 6)

Numbered; each step has an exit criterion. Aim: land a first NPU-drafted
token by step 7, end-to-end spec decode by step 10.

1. **Environment snapshot.** Run section 5 checks. Record versions
   to `results/npu_env_snapshot.txt`.
   Exit: ORT-QNN lists at least one NPU device under our user
   account; AI Hub returns "Snapdragon X2 Elite CRD" in
   `get_devices()`; QAIRT 2.45.40 present.

2. **ORT-QNN sidecar skeleton.** Create `scripts/npu_draft_sidecar.py`:
   load ORT env, register QNN EP, accept a dummy ONNX model,
   confirm session creation with NPU EP (not CPU fallback). Model
   can be a 3-op matmul graph.
   Exit: session.run() executes on NPU, confirmed by QNN logger
   output showing HTP backend. Round-trip latency < 5 ms.

3. **Qwen3-0.6B ONNX export.** Using `transformers` +
   `optimum.onnxruntime`, export `Qwen/Qwen3-0.6B` (BF16 HF
   checkpoint) with KV cache exposed as I/O tensors. Target shape:
   `(1, 1)` token input, `(num_layers, 1, kv_heads, CONTEXT_MAX,
   head_dim)` KV cache. CONTEXT_MAX = 2048 for first cut.
   Exit: `optimum` produces `model.onnx` + `model.onnx_data`; ORT-CPU
   inference with the export matches HF PyTorch logits within 1e-3
   for a 32-token prompt.

4. **AI Hub compile.** Submit via `qai_hub.submit_compile_job`
   mirroring voice_project/aihub_compile.py:66-74. Device
   "Snapdragon X2 Elite CRD", options `--target_runtime
   qnn_context_binary --compute_unit npu`.
   Exit: job returns `SUCCESS`, downloads
   `qwen3_0_6b_draft_htp.bin` (~600-800 MB).

5. **Load + shape validate on device.** Load the `.bin` via
   ORT-QNN in the sidecar. Inspect IO tensor shapes against the
   ONNX export; confirm KV-cache binding slots work. Run one
   forward pass with zero input; confirm no DMA errors (0x1774)
   and no signing errors (0x80000406).
   Exit: one forward pass completes without error. Latency
   recorded; target < 50 ms per token for 0.6B at CONTEXT_MAX=2048.

6. **Correctness vs CPU.** Run same prompt through NPU path and
   through `llama-cli` CPU path with greedy decoding + seed=1,
   Qwen3-0.6B-Q8_0.gguf. Compare first 64 tokens. Expect small
   numeric drift from W4A16/W8A16 quantisation but coherent text.
   Exit: >= 50% exact-match on first 16 tokens; no token
   catastrophes. If quantisation kills coherence, drop W4A16 ->
   W8A16 and rerun step 4.

7. **First NPU-drafted token into llama.cpp verify.** Simplest
   path: run `llama-server` with the target Qwen3-8B-Q4_K_M, hit
   `/completion` with a prompt and `n_predict=1`. Separately,
   sidecar produces one drafted token from same prompt. Compute
   accept/reject manually by comparing to target's greedy output.
   Exit: one drafted token returned, accept/reject decision made.
   (No end-to-end speedup yet; this is a plumbing checkpoint.)
   **[DONE, session 11]** `scripts/npu_spec_step7_plumbing.py`:
   at the step-6 511-token anchor, NPU Path A draft, CPU-ORT 0.6B
   reference, and Qwen3-8B llama-server target all greedy-pick
   token 264. Accept=True. Tokenizer sanity probe (HF Qwen3 BPE
   vs llama.cpp GGUF vocab) agreed 11/11 ids. Key design find:
   `/completion` accepts raw id arrays (`server-common.cpp:767`
   `json_is_array_of_numbers`), so the comparison is purely at
   the id level — no detok->retok in the middle. Log:
   `results/phase5_step7_plumbing.log`.

8. **llama.cpp spec-decode pipeline integration.** Investigate
   whether `llama-speculative` can accept an external drafter, or
   whether we run the sidecar as the outer loop and call
   `llama-server` per-verify. Likely the latter is easier.
   Port specula/scripts/sweep_speculative.ps1 to optionally drive
   this hybrid.
   Exit: scripted run produces a complete sequence with accept-
   rate logged per round. Numbers don't need to beat anything yet.
   **[DONE, session 11]** Sidecar-as-driver. Two scripts:
   `scripts/npu_short_prompt_probe.py` (gate: Path B-mask masking
   at prompt_len=16, cos=0.999960, 3/3 multi-step match) and
   `scripts/npu_spec_outer_loop.py` (end-to-end: humaneval p0,
   k=3, n_predict=64, **65.2% accept, 6.23 t/s**, coherent
   fibonacci output). NPU per-step latency (~63 ms, 5 calls/round)
   is the bottleneck: 6.9s of 10.4s wall is NPU-bound.
   Phase 2 CPU-spec's 40.2 t/s ceiling looks out of reach without
   NPU/target overlap. Log:
   `results/phase5_step8_outer_loop.log`.

9. **First NPU-spec number.** Run the sidecar-driven spec decode
   on specula's 10-prompt humaneval subset, k=3, greedy, same
   rig as Phase 2. Record mean decode t/s and accept rate.
   Exit: CSV row written to `results/spec-npu-...csv` with
   comparable schema to the Phase 2 outputs. First headline
   number known.
   **[DONE, session 11]** Full 40-cell sweep
   (`scripts/sweep_npu_spec.py`): k ∈ {2, 3, 4, 8} × 10 humaneval
   prompts × n_predict=256, 25.9 min wall. k=2 wins:
   **7.98 t/s mean, 81.0% accept** (best cell p8 at 8.44 t/s,
   92.2% accept). Phase 2's k=3 optimum shifts to k=2 on NPU
   because NPU per-step is 7× CPU-draft's cost, so fewer draft
   calls per round wins.

10. **Sweep + writeup.** If step 9 is within 20% of Phase 2's
    40.2 t/s (i.e. 32-48 t/s), sweep k ∈ {2, 3, 4, 8} and include
    JSON workload. Write results up in `docs/npu_results.md`
    (new). If step 9 is catastrophically lower (<10 t/s), diagnose
    overhead source: per-token bind latency vs NPU execute vs
    bridge IPC. trident's experience suggests bind-and-execute is
    fast; bridge IPC is the most likely culprit.
    Exit: Phase 5 closes with either a win, tie, or documented
    loss with root cause. Qwen3 graduates.
    **[DONE, session 11]** `docs/npu_results.md` landed:
    documented loss with root cause (NPU per-step latency ~63 ms,
    4 calls/round, sequential with target verify; 5.86 s NPU of
    9.32 s wall on humaneval p0). Draft quality is fine (cos
    0.999960, accept 81.0% matches CPU-spec's 82.3% at k=2). JSON
    workload NOT added — deferred to post-W4A16 rerun since the
    numerical story is already clear. w4a16 identified as biggest
    Phase 5.5 lever (2-3× NPU per-step speedup per Qualcomm's
    Qwen3-4B reference pattern). Phase 5 CLOSED; Qwen3 graduates
    after the stashed close-out items land (see
    `current_status.md`).

## 8. Open questions

- **(blocking step 3)** Does `optimum.onnxruntime` produce a
  Qwen3-0.6B ONNX export with KV cache that lowers cleanly on
  QAIRT 2.45? If not, we need a custom tracing wrapper.
- ~~**(blocking step 8)** Can `llama-server` accept externally-drafted
  tokens via its HTTP API, or do we need our own spec-decode
  outer loop that calls llama.cpp as a verifier? llama.cpp HEAD
  `e365e658f…` behaviour unverified.~~ **Answered session 11:**
  llama-server does NOT expose a "verify these draft ids" endpoint.
  `/completion` takes a prompt and emits its own greedy
  continuation. Step 8 outer loop will run the sidecar as driver
  and call `/completion` per round with `n_predict=k+1`, comparing
  returned ids one-for-one to our drafted ids for the standard
  greedy-accept rule. The `/completion` endpoint does accept
  raw id arrays as `prompt`, which removes tokenizer round-trip
  from the loop. Open sub-question: `cache_prompt=true` caches
  the committed prefix KV across rounds so each verify pays only
  the new-token cost — need to verify this works with id-array
  prompts (string prompts definitely work).
- **(blocking step 5)** Whole-model Qwen3-0.6B context binary total
  spill/fill -- does it fit on X2E HTP DMA in one context? trident's
  experience says yes for a single unified context, but 0.6B is
  our first empirical data point. May need to split into 2-3
  sub-graphs (prefill / token-step / final-projection) if one
  context is too big. Ballpark budget from voice_project's Whisper
  encoder context binary: 272 MB context blob, 50 MB spill/fill,
  112 MB IO tensors, 264 MB const (encoder_info_v81.json:22-124).
  Qwen3-0.6B weights ~600 MB at Q8_0, so the context binary may
  be larger.
- **(non-blocking)** W4A16 vs W8A16 for Qwen3-0.6B quality. Start
  W4A16 per trident/npu_path_back.md; fallback W8A16 if
  coherence suffers.
- **(non-blocking)** Does ORT-QNN's EP API let us share KV-cache
  buffers across session.run() calls without copy? If not, we're
  paying a per-token IO copy that caps throughput. Can revisit
  via `cl_qcom_ion_host_ptr`-style handoff only if we drop to
  raw QNN (Path E).
- **(non-blocking)** Does pip `onnxruntime-qnn==1.24.4` bundle QAIRT
  2.45 exactly? If it bundles a different version the AI Hub
  compile may err 1009 at load (trident/npu_path_back.md:66). If
  so, pin AI Hub SDK version to match whatever ORT bundles.
- **(non-blocking)** Snapdragon X2 Elite CRD vs X2 Elite Extreme --
  voice_project targets "Snapdragon X2 Elite CRD" device string on
  AI Hub and the binary works on our X2 Elite Extreme machine. AI
  Hub device list may have a more precise target string now
  (2026-04 vs voice_project's earlier work). Check
  `qai_hub.get_devices()` output in step 1.
- **(non-blocking)** trident/npu_path_back.md:67 flags Genie as
  Qualcomm's "blessed runtime" for LLMs. We're explicitly NOT using
  Genie because it assumes you shipped with Qualcomm's model zoo.
  If performance is marginal, Genie becomes a revisit target for
  Phase 5.5 or Phase 6.
- **(non-blocking)** For eventual Qwen3.5 graduation: the hybrid
  variant uses `gated_delta_net` and `ssm_conv` which are **not**
  in QAIRT's shipped op library. NPU-on-hybrid lands after custom
  op authoring (Path C) -- explicitly out of scope here. Pure-
  attention Qwen3.5 variants stay NPU-friendly.

## 9. Prior art review (2025 literature)

Five papers inform Phase 5 design. The raw notes + citations live in
`npu_thoughts_previous_examples.md` (repo root); this section is the
plan-impact summary.

### Mirror-SD (arXiv, Oct 2025)

NPU-drafter + GPU-verifier split on M2 Ultra datacenter hardware
(8x M2 Ultra GPUs for target, 8x NPUs for draft). Token channel
between draft and verify is minimal: top-k token IDs plus bf16
log-probs. That's the whole IPC surface.

**Plan impact:** step 8 bridge adopts this protocol. For greedy
spec (our baseline) we pass only token IDs; log-probs reserved for
if we ever go non-greedy.

### sd.npu (arXiv, Dec 2025)

First end-to-end NPU-offloaded spec decode on phones. 1.06-3.81x
energy reduction on Qwen2.5-0.5B / Qwen2.5-1.5B / LLaMA3.2-3B across
Redmi K60 Pro, K70 Pro, OnePlus 13. **Key finding: >75% of drafts
are <8 tokens but mobile NPUs need 32+ tokens to approach peak
utilization.** Fix: pad/recycle rejected tokens to keep the NPU
busy.

**Plan impact (big one for us):** our k=3 draft is deep in the
underutilization regime. Do NOT tune for best t/s at step 9; just
report the number. If step 10 is bandwidth-bound rather than
latency-bound, apply sd.npu padding as the first optimization lever
(see step-10 addendum below).

### HeteroLLM / HeteroInfer (SOSP 2025)

Tensor partition across CPU+GPU+NPU on mobile SoCs during both
prefill and decode. **Finding:** parallel execution leverages
aggregate SoC memory bandwidth exceeding what any single processor
can saturate alone.

**Plan impact:** strengthens the "working path has demo value"
framing. Even if NPU-draft alone ties CPU-spec, the architectural
proof opens a follow-on lane -- async draft || verify overlap --
that wins on aggregate bandwidth without needing NPU peak
throughput. Noted as post-step-10 optimization lever.

### Dovetail (EMNLP 2025)

CPU/GPU heterogeneous spec decode: draft on GPU, verify on CPU,
~5.86 tok/s for LLaMA2-Chat-7B with 3 GB VRAM. Not NPU, but same
handoff pattern we're building.

**Plan impact:** reference for the draft-verify handoff structure
if Mirror-SD's minimal token-channel proves too lean for our
accept/reject bookkeeping.

### OpenPangu (cited in sd.npu literature)

Solved the static-graph vs conditional-branch tension in NPU spec
decode via static lookup tables + zero-copy retrieval.

**Plan impact:** validates our fixed CONTEXT_MAX=2048 padding
approach (section 6.4). The tension is: spec decode's accept count
varies per round, NPUs want static graphs, so we pad. This is the
known-good shape; we're not alone on it.

### Structural tensions the literature confirms

1. **GEMV underutilization at batch=1.** NPU matrix units starve on
   autoregressive decode. Our k=3 spec decode doesn't fix this for
   the draft model -- draft is sequential, only verify is batched.
   Architectural ceiling, not a specula-specific bug.
2. **Static graph constraint.** NPUs prefer fixed shapes; spec
   decode has variable accept counts. Handled by padding to
   CONTEXT_MAX and masking (our plan) or static lookup tables
   (OpenPangu). We use the simpler option.
3. **Hexagon cDSP 32-bit VA space.** Caps single-context models at
   ~4B params per session. 0.6B draft is comfortably under; 1.7B
   also fits; **8B target will never run on NPU** regardless of
   our other work. This is why the target stays on CPU for
   Phase 5 -- architectural ceiling, not a preference.

### Landscape confirmation

No published work runs NPU spec decode on laptop-class Snapdragon X.
All existing NPU spec-decode work is either mobile Snapdragon
8 Gen 2/3 (sd.npu) or datacenter M2 Ultra (Mirror-SD). Windows on
ARM + laptop Hexagon + llama.cpp target is the specula-shaped hole
in the literature. This is why even a working NPU-draft path that
TIES CPU-spec is a shipping milestone -- not because speculative
decoding is novel, but because demonstrating it works on this
hardware surface is.

### Step-10 addendum: optimization ladder if NPU is starved

If step 9 lands catastrophically below Phase 2's 40.2 t/s, apply
in order:

1. **sd.npu pad/recycle.** Draft k+pad_tokens, execute the NPU at
   higher batch size, discard pad after verify. First lever
   because it targets the GEMV underutilization directly.
2. **Async draft || verify overlap.** NPU drafts token n+1 while
   CPU verifies tokens 1..n. Needs proper async dispatch both
   sides; converts serial latency into parallel latency. HeteroLLM
   aggregate-bandwidth play adapted to spec decode.
3. **Raw QNN with persistent context + rpcmem zero-copy** (Path E).
   If bridge IPC is the bottleneck, drop ORT-QNN's session-boundary
   copy and use shared ION buffers. Loses the signed-driver
   convenience of ORT-QNN; only pursue if instrumented proof says
   IPC is the actual cost.

All three levers are research work beyond the Phase 5 definition.
Phase 5 closes on "working path" regardless of whether any are
applied.

## 10. Sources

Inline citations throughout. Primary docs:

- `voice_project/current_status.md` -- end-to-end NPU working
  narrative, driver signing walls, ORT-QNN rescue.
- `voice_project/aihub_compile.py` -- AI Hub submit + poll pattern.
- `voice_project/encoder_info_v81.json` -- hardware targeting
  evidence (dspArch=81, socModel=88).
- `voice_project/run_qnn_converter.py`, `build_qnn_models.py`,
  `convert_whisper.py`, `fix_onnx_int64.py` -- local QAIRT
  converter pipeline reference.
- `trident/postmortem.md` -- ARM64 binary verification lesson
  (orthogonal but cited for completeness).
- `trident/npu_current_status.md` -- shelved NPU GEMM, DMA
  exhaustion, per-context overhead breakdown.
- `trident/npu_path_back.md` -- AI Hub revival plan, quantisation
  recipes, failure modes on re-entry.
- `trident/npu_optimizations_thoughts.md` -- five-way options
  matrix for the DMA problem (A-E).
- `specula/current_status.md` -- Phase 2 baseline (40.2 t/s @ k=3)
  and Phase 5 framing (session 5, NPU-first ordering).
- `specula/docs/reference-projects.md` -- cross-project pointer
  map already in specula.
- `specula/npu_thoughts_previous_examples.md` -- 2025 literature
  review (Mirror-SD, sd.npu, HeteroLLM, Dovetail, OpenPangu).
  Integrated into section 9 above.
