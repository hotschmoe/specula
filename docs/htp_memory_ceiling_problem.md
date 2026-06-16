# PROBLEM: breaking the ~10 GB Hexagon HTP memory ceiling for large LLMs

Self-contained brief for brainstorming. Goal: run **dense LLMs whose total
weights exceed ~10 GB** (Qwen3-14B now, **Qwen3.6-27B dense** as the real
target) on the Snapdragon X2 Elite Extreme **Hexagon NPU (HTP)**. We have hit
a hard runtime memory ceiling and need creative ways past it. Low-level work
(C/C++/Rust/Zig, raw QNN SDK, custom FastRPC) is on the table.

## Hardware + software stack

- **SoC:** Snapdragon X2 Elite Extreme laptop. **48 GB LPDDR5X unified @
  228 GB/s** (shared CPU / Adreno X2 GPU / Hexagon NPU). DSP arch `v81`,
  `soc_model 88`. Windows 11 ARM64 native (no WSL for NPU).
- **Runtime:** ONNX Runtime **1.24.4** with the **QNNExecutionProvider**,
  pointed at the system **QAIRT 2.45.40** `QnnHtp.dll`
  (`C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\`).
  We load **EPContext**-wrapped pre-compiled QNN context binaries (one
  `.bin` per model part), `embed_mode=0`, `htp_performance_mode=burst`,
  `enable_htp_fp16_precision=1`, `htp_arch=81`, `soc_model=88`.
- **Genie is NOT usable:** the Genie DSP transport is broken on this box
  (`qnn_open` fails `0x80000406`); ORT-QNN works on the same HTP. So we run
  our own ORT-QNN engine, not Genie.
- Model is split into parts (each part = embed, or N decoder layers, or
  lm_head), each compiled to its own HTP context binary via
  `qairt-converter → qairt-quantizer → qnn-context-binary-generator`.

## The three ceilings we have measured (Qwen3-14B, ctx=512)

The 14B is split into **10 parts**: embed (fp16, 1.56 GB) + 8×(5-layer
decoder, w8a16 **int8**, 1.66 GB each) + lm_head (fp16, 1.56 GB). Total
**~16.4 GB** of context binaries. Each part is its own QNN HTP context.

1. **Per-context SIZE ceiling ≈ 2 GB.** A single context binary larger than
   ~2 GB fails to **load** with **QNN error 1002**
   (`Failed to create context from binary`). Evidence: the *original*
   no-calibration build stored fp16 weights → 3.3 GB decoder contexts →
   every decoder part failed 1002 *individually*. Re-quantizing with
   calibration → genuine int8 → 1.66 GB → loads fine. The 1.56 GB
   embed/head contexts load. (So the load-size limit is between 1.66 and
   3.30 GB; Qualcomm's own shipped 7B parts are ≤1.09 GB.)

2. **Total RESIDENT-on-LOAD ceiling ≈ 10 GB / ~6 contexts.** Loading the 10
   parts as separate ORT-QNN sessions: **6 load OK (~9.9 GB resident), the
   7th fails QNN 1002.** Packing parts as multiple EPContext nodes in fewer
   ORT sessions does **not** help (3 paired sessions = 6 contexts load, the
   4th session fails) — so it is **not** an ORT-session-count limit; it is
   **total HTP context memory**. (The smaller 4B w4a16 bundle reached 7
   contexts because each is smaller — the wall is GB, not count.)

3. **EXECUTION ceiling is LOWER than the load ceiling.** Even when ~8 GB
   (5 contexts) load successfully, the **first inference** call fails:
   ```
   <E> DspTransport call failed, error 0x00000007
   <E> CONN Reset
   <E> Failed to destroy perf context: 1007
   ```
   i.e. the **FastRPC connection to the DSP resets** during execution
   (surfaces as a Windows exception `0xc00000aa`). Running with only 2–3
   small contexts resident executes fine. So weights + per-inference DSP
   scratch/working memory together blow a limit that pure loading does not.

**Net:** ORT-QNN can hold/run only ~**8–10 GB** of HTP context memory at
once on this box. Our 14B w8a16 (~16 GB) does not fit; a **27B dense even at
w4a16 (~14 GB weights)** will not fit either. We cannot just shrink our way
under it for the 27B — **we need to defeat or stream around the ceiling.**

## What we've ruled out

- **ORT session grouping / combined EPContext wrapper:** collapses ORT
  sessions but NOT the number of HTP contexts or total resident bytes →
  same wall. (It *does* solve a separate ~7-ORT-session limit, just not
  this memory one.)
- **Per-token streaming swap (load 5 parts, run, unload, load other 5):**
  the 5-part group (~8 GB) crashes at execution (ceiling #3). Smaller
  groups (≤2–3 GB resident) run, but mean many context load/unload cycles
  per token, which also churns the fragile DSP transport. Unproven at scale
  and likely very slow.
- **Genie's `use-mmap: true`:** Genie memory-maps weights so a >RAM bundle
  streams from DDR, but ORT-QNN's EPContext loader does **not** honor it,
  and Genie itself won't run on this box (broken DSP transport).

## Open question for brainstorming

**How do we execute a model whose total weights exceed ~10 GB on this
Hexagon HTP?** Concretely, two framings:

A. **Raise the ceiling.** *Why* is it ~10 GB? Candidates to investigate:
   - FastRPC / cDSP **user-PD memory limit** or default ION/heap pool size
     (is there an env var / property / API to grow it? e.g. RPC heap,
     `rpcmem`, `fastrpc` mmap budget, unsigned vs signed PD limits).
   - Hexagon DSP **virtual address space** limit (historically the cDSP
     user process had limited VA — is there a 4 GB-ish or other VA cap that
     the SMMU/`io-coherent` mode or a 64-bit addressing flag lifts?).
   - QNN HTP **context arena / weight-sharing** options: `spill-fill-bufsize`
     (currently 0), `mmap-budget` (0), `vtcm` config, `shared_buffer`
     memtype, `weight_sharing_enabled`, or a QNN graph/context config that
     enables on-demand weight paging.
   - ORT-QNN provider options we haven't set that raise memory limits, or a
     newer ORT-QNN / QAIRT that fixes it.

B. **Stream around it efficiently.** If the ceiling is immovable, what is
   the fastest correct way to run a >10 GB model with a ≤~8 GB resident
   working set? Candidates:
   - **Weight streaming / paging:** keep weights in CPU/DDR (mmap), DMA each
     part's weights into the HTP just-in-time, run, evict. Needs a custom
     runtime — likely the **raw QNN SDK in C++** (bypass ORT-QNN, which
     hides context memory control), or direct **FastRPC** programming, to
     control context create/free + shared buffers + RPC memory.
   - **Heterogeneous split:** run some layers on the **Adreno X2 GPU**
     (OpenCL/llama.cpp already gets 50 t/s on 4B there) or CPU, keeping HTP
     resident set small. Unified 48 GB memory @ 228 GB/s helps.
   - **Smarter swap:** double-buffer / prefetch the next part's context
     while the current runs; overlap DMA with compute; persistent PD to
     avoid transport churn.
   - **mmap'd context binaries:** can ORT-QNN or raw QNN load a context with
     the weights left memory-mapped (resident in DDR, not duplicated into a
     DSP-private arena)? The unified-memory architecture *should* allow the
     DSP to read weights in place via the SMMU.

## Key facts / constraints to respect

- Unified 48 GB LPDDR5X — capacity is NOT the constraint; the HTP-resident
  budget is. The DSP can in principle address DDR via the SMMU.
- We can rebuild bundles freely on a Threadripper build box (QAIRT 2.45,
  full pipeline). Re-split (layers/part), re-quantize (w8a16 / w4a16),
  re-compile context bins, change HTP config — all available.
- Engine is Python + ORT-QNN today (`npu_engine/engine_14b_q.py`,
  `engine_14b_swap.py`), but **C/C++/Rust/Zig + raw QNN SDK / FastRPC is
  explicitly allowed** if it unlocks memory control.
- Numerics already validated: calibrated int8 parts load + run correctly in
  isolation; a CPU fp32 reference exists for cos-sim.

## RESEARCH FINDINGS (2026-06-16) — the ceiling is real + documented, and llama.cpp already solves it

Web research (sources at bottom) confirms our measurements and points to a
clear path.

### Our ceiling is real and matches public limits
- **llama.cpp's Hexagon backend documents the same wall:** *"each Hexagon
  session has ~2 GB of dedicated memory"* and *"for models larger than
  ~3.5 GB multiple devices are allocated using multi-GPU layer-splitting."*
  This independently validates our per-context ~2 GB and total ~10 GB
  findings — it is a **real HTP/session memory limit, not a bug in our code.**
- ORT-QNN issue threads: quantized models can take minutes in
  `FinalizeGraphs` and consume >4 GB on HTP; QNN added features
  specifically to cut this.

### Two ORT-QNN knobs we are NOT using (stopgap, keeps our engine)
1. **`enable_htp_spill_fill_buffer=1`** (QNN ≥2.28; we have 2.45). For models
   with **multiple context binaries + weight sharing**, all contexts **share
   one spill/scratch buffer** (sized to the max) instead of each reserving
   their own. Targets exactly "large models with large context binaries."
   **Requires:** regenerate context bins offline (Linux — we have the box)
   with `enable_htp_spill_fill_buffer|1`, AND **merge the contexts into ONE
   ONNX** (multiple EPContext nodes); runtime loads the highest-`max_size`
   context first. Our bins were built with `spill-fill-bufsize: 0` (OFF) —
   this likely caused execution ceiling #3. **First thing to test.**
2. **`enable_htp_shared_memory_allocator=1`** — uses `rpcmem`/`libcdsprpc`
   for CPU↔HTP shared buffers; reduces fragmentation for multi-context.
   Pair with #1.
   (Also available: `vtcm_mb`, `htp_graph_finalization_optimization_mode 0-3`,
   `qnn_context_priority`.)

### RECOMMENDED PATH — adopt llama.cpp's Hexagon backend (purpose-built)
The official **`ggml-org/llama.cpp` Hexagon backend** (`GGML_HEXAGON=ON`)
already does everything we are hand-rolling, and better:
- Builds **`libggml-htp-v81.so` — our exact DSP arch (v81)**; Windows ARM64
  supported (VS + LLVM/Clang + **Hexagon SDK 6.6+** + OpenCL SDK 2.3+).
- Each HTP session = a GGML **device** (`HTP0,HTP1,…`); **`GGML_HEXAGON_NDEV`**
  allocates several and **splits layers across them** (multi-GPU style),
  defeating the per-session ~2 GB limit.
- **`-ngl` partitions layers between NPU and CPU/GPU** over the **unified
  48 GB memory (no copy)** — so a **27B** can put ~10 GB of layers on the HTP
  and the **rest on the Adreno GPU / CPU**. This is the realistic way past a
  hard total-HTP limit.
- Native **GGUF + Q4_0** (already our production default + we have the GGUF
  files), KV cache, sampler, graph reuse. Measured ~51 t/s TG on a 1B on one
  HTP session.
- Open source in C/C++ — we can **extend it** (the user OK'd C/C++/Rust/Zig):
  e.g. add true weight streaming / mmap if the per-session weights can stay
  resident in DDR and be read in place via the SMMU.

**Implication for the 27B goal:** stop fighting ORT-QNN's full-resident
contexts. Build llama.cpp with `GGML_HEXAGON` on the X2E, run our Q4_0 GGUFs
with `-ngl` tuned so the HTP holds what fits (~5 sessions) and the Adreno/CPU
take the rest. Then investigate raising per-session memory / true DDR weight
streaming inside the ggml-hexagon backend.

### Open sub-questions to brainstorm/verify
- Exact **per-session memory limit on v81 X2 Elite** (is it ~2 GB like
  Android, or higher on the 48 GB laptop?) and **max concurrent sessions**.
- Does ggml-hexagon keep weights **resident (repacked)** or **mmap-stream**
  them? (REPACK buffers ≈ weight size suggests resident — so total is still
  bounded; hybrid `-ngl` is the lever.)
- Can the **HTP read Q4_0 weights in place from DDR** (unified memory +
  SMMU) without a per-session resident copy? If yes → no ceiling. This is
  the highest-value thing to prove (possibly a ggml-hexagon patch).
- Is the FastRPC/cDSP user-PD limit configurable (signed vs unsigned PD,
  `rpcmem` heap, `fastrpc` mmap budget)?

## Concrete next step — build llama.cpp Hexagon on the X2E (Windows ARM64)

Box status (checked 2026-06-16): **llama.cpp already cloned**
(`~/Documents/GitHub/llama.cpp`, ARM64 build at `~/llama-cpp-arm64/`). NPU
driver present (ORT-QNN works). **Missing: Hexagon SDK + OpenCL SDK.**

Recipe (from llama.cpp `docs/backend/snapdragon/windows.md`):
1. **Download SDKs** (snapdragon-toolchain GitHub releases, arm64 WoS):
   - Adreno OpenCL SDK v2.3.2 → `C:\Qualcomm\OpenCL_SDK\2.3.2`
   - Hexagon SDK v6.6.0.0 → `C:\Qualcomm\Hexagon_SDK\6.6.0.0`
2. **DSP code-signing** (the HTP skel must be signed — FastRPC signed PD):
   - ⚠️ `bcdedit /set TESTSIGNING ON` (**system change + REBOOT** — user must
     approve), then self-signed cert:
     `makecert -r -pe -ss PrivateCertStore -n CN=GGML.HTP.v1 -eku
     1.3.6.1.5.5.7.3.3 -sv ggml-htp-v1.pvk ggml-htp-v1.cer` →
     `pvk2pfx` → import .pfx into Trusted Root + Trusted Publishers (`certlm`).
3. **Build:**
   ```
   $env:OPENCL_SDK_ROOT="C:\Qualcomm\OpenCL_SDK\2.3.2"
   $env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0"
   $env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
   $env:HEXAGON_HTP_CERT="...\ggml-htp-v1.pfx"
   cmake --preset arm64-windows-snapdragon-release -B build-wos
   cmake --build build-wos
   ```
4. **Run** (our Q4_0 GGUFs already on disk): set `--device HTP0` (+ multi via
   `GGML_HEXAGON_NDEV=N`, `--device HTP0,HTP1,…`), `-ngl` to tune how many
   layers go to the HTP vs Adreno/CPU. Start with the 4B Q4_0 we already
   benchmark, then 14B, then 27B with hybrid `-ngl`.

The test-signing + reboot + cert import are **system changes requiring user
action**; the SDK downloads + build can be automated.

## What a useful answer looks like

A concrete, testable hypothesis with the exact knob/API/approach and how to
verify it on this box — e.g. "set FastRPC property X / QNN config Y to raise
the arena to N GB, verify by loading 8 contexts and running inference", or
"load context with QNN flag Z to keep weights mmap'd, measure resident HTP
memory", or "minimal C++ QNN harness that creates+frees contexts per part
with a persistent PD, benchmark swap latency". Cite Qualcomm docs / QNN SDK
headers / forum threads where possible.
