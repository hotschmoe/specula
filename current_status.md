# specula -- current status

Last updated: 2026-04-22 (session 19 — **Phase 5.5 Lever C closes
NEGATIVE as a product: w8a16-local AC sweep mean 12.83 t/s k=2 vs
Lever B's 18.12 t/s (−29%), 71.65% vs 81.91% accept.** Correctness
pipeline is fully delivered (every stage works, local QAIRT compile
bypasses AI Hub's preserve-list bug entirely) but PTQ noise on a
0.6B draft costs ~10 pp of accept rate, and per-step latency savings
don't compensate. Lever B's 18.12 t/s fp16 pathbmask AC remains
Phase 5.5's high-water mark.

Sessions 15-18 delivered the entire Lever C runtime stack via x86
local QAIRT (plan `docs/phase5_local_qairt_compile.md`, findings
`docs/phase5_local_qairt_compile_findings.md`): pathb rotary-hoisted
w4a16 binaries load cleanly on ORT-QNN 1.24.4, quant formula
validated (RMS 0.001%), IS_LOCAL_COMPILE dispatcher pattern-matches
any `*-local` variant, quant_specs threaded through sync + async
outer loops + sweep. Session 17 differential probe localised the
w4 PTQ collapse to layer-1+ V-projection weights (value tensor cos
0.957 at layer 0 → 0.130 at layer 1 → <0.2 all the way through
layer 27; keys degrade gracefully via rotary smoothing). Session 18
shotgun (7 variants, 6 distinct MD5s): **w8a16-local = first full
gate pass** (cos 0.963/0.979, argmax ✓, multi-step 100%);
**w4a16-local-pr soft pass** (cos 0.888, 100% greedy match, 620 MB
— 32% smaller binary); w4a16 mse/tfe/cle all confirmed negative
(activation-cal not the lever; CLE is a no-op on MatMul graphs);
w8a16-local-pr soft pass (per-row hurts at w8). Session 19 on AC:
steady-state latency (`scripts/probe_npu_steady_state_latency.py`,
5 warmup + 25 measured per variant) shows all quantized variants
cluster 21-24 ms/step (w4a16-local-pr fastest at 21.4 ms),
fp16-local at ~50 ms — session-18's uniform "50 ms on battery" was
cold-HTP + thermal noise. The 40-cell AC sweep on w8a16-local
(async-pipelined, n_predict=200, 14.2 min) broke out as k=2 mean
12.83 / k=3 11.79 / k=4 10.12 / k=8 6.74 t/s. Best cell p2 k=2 =
14.39 t/s / 78.2% accept. Worst p6 k=8 = 4.73 / 26.9%.

Decision: ship Lever B's 18.12 t/s AC baseline as Phase 5.5's
final number. Document Lever C as a structurally-working PTQ
pipeline that didn't clear the throughput bar at 0.6B draft size —
forward-compatible with Qwen3.5 graduation where the draft is
larger (per-step costs grow, per-step savings become worth more
vs fixed HTTP verify overhead) and where the same local-QAIRT
toolchain drops in unchanged. w4a16-local-pr AC sweep queued in
background (may close some of the gap via 11% faster per-step +
100% greedy match); results append below when complete. Commits
48301d9 (quant_specs plumbing + steady-state probe), 435abf1
(session-18 shotgun probes), {pending} for session-19 sweep
writeups.)

Last updated: 2026-04-22 (session 14 -- **Phase 5.5 Lever C — pathb
w4a16 compile SUCCEEDED but runtime blocked by an AI Hub compile
driver bug.** Rotary hoisting cleared the op-validation failure that
killed j563xme75 (job `jg93r1jqg` reached SUCCESS in 100 min; .bin
876 MB at `models/qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-a.bin`).
But the ORT-QNN forward pass fails with *"ORT Tensor data size does
not match QNN tensor data size"*: AI Hub's driver mis-formats the
`--preserve_io_datatype` list for qairt-quantizer, dropping exactly
the first entry (`past_key_values.0.key`) — the converter gets 116
names, the quantizer gets 115. Layer-0 key is therefore uint8-quantized
at the IO boundary while every other past_kv stays fp32, causing a
4× byte-count mismatch at runtime. Evidence: direct grep of both
invocations in `results/aihub-compile-jg93r1jqg-pathb-w4a16-a/jg93r1jqg.log`.
fp16 binaries are unaffected (no quantizer step invoked).
Session-14 X2E plumbing is still sound and reusable: 61-input schema
wired through `compile_qwen3_ai_hub.py`, `prep_onnx_for_ai_hub.py`,
`capture_calibration_samples.py` (+ `rope_tables(pos)` with
rope_theta=1e6), `npu_load_qwen3_bin.py`, probes + sweep (commit
1423f6c). Bundle A calibration captured (60 samples × 61 inputs,
3.27 GB). Lever B's 18.12 t/s AC baseline remains Phase 5.5's high-
water mark. Next session picks a workaround from
`docs/qwen3_perf_levers_investigation.md` §Lever C: prepend a
sacrificial preserve-guard input, do ORT-side uint8 quant of just
past_kv.0.key, or file a Qualcomm AI Hub bug ticket. See commit
{pending} and the same doc for the full workaround matrix.)

Last updated: 2026-04-22 (session 13 -- **x86 delivered Path B
(rotary hoisted).** `models/qwen3-0.6b-pathb/`: 61 inputs (was 59),
7,131 nodes, **zero `/model/rotary_emb/*` nodes**. CPU-equivalence
probe vs optimum source: cos = 1.000000 on both pos=0 zero-KV and
pos=5 synthetic-past_kv probes (numerically exact, not just within
tolerance). Transferred to `Z:\exposed\junk\phase5_step12_pathb\`
with MD5 verified end-to-end. New scripts: `rewrite_qwen3_pathb.py`
(pure protobuf rewrite) + `probe_pathb_equivalence.py`. X2E follow-up
unchanged from session 12: extend `compile_qwen3_ai_hub.py` for the
pathb 61-input schema, regenerate calibration, submit `--quant w4a16`.
See `status_x86.md` session 2 for handoff details and the canonical
runtime cos/sin formula.)

Last updated: 2026-04-22 (session 12 -- **Phase 5.5 Lever C handed
off to x86.** Levers A + B closed on battery + AC (k=2 async-pipelined,
ctx=256): AC baseline **18.12 t/s mean, 19.07 best, 81.91% accept**
(+127% over Phase 5 baseline 7.98 t/s). Lever C W4A16 compile
attempted twice, both failed at AI Hub — root cause diagnosed by
inspecting Qualcomm's shipping Qwen3-4B w4a16 bundle: our graph
computes rotary_emb inline; Qualcomm hoists it out. Fix is a new
x86-side export (Path B: rotary hoisted + additive mask) per
`docs/phase5_export_on_x86.md` §"Path B implementation contract
(2026-04-22 revision)". See also Phase 5.5 section below and
Lever C detail in `docs/qwen3_perf_levers_investigation.md`.)

Last updated: 2026-04-21 (session 11 -- **Phase 5 CLOSED.** Full sweep
landed: 40 cells (k ∈ {2,3,4,8} × 10 humaneval prompts, n_predict=256)
in 25.9 min. **k=2 wins with 7.98 t/s mean, 81.0% accept (best cell
8.44 t/s at p8).** Structural regression vs Phase 2 CPU-spec 40.2 t/s,
driven by NPU per-step latency; accept rate matches CPU-spec exactly.
Writeup in `docs/npu_results.md`. w4a16 quantisation identified as
biggest Phase 5.5 lever.)

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

## Phase 5 status (session 9, 2026-04-21)

**Step 4 CLOSED.** `models/qwen3_0_6b_draft_v81_ctx512.bin` (1438 MB)
is on disk. AI Hub job `jgzx6xlz5` compiled cleanly on the first
attempt off the x86-produced `nomask` ONNX: CREATED → OPTIMIZING_MODEL
→ **SUCCESS at t=400s**. This cleared the 440-465s CTX-BIN wall that
killed the prior 8 attempts — the x86-side `nomask` variant
(onnxsim + aggressive mask-subgraph removal, commit `adbbbd4`)
eliminated `attention_mask`, `Where`, and `IsNaN` entirely, so HTP
had no BOOL tensors to reject. See `qlcom_compile_status.md` for the
full retro.

Current local state:
- `models/qwen3-0.6b-nomask/` — x86 handoff input (3 GB onnx + data)
- `models/qwen3-0.6b-nomask-ai-hub/` — staged for upload (2.87 GB)
- `models/qwen3_0_6b_draft_v81_ctx512.bin` — compiled HTP context
  binary, signed via AI Hub's QAIRT stack. Target for step 5.

Steps 5-10 (per `docs/npu_scoping.md` §7) still ahead:

```
[DONE]    1. Environment snapshot                 (commit 7230210)
[DONE]    2. ORT-QNN sidecar skeleton             (commit 282e84a)
[DONE]    3. Qwen3-0.6B ONNX sourced + CPU-valid  (commit 106c756)
[DONE]    4. AI Hub compile -> Hexagon .bin       (session 9 ★)
          5. Load .bin via NPUSession, shape-check <-- next
          6. Correctness vs CPU, single greedy prompt
          7. Pipe first drafted token through llama.cpp verify
          8. External-drafter bridge for llama.cpp spec decode
          9. First NPU-spec number on 10-prompt humaneval
         10. Sweep k values, write up, close phase
```

Step 5 risk: `NPUSession` currently loads `.onnx` files. Context
binaries need a different ORT-QNN load path (either via
`qnn_context_binary_file` provider option or via an ONNX-EPContext
wrapper that AI Hub emits alongside). Small extension expected.

**Step 5 CLOSED (2026-04-21).** First forward pass running on the
Hexagon NPU. **Step 6 diagnosis (session 10, 2026-04-21) REOPENED
STEP 4** — the binary runs cleanly but produces catastrophically
wrong logits. See "Step 6 diagnosis" section below.



```
session providers   : ['QNNExecutionProvider', 'CPUExecutionProvider']
inputs (58) / outputs (57) — match the binary signature exactly
run latency       : 109.81 ms   (decode step, empty KV, ctx 512)
logits shape      : (1, 1, 151936)
logits finite frac: 1.0000      (no NaN/Inf)
logits min/max    : -4.500 / 3.059
=== STATUS: ok ===
```

Per scoping doc §7 step 5 exit criterion ("one forward pass completes
without error") — done. 110 ms/step is in-line with NPU expectations
(slower than CPU's ~9 ms/tok for Qwen3-0.6B Q8_0; the value lands at
step 7 when drafts pipeline alongside CPU verify). Commit `<TBD>`.

**Two walls hit + cleared on the way:**

1. **ORT-QNN ↔ QAIRT version mismatch.** AI Hub default was QAIRT 2.45;
   `onnxruntime-qnn 1.24.4` bundles 2.42 → `LoadCachedQnnContextFromBuffer`
   error 5000. Bumping to `onnxruntime-qnn 2.1.0` (bundles 2.45.40,
   ships Genie.dll) cleared the version match but its context-binary
   loader has unrecoverable bugs on the X2E94100 driver (both
   file-mapping retry path AND embed_mode=1 path segfault with no
   Python traceback — only the plain-ONNX path works in 2.x). Working
   fix: stay on 1.24.4 + recompile via AI Hub with `--qairt_version 2.42`.
   Recompile (job `jp34dq03g`, also 400s) reused the upload from
   `mng5oj90m`. Full writeup in `docs/npu_ort_qnn_version_match.md`,
   cross-linked from `docs/npu_scoping.md` §3.8.

2. **EPContext wrapper IO names + dtypes had to match the compiled binary.**
   QAIRT's converter normalises dotted names to underscored
   (`past_key_values_0_key`, not `past_key_values.0.key`) and renames
   all outputs to `output_0..output_N` in declaration order. Also,
   `--preserve_io_datatype` keeps past_key_values at FP32 even when
   `--quantize_full_type float16` is set for the graph interior. Real
   names + dtypes captured from `qnn-context-binary-utility.exe`
   inspection (`results/bin_inspect.json`); wrapper builder in
   `scripts/npu_load_qwen3_bin.py` updated to match.

**Updated 10-step tracker (after session 11 step-7 close):**

```
[DONE]    1. Environment snapshot                 (commit 7230210)
[DONE]    2. ORT-QNN sidecar skeleton             (commit 282e84a)
[DONE]    3. Qwen3-0.6B ONNX sourced + CPU-valid  (commit 106c756)
[DONE]    4. AI Hub compile -> Hexagon .bin       (session 11 jperqy07g,
                                                   patha binary 1.4 GB)
[DONE]    5. Load .bin via NPUSession, shape-check (session 9, re-verified
                                                   session 11 on Path A wrapper)
[DONE]    6. Correctness vs CPU, single greedy    (session 11 Path A:
                                                   cos=0.9999, 100% match)
[DONE]    7. Pipe first drafted token through     (session 11 step-7
              llama.cpp verify                     plumbing script passed;
                                                   draft=target=264 at anchor)
[DONE]    8. External-drafter bridge for          (session 11: short-prompt
              llama.cpp spec decode                probe + outer loop;
                                                   6.23 t/s, 65% accept on
                                                   humaneval p0, coherent text)
[DONE]    9. First NPU-spec number on 10-prompt   (session 11: 40-cell sweep,
              humaneval                            k=2 optimal at 7.98 t/s
                                                   mean, 81.0% accept; best
                                                   cell 8.44 t/s at p8)
[DONE]   10. Sweep k values, write up, close      (session 11: docs/npu_results.md
              phase                                — documented loss, 0.31× of
                                                   CPU-alone TG, w4a16 lever
                                                   flagged for Phase 5.5)
```

## Phase 5.5 status (session 12, 2026-04-22)

Full detail in `docs/qwen3_perf_levers_investigation.md` — this is the
project-wide summary.

**Closed levers:**

| lever | commit | k=2 mean t/s | vs baseline | notes |
|-------|--------|-------------:|------------:|-------|
| Phase 5 baseline | 7e10670 | 7.98 | — | ctx=512 fp16 sync |
| Lever A (async draft∥verify) | 64de69f | 10.93 | +37% | pipelined verify-ahead landed in 56b375b |
| Lever B (ctx=256) × A | f755d6d | 14.28 | +79% | battery + CAD load; best cell 17.78 |
| Lever B AC rerun (this session) | 90594d9 CSV | **18.12** | **+127%** | clean AC, other programs closed; **new reference baseline** |
| R4 (zero-copy / shared-mem) | 557c59e | — | no win | parked; per-step dominated by compute not copy |

Battery→AC delta on identical binary is +26.9%, larger than any single
lever's gain — future comparisons must be AC-vs-AC.

**Lever C — W4A16 quantization — in flight, x86 handoff.**

Two AI Hub compile attempts this session, both failed:

1. `jp4x74ll5` (FAILED, ~120s): AI Hub's PTQ validator rejects
   `calibration_data` dicts whose key order doesn't match ONNX
   `graph.input` order. Fixed in commit 372e17a (compile script now
   iterates `specs` to rebuild DatasetEntries; capture script
   puts `attention_bias` AFTER past_kv to match graph order).
2. `j563xme75` (FAILED, 6010s): pipeline got deep — ONNX→DLC ✓,
   quantizer ✓, quantized DLC saved ✓, then QNN backend
   op-validation rejected `/model/rotary_emb/MatMul` with *"has
   incorrect Value 0, expected equal to -32768"* (INT16_MIN, the
   offset QNN's backend hard-codes for rotary outputs). Full AI Hub
   log archived at
   `results/aihub-compile-log-j563xme75-w4a16-a-FAILED.log`.

**Root cause confirmed by inspecting Qualcomm's shipping Qwen3-4B
w4a16 Genie bundle** (`models/qualcomm-qwen3-4b-ref/.../metadata.yaml`):
their graph does not contain rotary_emb internally. `position_ids_cos`
and `position_ids_sin` are declared as top-level graph inputs
(shape `[1,1,N,head_dim/2]`, dtype uint16, **offset -32768** —
exactly the value the AI Hub error expected). Same QAIRT 2.42, same
X2 Elite target. The conclusion: for w4a16 compile to succeed, our
export must hoist rotary out, matching Qualcomm's recipe.

**Infrastructure landed this session** (commits 90594d9, 372e17a,
11fe8fa):

- `scripts/capture_calibration_samples.py` — CPU FP32 prefill +
  greedy decode on humaneval + structured_json fixtures, snapshots
  model inputs at selected decode positions into stacked-per-input
  `.npz`. Reusable for Qwen3.5 cutover.
- `scripts/compile_qwen3_ai_hub.py` extended: `--quant
  {float16,w4a16,w8a16}`, `--calibration-npz`,
  `--calibration-dataset-id`, `--quant-tag`. fp16 path
  backward-compatible.
- `SPECULA_NPU_VARIANT` env var wired through
  `npu_load_qwen3_bin.py` + `npu_vs_cpu_correctness.py` so
  probe/outer_loop/sweep target variant binaries transparently.
  Mirrors `SPECULA_NPU_CTX`'s pattern.
- Calibration bundles (models/calibration/, gitignored): Bundle A
  (60 realistic samples, 3.27 GB) + Bundle B (20 step-0 samples,
  1.09 GB). Both at ctx=256 for the pathbmask schema; both need
  regeneration once pathb lands.

**x86 team work — DELIVERED (session 13).** Artifact:
`models/qwen3-0.6b-pathb/` (61 inputs, 7,131 nodes, zero
`/model/rotary_emb/*` nodes). CPU-equivalence cos = 1.000000 on
both probes vs optimum source. Shipped 3D shape
`[batch_size, sequence_length, 128]` for cos/sin (doc said 4D
`[1,1,1,128]` but that was for a different seam — see
`status_x86.md` session 2 for the seam choice). Bundle on NAS at
`Z:\exposed\junk\phase5_step12_pathb\qwen3-0.6b-pathb\` with MD5
verified.
CPU-equivalence probe gate: cos ≥ 0.9999 vs optimum source. ~0.5
session estimate.

**Next — X2E team work (after pathb arrives):**

1. Add `pathb` to `build_paths` + `build_input_specs` in
   `compile_qwen3_ai_hub.py` (61 inputs, includes cos/sin).
2. Regenerate Bundle A + B calibration for pathb schema (compute
   cos/sin per sample using Qwen3's rope_theta=1e6).
3. Submit `--quant w4a16 --calibration-npz
   bundle_a_pathb_ctx256.npz` — expected to succeed this time based
   on Qualcomm-reference alignment.
4. Wire cos/sin computation into the runtime caller (probe,
   outer_loop, sweep).
5. Correctness probe (cos ≥ 0.95 tolerated post-w4a16) + AC sweep
   vs 18.12 t/s baseline. Optionally run Bundle B for the
   cheap-vs-realistic calibration A/B.
6. Phase 5.5 writeup + close.

### Step 6 diagnosis (session 10, 2026-04-21)

**Summary: the compiled .bin loads + runs, but the nomask ONNX it
was compiled from is computationally broken. The NPU is faithfully
reproducing a corrupted graph.**

Harness: `scripts/npu_vs_cpu_correctness.py` — drives CPU prefill
on the optimum ONNX (standard-ops, FP32 KV) until past_len=511,
then compares one more decode step on both backends with identical
past_kv + input_ids + position_ids. Also runs a 16-step
sliding-window greedy comparison.

**Single-step result (prefilled KV):**
- cosine sim: **0.546** (expected > 0.99)
- argmax: CPU=264 (' a') vs NPU=133927 (Arabic glyph)
- top-5 overlap: **0/5**
- max |logit delta|: 23.97

**Zero-KV + BOS control probe (isolates graph vs KV-handoff):**
- cosine sim: **-0.183** (anti-correlated)
- max logit magnitude: CPU=+14.09, NPU=+4.80

Zero-KV failing rules out KV-handoff semantics — the NPU graph
itself is wrong. Localized bug:

1. **Root cause: `models/qwen3-0.6b-nomask/model.onnx` run on
   CPU-ORT gives cos = -0.18 vs its optimum source.** All earlier
   intermediate artifacts (`optimum`, `optimum-frozen`,
   `optimum-frozen-ortopt`, `optimum-ortopt`, `patched`) produce
   cos = +1.0000 with the source. Only `nomask` is broken.

2. **Bisected the two `simplify_qwen3_no_mask.py` transforms
   against the clean `patched` graph:**
   - Mask-promote-to-constant alone: **cos = +1.0000** (safe)
   - IsNaN/Where guard elision alone: **cos = +1.0000** (safe)
   - Mask-promote + onnxsim(with shape overrides): **cos = -0.18** (BROKEN)

3. **Breakage comes from `onnxsim.simplify()` folding with
   `overwrite_input_shapes` pinning + attention_mask pre-promoted
   to constant `[1,512]` all-ones.** Something in that combination
   constant-folds a position-dependent subgraph incorrectly. Both
   transforms are individually safe; their combination with
   onnxsim is not.

4. **Verified fix direction: skip onnxsim.** Starting from `patched`
   (2185 nodes) and applying mask-promote + isnan-elide only
   produces a graph with cos = +1.0000 vs source. But 2 BOOL Cast
   nodes remain in the attention_mask subgraph (HTP will reject),
   and both trace to ops whose inputs are now known constants.
   They need a targeted surgical constant-fold of just that
   subgraph (not a whole-graph onnxsim pass). See the updated
   `docs/phase5_export_on_x86.md` for the recommended x86-side fix.

### Qualcomm Qwen3-4B NPU reference (inspected session 10)

Downloaded `qualcomm/Qwen3-4B` Genie w4a16 bundle
(~3 GB zipped). Local copy at
`models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`.
Inspected via `metadata.yaml` + `genie_config.json`. Their
architecture choices are substantially different from ours:

- **4-part binary split** per variant (embed + 3 transformer
  chunks). Each part compiles as its own QNN context binary. This
  almost certainly keeps each individual AI Hub compile under the
  op-lowering complexity budget that bit us repeatedly at 4a-4f.
- **40 variants in one bundle**: 5 context tiers (512 / 1024 /
  2048 / 3072 / 4096) × 2 AR batch sizes (128 prefill, 1 decode)
  × 4 parts. Weight-sharing across all of them.
- **RoPE externalized.** `position_ids_cos` and `position_ids_sin`
  are INPUT tensors of shape `[1, 1, seq_len, 64]`, uint16
  quantized, pre-computed on CPU. Graph contains **zero**
  Cos/Sin/Range ops. This eliminates a whole class of HTP
  lowering issues.
- **Attention mask is runtime input, additive, uint16.** Shape
  `[1, 1, seq_q, seq_k]`, quant scale 0.00153 / offset -65535.
  Graph adds this to attention scores pre-softmax. **Zero BOOL
  tensors anywhere.** This is the mechanism for expressing the
  causal pattern without any of the Range/Gather/Cast/And/Where
  subgraph that onnxsim is supposed to fold for us.
- **Full w4a16 quantization.** Activations and KV at the IO
  boundary are uint8/uint16 quantized, not FP16 or FP32. Needs
  AIMET (or AI Hub's quant path) and ~50-100 calibration prompts.
- **Transposed key layout.** `past_key` is `[heads, batch,
  head_dim, seq]` (head_dim BEFORE seq); `past_value` is `[heads,
  batch, seq, head_dim]`. A Qualcomm-specific layout that the
  graph surgery must emit.
- **Tool versions:** QAIRT 2.42.0 (matches our pin for ORT-QNN
  1.24.4 compatibility).

Relevance to our fix: we cannot replicate the full Qualcomm
pipeline (w4a16 + 4-way split + 5 ctx tiers + 2 batch modes is
out of scope for a quick fix). But the two load-bearing
architectural choices — **externalized RoPE** and
**additive-FP16 attention mask** — are the principled fixes that
make the graph naturally HTP-friendly without needing
constant-folding tricks. Our current path (optimum + onnxsim)
takes a graph that contains these problem subgraphs and tries to
fold them out; the Qualcomm path never introduces them in the
first place.

The `docs/phase5_export_on_x86.md` doc now recommends two paths
for the x86 re-export:

- **Path A (minimal):** keep current pipeline (optimum +
  `--no-post-process`), drop onnxsim entirely, apply
  mask-promote + isnan-elide + targeted surgical fold of the
  residual attention_mask subgraph. Preserves cos = 1.0 and
  emits zero BOOL Casts. Smallest change; may still hit
  compile-time op-lowering issues on monolithic graph
  complexity.
- **Path B (Qualcomm-style):** externalize RoPE + use
  additive-FP16 attention mask. Larger surgery but the
  architecturally robust path. Separate from w4a16 quantization
  (which can layer on top later for the perf win).

Path A is the recommended starting point since it's a smaller
delta from what's known to produce a compileable graph (we
already got one compile through on nomask, just semantically
wrong).



### Session 11 (2026-04-21): first compile cycle on patha + pathbmask

X86 team delivered two CPU-ORT-verified ONNX variants (both
cos=1.0 vs optimum source, zero BOOL casts on both, zero BOOL
tensors on pathbmask). Staging + AI Hub compile scripts
parameterized by `--path {patha,pathbmask}`.

Both first-cycle compiles **failed**; both for the same root
cause — **dynamic shapes in the uploaded ONNX**. Full retro in
`docs/phase5_step6_compile_retro.md`. Load-bearing findings:

1. **AI Hub's OverrideFoldConstantsPass folds BOOL subgraphs
   cleanly** when the shapes around them are concrete. Path A's
   compile log shows Cast 348→1, ConstantOfShape 60→0, Equal
   58→0, Where 86→0, Range 3→0 in a single pass. **The x86
   team's "BOOL rejection" hypothesis was wrong** — surgical
   BOOL removal was nice-to-have, not required.
2. **AI Hub's compiler has a pass that doesn't handle SymbolicDim**
   (`_op_identity.py:30` calls `np.broadcast_shapes` on dims).
   Our ONNX declares `input_ids: (-1, -1)`, past_kv: `(-1, 8, -1, 128)`,
   etc. — the rewriter blows up on the first symbolic dim it sees
   after the fold pass.
3. **Session 9's nomask compile only worked because onnxsim had
   pinned shapes statically**. The x86 team replaced onnxsim
   (correctly — it corrupted numerics) with pure protobuf surgical
   folds, but those preserved dynamic shapes.

Fix: a 30-LOC pure protobuf edit in `prep_onnx_for_ai_hub.py` that
pins every graph input's `TensorShapeProto` dims to the static
values from `compile_qwen3_ai_hub.build_input_specs`. Idempotent,
no numerical impact (session 10 bisection established that
protobuf-only edits preserve cos=1.0).

Jobs (both artifacts):

| path      | upload id  | job id     | status  | failure kind               |
|-----------|------------|------------|---------|----------------------------|
| patha     | mnzpwoe6q  | jp83n13kg  | FAILED  | SymbolicDim in rewriter    |
| pathbmask | mqe19804m  | jp01w619g  | FAILED  | pre-compile shape mismatch |

Local compile logs: `results/phase5_step6_compile_{patha,pathbmask}.log`.
Workbench logs: `results/ai_hub_logs/{jp83n13kg,jp01w619g}.log/`.

**Next:** resubmit both serially with the pinned ONNX. Serial avoids
halving upload bandwidth on this link (~25 min/path single-stream
vs ~50 min parallel). Predicted outcome: both compile (the fold pass
already handled the BOOL region for Path A; pinning resolves the
SymbolicDim bug for both).

### Session 11 cont. — what actually unblocked the compile

Predicted wrong. Shape pinning + dim_param resolution was **not
sufficient** — iterations v2, v3, v4 produced byte-for-byte
identical op histograms on the failing compile side. AI Hub's
pipeline discards or re-derives our provided value_info and
re-triggers the same SymbolicDim crash in its identity-op
rewriter.

**The load-bearing fix was ORT's `ORT_ENABLE_BASIC` graph-optimization
pass applied locally before upload.** Sequence that worked (v5):
1. pin_input_shapes — graph.input dims concrete
2. resolve_dim_params — substitute `batch_size`/`sequence_length`/
   `past_sequence_length + sequence_length` with concrete ints
3. **ORT-BASIC constant-fold** (node count 7580 → 2061)
4. resolve any `unk__N` placeholders ORT generated → 1
5. final onnx.shape_inference(data_prop=True) — 0 tensors with
   symbolic dims

Why ORT-BASIC wins where AI Hub's own fold pass didn't: ORT
actually *replaces* Range/Shape/ConstantOfShape/Gather nodes with
Constant initializers and eliminates them from the graph.
AI Hub's `OverrideFoldConstantsPass` also folds them but keeps
residual tensor shape annotations marked SymbolicDim from its own
shape inference, which the downstream rewriter then chokes on.
By pre-folding with ORT, the symbolic-shape annotations never
enter AI Hub's pipeline in the first place.

Why not use ORT `ENABLE_EXTENDED`: that level introduces operator
fusions (GELU/LayerNorm/Attention) that emit `com.microsoft`
ops — the exact ones we spent sessions 7-9 removing. BASIC is
strictly constant-folding + redundant-node elimination, numerically
equivalent to source per session 10's `optimum-ortopt` cos=1.0
probe.

Jobs in session 11:

| path      | job id      | result                              |
|-----------|-------------|-------------------------------------|
| patha v1  | jp83n13kg   | FAILED (SymbolicDim in rewriter)    |
| pathbmask v1 | jp01w619g | FAILED (pre-compile shape mismatch) |
| patha v2  | jgj09qwep   | FAILED (same)                       |
| patha v3  | jg93rddmg   | FAILED (same)                       |
| patha v4  | jpr4rw7vg   | FAILED (same)                       |
| patha v5  | **jperqy07g** | **SUCCESS @ 430s, 1.4 GB binary** |
| pathbmask v5 | j563xwkv5 | FAILED (input-spec *order* mismatch) |
| pathbmask v6 | **jpx7q4o9g** | **SUCCESS @ 400s, 1.4 GB binary** |

Correctness on both v5/v6 binaries (`npu_vs_cpu_correctness.py --path {patha,pathbmask}`):
- Single-step prefilled KV: **cos = 0.999916** (gate ≥ 0.95) — both paths
- 16-step sliding-window greedy: **100% match rate** (gate ≥ 50%) — both paths
- NPU text matches CPU verbatim: `" a 5G network. It is a smartphone with a smartphone"`
- Path A and Path B-mask produce **byte-identical NPU output** (expected — same 2061-node graph after ORT fold)
- Zero-KV + BOS probe: cos 0.94 (edge; non-BOS zero-KV = cos 1.0000)

**Step 6 CLOSED on both paths.** x86's 2×2 hypothesis matrix
(Path A: "BOOL casts removed, tensors remain" vs Path B-mask:
"zero BOOL tensors") is answered: HTP accepts either. The
load-bearing issue was graph complexity (dynamic shapes + unfolded
Range/Shape/Expand chains), not BOOL op types.

**Operational recommendation:** Path A is the primary NPU binary
(one less runtime feed — no `attention_bias` zeros tensor per
step). Path B-mask stays in the repo as the documented equivalent
path, useful if a future regime needs a non-zero additive bias
(partial-window prefill with padding).

Step 7 (llama.cpp verify wiring) now unblocked.

### Session 11 cont. — step 7 plumbing checkpoint PASSED

`scripts/npu_spec_step7_plumbing.py` drives the three-way comparison
the scoping doc §7 step 7 calls for: NPU draft vs CPU reference vs
llama-server target, all at the step-6 validated anchor position.

**Design choice:** pass raw token ids (not detokenized text) to
llama-server's `/completion` endpoint. `llama.cpp/tools/server/
server-common.cpp:767` accepts `"prompt"` as a JSON array of ints via
`json_is_array_of_numbers(json_prompt)`, so both sides compare purely
at the id level with zero risk of detok->retok divergence between
HF's Qwen3 BPE (what the NPU draft uses) and the llama.cpp GGUF vocab
(what the target uses). A lightweight sanity probe up front confirms
the two tokenizers agree on ids for a sample string — they match byte-
for-byte on the 11-id Fibonacci probe.

**Run outcome (`results/phase5_step7_plumbing.log`):**

```
tokenizer probe      : 11/11 ids match between HF 0.6B and server 8B
NPU draft (Path A)   : token 264 (' a')
CPU reference (0.6B) : token 264 (' a')   [step-6 anchor, known good]
Target (8B, CPU)     : token 264 (' a')
draft == target      : True   (accept)
draft == CPU 0.6B    : True   (sanity)
```

All three backends converge on the same next-token id at the
511-token anchor. The plumbing exit criterion — one drafted token
returned, one target token returned, one accept/reject decision
logged — is met. The `accept=True` here is a bonus: it demonstrates
the small draft genuinely predicting what the large target would,
not a coincidence of anchoring.

**Invocation recipe** (captured in the script):

```
llama-server.exe
  -m models/Qwen3-8B-Q4_K_M.gguf
  --host 127.0.0.1 --port 8088
  -c 576                 # CONTEXT_MAX (512) + n_predict slack
  -t 18                  # match Phase 2 CPU-spec baseline
  --no-warmup
```

with `/completion` body
`{"prompt": [512 ids], "n_predict": 1, "temperature": 0.0, "top_k": 1,
  "seed": 1, "cache_prompt": false, "return_tokens": true}`.
Target call latency: **4.60 s** for PP=512 + 1 generated token (about
111 t/s prompt-eval, matching Phase 1's 8B Q4_K_M CPU PP512 of
164 t/s once server overhead is accounted for).

**Caveat carried into step 8:** the script anchors at past_len=511
so the NPU never has to mask out invalid KV positions. The
production spec-decode outer loop wants short prompts (20-50 tokens,
drafting 3+ tokens per round), which needs Path B-mask's non-zero
`attention_bias` with -65504 for invalid slots. That masking pattern
is numerically unvalidated on the NPU today; step 8's outer loop
has to prove it out before the first end-to-end run.

Step 8 (external-drafter outer loop) now unblocked.

### Session 11 cont. — step 8 PASSED end-to-end

Two scripts landed, both green:

**`scripts/npu_short_prompt_probe.py`** — short-prompt NPU probe
gate. Encodes humaneval p0 (16 tokens), CPU-prefills, then NPU
single-step at position 16 with slots 16..510 zero-padded and
`attention_bias` set to `-65504` over padded slots + 0 over
valid slots + 0 over the self-slot. Also validates the multi-step
KV rearrangement primitive (3 consecutive NPU steps, each growing
valid_past_len by 1, moving the K/V from slot 511 to slot P after
each step). Both gates passed byte-clean: **single-step cos =
0.999960**, argmax match, top-5 5/5; **multi-step 100% match**
(3/3 tokens identical to CPU greedy), NPU text `"    if n"`.

**`scripts/npu_spec_outer_loop.py`** — first NPU-drafted spec
decode end-to-end. Sidecar-as-driver; per round:

1. Draft k tokens on NPU via short-prompt mask + slot-511→slot-P
   rearrangement between steps. Keep k+1 past snapshots so any
   accept count j ∈ [0, k] can roll back cleanly.
2. POST committed ids to `/completion` with `n_predict=k+1`,
   `cache_prompt=true`, greedy pinned. Read k+1 target tokens.
3. Longest-common-prefix accept: j matching drafts + 1 bonus
   target token committed.
4. Absorb bonus into past via one more NPU step → next round's
   state + first candidate for next round.

Run on humaneval p0, k=3, n_predict=64:

```
rounds              : 22
decoded tokens      : 65
mean accept rate    : 65.2%   (43/66 drafts accepted)
wall generate       : 10.43 s
  NPU draft total   :  5.22 s
  target verify     :  3.53 s
  NPU absorb        :  1.67 s
decode rate         : 6.23 t/s
```

**Generated text** (the NPU-drafted, target-verified continuation of
the Fibonacci stub):

```
    # Initialize a memoization dictionary
    memo = {0: 0, 1: 1}

    # Define a helper function to compute the Fibonacci number
    # recursively with memoization
    def fib(n):
        if n not in memo:
            memo[n] = fib(n-1) + fib
```

Functionally correct memoized Fibonacci. Proves draft + verify + KV
rearrangement are all semantically sound; the question now is purely
performance.

**Why 6.23 t/s (not faster than 25.91 t/s CPU-alone, let alone
Phase 2's 40.2 t/s CPU-spec)?**

Bottleneck is NPU per-call latency:

- **110 NPU calls / 22 rounds = 5 calls/round** (3 drafts + 1 final-
  snapshot step that only pays off when j == k + 1 absorb-bonus).
- At ~63 ms/call that's 6.9 s of the 10.4 s wall budget.
- Target verify is 3.53 s = 160 ms/round. At CPU target TG ~26 t/s
  and k+1=4 tokens per round, that's ~155 ms of decode + negligible
  HTTP. Cache-prompt keeps prefill cost off the critical path after
  round 1.
- Draft phase of 5.22 s + absorb of 1.67 s = 6.89 s strictly
  sequential with the target verify's 3.53 s. 10.4 s total.

**CPU-spec baseline used** ~25 ms/token-generated with 0.6B at
~9 ms/draft-step on CPU. NPU's 63 ms/step is ~7× slower per step.
Without overlap or a fatter draft tree, there's no way to recover
the cost of running draft + target sequentially when draft alone
costs more than target's per-token decode.

**Step 9/10 levers (in priority order):**

1. **Drop the final-snapshot step when j < k.** Saves 1 NPU call
   (22 × 63 ms = 1.4 s in this run). Easy lazy-compute refactor —
   only materialise `past_snapshots[k]` if j==k actually happens.
   Would push us toward ~7.2 t/s, still below 25.91 CPU-alone TG.
2. **Overlap NPU draft with target verify.** Kick off `/completion`
   for round N's verify and NPU draft for round N+1 in parallel.
   Caveat: round N+1's draft depends on round N's accepted tokens,
   so this only overlaps the target-side portion of round N with
   round N+1's BEFORE-accept drafts (which we'd speculatively compute
   assuming drafts[0] accepted). Saves up to 3.5 s on the verify
   side. Complex but can double throughput if it works.
3. **Pipelined k+K drafting.** Draft k tokens, but also opportunistically
   start the next-round's first draft step during verify. Structural.
4. **Reduce NPU per-step latency.** Would need ORT-QNN EP-side work
   (cl_qcom_ion_host_ptr-style zero-copy KV, per-call bind caching)
   or a recompiled binary with a smaller compiled context. Deep dive,
   Phase 5.5 or 6 territory.

Step 9 (sweep k + multi-prompt) will produce the real CSV for the
writeup. Lever (1) is easy and should land before step 9 so the
numbers we report don't carry an obvious waste. Levers (2-4) are
explicitly out-of-scope for Phase 5 close; noted for Phase 6.

## Immediate next steps (next session)

**Phase 5 is CLOSED.** 40-cell sweep banked, writeup in
`docs/npu_results.md`. Headline: k=2, 7.98 t/s mean (8.44 best),
81.0% accept — a 5× structural regression vs Phase 2 CPU-spec
40.2 t/s, with accept rate identical to CPU-spec (81.0% NPU vs
82.3% CPU at k=2). NPU per-step latency is the root cause, not
drafter quality.

Two branches open next. Both can be pursued independently; the
Qwen3 close-out items have priority so Qwen3.5 graduation
(Phase 4) is unblocked.

**(A) Phase 5.5 — NPU performance levers (if we want a better
NPU-spec number before graduation).** Ranked by impact × effort:

1. **W4A16 quantisation** — biggest per-step lever. Qualcomm's own
   Qwen3-4B Genie bundle ships W4A16 (`models/qualcomm-qwen3-4b-ref/`);
   expected ~2-3× NPU per-step speedup + 4× weight BW reduction.
   Would push decode to ~20 t/s (model). Cost: AIMET or AI Hub
   quant pipeline run against humaneval + structured_json as
   calibration set. ~1 session.
2. **Async NPU-draft ↔ target-verify overlap.** Makes wall =
   max(NPU_round, verify_round) rather than sum. Expected ~11 t/s
   at FP16 (57% improvement); stacks with W4A16 to ~28 t/s.
   Cost: ~200 LOC async rework. ~1 session.
3. **Smaller past_len compile tier.** Our binary bakes
   past_len=511. For code drafting at ~256-token generation, a
   past_len=256 tier would ~halve attention FLOPs per step.
   Cost: one AI Hub recompile + tiered loader. ~half session.
4. **Zero-copy KV handoff** via `cl_qcom_ion_host_ptr`. Small
   (~10%) NPU-side win but unblocks DFlash-on-OpenCL's later
   Phase-4 perf tuning. Phase 6 territory.

**(B) Qwen3 close-out before graduation** (the stashed Phase-2
items that become orphaned at Qwen3.5 cutover):

- [ ] `--draft-p-min` tightening at k=3 on CPU-spec (kept; cheap
      data point to sharpen our CPU-spec baseline).
- [ ] `prompts/prose_longform.jsonl` + `prompts/chat_multiturn.jsonl`
      stub content (fill the four-workload matrix before writing up).
- [ ] Ngram spec (`--spec-type ngram-*`) A/B on JSON (floor
      baseline for "dumbest draft").
- [ ] Negative-result upstream contribution to llama.cpp — the
      NPU-spec + OpenCL-spec stories together are publishable data
      on the Snapdragon X2 heterogeneous-exec question.

**Recommended order:** (B) close-out items first (cheap, graduates
Qwen3 cleanly), then decide whether (A) W4A16 is worth chasing
on Qwen3 or save for Qwen3.5. Given production-target is Qwen3.5/6
and Phase 4 DFlash on Qwen3.5 is the next big milestone, the
W4A16 lever probably lands on Qwen3.5's draft directly rather than
Qwen3's.

**Strategy (session 5, 2026-04-20): close out Qwen3, then graduate
fully to Qwen3.5 for all further work.**

The Qwen3 family has been our scaffolding model because it has more
public literature (EAGLE PRs, community benchmarks, llama.cpp coverage).
Production target is Qwen3.5 → Qwen3.6 -- see
`memory/project_target_model.md`. Rather than running two parallel
tracks (keep poking at Qwen3, start Qwen3.5), we treat this as a
clean graduation:

- Use the Qwen3 window to land everything *adventurous* that only
  makes sense on pure-attention + smaller models: **NPU drafting** is
  the headline item (simplest path into QAIRT), but any stashed
  Phase-2 experiments (see below) that we care about should finish
  in this window too -- they'll be orphaned after graduation.
- Once NPU drafting on Qwen3-0.6B + Qwen3-8B has a stable baseline
  (win, lose, or tie against Phase 2's 1.55× CPU-spec), declare Qwen3
  *closed* and move all subsequent phases (DFlash, NPU-on-dense-small,
  MoE) to Qwen3.5+.
- This keeps the codebase from accumulating two model-family branches
  and keeps our attention on Qwen3-era questions while the tooling is
  still fresh.

**Revised phase order: NPU-first (close out Qwen3), then Phase 4
DFlash (opens Qwen3.5 era).**
Session 5 scoped Phase 4 against lucebox-hub and then re-sequenced:

- **Phase 5 (NPU drafting on Qwen3) moves ahead of Phase 4.** The
  Phase-2 mixed-device negative result told us CPU↔OpenCL sync breaks
  small-batch heterogeneous exec on this hardware. It did NOT tell us
  whether heterogeneous exec works *at all* -- only that ggml-opencl's
  per-round launch profile loses in the specific CPU-draft +
  OpenCL-target pairing. The NPU path is the structurally different
  bet: pipelined async dispatch + ION-backed buffers + parallel-to-
  target draft. Answering "does heterogeneous work on X2 at all" is
  a prior-scope question that gates DFlash's eventual Phase-5-style
  async design. If NPU-draft also regresses, we learn that before
  committing to a DFlash-on-OpenCL port with the same sync shape.
- **Phase 4 (DFlash + DDTree on Qwen3.5) follows NPU work.** The
  scoping pass already landed (see "Phase 4" section below); the
  assets can be downloaded in parallel with NPU bring-up so Phase 4
  is ready to start the moment NPU drafting answers its core
  question. Session 5 scoping remains valid regardless of ordering.
- CPU spec peaks at **1.55× (40.2 t/s) at k=3**; ceiling ~1.6×. This
  remains the target-side baseline for both phases.
- lucebox-hub hit 3.43× on RTX 3090 with DFlash+DDTree (pure-CUDA,
  single device). Our win condition is different: we're stitching
  heterogeneous compute (NPU + CPU + GPU) via shared LPDDR5X, not
  competing on single-device throughput.
- Phase-2 stretch items (`--draft-p-min`, 14B target, prose/multi-
  turn, ngram spec, upstream writeup) stay stashed -- none break 1.6×.

### Phase intersection — why NPU-first is cheap information

The two tracks share less code than they look to at first, but the
one thing they DO share is the work that matters most.

- **Shared, high-value: zero-copy buffer model on shared LPDDR5X.**
  NPU drafting needs NPU↔CPU/GPU shared allocations via
  `cl_qcom_ion_host_ptr` or QAIRT buffer handoff so the NPU draft
  doesn't pay a cache-flush round trip to hand tokens back to the
  target. DFlash-on-OpenCL needs the *same* pattern for its per-
  round `target_feat` → draft → verify loop -- the Snapdragon
  analogue of the D2D copy that gave lucebox +3.3% on PCIe. Doing
  NPU first forces us to solve ION-backed allocation + async
  dispatch first; Phase 4's OpenCL port then inherits that
  infrastructure intact. This is the single biggest intersection.
- **Shared, low-value: benchmark harness + prompt fixtures + metric
  logging.** `sweep_speculative.ps1`, the humaneval/json JSONL
  fixtures, the AL/accept/tok-s columns in our CSVs all carry over.
  Already built in Phase 2.
- **NOT shared: kernel ports.** NPU drafting doesn't touch
  ggml-opencl (draft runs on QAIRT, target uses stock llama.cpp).
  DFlash-on-OpenCL doesn't touch Hexagon. Two independent skill
  trees; doing one doesn't accelerate the other's kernel work.
- **NOT shared: target-side code.** NPU spec decode uses llama.cpp's
  stock `--draft` pipeline -- zero target patching, just swap which
  binary runs the draft forward. DFlash needs the custom
  non-libllama target loader + 5-layer hidden capture hooks.
- **NOT shared: draft/target pairing.** NPU standard spec decode
  only needs tokenizer compatibility (Qwen3-0.6B-draft +
  Qwen3-8B-target works). DFlash requires the drafter to be trained
  on the specific target's hidden states -- Qwen3-trained drafter
  cannot verify a Qwen3.5 target. So NPU-on-Qwen3 and
  DFlash-on-Qwen3.5 are independent experiments, not a shared lane.

### Session 5.5 prep (do in parallel with NPU bring-up)

To keep Phase 4 unblocked once NPU drafting answers its core
question, stage these in the background:

- [ ] Download `unsloth/Qwen3.5-27B-GGUF` Q4_K_M (~16 GB) into
      `models/`. Verify `arch=qwen35` in the GGUF metadata.
- [ ] Download `z-lab/Qwen3.5-27B-DFlash` safetensors (~3.5 GB BF16).
- [ ] Read `delta_net_chunked.cpp` (237 LOC) and list every ggml op
      it calls. Cross-check against ggml-opencl's `supports_op`.
      This is the Phase-4 gating item.
- [ ] Confirm `ggml_rope_ext` section-mode support on OpenCL (Qwen3.5
      M-RoPE uses sections `[11,11,10,0]`; plain NEOX is fine for the
      draft and verified working).

### NPU drafting on Qwen3 — why Qwen3 (not Qwen3.5) for this phase

Two sub-questions people conflate:

1. **NPU porting difficulty.** Pure-attention (Qwen3 small variants)
   maps cleanly onto QNN's shipped op library (matmul + rmsnorm + rope
   + swiglu). Hybrid variants (Qwen3.5 with `gated_delta_net` +
   `ssm_conv`) require persistent-state handling on NPU that isn't in
   the sample ops. **Hybrid is the blocker, not Qwen3 vs Qwen3.5.**
   Pure-attention variants of either family work equivalently.
2. **Draft/target pairing** for standard speculative decoding (not
   DFlash): only needs tokenizer compatibility. Qwen3 and Qwen3.5
   share the same tokenizer (Qwen family policy), so a tokenizer-
   compatible cross-family pairing is valid. But for apples-to-apples
   baselines it's cleaner to stay one-family-per-experiment.

So the Phase 5 NPU plan is: bring-up on Qwen3-0.6B-Q8_0 draft +
Qwen3-8B-Q4_K_M target (same pair as Phase 2). Once it works, swap
to Qwen3.5-*dense-small* draft + Qwen3.5-*dense* target for the
production-target experiment. We **don't** attempt NPU + hybrid
(Qwen3.5-27B-hybrid) together yet -- layering two unknowns. That
combination lands after both Phase 4 (DFlash on hybrid) and Phase 5
(NPU on pure-attention) have baselines.

### Caveats carried into Phase 5

- **Scoping doc is canonical:** `docs/npu_scoping.md` (session 5,
  2026-04-20) has the 10-step bring-up plan, toolchain pins
  (QAIRT 2.45.40, ORT-QNN 1.24.4), known-failure-mode catalog, and
  prior-art review. The bullets below are the short version; the
  doc is the single source of truth for Phase 5 execution.
- Hedge docs already absorbed into the scoping doc:
  `voice_project/current_status.md` + trident's `postmortem.md` /
  `npu_path_back.md` / `npu_current_status.md` /
  `npu_optimizations_thoughts.md`.
- Pin the NPU bring-up draft to **Qwen3-0.6B-Q8_0** (not 1.7B).
  Smaller compile iterations + fewer custom-op surprises on the
  first pass.
- Keep target side identical to Phase 2 winning config: Qwen3-8B-
  Q4_K_M on CPU at 18 threads (25.91 t/s TG baseline, 40.2 t/s
  CPU-spec at k=3). That way the first NPU-spec number is directly
  comparable to the CPU-spec result; we can say cleanly whether
  NPU-as-draft wins, loses, or ties vs CPU-as-draft.
- Hexagon arch target is **v81** on X2E -- proven by voice_project's
  working AI Hub compile (`dspArch: 81, socModel: 88`). Session 6
  step 1 still does a QAIRT device-enum sanity check before any
  code, but v81 is the pin.
- **Primary path = AI Hub cloud compile + ORT-QNN EP 1.24.4 runtime.**
  Not raw `QnnContext_createFromBinary`. voice_project hit three
  driver-signing walls on the raw path and only escaped via ORT's
  bundled signed QAIRT stack. Our Phase 5 target/draft integration
  needs an external-drafter sidecar (no QNN backend in llama.cpp),
  tracked as blocking in npu_scoping.md §8.
- **Prior-art review integrated** from
  `npu_thoughts_previous_examples.md` (sd.npu, Mirror-SD, HeteroLLM,
  Dovetail, OpenPangu). Structural plan unchanged; key post-bring-up
  lever is sd.npu's pad/recycle trick for <8-token drafts (our k=3
  is deep in that regime). Full analysis in npu_scoping.md §10.

### Stashed -- now reframed as "Qwen3 close-out" candidates

After the session-5 re-sequencing, these items fall into the Qwen3
window that closes with Phase 5 NPU. Anything we don't do here
gets orphaned at graduation -- pick deliberately. Priority marks:
**[keep]** = worth doing before graduation, **[drop]** = fine to
skip, **[carry]** = trivially portable to Qwen3.5 so not time-
pressured.

- **[keep] Tighten `--draft-p-min`.** Default 0.75 over-drafts on
  low-confidence streaks (the `flatten` pathology). Try 0.80, 0.85,
  0.90 at k=3. Cheap data point; improves our CPU-spec baseline
  before we compare it to NPU-draft. Actively useful inside the
  Qwen3 window.
- **[carry] 14B target + 0.6B draft (or 1.7B draft).** Draft/target
  compute ratio drops ~2×; break-even k shifts higher. Needs
  `Qwen3-14B-Q4_K_M.gguf` download. Not time-pressured -- we'll do
  the equivalent exercise on Qwen3.5 dense variants after
  graduation, so skipping on Qwen3 loses no learning. Do only if
  convenient alongside Phase 5 NPU.
- **[keep] Prose + multi-turn workloads.** `prompts/prose_longform.jsonl`,
  `prompts/chat_multiturn.jsonl` stubs still empty. Worth filling
  before NPU-draft data collection so the first NPU-spec numbers
  already cover the full workload matrix we'd want for a writeup.
- **[keep] Draftless ngram spec** (`--spec-type ngram-*`). Memory-
  free; should fly on JSON. Quick A/B, closes a gap in the Qwen3
  spec-decode story and sets a useful floor for "dumbest possible
  draft" that NPU-draft and DFlash both need to beat.
- **[keep] Negative-result contribution upstream.** The Adreno-
  OpenCL spec-decode story (no win at any k or placement) is worth
  a llama.cpp docs/discussion post. Write after Phase 5 NPU so we
  can contribute "here's what does work on heterogeneous X2"
  alongside. High visibility, low effort once the NPU number is in.
- **[drop] EAGLE-3 viability probe.** Was Phase 3 anchor, demoted
  already. Lucebox paper showed chain-over-tree gives +15% on Q4_K_M
  (quantization flattens draft softmax); EAGLE-3 alone won't break
  our 1.6× ceiling. Not worth Qwen3-window time. Revisit on Qwen3.5
  only if DFlash underperforms.

### Phase 4 (DFlash + DDTree on Qwen3.5) -- queued behind Phase 5 NPU

Reference impl to mine: `C:\Users\hotschmoe\Documents\GitHub\lucebox-hub/dflash/src/`
(sibling checkout pulled 2026-04-20). See
`docs/reference-projects.md` for file-level guidance on what to read
first.

#### Session 5 (2026-04-20) scoping findings

Read: `dflash_graph.h`, `qwen3_dflash_graph.cpp` (168 LOC, draft
graph), `safetensors_draft.cpp` (407 LOC, draft weights),
`internal.h` (288 LOC, shared state/cache schema),
`qwen35_target_graph.cpp` (806 LOC, hybrid target),
`delta_net_chunked.h` (chunked delta-net entry),
`gguf_target_loader.cpp` (386 LOC, non-libllama GGUF loader),
`RESULTS.md`.

**Kernel coverage on current ggml-opencl backend**
(`llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp`, HEAD
`e365e658f…`):

- `GGML_OP_FLASH_ATTN_EXT` -- IMPLEMENTED (`ggml_cl_flash_attn` at
  line 9265, dispatch at 14033). Supports_op gate at 4166-4200:
  supported `{dk,dv}` pairs include `{128,128}` which matches our
  draft head_dim; supported dtypes F32/F32, F16/F16, F32+F16KV.
  BF16 input not supported so we convert at load, same trick the
  lucebox CUDA port uses for norms.
- `GGML_OP_SSM_CONV` -- IMPLEMENTED (4085, 13934). Hybrid
  delta-net 1D causal conv already runs on OpenCL.
- `GGML_OP_GATED_DELTA_NET` -- NOT implemented on OpenCL.
  This is the Qwen3.5 SSM recurrence kernel. Options:
  (i) use lucebox's `delta_net_chunked.cpp` path, which
  re-expresses the recurrence in already-supported primitives
  (237 LOC); (ii) CPU fallback for delta-net layers only; (iii)
  write an OpenCL kernel ourselves. (i) is the only tractable
  short-term path -- if its ops are all ggml-opencl-backed, we
  get hybrid on OpenCL for free.
- M-RoPE with `rope_sections [11,11,10,0]` -- still need to verify
  OpenCL supports section-mode `ggml_rope_ext` (Qwen3.5 needs it;
  pure Qwen3 uses plain NEOX which works).

**Non-kernel blockers**
- `gguf_target_loader.cpp` is POSIX (mmap / open / fstat / munmap).
  Same ~30-LOC Windows swap as `safetensors_draft.cpp`
  (`CreateFileMapping`/`MapViewOfFile`). Do both at once.
- Target arch string is `qwen35` (not `qwen3`); loader validates
  this. GGUF we download must match.
- Token embedding stays on CPU (CUDA port notes CUDA get_rows
  can't handle k-quants; OpenCL likely same). `CpuEmbedder` in
  `internal.h` already handles this -- portable as-is.
- Target graph uses `capture_layers` mode to sink 5 specific
  layer hiddens into a 4096-slot `target_feat` ring
  (`qwen35_target_graph.cpp:694` — `CAPTURE_LAYERS[]`). The draft
  reads this as its "5*hidden" input. Keep as-is; no llama.cpp
  patching needed since we use the non-libllama loader.
- `DDTree` adds per-delta-layer `ssm_intermediate` (F16,
  `[S_v, S_v, H_v, max_verify_tokens]`) and
  `conv_input_cache`. Hybrid-only overhead. Pure-attention
  targets pay ZERO per-node memory tax (RESULTS.md, "Memory
  ceiling notes") — the published DFlash paper runs budgets up
  to 1024 on pure-attention Qwen3-8B/30B. This is the single
  biggest argument for eventually doing our own pure-attention
  drafter (option B below).

**Draft graph port cost (option A, using z-lab 27B drafter):**
~near zero. Every op (`ggml_mul_mat`, `rms_norm`, `mul`,
`reshape`, `concat`, `rope_ext` NEOX, `permute`, `cont`,
`flash_attn_ext`, `silu`, `add`) already on ggml-opencl.
Weights loader is pure I/O + a hand-rolled safetensors JSON
parser. BF16-on-disk is fine; norms get converted to F32 at
load (already in the code).

#### Path decision: A (today) → B (later). Documented below.

lucebox-hub's reference impl is glued to Qwen3.5-27B-hybrid.
z-lab only publishes DFlash drafter weights for 27B. We pick A
first to de-risk (reference impl exists, working numbers exist,
no training needed) and carry B in the backlog.

**Option A — Qwen3.5-27B hybrid target + z-lab 27B DFlash drafter (chosen for now)**

Plan:
1. Download assets:
   - `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB) into `models/`.
   - `z-lab/Qwen3.5-27B-DFlash` (safetensors, BF16, ~3.5 GB).
2. Port `safetensors_draft.cpp` + `gguf_target_loader.cpp`
   mmap→Win32 mapping. ~1 day.
3. Standalone drafter smoke: port `qwen3_dflash_graph.cpp`
   against ggml-opencl backend. No target yet, just prove the
   drafter forward runs and dims match.
4. Target forward with full-attention layers only (16 of 64,
   every 4th). Confirm logits plumbing; garbage output is
   expected because delta-net layers are stubbed.
5. Delta-net layers via `delta_net_chunked` primitives path.
   If every op has OpenCL coverage → hybrid on OpenCL directly.
   Otherwise CPU-partition delta-net layers.
6. Chain verify (q_len=16) end-to-end. Expected AL 6-7 (lucebox
   Math500/GSM8K numbers).
7. DDTree verify. Expected AL 8+, ~3× target.
8. Perf-tune: f16 intermediate, `gated_delta_net_tree_persist`
   equivalent, D2D target_feat replacement (Snapdragon shared-
   LPDDR analogue: avoid cache flush pairs).

Upsides: reference impl exists, weights exist, results are
published (3.43× HumanEval on 3090). Strong alignment with
stated production target (see user memory:
`project_target_model.md`).

Downsides: 27B Q4_K_M (~16 GB) + drafter (~3.5 GB BF16) + KV +
intermediates leaves comfortable margin in 48 GB LPDDR5X --
less than Qwen3-8B (~4.5 GB weights) but nowhere near tight.
Long-context 128K via Q4_0 KV would add ~8 GB and still fit.
Hybrid port is more unknown than pure-attention. Per-node
DDTree memory tax caps budget ~22-26 on hybrid only (a
pure-attention target would be uncapped).

**Option B — Train our own DFlash drafter for a pure-attention Qwen (revisit later)**

Plan:
1. Pick target: Qwen3-8B (scaffolding) OR Qwen3.5-8B-dense
   (closer to production, no hybrid).
2. Build training infra: 5-layer DFlash drafter per the paper
   (`docs/reference-projects.md` cites Qwen3-8B/30B-MoE numbers
   from the original DFlash paper at 4-5× on HumanEval).
3. Run distillation against target hiddens on a cloud GPU
   (BF16 training -- not feasible locally on the X2). A few
   hundred GPU-hours ballpark.
4. Port the drafter weights + dims into our (by-then-working)
   DFlash pipeline. Pure-attention target skips the delta-net
   port entirely. Per-node memory tax is zero -> budgets up to
   1024 per the paper.
5. Retest DDTree at large budgets; upper bound on speedup
   likely higher than 27B-hybrid because verify-batch memory
   isn't hybrid-capped.

Upsides: pure-attention path removes the deltanet port
entirely; matches any Qwen3.5-*dense* production target;
higher budget ceiling. Weights are ours (no z-lab license
considerations).

Downsides: training cost; no existing reference impl at our
exact dims; gap between "has a drafter" and "has a *good*
drafter" (distillation quality drives AL).

**When to pivot A→B:** after Option A lands end-to-end and we
have a clean hybrid DDTree baseline on Snapdragon. At that
point we know the per-round overhead floor on our hardware and
can quantify what training our own drafter buys. Until then,
option A is cheaper information.

#### Dependency queue (session 5 → session 6)

Before writing any port code:

- [ ] Download `unsloth/Qwen3.5-27B-GGUF` Q4_K_M (~16 GB) into
      `models/`. Bandwidth budget.
- [ ] Download `z-lab/Qwen3.5-27B-DFlash` safetensors into
      `models/` (or a sibling dir; it's not a GGUF).
- [ ] Skim `delta_net_chunked.cpp` (237 LOC) and list every
      ggml op it calls; cross-check each against ggml-opencl
      `supports_op`. This is the gating item for whether
      hybrid-on-OpenCL is tractable without writing a new
      kernel.
- [ ] Confirm `ggml_rope_ext` section-mode support on OpenCL
      (qwen3.5 M-RoPE). If unsupported, plan a CPU shim for
      rope only.
- [ ] Decide on build layout: separate `phase4/` source tree
      (lucebox-shaped standalone, non-libllama) vs. integrate
      into the llama.cpp fork. Lean standalone -- inherits
      lucebox's structure 1:1 and sidesteps llama.cpp
      graph-capture patching.

#### High-level plan (as a checklist, unchanged structure)

1. **Scoping pass** (no code). *Done this session.* ✓
2. Drafter-weight pipeline: port `safetensors_draft.cpp`
   mmap→Win32.
3. Target-weight pipeline: port `gguf_target_loader.cpp`
   mmap→Win32 and confirm `arch=qwen35` handling.
4. Minimum-viable DFlash (chain-verify, no DDTree). Expected AL
   6-7, comparable to session-4 CPU chain-spec in rate but
   higher AL.
5. Add DDTree verify. Target: clear 2× end-to-end.
6. Port any delta-net ops that aren't already on OpenCL (see
   `delta_net_chunked.cpp` survey above).
7. OpenCL perf tuning (f16 intermediate, target_feat copy,
   tree-persist kernel).

### Phase 6 -- Standalone lite harness (flagged for way later)

If Phase 5 closes with llama.cpp's backend model still leaving perf on
the table (specifically the OpenCL buffer model -- no zero-copy on
shared LPDDR5X -- and uncoalesced kernel dispatch), spin a narrow
lucebox-shaped harness (~2000 LOC, no libllama link) rather than
graduating to a full runtime like trident. See README Phase 6 for
scope. Lever: `CL_MEM_USE_HOST_PTR` / `clSVMAlloc` /
`cl_qcom_ion_host_ptr` as allocation invariants, plus fused kernels
and direct QAIRT↔OpenCL ION-buffer handoff for the NPU draft path.

### Phase 7+ -- MoE targets (Qwen3.6-35B-A3B), explicitly deferred

Holding until the fundamentals land on dense/hybrid. Rationale:

- We're doing fundamental exploration on niche hardware (X2 Elite
  Extreme is an almost unstudied platform for spec decode). Stacking
  architectural novelty (MoE expert routing) on top of platform
  novelty multiplies the unknowns without adding information.
- MoE expert routing is a **new op class on NPU**. Qualcomm's shipped
  sample ops don't cover it; custom-op development would be an
  unrelated rabbit hole that delays the platform-characterization
  results we're actually trying to get.
- Spec-decode research on MoE is still unsettled in the literature --
  expert-miss rate adds a second acceptance axis beyond token-accept,
  and the interactions with tree-verify/DFlash aren't
  well-characterized.
- Our 48 GB LPDDR5X has room for Qwen3.6-35B-A3B Q5_K_M (reference
  numbers in `gguf_models/LOCAL_LLM_NOTES.md`), so the only thing
  deferring Phase 7 costs us is research novelty -- not hardware
  reach.

What triggers unblocking Phase 7:

- DFlash+DDTree on Qwen3.5 hybrid target has a stable baseline
  (Phase 4 closed).
- NPU drafting has a stable zero-copy pipeline on pure-attention
  (Phase 5 closed, regardless of whether it wins vs CPU-spec).
- At least one of (a) Phase 6 lite harness lands, or (b) we can
  articulate a concrete bandwidth-budget model that predicts MoE
  expert routing's cost on LPDDR5X.

Then Phase 7 is additive: reuse the existing DFlash target-loader
path (qwen3.6 GGUF → extend `gguf_target_loader.cpp` with MoE tensor
naming), add expert-selection to the target graph, and treat it as
an incremental A/B vs Phase 4's 27B result.

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
