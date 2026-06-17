# Next session — on-device HMX single-tile probe (w4a16 verify + w4a8 validate)

Copy the **KICKOFF PROMPT** block below into a fresh session. Everything it
needs is referenced. The machine is dedicated — DSP crashes are fine; kill and
retry freely.

---

## KICKOFF PROMPT (paste this)

You are continuing frontier NPU work on a Snapdragon X2 Elite Extreme laptop
(Windows 11 ARM64, Hexagon DSP arch **v81**). The big wins are already landed
(Qwen3.6-27B + 35B-MoE run on the Hexagon NPU via llama.cpp). Your job now is
the **perf kernel**: settle, on real silicon, whether a **native w4a16**
(int4-weight × fp16-activation, hardware weight-decompress) HMX matmul exists,
and validate the **w4a8** (int4-weight × uint8-activation → int32) kernel
draft. Both are settled by ONE thing: an **on-device single-tile HMX probe**
compared to a CPU reference. The HMX *matrix* ISA is undocumented, so this is
empirical reverse-engineering. **The machine is yours — crashes are expected;
kill stuck processes and retry. Never run an unguarded HMX kernel in the full
model; probe single tiles first.**

### Read first (context + decisions, in order)
- `specula/docs/llama_hexagon_qwen35_w4a16_plan.md` — THE plan. Read the top
  sections: "REOPENED — native w4a16 primitive PLAUSIBLE", "w4a8 KERNEL —
  compile-complete, 3 on-device unknowns", and the kernel spec. (specula commit
  `d1dd35d` and ancestors.)
- `specula/docs/llama_hexagon_build_setup.md` — exact build/sign/run recipe +
  the `ADSP_LIBRARY_PATH` requirement + WSL/Docker won't-work note.
- Memory `reference_llamacpp_hexagon_npu_works` — the working NPU setup + the
  perf-path correction (do NOT close the w4a16-primitive question).
- `specula/docs/htp_memory_ceiling_problem.md` — the ~10 GB / 4-PD ceiling
  context (why this matters).

### Code references
- **llama.cpp repo:** `C:\Users\hotschmoe\Documents\GitHub\llama.cpp`
  (integration branch `hotschmoe-npu-work`, base master `45cac7c`).
- **w4a8 draft (your starting point):** worktree
  `C:\Users\hotschmoe\Documents\GitHub\llama-int4a16`, branch
  `npu-int4a16-hmx`, commits `2954950` (kernel) + `2aad888` (stubs filled).
  Key file: `ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c`:
  - `core_dot_chunk_int4_int8()` (line ~773) — the integer-HMX inner loop.
  - `repack_q4_0_x4x2_to_int4_tiles()` (~1724) — int4 weight pack (UNKNOWN #1:
    nibble order).
  - `hmx_uh_2x2_elem_off()` (~1782) — int readout offset (UNKNOWN #2: the
    acc(row,col)→(h,w,d) axis assignment; formula itself is verified).
  - Per-block Q4_0 scale (UNKNOWN #3, structural): the single full-K `uh_2x2`
    readout sums over K so per-32-block fp16 scales collapse → needs per-block
    partial readouts (or push scale into `mxmem2` column-bias).
  - Gated by `GGML_HEXAGON_W4A16` (default OFF) + a hard `-1` guard until
    `HTP_W4A16_VALIDATED` is defined — keep this guard; only flip it after a
    single tile matches the CPU reference.
- **fp16 reference kernel (proven, mirror its structure):** same file,
  `core_dot_chunk_fp16()` (~ line 884 region) + `transfer_output_chunk_fp16_to_fp32()`.
- **HMX tile layouts (authoritative):**
  `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\QNN\HTP\core\memory_layout.h`
  (`R4Weights8x4Layout` line 340 = weight pack; `R4Crouton2x2Layout` line 319 =
  `uh_2x2` readout; `R4CroutonLayout` = fp16 8×8×32) and `tile_extract.h`.
- **HMX intrinsics:** `C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07\Tools\target\hexagon\include\hmx_hexagon_protos.h`
  (weight loaders `Q6_weight_ubit/sbit/n/b` ~425; activation `Q6_activation_ub/hf`
  ~371/35; `Q6_mxclracc`/`_hf` 26/35; readout `Q6_mxmem_AR_after_uh_2x2`/`_hf`
  ~970/152). `enable_native_mixed_precision_ops` is in QAIRT
  `include/QNN/HTP/core/optimize_flags.h:65`.

### Build / sign / run (known-good; from the build-setup doc)
Cert already created + trusted: `C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx`
(test-signing is ON). Env for build:
```
$env:OPENCL_SDK_ROOT="C:\Qualcomm\OpenCL_SDK\2.3.2"
$env:HEXAGON_SDK_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0"
$env:HEXAGON_TOOLS_ROOT="C:\Qualcomm\Hexagon_SDK\6.6.0.0\tools\HEXAGON_Tools\19.0.07"
$env:WINDOWS_SDK_BIN="C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0"
cmake --preset arm64-windows-snapdragon-release -B build-w4 -DGGML_HEXAGON_W4A16=ON -DGGML_HEXAGON_HTP_CERT="C:\Users\hotschmoe\Certs\ggml-htp-v1.pfx"
cmake --build build-w4 --target htp-v81 libggml-htp-cat llama-bench llama-cli
```
Run (REQUIRED for any HTP run): set
`$env:ADSP_LIBRARY_PATH="<build-w4>\ggml\src\ggml-hexagon"` (the dir with the
skels + signed `libggml-htp.cat`). Bench model:
`C:\Users\hotschmoe\Documents\Github\specula\models\Qwen3-4B-Q4_0.gguf`,
`--device HTP0 -ngl 99 -fa 0 -t 16`. Baseline to beat: fp16 path **pp512 ~175
t/s** (plateau); Qualcomm w4a16 reference **~2224 t/s** PP.

### The probe (do this — it answers everything on one tile)
Add a tiny **on-device probe op/path** (FARF/`GGML_LOG` dumps) that runs ONE HMX
matmul tile (m=32, k=256, n=32) with KNOWN inputs and dumps the raw output
tile, so you can brute-force the unknowns against a CPU reference computed in
the same process. Iterate ONE unknown at a time:

1. **w4a16 primitive test (highest value):** `Q6_mxclracc_hf` →
   `Q6_weight_ubit`(int4) + `Q6_activation_hf`(fp16) in the SAME packet →
   `Q6_mxmem_AR_after_hf` readout. If the dumped tile matches CPU
   `sum_k(dequant_int4(w)·fp16(a))` (after resolving the weight nibble order +
   readout axis), **the native w4a16 primitive EXISTS** → make it the priority
   (accuracy-free, beats w4a8). If it errors/garbage regardless of layout, w4a16
   is not a mixed primitive → fall back to w4a8.
2. **w4a8 path (the draft):** `Q6_mxclracc` → `Q6_weight_ubit`(int4) +
   `Q6_activation_ub`(uint8) → `Q6_mxmem_AR_after_uh_2x2` (int32). Resolve, vs
   CPU ref: (a) int4 nibble order in `repack_q4_0_x4x2_to_int4_tiles`; (b) the
   acc(row,col)→(h,w,d) axis in `hmx_uh_2x2_elem_off`; (c) the 4-term rescale
   `dot = acc −8·Σq −128·Σv +8·128·K` + per-block scale (start single-block
   K=256 to sidestep #3, then add per-block partial readouts).

When a single tile matches CPU (cos≈1, max-abs-err tiny), define
`HTP_W4A16_VALIDATED`, run the full `llama-bench` on Qwen3-4B-Q4_0, confirm
numerics (cos vs fp16-path build) AND measure PP vs the 175/2224 references.
Use `GGML_HEXAGON_PROFILE=1|2` (output goes to on-device FARF) to confirm the
dequant cost disappears.

### Deliverables + discipline
Commit often on `npu-int4a16-hmx` (w4a8) and a new `npu-w4a16-hmx` branch off
`hotschmoe-npu-work` if the w4a16 primitive is confirmed. Update
`specula/docs/llama_hexagon_qwen35_w4a16_plan.md` + the
`reference_llamacpp_hexagon_npu_works` memory with results (especially: does the
w4a16 primitive exist? final PP numbers?). Clean up build dirs / intermediate
GGUFs to respect disk (8 TB SSD on the Threadripper `192.168.10.5` is available
via `specula/end-to-end/build_server/boxssh.py` if you need room).

### Success criteria
(1) Definitive yes/no on the native w4a16 (int4×fp16) HMX primitive, proven by a
single-tile match. (2) If yes: a w4a16 matmul beating the fp16 175 t/s plateau
on the 4B, numerics cos>0.99 vs fp16. (3) w4a8 single-tile validated (or a clear
blocker). Document which path wins for the 27B/35B.
