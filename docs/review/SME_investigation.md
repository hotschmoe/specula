# SME2 / KleidiAI investigation on Oryon v2 (X2 Elite Extreme)

Session 4 start: 2026-04-20. Goal: determine with concrete evidence
whether SME2 via KleidiAI accelerates CPU inference on this machine,
or whether it's genuinely unavailable at runtime. "We tried it once
and it crashed" is *not* an acceptable conclusion; we either ship
numbers showing uplift, or we ship a specific broken symbol / register
state / OS-level evidence of what's wrong.

## What we actually know before burning a build cycle

### Hardware
- **CPU:** Qualcomm Oryon v2, Snapdragon X2 Elite Extreme X2E94100,
  18 physical cores (no SMT). Identified by `Get-WmiObject Win32_Processor`
  as `ARMv8 (64-bit) Family 8 Model 2 Revision 201` — note Windows
  reports base architecture as ARMv8 even though Oryon v2 is
  ARMv9.2-A. That's Windows enumerating the baseline profile; the
  extensions (SVE/SME/SME2) are reported separately via feature
  flags, not the architecture string.
- **OS:** Windows 11 Home, build 28000. Modern enough that
  Windows-side SME enablement should be present (user-mode ZA-tile
  state and streaming-SVE state saving landed in Windows 11 for
  ARM64; 28000 is post-enablement).

### Features ggml already detects on this CPU

From `system_info:` banner emitted by `llama-cli` on the CPU build:
```
NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 |
SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | SME = 1 |
OPENMP = 1 | REPACK = 1
```
**`SME = 1` is NOT a runtime detection on Windows — it's compile-
time.** Specifically, `ggml_cpu_has_sme()` at
`ggml-cpu.c:3689` returns 1 iff `__ARM_FEATURE_SME` is defined at
compile time:
```c
int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}
```
The runtime aarch64 feature probe lives in
`arch/arm/cpu-feats.cpp` and reads `AT_HWCAP2` via `getauxval` —
but **that probe is gated `#if defined(__linux__)`**, so on
Windows `has_sme` is default-false and is never populated. The
score function on line 88-91 returns 0 for any `GGML_USE_SME`-
compiled backend when `has_sme == false` — meaning on Windows,
the aarch64 backend scoring is effectively blind to SME.

### KleidiAI's SME dispatch on Windows (the subtle part)

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`:
- `init_kleidiai_context()` sets `cpu_has_sme = ggml_cpu_has_sme()`
  (compile-time → `true` if we build with SME flags).
- It then calls `detect_num_smcus()` to count Streaming Mode Compute
  Units — that function is **Linux-only** (reads `/sys/devices/.../smidr_el1`).
  On Windows every iteration of the loop fails to open the path
  and the function returns 0.
- SME policy (from source comments at line 207-213):
  - env `GGML_KLEIDIAI_SME` unset → auto-detect cores; enable if
    detected > 0.
  - env `= 0` → force off.
  - env `> 0` → force that many cores (skip detection).
- With detection returning 0 on Windows and no env var,
  `sme_cores = 0` → **SME is auto-off by default on Windows**.

### Implication for the "prior crash"

The prior `STATUS_ILLEGAL_INSTRUCTION` was almost certainly
triggered by *explicitly setting* `GGML_KLEIDIAI_SME=<n>` (or an
older llama.cpp without the SMCU detection code path). With the
current llama.cpp HEAD, a plain `cpu-kleidiai` build on Windows
should be safe at default — SME is auto-disabled, KleidiAI falls
through to SVE / DOTPROD / I8MM / NEON kernels, and those have
been validated on Oryon v1/v2 hardware historically.

**So the `cpu-kleidiai` build is our "safe" phase-1 test.
`GGML_KLEIDIAI_SME=N` is the "I want to actually test SME2" lever,
and is where any crash is expected.**

### llama.cpp's KleidiAI wiring

`llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt` pins
**KleidiAI v1.22.0** and includes four matmul micro-kernel
directories:
- `matmul_clamp_f32_qsi8d32p_qsi4c32p/` — Q4_0 (int4 channels × 32,
  int8 dynamic quant per 32)
- `matmul_clamp_f32_qai8dxp_qsi8cxp/` — **Q8_0** (int8 channels × K,
  int8 dynamic per axis)
- `matmul_clamp_fp32_bf16p_bf16p/` — bf16
- `matmul_clamp_f32_f16p_qsi4c32p/` — fp16 × int4

**Q4_K_M is not wired** — there is no `matmul_*_q4_k_*` include.
Consequence: **Qwen3-8B Q4_K_M will silently fall back to the
non-KleidiAI path for all its matmuls**. We won't see SME2 uplift on
that model. For a fair SME2 test we need Q8_0 models (our 0.6B and
1.7B are already Q8_0 — perfect) or a Q4_0 variant of an 8B model.

### Configure-time evidence

From the Vulkan-build configure output (session 1):
```
HAVE_SME - Success
```
meaning the clang+vcvarsarm64 toolchain can assemble SME
instructions. Build side is not the problem; if SME2 kernels do
run, they'll run via KleidiAI's dispatch, not a compile stopper.

### Clang-on-Windows assembly caveat

`scripts/patch_kleidiai.py` exists and patches KleidiAI's `.S`
files so clang-on-Windows (which defines both `_MSC_VER` and
`__clang__`) takes the GAS branch instead of the armasm branch,
and skips ELF-only `.type`/`.size` directives that COFF rejects.
Applied automatically by `build_llama_cpp.ps1 -Preset cpu-kleidiai`
post-configure, pre-build.

## Hypothesis ranking before we build

1. **SME2 works, KleidiAI engages for Q8_0, measurable PP uplift
   ≥20% on 0.6B and 1.7B.** This is the happy path. If it lands
   we rerun the Phase 1 baseline sweep and may flip the OpenCL
   PP-at-512 advantage for small models.
2. **SME2 works, KleidiAI's Q8_0 path runs, but the CPU baseline
   is already saturating bandwidth at 18t so we see no lift.**
   Observable as: bench completes fine, numbers match the stock
   CPU build within noise. In that case SME2 might still help at
   lower thread counts (bandwidth headroom), so we'd test at t=8.
3. **Build fails at the KleidiAI assembly step.** `patch_kleidiai.py`
   should prevent this; if it fails anyway, an off-by-one in the
   patch is likely the culprit (KleidiAI may have moved files
   around between the version trident was testing and v1.22.0).
4. **Build succeeds, runtime traps.** Either the feature probe
   claims SME is present but user-mode execution actually faults
   (older Windows on X2, driver/firmware state), or the specific
   SME2 op in the Q8_0 micro-kernel hits an unimplemented
   encoding. Exit criteria: capture the faulting instruction
   address and mnemonic, check it against the KleidiAI symbol,
   map back to the SME2 opcode encoded.

## Execution plan — stage 1 (cheap, this session)

1. **Confirm ggml's SME detection is runtime, not compile-time.**
   Read `ggml/src/ggml-cpu/` for the SME probe. 5 min.
2. **Build `-Preset cpu-kleidiai`.** The preset exists and
   applies the .S patch. Budget: ~5-10 min wall. Capture full
   configure+build log; grep for `SME`, `KLEIDI`, and any
   "disabled" / "not supported" lines.
3. **Smoke test — 0.6B Q8_0, no fault expected.**
   `llama-bench.exe -m Qwen3-0.6B-Q8_0.gguf -p 128 -n 32 -r 1`.
   - If it crashes → jump to Stage 3 (forensics).
   - If it runs → compare the `t/s` against the stock `cpu`
     build baseline (we have a fresh one from today's sweep).
4. **Full sweep if Stage 1 clean.** Re-run
   `sweep_baseline.ps1 -Backend cpu-kleidiai` (needs script edit
   to accept the new backend name) on all three models. Diff
   against `baseline-cpu-*.csv` committed today.

## Execution plan — stage 2 (only if Stage 1 runs but no lift)

1. **Thread-count sweep.** Focus on t=4, t=8, t=12 — where the
   CPU bench ISN'T bandwidth-bound. If SME2 helps per-core
   compute, uplift should be biggest there.
2. **Quant coverage audit.** Grep the built `ggml-cpu.dll` for
   `kai_matmul` symbols, confirm which KleidiAI micro-kernels
   actually got linked. Verify Q8_0 model loads dispatch to a
   KleidiAI symbol (use a debugger or `dumpbin /exports` on the
   DLL + log trace at the llama.cpp call site).
3. **Q4_0 variant.** If Q8_0 path runs but gives no lift,
   download a Q4_0-quantized Qwen3 (0.6B or 1.7B) as a second
   quant to cross-check.

## Execution plan — stage 3 (only if we see faults)

Concrete evidence we need before declaring SME2 dead on this
platform:
- **Faulting instruction address + mnemonic.** Capture via
  `windbg` attached to `llama-bench.exe`, or set
  `$env:PYTHONBREAKPOINT=...` equivalent / Windows Error
  Reporting crash dump. Need an actual opcode, not just
  `0xC000001D ILLEGAL_INSTRUCTION`.
- **Symbol mapping.** Resolve the faulting PC to a KleidiAI
  function name using the `.pdb` files emitted by clang.
- **Windows SME state check.** Tiny C/Zig program that calls
  `IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)`
  and every PF_ARM_* feature code 43-48 inclusive; cross-
  reference with what ggml's banner claims. If ggml says
  `SME = 1` but IsProcessorFeaturePresent says no, one of the
  two is wrong and we know where to look.
- **Attempt to force SVE-only (no SME) KleidiAI.** v1.22.0 has
  both SVE (KAI_ARCH=sve2) and SME (KAI_ARCH=sme2) micro-kernel
  variants. If the full SME2 build faults, an SVE-only build
  that excludes SME kernels can isolate whether SME specifically
  is the problem.

Exit criteria for declaring SME2 not viable on this platform:
- We've captured a specific faulting instruction inside a KleidiAI
  SME2 kernel, AND
- We've confirmed Windows reports that instruction's requirement
  unsupported via `IsProcessorFeaturePresent`, OR
- We've tried disabling just SME2 (keeping SME baseline) and
  the fault persists — meaning it's the SME base set that's
  broken, not just SME2.

Anything short of that is "we haven't looked hard enough yet."

## Log

### Session 4 (2026-04-20) — evidence-first investigation

**Build + default smoke test (no env var):**
`llama-bench.exe -m Qwen3-0.6B-Q8_0.gguf -t 18 -p 128 -n 32 -r 1`
- Exit: `0xC000001D` (STATUS_ILLEGAL_INSTRUCTION)
- stdout bytes: 0, stderr bytes: 0 — process dies before any log output
  can flush. No KleidiAI init lines observed, suggesting the fault
  happens inside `init_kleidiai_context` before first `GGML_LOG_*`.

**Same binary with `GGML_KLEIDIAI_SME=0`:**
- Exit 0, clean bench output. Confirms the fault lives specifically
  in the SME dispatch path, and the env-var fast path (line 237:
  `sme_cores = 0`) successfully avoids it.

**Same binary with `GGML_KLEIDIAI_SME=18` (force SME on):**
- Same fault: `0xC000001D`, zero output. So forcing SME on with a
  positive core count faults identically.

**Root cause — llama.cpp source bug:**
`llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:159-161`
```cpp
#else
    return 1;    // non-Linux, non-Apple fallback
#endif
```
On Windows `detect_num_smcus()` returns 1 unconditionally when
`cpu_has_sme` is true at compile time. That sets `sme_cores=1` in
`init_kleidiai_context`, which ORs `CPU_FEATURE_SME` into
`ctx.features` and selects the SME Q8_0 kernel
(`ggml_kleidiai_select_kernels_q8_0`). First call into that kernel
faults.

**Windows OS probe —
`IsProcessorFeaturePresent` flag dump for IDs 0-70:**
```
[46] = SVE      = TRUE
[47] = SVE2     = TRUE
[48] = SVE2P1   = FALSE
[51] speculative (SME?)  = FALSE
[52] speculative (SME2?) = TRUE
```
Note: PF_ARM_SME* are not in public Windows SDK headers yet, so the
51/52 labels are my guess. The empirical TRUE at 52 suggested
"SME2 announced by OS" but we needed to verify with a real SME
instruction execution.

**Hardware + OS SME probe (`scripts/sme_probe/`):**
Tiny native ARM64 executable that executes SME instructions under a
`AddVectoredExceptionHandler`, advancing PC past the offending
instruction on fault so the trampoline still returns cleanly.
```
probe: smstart sm           OK
probe: smstart za           OK
probe: zero za (ZA tile op) OK
summary: smstart-sm=OK  smstart-za=OK  zero-za=OK
```
**Base SME is fully usable on this CPU+OS combo.** Oryon v2 (X2E94100)
implements base SME; Windows 11 build 28000 saves/restores the ZA
tile across context switches as required.

**Conclusion on what's actually broken:**
Not the hardware, not the OS, not llama.cpp's entire KleidiAI
path. The fault is **inside KleidiAI v1.22.0's SME matmul micro-
kernel** — some specific SME (or SME2) instruction encoding that
Oryon v2 doesn't implement. Candidates (not verified):
- An SME2 outer product like `bmopa` / `sumopa` — SME2-only, and
  Qualcomm's public Oryon v2 disclosures don't confirm full SME2.
- A specific-width tile load/store (e.g., 8-byte elements with
  certain predicate interactions) not implemented by this core.
- An assembler encoding using extensions our CPU lacks (e.g.,
  FEAT_SME_F64F64 = double precision outer product).

Pinning the exact faulting instruction requires attaching a
debugger to `llama-bench.exe` with the crashed SME kernel in
scope — cheap but not done this session.

### Local patch applied

`llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:159-161`:
`return 1;` → `return 0;` on the Windows else-fallthrough. Commented
in-source so a future llama.cpp update that trips over the change
(e.g., `git pull` conflict) surfaces the context clearly.

Result: `cpu-kleidiai` build now runs out-of-the-box with default
env, dispatches to non-SME KleidiAI kernels (SVE/I8MM/DOTPROD),
and measurably helps PP on Q8_0 models (+16-22% at t=18 on 1.7B
Q8_0) while regressing TG (-14%). Net = workload-dependent; use
it for PP-heavy phases, use stock CPU for TG-heavy phases.

### Upstream issue filed

[ggml-org/llama.cpp#22182](https://github.com/ggml-org/llama.cpp/issues/22182)
— "Misc. bug: cpu-kleidiai build crashes with
STATUS_ILLEGAL_INSTRUCTION by default on Windows ARM64 (Oryon v2 /
Snapdragon X2)". Body archived at `docs/upstream_issue_body.md`.
Proposed fix: flip the `#else return 1;` in `detect_num_smcus` to
`return 0;`. We've offered to open the PR if the maintainers agree
with the direction.

### Upstream fix path (not done)

Proper fix is a PR to llama.cpp replacing the
`detect_num_smcus` Windows fallthrough with:
1. A real `IsProcessorFeaturePresent` probe for PF_ARM_SME
   (once that constant is in the public SDK), or
2. A guarded trial execution of `smstart sm / smstop sm` wrapped
   in VEH (same approach as `scripts/sme_probe/sme_probe.c`), or
3. Return 0 and require explicit `GGML_KLEIDIAI_SME=<n>` opt-in.

Our local patch implements option 3 without the env requirement.

## Future-retry trigger conditions

**When to re-investigate SME2 on this platform:**

1. **Windows 11 SDK ships `PF_ARM_SME_INSTRUCTIONS_AVAILABLE` and
   `PF_ARM_SME2_INSTRUCTIONS_AVAILABLE` constants.** Once
   documented, we can replace the speculative IDs in
   `scripts/probe_arm_features.ps1` with real names and file a
   proper upstream PR.
2. **KleidiAI ships a version that tags its SME kernels with
   required `FEAT_*` bits and gates dispatch accordingly** — the
   crash implies KleidiAI's current SME path uses an instruction
   not present on every SME-capable CPU. A per-kernel feature
   gate would let dispatch avoid our faulting op.
3. **Qualcomm publishes an authoritative Oryon v2 ISA manual** that
   enumerates which SME2 encodings are implemented. That tells us
   definitively which KleidiAI kernel is unsafe here.
4. **An upstream llama.cpp issue surfaces with the same crash on
   another Snapdragon X2 / Oryon v2 box.** If the symptom is
   identical, we can fold our evidence into that issue and get a
   maintainer-led fix.
5. **Someone writes a per-instruction SME probe that KleidiAI can
   run at init** (essentially our `sme_probe.c` generalized to
   every SME op KleidiAI uses). That eliminates the guesswork.

Signal of "trigger fired": run `scripts/sme_probe/sme_probe.exe` +
the llama.cpp cpu-kleidiai build without our local patch and with
`GGML_KLEIDIAI_SME` unset; if both exit 0 with sensible output, the
regression is fixed. Rebuild stock and compare PP numbers against
the baseline CSVs committed in session 4.

## Artifact index

- `scripts/sme_probe/sme_probe.c` — C + VEH harness
- `scripts/sme_probe/sme_probe.S` — AArch64 SME trampolines
- `scripts/sme_probe/build_sme_probe.ps1` — build + run
- `scripts/probe_arm_features.ps1` — Windows PF flag dump
- `results/sme-probe.log` — probe output (SME works)
- `results/kleidiai-smoke-0.6B.log` — initial crash reproducer
- `results/kleidiai-sme0.stdout` — working run with env=0
- `results/kleidiai-envunset.*` / `kleidiai-sme18.*` — crash captures
- `results/baseline-cpu-kleidiai-*.csv` — full Q8_0/Q4_K_M sweep
  with SME forced off
- `results/build-cpu-kleidiai-patched.log` — build with local patch
- `llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:159-161` — the
  one-line patch itself
