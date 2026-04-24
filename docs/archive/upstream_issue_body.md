### Summary

On Windows ARM64, a default `GGML_CPU_KLEIDIAI=ON` build crashes with
`STATUS_ILLEGAL_INSTRUCTION` (`0xC000001D`) the first time a Q8_0
matmul is dispatched. The process dies before any log output can be
flushed. Setting `GGML_KLEIDIAI_SME=0` at runtime avoids the crash.

### Root cause

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` line 159-161:

```cpp
#else
    return 1;
#endif
```

`detect_num_smcus()` returns `1` unconditionally on every platform
that is neither `__linux__` nor `__APPLE__` (i.e. Windows). With
`cpu_has_sme` true at compile time (the build enabled SME), this
sets `sme_cores = 1`, ORs `CPU_FEATURE_SME` into `ctx.features`, and
`ggml_kleidiai_select_kernels_q8_0` picks the SME kernel. The first
call into that kernel faults.

First bad commit: `0cd4f4720` (PR #20070, "kleidiai : support for
concurrent sme and neon kernel execution"). Prior versions either
didn't enter this path or returned 0 by default.

### Reproducer

Hardware: Snapdragon X2 Elite Extreme (X2E94100, Qualcomm Oryon v2),
18 cores. Windows 11 Home, build 28000. llama.cpp HEAD `cf8b0dbda`
(b8861). Build uses clang 22.1.3 from an MSVC `vcvarsarm64` shell
with `GGML_CPU_KLEIDIAI=ON`.

```
.\build\bin\llama-bench.exe -m Qwen3-0.6B-Q8_0.gguf -t 18 -p 128 -n 32 -r 1
```

Exit: `0xC000001D` (`STATUS_ILLEGAL_INSTRUCTION`). Zero stdout, zero
stderr.

Setting `GGML_KLEIDIAI_SME=0` at runtime makes the same command
succeed (dispatches the SVE/I8MM/DOTPROD KleidiAI kernel instead).
Setting `GGML_KLEIDIAI_SME=18` (force SME on) fails identically to
the default.

### What we verified (not just a "my CPU lacks SME" report)

The Oryon v2 in this machine **does** implement base SME, and
Windows 11 build 28000 has user-mode ZA-tile state enablement
working. We wrote a small native-asm probe that executes `smstart
sm`, `smstart za`, and `zero {za}` under a vectored exception
handler:

```
probe: smstart sm           OK
probe: smstart za           OK
probe: zero za (ZA tile op) OK
```

All three pass cleanly. So the crash is not "no SME on this box" —
it's specifically inside KleidiAI's SME matmul kernel, which
evidently uses at least one encoding (likely an SME2 instruction)
that this implementation of SME does not have. Qualcomm has not
publicly confirmed the full SME2 feature set on Oryon v2.

Windows `IsProcessorFeaturePresent` reports `PF_ARM_SVE=1`,
`PF_ARM_SVE2=1`. The `PF_ARM_SME*` IDs are not in the publicly
documented Windows SDK headers yet, so we can't cleanly cross-check
what the OS claims about SME feature sub-flags.

### Proposed fix

Replace the `#else return 1;` fallthrough with `return 0;` (make
Windows opt-in rather than opt-out for SME dispatch), with a runtime
probe added later as Windows ships `PF_ARM_SME*` constants or as
KleidiAI gains per-kernel feature-bit gating.

A minimal patch (what we carry locally):

```diff
-#else
-    return 1;
-#endif
+#else
+    // Windows: return 0 until a proper runtime SME probe lands.
+    // Upstream's `return 1;` falsely claims a Streaming Mode Compute
+    // Unit is present, causing KleidiAI to dispatch its SME kernel
+    // and fault on hardware that implements base SME but not the
+    // specific SME2 encodings KleidiAI uses (observed on Oryon v2 /
+    // Snapdragon X2 Elite Extreme on Windows 11). Users can still
+    // opt in via `GGML_KLEIDIAI_SME=<N>`.
+    return 0;
+#endif
```

With that patch applied, the `cpu-kleidiai` build runs cleanly by
default and delivers measurable PP uplift on Q8_0 models via the
SVE/I8MM/DOTPROD KleidiAI path (e.g. +16-22% PP at t=18 on
Qwen3-1.7B Q8_0 vs a stock non-KleidiAI CPU build). Happy to open a
PR if the maintainers agree with the direction.

### Side note

The `system_info:` banner from llama-cli prints `SME = 1` on this
machine because `ggml_cpu_has_sme()` is a compile-time check. That
banner is not load-bearing evidence of runtime SME availability on
any platform (the runtime probe in `arch/arm/cpu-feats.cpp` is
`__linux__`-only). Worth noting in the troubleshooting docs so other
users don't mis-diagnose their own SME crashes as "the CPU claims
SME so this must be working."
