# Adreno X2-90 Vulkan debugging

Session 1 start: 2026-04-19. Session 2 (same day) added the fp16
finding below. Living doc for the Adreno Vulkan correctness +
performance investigation.

## TL;DR of where we are (post-session 2, after manual correctness run)

- **The Adreno Vulkan backend produces garbage text on every env-var
  combination tried**, on this driver. Manual llama-cli runs with
  greedy sampling and a fixed seed on Qwen3-0.6B Q8_0 returned
  incoherent token salad for B0 (baseline), B1 (`DISABLE_COOPMAT`),
  and B4 (`DISABLE_F16` + `DISABLE_COOPMAT` + `DISABLE_COOPMAT2`).
  The CPU build on the same model / seed / prompt produced the
  expected `[Start thinking] Okay, the user is asking...` Qwen3 chain.
- **Perf and correctness diverged.** `DISABLE_F16` gives a ~30× PP
  speedup (20 → 600 t/s, table below), but the fast path is fast
  *and wrong*. Bench perf alone is not a correctness signal on this
  backend. Earlier session-2 TL;DR that suggested locking in B3/B4
  as the Adreno baseline is retracted.
- **Memory-accounting underflow** at shutdown:
  `unaccounted | 17592186039033` (~2⁴⁴) in the Vulkan memory
  breakdown. Classic size_t underflow — suggests the Vulkan backend
  mis-accounts buffer state on this driver, not just a shader-level
  issue.
- **Decision point reached.** Three distinct shader-path configs all
  produce garbage, plus a memory-accounting bug at teardown ⇒ the
  bug is structural or driver-level, not a single kernel. Options:
  (a) try one more env-var lever (`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`)
  since `int dot: 1` is active on the device, (b) bisect llama.cpp to
  a Vulkan-on-Adreno known-good commit, (c) pivot primary GPU
  attention to OpenCL. **Plan of record: (c), with (a) as one cheap
  manual sanity check before giving up.**
- **Vendor-path note.** Qualcomm engineers actively maintain
  llama.cpp's `ggml-opencl` backend for Adreno; the Vulkan backend is
  generic and Qualcomm does not tune it. The garbage-out behavior on
  Vulkan doesn't surprise the vendor-support picture.
- **Tooling caveat discovered during manual runs.** This llama.cpp
  build's `llama-cli` silently ignores `-no-cnv` and always enters
  conversation mode; it explicitly suggests using `llama-completion`
  instead, which is not in our build's tool set. To get a scripted
  correctness assay we need to rebuild with a wider `LLAMA_BUILD_TOOLS`
  (or use `llama-server` over HTTP).

## Adreno Vulkan device info (as reported by llama.cpp)

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Qualcomm(R) Adreno(TM) X2-90 GPU
  (Qualcomm Technologies Inc. Adreno Vulkan Driver)
  | uma: 1
  | fp16: 1
  | bf16: 0
  | warp size: 64
  | shared memory: 32768
  | int dot: 1
  | matrix cores: KHR_coopmat
```

Extensions the builder saw available at configure time:
`GL_KHR_cooperative_matrix`, `GL_NV_cooperative_matrix2`,
`GL_EXT_integer_dot_product`, `GL_EXT_bfloat16` (reported by glslc; not
all are necessarily usable at runtime on Adreno).

## What the numbers imply

- **TG on Adreno ≈ TG on CPU (102 vs 111 t/s).** At 0.6B Q8_0 (~0.6 GB
  weights) on this 228 GB/s UMA system, the bandwidth-limited ceiling
  for single-stream TG is ~380 t/s. Both backends hit ~27–30% of that
  ceiling, which is plausible for a cold-cache single-sequence run.
  This is not where the bug is showing up; at small model sizes TG is
  bandwidth-bound and indifferent to which engine does the math.
- **PP on Adreno is wildly low.** 137 t/s is worse than CPU on a 0.6B
  model where Adreno's compute should dominate by 5–10×. This is the
  loud signal. Combined with the garbled output, the working hypothesis
  is that the Vulkan matmul kernel used for the large GEMMs in prompt
  processing is both **wrong and slow** on Adreno — likely the
  KHR_coopmat path, since that's the newest and most backend-specific
  code.

## Top suspects (ranked) — updated after session 2

1. **General fp16 shader path on Adreno Vulkan.** *Promoted to #1.*
   Matrix below shows PP jumps ~30× when fp16 entry points are
   disabled; coopmat's state (on or off) is irrelevant once that
   happens. Consistent with the Adreno Vulkan driver either
   miscompiling fp16 matmul shaders or falling back to a very slow
   path for them. Whether this is the same as the garble-output bug
   needs a separate correctness run (pending).
2. ~~**KHR_coopmat fp16 matmul on Adreno.**~~ *Ruled out by session 2
   matrix.* Disabling `GGML_VK_DISABLE_COOPMAT` alone (B1/B2)
   produced ~0 change vs baseline: PP128 37.5 → 38.8, PP512 20.4 →
   21.0. The coopmat kernel is not the hot path — or at least isn't
   the cause of the pathology.
3. **Unified-memory buffer aliasing / sync.** `uma: 1` means the GPU
   shares host RAM; if any shader is reading weights before the host
   write has been made visible, you'd see garbage. Less likely
   because the same path works on other backends with UMA (Apple
   Silicon), and the fp16-off configs get fast + stable numbers that
   are inconsistent with a sync bug.
4. **Driver bug in Adreno Vulkan driver** (outside our control). Now
   more plausibly the root cause: the driver-string reported is
   `Qualcomm Technologies Inc. Adreno Vulkan Driver` and the fp16
   GEMM kernels specifically misbehave. Worth capturing the exact
   version for any upstream issue we file.

## Escape hatches in llama.cpp (env vars)

These are runtime-only — no rebuild required. Set in PowerShell via
`$env:NAME = '1'` before invoking the binary; clear with
`Remove-Item env:NAME`.

- `GGML_VK_DISABLE_COOPMAT` — disable `KHR_cooperative_matrix` matmul
  path. Falls back to a scalar/subgroup matmul shader.
- `GGML_VK_DISABLE_COOPMAT2` — disable `NV_cooperative_matrix2`
  (separate, NV-only extension; probably a no-op on Adreno but cheap to
  set for completeness).
- `GGML_VK_DISABLE_F16` — disable fp16 shader entry points, force fp32
  shaders. Significant performance hit if it works, but a useful
  signal.
- `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` — disable the integer dot
  product extension (used for some quant paths). Only relevant if we
  still see garbage at quant boundaries.
- `GGML_VK_VISIBLE_DEVICES` — pin which Vulkan device is used. Single
  device on this machine, but useful sanity check.

(Exact name may drift between llama.cpp versions. At the top of any
debug session, grep `ggml/src/ggml-vulkan/` for `getenv` to confirm.)

## Test matrix for next session

Goal: isolate the layer at which correctness returns, and whether PP
recovers when it does.

All runs use `Qwen3-0.6B-Q8_0.gguf` first (small and fast) and then
repeat at `Qwen3-8B-Q4_K_M.gguf` once the small-model config is clean.

### Phase A — correctness

Deterministic prompt, greedy sampling, short generation. Pass = coherent
English token stream.

Template:
```powershell
$env:<VAR> = '1'  # per row
.\llama.cpp\build-vulkan\bin\llama-cli.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -ngl 99 -no-cnv --temp 0 --seed 1 2>&1 |
  Tee-Object .\results\adreno-corr-<label>.log
Remove-Item env:<VAR>  # clean up between runs
```

| Row | Env vars set                                   | Hypothesis                                | Pass? |
|:---:|:-----------------------------------------------|:------------------------------------------|:------|
| A0  | *(none — baseline, reproduce the garbling)*    | Establishes failure signature              |       |
| A1  | `GGML_VK_DISABLE_COOPMAT=1`                    | Coopmat kernel is the correctness bug      |       |
| A2  | `GGML_VK_DISABLE_COOPMAT=1`, `..._COOPMAT2=1`  | Belt-and-braces kill both coopmat paths    |       |
| A3  | `GGML_VK_DISABLE_F16=1`                        | fp16 shader path at large is the problem   |       |
| A4  | all three above set                            | Maximally conservative shader path          |       |
| A5  | `-ngl 0` (no env vars)                         | Control — CPU path via Vulkan binary; should match CPU baseline |   |

Record for each row: first ~40 tokens of output (paste into the log),
pass/fail, any Vulkan warnings in stderr.

### Phase B — performance (only for rows that pass correctness)

Use `llama-bench` to get clean numbers; repeat with the same env vars
that passed Phase A.

```powershell
$env:<VAR> = '1'
.\llama.cpp\build-vulkan\bin\llama-bench.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -ngl 99 -p 32,128,512 -n 16,64,128 -r 3 2>&1 |
  Tee-Object .\results\adreno-perf-<label>.log
Remove-Item env:<VAR>
```

Then repeat with `Qwen3-8B-Q4_K_M.gguf` to see whether the PP gap
closes at a more GPU-realistic model size (8B has much larger GEMMs
per prompt token — if PP is low there too, the issue is per-kernel
throughput, not launch overhead on small GEMMs).

Record columns (from llama-bench output): model, test (pp32, pp128,
pp512, tg16, tg64, tg128), t/s, stddev. Target sheet:

| Model        | Config      | pp32 | pp128 | pp512 | tg16 | tg64 | tg128 |
|--------------|-------------|-----:|------:|------:|-----:|-----:|------:|
| 0.6B Q8_0    | A0 baseline |  137 |   ?   |   ?   |  102 |   ?  |   ?   |
| 0.6B Q8_0    | A1 nocoop   |   ?  |   ?   |   ?   |   ?  |   ?  |   ?   |
| ...          | ...         |  ... |  ...  |  ...  |  ... |  ... |  ...  |
| 8B Q4_K_M    | A0 baseline |   ?  |   ?   |   ?   |   ?  |   ?  |   ?   |

### Phase C — interpretation

- If any Phase A row passes: we have a known-good Adreno config. Use it
  as the Vulkan baseline going forward; open upstream issue against
  llama.cpp with driver version string + failing kernel.
- If none of Phase A rows pass: the problem is deeper (driver or a
  llama.cpp Vulkan code path we haven't disabled). Options: downgrade
  llama.cpp to an older known-good commit for Adreno (bisect), or
  switch to OpenCL/Adreno as the primary GPU backend.
- If Phase A passes but PP stays low in Phase B: per-shader throughput
  is the problem — profile via RenderDoc or Qualcomm Snapdragon
  Profiler, or compare specific matmul shapes against Qualcomm's
  published Adreno matmul numbers to see where the gap is.

## Reference data from this machine

From `gguf_models/LOCAL_LLM_NOTES.md`, CPU (18 threads, KleidiAI off):

| Model             | Quant   | PP t/s | TG t/s |
|-------------------|---------|-------:|-------:|
| Qwen3-4B          | Q4_K_M  | 248    | 42     |
| Gemma-4-26B-A4B   | Q4_K_M  | 168    | 31     |
| Qwen3.6-35B-A3B   | Q5_K_M  | 145    | 29.6   |

For this session (Qwen3-0.6B Q8_0, CPU 18t from earlier smoke test):
PP ~826 / TG 111. Any Adreno config worth considering must at least
match CPU TG and ideally clear CPU PP by 3–5×.

## Session 2 results (2026-04-19)

### Blocker found: llama-cli hang

`llama.cpp\build-vulkan\bin\llama-cli.exe` does not exit cleanly after
generation finishes on this machine, even with `-no-cnv --temp 0 --seed
1 -n 64`. The process hangs; scripted A0–A5 correctness runs couldn't
proceed. Work pivoted to `llama-bench` (perf signatures) as the
primary signal, with manual llama-cli runs reserved for one-off
human-eyeballed correctness checks. A rebuild that includes
`llama-perplexity` would restore a clean scripted correctness assay.

### Phase B results — Qwen3-0.6B Q8_0 (`llama-bench -p 128,512 -n 64 -r 2`)

Script: `scripts/adreno_bench_matrix.ps1`. Per-row logs in
`results/adreno-perf-B*.log`. Driver string per row: `Qualcomm
Technologies Inc. Adreno Vulkan Driver`.

| Row | Env vars                                                   | fp16 | matrix_cores | PP128 t/s   | PP512 t/s   | TG64 t/s     |
|:---:|:-----------------------------------------------------------|:----:|:-------------|------------:|------------:|-------------:|
| B0  | *(none — baseline)*                                        |  1   | KHR_coopmat  | 37.5 ± 1.6  | 20.4 ± 0.8  | 104.8 ± 1.0  |
| B1  | `GGML_VK_DISABLE_COOPMAT=1`                                |  1   | none         | 38.8 ± 1.0  | 21.0 ± 0.6  |  99.9 ± 2.2  |
| B2  | `GGML_VK_DISABLE_COOPMAT=1` + `..._COOPMAT2=1`             |  1   | none         | 40.4 ± 3.1  | 20.6 ± 0.1  |  98.6 ± 3.3  |
| B3  | `GGML_VK_DISABLE_F16=1`                                    |  0   | KHR_coopmat  | **599.8 ± 6.2** | **604.2 ± 1.2** | 100.3 ± 0.7 |
| B4  | `DISABLE_F16` + `DISABLE_COOPMAT` + `DISABLE_COOPMAT2`     |  0   | none         | **593.4 ± 9.4** | **599.5 ± 7.5** | 101.1 ± 1.7 |
| B5  | *(none)*, `-ngl 0` (CPU path via Vulkan binary)            |  1   | KHR_coopmat  | 58.7 ± 1.0  | 30.9 ± 0.5  | 145.9 ± 0.5  |

CPU build reference on the same model (from session 1, 18 threads):
**PP ~826 / TG 111**.

### What the matrix says

- **Hypothesis rank flipped:** fp16 shaders are the problem; coopmat
  alone is clean (B1/B2 ≈ B0). If we had only run the originally
  ranked-#1 experiment, we would have concluded "not the coopmat
  kernel, issue goes deeper" and missed the fp16 lever entirely.
- **PP scaling:** on the broken path (B0–B2) PP actually *decreases*
  from pp128 to pp512 (37 → 20), which is not a normal compute curve —
  fp16 shader is either spilling, recompiling per shape, or hitting
  some size-dependent fallback. On the fp32 path (B3/B4) PP is flat
  ~600 across shapes, as expected for a compute-bound kernel.
- **TG is not a useful signal here.** At 0.6B Q8_0, TG is
  bandwidth-bound at ~100 t/s across all configs including CPU
  (145 t/s via ngl=0 inside the Vulkan binary). TG will only become
  informative on larger models where KV + weights stress bandwidth
  differently per backend.
- **B5 (ngl=0) is weirdly fast** (145 TG vs 111 CPU-build). Inside
  the Vulkan binary the CPU path picks up something — either a
  different thread count default or different ggml-cpu codegen than
  `build-cpu`. Worth a footnote, not a priority.

### Adreno status going forward

B3/B4 correctness **failed** in manual testing (see Correctness
section below). Fp16 was not the only broken kernel; disabling every
documented escape hatch short of integer-dot-product still produced
garble. Vulkan on this Adreno driver is unusable for correct
inference regardless of env-var tuning we have tried.

Decision: **pivot primary GPU attention to OpenCL** (the
vendor-maintained path). Leave Vulkan built so we can revisit
cheaply after a Qualcomm driver update, but don't sink more tuning
hours into it. Single remaining cheap check: `DISABLE_INTEGER_DOT_PRODUCT`
(manual), then drop it.

### Correctness results (manual llama-cli runs, 2026-04-19)

Prompt: `"The Snapdragon X2 Elite Extreme is"`, greedy sampling
(`--temp 0 --seed 1 -n 64`), model `Qwen3-0.6B-Q8_0.gguf`. Note
that `-no-cnv` is silently ignored in this llama.cpp build — each
run landed in conversation mode and took the prompt as the first
user turn; that was consistent across CPU and Vulkan runs, so
doesn't confound the comparison.

| Config | Output (first ~20 tokens)                                              | Verdict |
|:-------|:-----------------------------------------------------------------------|:--------|
| B0 baseline | `iciesäre有条件ebraoramesarapixelewith жеinizquel락iage...`       | 🔴 garble |
| B1 DISABLE_COOPMAT | `iciesäre有条件ebraoramesarapixelewith жеinizquel清明...`   | 🔴 garble |
| B4 all three disabled | `weepMOST TSR下发ONGLinjaermland cheapestilly<<<<...`   | 🔴 garble |
| CPU build | `[Start thinking] Okay, the user is asking about the Snapdragon X2 Elite Extreme. First, I need to confirm if this is a real product...` | ✅ coherent |

Full raw session logs (uncaptured to file, taken from terminal
scrollback) — summarized here. Stands as evidence for the pivot.

### Memory-accounting red flag

At Vulkan process shutdown, llama.cpp printed:

```
llama_memory_breakdown_print: | memory breakdown [MiB] ...
  - Vulkan0 (Qualcomm(R) Adreno(TM) X2-90 GPU) | 28490 = 28490 + (5382 =   604 +    4480 +     298) + 17592186039033
  - Host                                       |                   241 =   157 +       0 +      84
```

`unaccounted = 17592186039033 MiB ≈ 2⁴⁴` — a wraparound. The Vulkan
backend's size_t arithmetic around buffer accounting underflows on
this driver. That isn't a clean kernel bug; something more
structural about how the backend is tracking allocations is off.
Worth including in any upstream bug report we open.

### Manual correctness check (paste output back into the PR / log)

User to run the following by hand in PowerShell (llama-cli hangs
after generation — just Ctrl-C when the next-line appears to stall,
the preceding generated text is already flushed):

```powershell
# B0 -- baseline (expected: garble)
.\llama.cpp\build-vulkan\bin\llama-cli.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -ngl 99 -no-cnv --temp 0 --seed 1

# B3 -- DISABLE_F16 (expected: coherent if fp16 was the bug)
$env:GGML_VK_DISABLE_F16 = '1'
.\llama.cpp\build-vulkan\bin\llama-cli.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -ngl 99 -no-cnv --temp 0 --seed 1
Remove-Item env:GGML_VK_DISABLE_F16

# B4 -- all three disabled (expected: same as B3)
$env:GGML_VK_DISABLE_F16      = '1'
$env:GGML_VK_DISABLE_COOPMAT  = '1'
$env:GGML_VK_DISABLE_COOPMAT2 = '1'
.\llama.cpp\build-vulkan\bin\llama-cli.exe `
  -m .\models\Qwen3-0.6B-Q8_0.gguf `
  -p "The Snapdragon X2 Elite Extreme is" `
  -n 64 -ngl 99 -no-cnv --temp 0 --seed 1
Remove-Item env:GGML_VK_DISABLE_F16,env:GGML_VK_DISABLE_COOPMAT,env:GGML_VK_DISABLE_COOPMAT2
```

Record per run: coherent vs garble, first ~40 tokens of output.

## Upstream prior art to search before filing

If we end up wanting to file: check llama.cpp GitHub issues and
discussions filtered to `vulkan adreno` and to `coopmat adreno`.
Confirmed-similar reports make a better bug report and may already have
workarounds or fixed commits to cherry-pick.

Specific searches worth running:
- llama.cpp issues: "Adreno X2", "Adreno garbage output vulkan",
  "coopmat Adreno"
- llama.cpp discussions: "Snapdragon Vulkan"
- Qualcomm Adreno Vulkan driver release notes for the version string
  reported above.
