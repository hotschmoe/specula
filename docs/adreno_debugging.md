# Adreno X2-90 Vulkan debugging

Session start: 2026-04-19. Living doc for the Adreno Vulkan correctness +
performance investigation. Pick up here after the session break.

## TL;DR of where we are

- `-Preset vulkan` build **configures, links, and runs** — llama-bench
  completes without crashing.
- Device enumeration is correct: `Qualcomm Adreno X2-90 GPU` on the
  native Qualcomm Adreno Vulkan driver. Not a software fallback.
- `llama-cli` text generation on Adreno produces **garbled token salad**,
  not coherent English. Same model on CPU (same binary / same build
  tree) produces coherent text.
- `llama-bench` on Adreno gives **PP 137 t/s, TG 102 t/s** on Qwen3-0.6B
  Q8_0. CPU baseline on the same model is **PP ~826 t/s, TG 111 t/s**.
  Adreno is **6× slower on prompt processing** than CPU — backwards.
- Both facts together point at a broken compute path, most likely in
  the fp16 / cooperative-matrix matmul kernels, not the runtime wiring.

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

## Top suspects (ranked)

1. **KHR_coopmat fp16 matmul on Adreno.** llama.cpp has had multiple
   correctness regressions on this path for Adreno specifically; it
   expects certain accumulation semantics that some drivers don't honor
   the same way. Also the first thing that explains garbled output
   *plus* low PP throughput in one bug.
2. **General fp16 shader path** (broader than just coopmat). Lower
   probability but covered by the same escape hatch.
3. **Unified-memory buffer aliasing / sync.** `uma: 1` means the GPU
   shares host RAM; if any shader is reading weights before the host
   write has been made visible, you'd see garbage. Less likely because
   the same path works on other backends with UMA (Apple Silicon).
4. **Driver bug in Adreno Vulkan driver** (outside our control). Lowest
   priority because we can't fix it — but if a known-bad driver
   version, a driver update from Qualcomm is the fix. Worth noting the
   exact driver string in the bench output for any upstream issue we
   file.

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
