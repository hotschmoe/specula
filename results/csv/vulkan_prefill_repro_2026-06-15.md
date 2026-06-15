# Vulkan prefill collapse — A2 repro data (clean)

build `e37abd6b5` (9617), AC / Balanced, `-fa 0` pinned, `llama-bench`.
Device: Adreno X2-90, Qualcomm Adreno Vulkan Driver, Vulkan API 1.4.295,
driverVersion 0.863.0 (build b3549aaa68), uma=1 fp16=1 bf16=0 coopmat=KHR.
Raw: `output/vulkan_prefill_repro_2026-06-15.md`.

## Prompt-length cliff — Qwen3-4B Q4_0, Vulkan -ngl99 -t16

| test | t/s |
|---|---:|
| pp8   | 117.68 |
| pp32  | 16.24 |
| pp64  | 35.47 |
| pp128 | 7.84 |
| pp256 | 6.56 |
| pp512 | 6.36 |

Throughput collapses ~18× as prompt grows (should increase). Decode is
unaffected: tg128 = 36.55 t/s.

## ubatch does not matter (pp512, -ngl99)

| n_ubatch | pp512 t/s |
|---|---:|
| 256 | 6.26 |
| 512 | 6.17 |

## Cross-model (not Qwen-specific) — Llama-3.2-3B Q4_0, Vulkan -ngl99

| test | t/s |
|---|---:|
| pp128 | 11.57 |
| pp512 | 7.90 |

## Cross-backend scale (Qwen3-4B Q4_0 pp512, -fa 0)

| backend | pp512 t/s | vs Vulkan |
|---|---:|---:|
| Vulkan -ngl99      | 6.36 | 1× |
| CPU -ngl0          | 375.97 | 59× |
| OpenCL -ngl99 -ub512 | 587.92 | 92× |
