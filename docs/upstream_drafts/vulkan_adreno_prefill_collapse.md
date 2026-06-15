<!-- DRAFT upstream issue for github.com/ggml-org/llama.cpp — NOT yet filed.
     Post under the user's GH identity. Fill the COOPMAT CONTROL line first. -->

# Vulkan: dense prefill collapses ~18× with prompt length on Adreno X2-90 (Snapdragon X2 Elite); decode unaffected

## Summary

On the native Qualcomm Adreno Vulkan driver (Adreno X2-90, Snapdragon X2
Elite Extreme, Windows 11 ARM64), **prompt processing throughput
collapses as the prompt grows** — from ~118 t/s at `-p 8` to ~6.4 t/s at
`-p 512` for a 4B model — whereas token generation is unaffected
(~36 t/s). Prefill is ~60–90× slower than the OpenCL and CPU backends on
the *same* machine and model, which makes the Vulkan backend unusable for
any non-trivial prompt. The effect is independent of flash-attention,
`-ub`/ubatch, and model architecture.

## Environment

- GPU: Qualcomm Adreno X2-90, `uma=1 fp16=1 bf16=0`, warp 64,
  `matrix cores: KHR_coopmat`
- Driver: Qualcomm Technologies Inc. Adreno Vulkan Driver, Vulkan API
  **1.4.295**, driverVersion **0.863.0** (Driver Build b3549aaa68)
- OS: Windows 11 ARM64 (native, not WSL)
- llama.cpp: build `e37abd6b5` (9617), `-DGGML_VULKAN=ON`, clang via
  vcvarsarm64
- Tool: `llama-bench`, `-fa 0` pinned

## Reproduce

```
llama-bench -m Qwen3-4B-Q4_0.gguf -p 8,32,64,128,256,512 -n 0 -r 2 -ngl 99 -t 16 -fa 0
```

Observed (Qwen3-4B Q4_0, `-ngl 99`):

| test | t/s |
|---|---:|
| pp8   | 117.68 |
| pp32  | 16.24 |
| pp64  | 35.47 |
| pp128 | 7.84 |
| pp256 | 6.56 |
| pp512 | **6.36** |
| tg128 | 36.55 (unaffected) |

Throughput should *increase* with prompt length; instead it drops ~18×.

## It is not the usual suspects

- **Not flash-attention:** measured with `-fa 0`; `-fa auto`/`-fa 1`
  give the same ~6.4 t/s.
- **Not ubatch:** `-ub 256` → 6.26, `-ub 512` → 6.17.
- **Not model-specific:** Llama-3.2-3B Q4_0 collapses the same way
  (pp128 11.57 → pp512 7.90).
- **Prefill-specific:** decode (tg) is fine.

## Scale vs other backends (same machine, Qwen3-4B Q4_0, pp512, -fa 0)

| backend | pp512 t/s |
|---|---:|
| Vulkan `-ngl 99`        | 6.36 |
| CPU `-ngl 0`            | 375.97 (59×) |
| OpenCL `-ngl 99 -ub512` | 587.92 (92×) |

## Coopmat isolation — ruled out

Disabling cooperative-matrix (`GGML_VK_DISABLE_COOPMAT=1`; device then
reports `matrix cores: none`) does **not** recover prefill:
pp128 = 8.25 t/s vs 7.84 with coopmat on — unchanged. So the collapse is
**not** in the `KHR_coopmat` kernel; it is the general (scalar/F16)
large-M matmul path.

## Hypothesis

The fast→slow transition between `pp8` (117) and `pp128`+ (~7) suggests
the batched (M>1) prefill matmul path is taking a pathological route on
this driver, while the per-token (M=1) decode path is fine. Coopmat is
ruled out (above), so the suspect is the **general F16/scalar mul_mat
path for large M**. `-ub` invariance argues against ubatch tiling.

## Ask

Is this a known issue with the Adreno Vulkan driver's coopmat / large-M
matmul path? Happy to run further controls (coopmat off, F16 off,
specific shapes, a RenderDoc/profiler capture) on this hardware — there
is very little public llama.cpp data on Snapdragon X2 Elite + Adreno
X2-90, and I can iterate on the device.
