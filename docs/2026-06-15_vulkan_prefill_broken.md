# 2026-06-15 — Vulkan prefill collapses on Adreno X2-90 (A2)

Session 36. Re-checked the session-35 "Vulkan prefill broken (6.36 t/s)"
finding against the `-fa` confound that explained the OpenCL "regression"
(see `docs/2026-06-15_flash_attn_prefill_slowdown.md`). **It is NOT an
`-fa` artifact** — pinning `-fa 0` leaves Vulkan prefill at 6.36 t/s.
Vulkan prefill is genuinely broken on the Adreno Vulkan driver. This doc
characterizes it and holds the draft upstream issue
(`docs/upstream_drafts/vulkan_adreno_prefill_collapse.md`).

## Finding

On the Adreno X2-90 native Vulkan driver, **dense prefill throughput
collapses ~18× as prompt length grows** (pp8 117.7 → pp512 6.36 t/s),
while decode is unaffected (tg128 ~36.5). It is:

- **not `-fa`** — measured with `-fa 0` (and matches the `-fa auto`
  number);
- **not ubatch** — `-ub 256`/`512` both ~6.2 t/s;
- **not model-specific** — Llama-3.2-3B collapses the same way
  (pp128 11.6 → pp512 7.9);
- **prefill-specific** — decode (tg) is fine and competitive.

Numbers + env: `results/csv/vulkan_prefill_repro_2026-06-15.md`.

## Why it matters

Same model (Qwen3-4B Q4_0), same machine, pp512: Vulkan 6.36 vs CPU 376
vs OpenCL 588 — Vulkan is **59–92× slower for prefill** while its decode
(36.5 tg) is in the same league as the other backends. So Vulkan is a
viable decode path but unusable for any prompt longer than a few tokens.
That blocks Vulkan as a general backend on WoA, which otherwise has
attractive properties (broad GPU coverage, concurrency).

## Hypothesis (for the maintainers, not asserted)

The fast→slow transition between pp8 (117) and pp128+ (~7) points at the
batched (M>1) prefill matmul path taking a pathological route on this
driver, rather than the per-token decode path (which is fine). `-ub`
invariance rules out ubatch tiling. **Coopmat is ruled out:** with
`GGML_VK_DISABLE_COOPMAT=1` (device reports `matrix cores: none`)
pp128 = 8.25 vs 7.84 with coopmat on — unchanged. So the suspect is the
general F16/scalar `mul_mat` path for large M, not the cooperative-matrix
kernel.

## Status

- A2 characterized; coopmat ruled out. Draft issue ready to post under
  the user's GH identity: `docs/upstream_drafts/vulkan_adreno_prefill_collapse.md`
  (parked TODO in `current_status.md`).

Companion: `docs/2026-06-15_flash_attn_prefill_slowdown.md` (A1 spin-off),
`results/csv/backend_refresh_2026-06-12.md`.
