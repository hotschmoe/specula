# AI Hub Model Zoo check for Snapdragon X2 Elite

Probed 2026-04-20 before spinning up the x86 ONNX export.

**Question:** does Qualcomm already publish a pre-compiled Qwen3-0.6B
(or similar small LLM) targeted at Snapdragon X2 Elite? If yes, we
skip the export + AI Hub compile entirely.

**Short answer:** no Qwen3-0.6B, but X2 Elite support does ship for
Qwen3-4B, Llama-v3.2-1B, and likely others. All pre-compiled models use
the **Genie runtime**, not ORT-QNN, which changes the Phase 5
integration story (scoping doc §8 flags Genie as a Phase 5.5 revisit
target).

## Method

- `qualcomm` organisation on Hugging Face Hub is the public mirror of
  AI Hub's Model Zoo. Repos contain a `release_assets.json` and/or a
  README with target-device benchmark tables.
- Probed via HF API `/api/models?author=qualcomm&search=<term>` +
  `/raw/main/release_assets.json` + `/raw/main/README.md`.

## Relevant hits (X2 Elite targeted)

### qualcomm/Qwen3-4B (also Qwen3-4B-Instruct-2507)

- **X2 Elite support confirmed.** `release_assets.json` lists
  `qualcomm-snapdragon-x2-elite` among six chipsets.
- Precision: **w4a16** only published.
- Runtime: **Genie** (QAIRT 2.42.0).
- Direct download URL (~2-3 GB zipped):
  `https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_4b/releases/v0.50.2/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite.zip`
- No published t/s benchmark on X2E in the README (only generic
  chipset list).
- **Use case for specula:** too big to be useful as a draft model
  (4B vs our 8B target; <2× ratio erases spec-decode savings). Decent
  stand-alone "LLM-on-X2E-NPU works" demo.

### qualcomm/Llama-v3.2-1B-Instruct

- **X2 Elite support confirmed, with published t/s numbers.** README
  table:

  | Precision | Chipset | Context | t/s | TTFT |
  |-----------|---------|--------:|----:|-----:|
  | w4a16 | Snapdragon X2 Elite | 4096 | **90.36** | 0.05-1.61 s |
  | w4    | Snapdragon X2 Elite | 4096 | 27.95 | 0.06-1.87 s |

- Runtime: Genie.
- **Draft-size-correct for a Qwen3-8B or Llama-8B target.** But
  Llama tokenizer is incompatible with Qwen3 (different vocab), so
  pairing with our Phase 2 Qwen3-8B-Q4_K_M target for speculative
  decoding is not possible.

### qualcomm/Qwen2.5-1.5B-Instruct

- Pre-compiled, but README only shows **Snapdragon X Elite (X1)**
  benchmarks — not X2 Elite. X1 w4 Genie: 12.99 t/s.
- No `release_assets.json` at HF (unlike Qwen3-4B).
- Could plausibly be recompiled for X2 by Qualcomm later; for now
  treat as X1-only.

### Other Qualcomm org LLMs surveyed

- qualcomm/Qwen2-7B-Instruct — not X2
- qualcomm/Qwen2.5-7B-Instruct — not X2
- qualcomm/Llama-v2-7B-Chat, v3-8B-*, v3.1-8B-*, v3.2-3B-*, etc. —
  X2 support not confirmed in README spot-check.
- qualcomm/Phi-3.5-mini-instruct, Mistral-3B — not checked further.

## Implication for Phase 5

**Primary plan unchanged: produce Qwen3-0.6B ONNX on the x86 machine**
per `docs/phase5_export_on_x86.md`, then compile via AI Hub. The
pre-compiled catalogue doesn't cover our target draft size + Qwen
tokenizer combination.

**New side-project opportunity (Phase 5.X):** drop in
`qualcomm/Llama-v3.2-1B-Instruct` as a draft + pair with a Llama-8B
target. Advantages:

- No compile step; download the .zip and run.
- Published 90 t/s draft speed (vs our CPU Qwen3-0.6B at 111 t/s — NPU
  is a bit slower per-token but frees the CPU for target verification
  in parallel).
- Tests the Genie runtime path end-to-end.

Trade-offs:

- Genie runtime ≠ ORT-QNN; the `NPUSession` wrapper built at step 2
  doesn't apply. Would need a new Genie-based sidecar.
- New target baseline required (Llama-8B-Q4_K_M CPU perf unknown).
- Tokenizer + prompt-format mismatch with the Phase 2 Qwen3 fixtures.

**Genie as a Phase 5 runtime (bigger pivot):** scoping doc §8 marked
Genie as a Phase 5.5 revisit target because "it assumes you shipped
with Qualcomm's model zoo." The zoo now includes X2 Elite, so that
assumption is increasingly safe — Genie may deserve an earlier probe.
If the x86 export + AI Hub compile pipeline turns out to be brittle
after a few more attempts, pivoting to Genie + the existing
qwen3-4b-genie-w4a16 bundle is a faster path to a working NPU LLM on
this hardware.

## Decision recorded

1. Continue with x86 ONNX export as the planned Phase 5 path.
2. Keep `qualcomm/Qwen3-4B-Genie-w4a16` in the back pocket as a "demo
   hardware works" escape valve if the export + compile chain stalls.
3. Keep `qualcomm/Llama-v3.2-1B-Genie-w4a16` in mind as a fast-track
   to a tokenizer-Llama spec-decode experiment, explicitly outside
   the current Phase 5 scope.

## Re-probe when

- `qualcomm/Qwen3-0.6B` appears in the HF Hub `qualcomm` org listing.
- Qwen2.5-1.5B gets an X2 Elite asset added to its `release_assets.json`.
- Genie SDK ships with ORT-QNN-compatible export (unlikely short-term;
  Genie's whole point is a different runtime).
