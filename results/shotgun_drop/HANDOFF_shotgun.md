# Phase 5.5 Lever C — w4a16/w8a16 shotgun — HANDOFF

**Produced:** 2026-04-23 early UTC on x86 box (`99-Luftballons`), QAIRT SDK
**2.42.0.251225**. Response to `docs/phase5_lever_c_x86_ask.md` §"Primary
ask now — A.6 w8a16" plus the broader "shotgun everything cheap" guidance.

Seven parallel PTQ variants built against the **same fp32 DLC** and
**same Bundle A input_list** (60 samples) — only the quantizer flags
change. Every `.bin` shares the Qualcomm magic-bytes prefix and the
same Step-4 context-binary-generator invocation. Total wall: ~11 min.

## Deliverables

MD5-verified in `SHOTGUN_SUMMARY.txt`. Sizes in (compressed-HTP-binary) bytes.

| # | variant | flags (beyond `--act_bitwidth 16 --input_list …`) | bin MD5 | enc MD5 | bin B |
|---|---|---|---|---|---:|
| 1 | **w8a16-local** | `--weights_bitwidth 8` | `0d40493513120fb4e3965280ed6358f4` | `d95516f4b95c86e75dfb4fdbc30f284a` | 917,909,504 |
| 2 | **w4a16-local-pr** | `--weights_bitwidth 4 --use_per_row_quantization` | `caff709da351b3db12ae5108e4b1ce2a` | `427dad31aded3b422f3b3d78561f31ea` | 620,146,688 |
| 3 | w4a16-local-cle | `--weights_bitwidth 4 --apply_algorithms cle` | `b8d8f3b7a4df9a6825af9b969f631228` | `9b9457c39e4e2139c76bcb9fe0b124b9` | 917,934,080 |
| 4 | **w4a16-local-pr-cle** | `--weights_bitwidth 4 --use_per_row_quantization --apply_algorithms cle` | `caff709da351b3db12ae5108e4b1ce2a` | `427dad31aded3b422f3b3d78561f31ea` | 620,146,688 |
| 5 | w4a16-local-sqnr | `--weights_bitwidth 4 --act_quantizer_calibration sqnr` | `96667934cbf9dfdcbddf2f1fe93f13a9` | `3dda18331e8fff2e11e9c2ee144bef12` | 917,946,368 |
| 6 | **w4a16-local-mse** | `--weights_bitwidth 4 --act_quantizer_calibration mse` | `fe89b7edf7f3dcfdabc78f9772f0f50b` | `af6ded42613fb78bb2f8bc8623285019` | 917,966,848 |
| 7 | **w8a16-local-pr** | `--weights_bitwidth 8 --use_per_row_quantization` | `ebdf9be6ef34576d5c943d47fc484f5f` | `b2ea616a61bd7b927361230aefbee5c3` | 917,929,984 |

Bolded variants are the ones worth probing first — see §"Suggested probe
order" below.

## Important MD5 collisions — rule these out of your probe plan

Three cross-variant collisions fell out of the sweep. They confirm
internal equivalences without costing an ARM64 probe run:

- **`w4a16-local-cle` ≡ baseline `w4a16-local-qairt242`**
  (MD5 `b8d8f3b7a4df9a6825af9b969f631228`). `--apply_algorithms cle`
  produced a byte-identical binary to no-CLE baseline. CLE was a
  silent no-op on our graph. Hypothesis: CLE needs a specific adjacent-
  conv-pair pattern to act on; our MatMul-heavy transformer may have
  nothing it recognises as "equalise-able" (Qualcomm's CLE examples are
  conv/BN-heavy CV models).
- **`w4a16-local-pr-cle` ≡ `w4a16-local-pr`**
  (MD5 `caff709da351b3db12ae5108e4b1ce2a`). Adding CLE on top of per-row
  was also a no-op. Same hypothesis.
- **`w4a16-local-sqnr` ≡ `w4a16-local-tfe` (the earlier "enhanced" run)**
  (MD5 `96667934cbf9dfdcbddf2f1fe93f13a9`). The legacy `--act_quantizer
  enhanced` flag internally maps to the same algorithm as the modern
  `--act_quantizer_calibration sqnr`. Prior ARM64 cos=0.36 result on
  `w4a16-local-tfe` applies to this binary too — no new probe needed.

**Net:** only **5 distinct binaries** in the shotgun. If ARM64 has already
probed the TFE binary (cos=0.36) and the baseline w4a16 (cos=0.33), the
**fresh probes needed are**:

1. `w8a16-local` (bin 1)
2. `w4a16-local-pr` (bin 2/4) — **most likely winner per the V-proj diagnosis**
3. `w4a16-local-mse` (bin 6) — different from tfe/sqnr
4. `w8a16-local-pr` (bin 7) — kitchen sink

## Binary-size surprise

Per-row variants are **32% smaller on disk**: 591 MB vs 876 MB for
non-per-row w4a16. Unexpected — per-row stores **more** scale/offset
metadata (one per output row), and indeed the `encodings.json` for
per-row variants is 124 MB vs 3.5 MB for non-per-row, confirming the
extra metadata exists. The `.bin` saving must come from a denser weight
packing the per-row path triggers.

w8 vs w4 is also a near-wash in binary size (917 MB vs 918 MB) — HTP
compressed-weight layout appears to amortize bitwidth almost entirely
into a fixed per-tensor overhead.

## ARM64 runtime contract

Per-variant observations:

- **Non-per-row variants (1, 3, 5, 6)** — identical IO contract to the
  earlier `w4a16-local-qairt242` build: 60 UINT16 inputs
  (+ `input_ids` int32), 57 UINT16 outputs. The existing
  `_describe_inputs_pathb_local(cfg, UINT16)` path applies verbatim;
  just swap the QuantSpec table from the new `encodings.json`.
- **Per-row variants (2, 4, 7)** — **check before you plumb these.**
  Per-row quantization places **per-row scales** on weight tensors but
  the IO tensors (past_kv, attention_bias, cos/sin) should still be
  per-tensor UINT16. The `encodings.json` is 124 MB (vs 3.5 MB) because
  it contains per-row entries for every MatMul weight. Graph IO scales/
  offsets — the ones we actually consume on ARM64 — should still be
  one-per-tensor. Confirm via: `jq '.graph.tensors["past_key_values_0_key"]
  .quant_params.scale_offset' qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-pr.encodings.json`
  — if that returns a single `{scale, offset}` object (not an array),
  the existing per-tensor QuantSpec plumbing works unchanged.
- `dlc_info_<tag>.txt` is included for each build as the authoritative
  per-tensor layout reference.

Suggested `SPECULA_NPU_VARIANT` values (all match the "`-local` suffix →
PTQ path" dispatcher you landed in session 17):
- `w8a16-local`, `w4a16-local-pr`, `w4a16-local-mse`, `w8a16-local-pr`,
  `w4a16-local-cle`, `w4a16-local-sqnr`, `w4a16-local-pr-cle`.

## Suggested probe order

Given the V-projection weight-precision diagnosis (differential probe
showed V-tensor cos collapsing at layer 1), probe in this order:

1. **`w4a16-local-pr`** first — directly addresses the diagnosis
   (per-row MatMul scales give the V-projection its own per-output-channel
   range, which is exactly what the `cos ~0.13` value collapse needs).
   If this doesn't lift cos substantially, per-row alone isn't enough.
2. **`w8a16-local`** second — the brute precision-ceiling test. If cos
   ≥ 0.95, weights were the ceiling; consider shipping w8a16, or using
   A.5 per-tensor overrides to cherry-pick 8-bit weights only on V/O
   projections and keep the rest at w4.
3. **`w8a16-local-pr`** third — if neither (1) nor (2) alone hits 0.95,
   the kitchen-sink combo tells us whether w8 + per-row stacks.
4. **`w4a16-local-mse`** fourth — activation-calibration ceiling check.
   If this moves cos more than sqnr (which matched tfe at cos=0.36),
   activation range still matters at the margin.

## Per-variant compile timings

| variant | PTQ | bingen | notes |
|---|---:|---:|---|
| w8a16-local | 32 s | 9 s | baseline-equivalent speed |
| w4a16-local-pr | 172 s | 26 s | per-row is ~4× PTQ + ~3× bingen vs baseline (more per-row calibration + HTP laying out denser weights) |
| w4a16-local-cle | 33 s | 9 s | CLE was no-op; timing matches baseline |
| w4a16-local-pr-cle | 168 s | 25 s | identical to -pr |
| w4a16-local-sqnr | 47 s | 9 s | same algorithm as tfe |
| w4a16-local-mse | 226 s | 9 s | MSE calibration is the slowest — runs more iterations per tensor |
| w8a16-local-pr | 143 s | 14 s | per-row speedup small when weights already 8-bit |

## Logs, configs

```
phase5_step15_local_qairt_out_shotgun/
├── HANDOFF_shotgun.md                                           ← this file
├── SHOTGUN_SUMMARY.txt                                          ← machine-readable table
├── config_main.json / htp_backend_ext_inner.json                ← (verbatim from prior drops)
├── qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin                  + .encodings.json + dlc_info + logs
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-pr.bin               + .encodings.json + dlc_info + logs
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-cle.bin              + .encodings.json + dlc_info + logs
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-pr-cle.bin           + .encodings.json + dlc_info + logs
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-sqnr.bin             + .encodings.json + dlc_info + logs
├── qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local-mse.bin              + .encodings.json + dlc_info + logs
└── qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local-pr.bin               + .encodings.json + dlc_info + logs
```

Per-variant: `qairt_compile_log.<tag>.txt`, `qairt_quantizer.<tag>.log`,
`qnn_ctx_bin_gen.<tag>.log`, `dlc_info_<tag>.txt`.

## What remains in-flight on the x86 side

None. All DLCs were deleted after bin+encodings extraction to keep disk
under control (5 distinct binaries × ~900 MB + 7 × DLCs at ~900 MB each
would have hit 14 GB; kept artifacts are ~6 GB).

Still staged locally at `C:\work\specula-qairt\`:
- `qwen3-0.6b-pathb-ai-hub-ctx256/` ONNX (2.9 GB)
- `qwen3_0_6b_draft_v81_ctx256.pathb.fp32.dlc` (3 GB) — reusable input
- `calibration_raw/` + `input_list.txt` (3.5 GB) — Bundle A derivable

Re-running any variant is ~60-180 s if you need a repeat or a tweaked
flag.

## Follow-up asks if none of these hit cos ≥ 0.95

In decreasing order of "likely to help":

- **A.5 per-tensor overrides** — hand-pin V/O projections to 8-bit,
  everything else stays w4. Needs the override JSON; ARM64 owns
  authoring once the per-layer V-proj cos data is in hand. The per-row
  and w8a16 probes above should narrow whether it's just V/O or
  broader.
- **A.3 Bundle B calibration** — swap to step-0-only samples. Low prior
  given weight-precision diagnosis but cheap if Bundle B gets pushed.
- **C.1 AIMET pre-quant** — ~1 session lift; gives full per-tensor
  control with explicit QDQ pairs in the ONNX. Orthogonal to QAIRT-PTQ
  entirely.
- **C.2 `qairt-accuracy-debugger`** — Linux-only tool that identifies
  the first op where numerical drift exceeds a threshold. WSL2 on this
  box would work; I can install on request.

## Terminology footnote

Three minor corrections from prior asks, now in the rearview:

- `--act_quantizer tf_enhanced` → 2.42 uses `--act_quantizer enhanced`
  (legacy), or equivalently `--act_quantizer_calibration sqnr` (modern).
  Verified byte-identical output in this sweep.
- `--use_adjusted_weights_quantizer` is not a 2.42 flag; CLE triggers
  via `--apply_algorithms cle`. Documented in HANDOFF_tfe.md and
  confirmed here (though CLE didn't actually do anything in our case).
- `--model libQnnHtpV81Prepare` and `--backend libQnnHtp` are Linux
  names; on Windows we pass full paths to `HtpPrepare.dll` and
  `QnnHtp.dll`. Same fix as prior handoffs.
