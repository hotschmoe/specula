# Phase 5.5.1 ARM-side plumbing smoke test

Session 20, 2026-04-23. Validates that the A.2 / A.1 full-quant-IO
plumbing in `scripts/npu_load_qwen3_bin.py` dispatches correctly for
the new variants without a binary, and that existing variants keep
their uint16-everywhere schema.

Pure-Python validation (no ORT-QNN session created, no binary
required). Covers: flag detection, describe_inputs/outputs dtype
assignment, quant/dequant bitwidth dispatch, quantized_zero
bitwidth-agnostic behavior.

## New variant — `w4a16-local-fqio` (A.2)

```
VARIANT=w4a16-local-fqio
IS_LOCAL_COMPILE=True
IS_LOCAL_W4A16=True
IS_LOCAL_FULL_QUANT_IO=True

describe_inputs (pathb) — 60 total
  INT32:  1 — input_ids
  UINT8: 56 — past_key_values_{0..27}_{key,value}
  UINT16: 3 — attention_bias, position_ids_cos, position_ids_sin

describe_outputs (pathb) — 57 total
  UINT16: 1 — logits
  UINT8: 56 — present_{0..27}_{key,value}
```

Matches Qualcomm's Qwen3-4B Genie metadata.yaml IO convention
exactly (uint8 per-layer past_kv + uint16 attention + uint16
rotary + uint16 logits/hidden).

## Quant dispatcher

```
bitwidth=8 spec (scale=0.0078125, offset=-128, qmax=255):
  input:     [-0.5  0.0  0.5  1.0]
  quantized: dtype=uint8 values=[64 128 192 255]
  dequant:   [-0.5  0.0  0.5  0.9921875]
  max abs err: 0.0078 (=scale, as expected for input outside spec range)

bitwidth=16 spec (scale=1.5e-05, offset=-32768, qmax=65535):
  quantized: dtype=uint16 (expect uint16)
  max abs err: 0.508 (test array exceeded spec calibration range;
                      not a dispatcher bug)

quantized_zero (bitwidth-agnostic):
  bw=8,  offset=-128:   returns 128   ✓
  bw=16, offset=-32768: returns 32768 ✓
```

## Regression — existing `w4a16-local` variant

```
VARIANT=w4a16-local
IS_LOCAL_W4A16=True
IS_LOCAL_FULL_QUANT_IO=False

describe_inputs (pathb) — 60 total
  INT32:  1 — input_ids
  UINT16: 59 — all past_kv + attention_bias + cos + sin

UINT8 count: 0 (expected 0; fqio path gated off)
```

Existing variants (`w4a16-local`, `w4a16-local-pr`, `w4a16-local-tfe`,
`w4a16-local-mse`, `w8a16-local`, `w8a16-local-pr`, `fp16-local`) keep
their exact prior schema. The new code path is explicit-whitelist-
gated on `VARIANT in {"w4a16-local-fqio", "w4a16-local-mixed"}`.

## AST parse check

All five modified scripts parse OK with `ast.parse`:
- `scripts/npu_load_qwen3_bin.py`
- `scripts/npu_short_prompt_probe.py`
- `scripts/probe_npu_steady_state_latency.py`
- `scripts/probe_w4a16_quant_roundtrip.py`
- `scripts/probe_w4a16_vs_fp16_differential.py`

## Status

- **ARM-side plumbing: verified green.**
- Next: awaiting A.2 + A.1 binaries from x86 via NAS
  (`Z:\exposed\junk\phase5_step15_local_qairt_out_qairt242_{fqio,mixed}\`).
- On binary arrival: MD5 verify, copy to `models/`, run the
  post-compile protocol in `docs/w4a16_investigation_continued.md`
  §"Pending — binaries from x86 + ARM-side measurement".
