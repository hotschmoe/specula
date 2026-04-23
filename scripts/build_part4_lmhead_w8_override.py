"""Emit a `--quantization_overrides` JSON that pins the lm_head weight
(onnx::MatMul_12758, shape [2560, 151936]) to w8 per-channel symmetric,
while leaving the other 12 layers' weights to qairt-quantizer's default
w4. This matches Qualcomm's apparent Part 4 structure (our 5l full-w8
Part 4 is 1613 MB; selective lm_head-only w8 should drop to ~1000 MB,
comparable to Qualcomm's 1020 MB).

Run:
    .venv/Scripts/python.exe scripts/build_part4_lmhead_w8_override.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


REPO = Path(__file__).resolve().parents[1]
PART4_ONNX = REPO / "models" / "qwen3-4b-arm-pathb-ctx512-part4" / "model.onnx"
OUT_JSON = REPO / "results" / "phase5_qwen3_4b_bundle" / "part4_encoding_overrides.json"

LM_HEAD_WEIGHT = "onnx::MatMul_12758"  # shape [2560, 151936]


def main() -> int:
    print(f"loading {PART4_ONNX} (with external data)")
    model = onnx.load(str(PART4_ONNX))

    w: np.ndarray | None = None
    for init in model.graph.initializer:
        if init.name == LM_HEAD_WEIGHT:
            w = numpy_helper.to_array(init)
            break
    if w is None:
        print(f"FATAL: couldn't find initializer {LM_HEAD_WEIGHT}")
        return 2
    print(f"  lm_head weight: dtype={w.dtype}  shape={w.shape}")
    print(f"    global range: [{w.min():.6f}, {w.max():.6f}]")

    # Per-output-channel encoding. w.shape = [2560, 151936], so output axis
    # is 1. Each of 151936 columns gets its own symmetric w8 scale.
    # For AIMET encoding JSON format with per-channel: emit a LIST of 151936
    # entries, in channel order.
    per_channel_max = np.max(np.abs(w), axis=0)  # [151936]
    # Symmetric w8: scale = max(|w|) / 127. We'll emit encodings with
    # min = -max, max = +max so the signed int8 range (-128..127) maps
    # symmetrically.
    encodings = []
    for c in range(w.shape[1]):
        m = float(per_channel_max[c])
        scale = m / 127.0 if m > 0 else 1e-8
        encodings.append({
            "bitwidth": 8,
            "is_symmetric": "true",
            "min": -m,
            "max": m,
            "offset": -128.0,
            "scale": scale,
        })
    print(f"  per-channel encodings: {len(encodings)}")
    print(f"    scale range: [{min(e['scale'] for e in encodings):.6e}, "
          f"{max(e['scale'] for e in encodings):.6e}]")

    out = {
        "version": "0.6.1",
        "activation_encodings": {},
        "param_encodings": {LM_HEAD_WEIGHT: encodings},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
