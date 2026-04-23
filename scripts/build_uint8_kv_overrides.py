"""Build quantization-override JSON files that pin per-layer KV cache
tensors (past_key_values.N.key/.value and present.N.key/.value) to
uint8 symmetric encodings matching Qualcomm's shipping bundle scales
exactly, per `qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/metadata.yaml`.

Rationale: Qualcomm's KV is uint8 with per-layer scale and offset=-128
(symmetric). Ours is uint16 with asymmetric encoding. For structural
match to their bundle, this script emits overrides that force uint8
for every past_kv input and present_kv output in parts 2/3/4.

One JSON per part so we can layer it with our existing
per-channel-weight convert step.

Output:
    results/phase5_qwen3_4b_bundle/part2_kv_uint8_overrides.json
    results/phase5_qwen3_4b_bundle/part3_kv_uint8_overrides.json
    results/phase5_qwen3_4b_bundle/part4_kv_uint8_overrides.json

Run:
    .venv/Scripts/python.exe scripts/build_uint8_kv_overrides.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (REPO / "models" / "qualcomm-qwen3-4b-ref"
          / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite")
OUT_DIR = REPO / "results" / "phase5_qwen3_4b_bundle"

NUM_LAYERS = 36
LAYERS_PER_PART = 12


def qualcomm_kv_encodings() -> dict[int, dict[str, tuple[float, int]]]:
    """Returns {layer_idx: {'key': (scale, offset), 'value': (scale, offset)}}
    derived from the AR1 CL512 components of Qualcomm's metadata.yaml. Every
    Qualcomm KV is uint8 symmetric (offset=-128)."""
    meta = yaml.safe_load((BUNDLE / "metadata.yaml").read_text())
    enc: dict[int, dict[str, tuple[float, int]]] = {}
    for part_idx in (2, 3, 4):
        part = meta["components"][f"ar1_cl512_{part_idx}_of_4"]
        # inputs include past_key_N_in and past_value_N_in
        for name, spec in part["inputs"].items():
            if name.startswith("past_key_") and name.endswith("_in"):
                li = int(name.split("_")[2])
                qp = spec["quantization_parameters"]
                enc.setdefault(li, {})["key"] = (float(qp["scale"]), int(qp["offset"]))
            elif name.startswith("past_value_") and name.endswith("_in"):
                li = int(name.split("_")[2])
                qp = spec["quantization_parameters"]
                enc.setdefault(li, {})["value"] = (float(qp["scale"]), int(qp["offset"]))
    return enc


def encoding_entry(scale: float, offset: int) -> dict:
    """AIMET-compatible uint8 symmetric encoding for QAIRT's
    --quantization_overrides. Qualcomm uses offset=-128 (symmetric around 0
    in the signed-int8-equivalent space)."""
    # Symmetric uint8: f = (q + offset) * scale, offset=-128. So the
    # representable range is [-128*scale, 127*scale].
    lo = offset * scale
    hi = (255 + offset) * scale
    return {
        "bitwidth": 8,
        "is_symmetric": "true",
        "min": float(lo),
        "max": float(hi),
        "offset": float(offset),
        "scale": float(scale),
    }


def main() -> int:
    enc = qualcomm_kv_encodings()
    print(f"Extracted Qualcomm KV encodings for {len(enc)} layers")

    # Our DLC uses these tensor names. For each part we emit its
    # layer-range's past_key_values.N.key/.value (inputs) AND
    # present.N.key/.value (outputs).
    for part_idx in (2, 3, 4):
        layer_start = (part_idx - 2) * LAYERS_PER_PART
        layer_end = layer_start + LAYERS_PER_PART

        acts: dict[str, list[dict]] = {}
        for li in range(layer_start, layer_end):
            for kind in ("key", "value"):
                scale, offset = enc[li][kind]
                entry = [encoding_entry(scale, offset)]
                acts[f"past_key_values.{li}.{kind}"] = entry
                # PRESENT tensors typically share scale with PAST per Qualcomm's
                # convention (confirmed Phase 0: in/out scale identical so KV
                # can be concatenated without requantize).
                acts[f"present.{li}.{kind}"] = entry

        out = {
            "version": "0.6.1",
            "activation_encodings": acts,
            "param_encodings": {},
        }
        out_path = OUT_DIR / f"part{part_idx}_kv_uint8_overrides.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path.name} ({len(acts)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
