"""Extract CPU-ORT per-layer residual ranges for Part 2 and emit an
encoding-overrides JSON that covers every `/model/layers.N/Add_1_output_0`
(N=0..11) plus the existing L11 seam.

Phase 5h probe found qairt-quantizer's Part 2 calibration cascade-clips
layer-by-layer: layer 0's residual encoded at ±5, layer 1 at ±5.5, etc.
Since each layer's narrow encoding clips its output during the calibration
forward, the NEXT layer's observed range is also narrow, producing a
snowball that caps at ±1443 by layer 6 (vs CPU-ORT ±16000 at L11 for pos=0
BOS). Passing `--quantization_overrides` at convert time pins the
encodings to the true CPU-ORT-observed ranges, side-stepping the cascade.

Output: results/phase5_qwen3_4b_bundle/part2_encoding_overrides.json
(overwrites existing L11-only override).

Run:
    .venv/Scripts/python.exe scripts/build_part2_residual_overrides.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"

NUM_LAYERS_IN_PART2 = 12


def augment_onnx_with_outputs(src_onnx: Path, tensor_names: list[str], dst_onnx: Path) -> None:
    """Load src ONNX (external data references preserved), add `tensor_names`
    as graph outputs, save to `dst_onnx` in the SAME directory as `src_onnx`
    so the existing external-data file references still resolve. Weights
    themselves are NOT rewritten."""
    if dst_onnx.parent != src_onnx.parent:
        raise RuntimeError(
            f"dst_onnx must live in the same directory as src_onnx to keep "
            f"external-data references valid. Got src={src_onnx.parent}, "
            f"dst={dst_onnx.parent}"
        )
    model = onnx.load(str(src_onnx), load_external_data=False)
    existing_out_names = {o.name for o in model.graph.output}
    for name in tensor_names:
        if name in existing_out_names:
            continue
        vi = onnx.helper.make_tensor_value_info(name, 1, None)  # dtype=FLOAT
        model.graph.output.append(vi)
    # Save protobuf-only (weights stay in the existing external-data file,
    # referenced by the initializer's data_location pointers).
    onnx.save(model, str(dst_onnx), save_as_external_data=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-onnx", type=Path,
                        default=MODELS / "qwen3-4b-arm-pathb-ctx512-part2" / "model.onnx")
    parser.add_argument("--npz", type=Path,
                        default=MODELS / "calibration" / "qwen3_4b_ctx512_a.npz")
    parser.add_argument("--out-overrides", type=Path,
                        default=RESULTS / "part2_encoding_overrides.json")
    parser.add_argument("--aug-onnx", type=Path,
                        default=MODELS / "qwen3-4b-arm-pathb-ctx512-part2" / "model.augmented.onnx",
                        help="Temp ONNX with per-layer residual tensors as extra outputs.")
    parser.add_argument("--part1-onnx", type=Path,
                        default=MODELS / "qwen3-4b-arm-pathb-ctx512-part1" / "model.onnx")
    args = parser.parse_args()

    residual_names = [f"/model/layers.{li}/Add_1_output_0" for li in range(NUM_LAYERS_IN_PART2)]
    print(f"augmenting {args.src_onnx}")
    print(f"  adding {len(residual_names)} graph outputs -> {args.aug_onnx}")
    augment_onnx_with_outputs(args.src_onnx, residual_names, args.aug_onnx)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    print(f"loading augmented part 2 ONNX ...")
    t0 = time.perf_counter()
    p2 = ort.InferenceSession(str(args.aug_onnx), sess_options=so,
                              providers=["CPUExecutionProvider"])
    print(f"  loaded in {time.perf_counter() - t0:.1f}s "
          f"({len(p2.get_inputs())} inputs / {len(p2.get_outputs())} outputs)")

    print(f"loading part 1 ONNX (for embed lookup) ...")
    p1 = ort.InferenceSession(str(args.part1_onnx), sess_options=so,
                              providers=["CPUExecutionProvider"])

    print(f"loading calibration npz: {args.npz}")
    d = np.load(str(args.npz))
    n_samples = d["input_ids"].shape[0]
    print(f"  {n_samples} samples")

    # Aggregate per-layer min/max across all samples.
    agg_min = {n: np.inf for n in residual_names}
    agg_max = {n: -np.inf for n in residual_names}

    for idx in range(n_samples):
        # Embed via part 1.
        embed = p1.run(["/model/embed_tokens/Gather_output_0"],
                       {"input_ids": d["input_ids"][idx]})[0]
        feed = {
            "/model/embed_tokens/Gather_output_0": embed,
            "attention_bias": d["attention_bias"][idx],
            "position_ids_cos": d["position_ids_cos"][idx],
            "position_ids_sin": d["position_ids_sin"][idx],
        }
        for li in range(NUM_LAYERS_IN_PART2):
            feed[f"past_key_values.{li}.key"] = d[f"past_key_values.{li}.key"][idx]
            feed[f"past_key_values.{li}.value"] = d[f"past_key_values.{li}.value"][idx]
        outs = p2.run(residual_names, feed)
        for name, arr in zip(residual_names, outs):
            agg_min[name] = min(agg_min[name], float(arr.min()))
            agg_max[name] = max(agg_max[name], float(arr.max()))
        pos = d["position_ids"][idx].item()
        iid = d["input_ids"][idx].item()
        l11_min = agg_min[residual_names[-1]]
        l11_max = agg_max[residual_names[-1]]
        print(f"  sample {idx:2d} pos={pos:2d} id={iid:6d}: "
              f"L11 running [{l11_min:9.2f}, {l11_max:9.2f}]")

    print("\nPer-layer observed CPU-ORT range across all samples:")
    for name in residual_names:
        lo, hi = agg_min[name], agg_max[name]
        print(f"  {name}: [{lo:10.2f}, {hi:10.2f}]")

    # Build overrides JSON in AIMET format.
    # uint16 asymmetric: scale = (max - min) / 65535; offset = round(min / scale)
    # (offset is stored as the uint16 value that corresponds to fp value 0,
    # which equals round(-min / scale) per QAIRT convention).
    activations = {}
    for name in residual_names:
        lo, hi = agg_min[name], agg_max[name]
        # Pad slightly so the encoding doesn't saturate on the exact observed boundary.
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad
        scale = (hi - lo) / 65535.0
        offset = float(int(round(lo / scale)))  # QAIRT stores offset as negative uint
        activations[name] = [{
            "bitwidth": 16,
            "is_symmetric": "false",
            "min": float(lo),
            "max": float(hi),
            "offset": offset,
            "scale": float(scale),
        }]

    out = {
        "version": "0.6.1",
        "activation_encodings": activations,
        "param_encodings": {},
    }
    args.out_overrides.parent.mkdir(parents=True, exist_ok=True)
    args.out_overrides.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(activations)} activation overrides -> {args.out_overrides}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
