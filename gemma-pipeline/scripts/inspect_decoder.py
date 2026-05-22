"""Inspect a Gemma 4 ONNX decoder graph — IO names, shapes, dynamic dims.

First real step of Route 1: before we can pin shapes or submit an AI
Hub compile job, we need to know the decoder's exact input/output
contract — which dims are symbolic, how the KV cache is named, whether
the `per_layer_inputs` (PLE) tensor is present, and what attention-mask
inputs exist.

Loads the graph WITHOUT the multi-GB external weight files
(`load_external_data=False`), so it runs instantly and needs no RAM.

Usage:
    python scripts/inspect_decoder.py \
        ../models/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_fp16.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto

_DT = {v: k for k, v in TensorProto.DataType.items()}


def _shape(t) -> str:
    dims = []
    for d in t.type.tensor_type.shape.dim:
        if d.HasField("dim_param"):
            dims.append(d.dim_param)            # symbolic
        elif d.HasField("dim_value"):
            dims.append(str(d.dim_value))       # fixed
        else:
            dims.append("?")
    dt = _DT.get(t.type.tensor_type.elem_type, "?")
    return f"{dt}[{', '.join(dims)}]"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"FATAL: not found: {path}", file=sys.stderr)
        return 2

    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    print(f"onnx: {path.name}")
    print(f"  ir_version={model.ir_version}  "
          f"opset={[o.version for o in model.opset_import]}")
    print(f"  nodes={len(g.node)}  inputs={len(g.input)}  "
          f"outputs={len(g.output)}  initializers={len(g.initializer)}")

    # Bucket inputs by role so the KV-cache + PLE structure is obvious.
    def bucket(name: str) -> str:
        n = name.lower()
        if "past" in n or "present" in n:
            return "kv"
        if "per_layer" in n or "ple" in n:
            return "ple"
        if "mask" in n:
            return "mask"
        if "position" in n or "rope" in n or "cos" in n or "sin" in n:
            return "pos"
        return "main"

    print("\n[inputs]")
    kv_in = 0
    for t in g.input:
        b = bucket(t.name)
        if b == "kv":
            kv_in += 1
            if kv_in <= 4 or kv_in % 20 == 0:
                print(f"  ({b}) {t.name:<44} {_shape(t)}")
        else:
            print(f"  ({b}) {t.name:<44} {_shape(t)}")
    if kv_in:
        print(f"  ... {kv_in} total KV-cache inputs")

    print("\n[outputs]")
    kv_out = 0
    for t in g.output:
        if bucket(t.name) == "kv":
            kv_out += 1
            if kv_out <= 2:
                print(f"  (kv) {t.name:<44} {_shape(t)}")
        else:
            print(f"  (--) {t.name:<44} {_shape(t)}")
    if kv_out:
        print(f"  ... {kv_out} total KV-cache outputs")

    # Symbolic dims — these are what pin_shapes_gemma4.py must fix.
    syms = set()
    for t in list(g.input) + list(g.output):
        for d in t.type.tensor_type.shape.dim:
            if d.HasField("dim_param"):
                syms.add(d.dim_param)
    print(f"\n[symbolic dims] {sorted(syms)}")
    print("  -> pin_shapes_gemma4.py must bind every one of these before "
          "AI Hub\n     submit_compile_job (QNN context binary needs static "
          "shapes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
