"""Programmatic onnxsim wrapper that keeps external data external.

The `onnxsim` CLI inlines the optimum export's 3 GB of weights into the
in-memory ModelProto, which then exceeds protobuf's 2 GB serialization
limit and aborts at `SerializeToString()` before any folding happens.

This script loads the model with `load_external_data=False` so weights
stay on disk as external-data references; the attention-mask subgraph
we want folded (Range/ConstantOfShape/Cast/Gather/And/Where) is
shape-/constant-only and does not need weight bytes to simplify.

See docs/phase5_export_on_x86.md for the full pipeline.
"""

import os
from pathlib import Path

import onnx
from onnx.external_data_helper import load_external_data_for_tensor
from onnxsim import simplify


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
SRC = SRC_DIR / "model.onnx"
DST_DIR = REPO_ROOT / "models" / "qwen3-0.6b-simplified"
DST_ONNX = DST_DIR / "model.onnx"
DST_DATA = "model.onnx_data"
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
CTX_LEN = 512


def build_shape_overrides() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {
        "input_ids": [1, 1],
        "attention_mask": [1, CTX_LEN],
        "position_ids": [1, 1],
    }
    past = CTX_LEN - 1
    for i in range(NUM_LAYERS):
        for k in ("key", "value"):
            shapes[f"past_key_values.{i}.{k}"] = [1, NUM_KV_HEADS, past, HEAD_DIM]
    return shapes


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {SRC} (external data kept as references)")
    model = onnx.load(str(SRC), load_external_data=False)

    overrides = build_shape_overrides()
    print(f"overriding {len(overrides)} input shapes")

    # onnxsim's C++ backend resolves external_data paths relative to CWD,
    # not relative to the original ONNX file. Chdir so it can find the
    # sidecar data file.
    print(f"chdir to {SRC_DIR} so onnxsim can locate {DST_DATA}")
    os.chdir(SRC_DIR)

    # The `eliminate_duplicate_initializer` pass WRONGLY merges every layer's
    # weights into one when external data isn't inlined (it hashes raw_data
    # which is empty for external tensors, so all initializers hash equal).
    # Skip just that pass; keep the rest of the optimizer (needed to fold
    # the attention-mask Cast-to-BOOL subgraph).
    skip_list = ["eliminate_duplicate_initializer"]
    print(f"simplifying (skipped_optimizers={skip_list})...")
    simplified, check_ok = simplify(
        model,
        overwrite_input_shapes=overrides,
        skipped_optimizers=skip_list,
    )
    if not check_ok:
        print("WARNING: onnxsim self-check reported not OK")

    # `onnx.save(..., save_as_external_data=True)` only re-materializes
    # tensors that have `raw_data` set. After simplify, the surviving
    # external initializers still point at offsets in the SOURCE sidecar;
    # without loading their data, onnx.save leaves them referencing a file
    # that doesn't exist in DST_DIR. Pull each external tensor's bytes
    # into raw_data now, mark them non-external, and let save_model
    # consolidate every weight into the new sidecar.
    external_tensors = [
        i for i in simplified.graph.initializer
        if i.data_location == onnx.TensorProto.EXTERNAL
    ]
    print(f"inlining {len(external_tensors)} external tensors from source sidecar")
    for t in external_tensors:
        load_external_data_for_tensor(t, str(SRC_DIR))
        t.data_location = onnx.TensorProto.DEFAULT
        t.ClearField("external_data")

    # Chdir to the destination so `convert_model_to_external_data` writes
    # the sidecar next to the new ONNX rather than next to the source.
    os.chdir(DST_DIR)

    print(f"writing {DST_ONNX} (+ {DST_DATA} sidecar, consolidating all weights)")
    onnx.save(
        simplified,
        str(DST_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=DST_DATA,
        size_threshold=1024,
    )

    before = len(model.graph.node)
    after = len(simplified.graph.node)
    print(f"\nnode count: {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
