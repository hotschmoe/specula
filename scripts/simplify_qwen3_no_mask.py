"""Variant of simplify_qwen3_onnx.py that kills the attention_mask BOOL cast.

The base simplify leaves exactly one `Cast(to=BOOL)` on the graph —
`attention_mask (INT64) -> Cast(BOOL) -> Flatten -> Gather -> Reshape
-> 28 x Slice -> 28 x Where`. attention_mask is a runtime input so no
amount of shape/constant folding eliminates that cast. HTP has no BOOL
support anywhere, so AI Hub compile rejects the model.

For speculative-decoding draft use we only ever run with a full KV
cache and all 512 positions valid (no padding). In that regime
`attention_mask` is always [1]*512. If we turn it from a graph input
into a constant initializer before simplify, the Cast output becomes
a known all-True tensor; Flatten/Gather/Reshape/Slice fold into
constant BOOL tensors; and the 28 `Where(cond=True, a, b)` ops
collapse to `a`. The whole mask subgraph disappears.

If the caller ever needs variable-length / padded attention at
decode time, this path is not valid — rebuild from the regular
`simplify_qwen3_onnx.py` output and handle the BOOL wall some other
way (e.g. FP16 multiplicative masking).
"""

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_tensor
from onnxsim import simplify


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
SRC = SRC_DIR / "model.onnx"
DST_DIR = REPO_ROOT / "models" / "qwen3-0.6b-nomask"
DST_ONNX = DST_DIR / "model.onnx"
DST_DATA = "model.onnx_data"
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
CTX_LEN = 512


def build_shape_overrides() -> dict[str, list[int]]:
    # attention_mask is intentionally absent — it's now a constant
    # initializer, not a graph input.
    shapes: dict[str, list[int]] = {
        "input_ids": [1, 1],
        "position_ids": [1, 1],
    }
    past = CTX_LEN - 1
    for i in range(NUM_LAYERS):
        for k in ("key", "value"):
            shapes[f"past_key_values.{i}.{k}"] = [1, NUM_KV_HEADS, past, HEAD_DIM]
    return shapes


def elide_isnan_where_guards(model: onnx.ModelProto) -> int:
    """Rewrite `Where(IsNaN(x), const, x) -> x`.

    optimum emits an extra NaN-safety guard around each layer's softmax
    output. Where takes a BOOL condition, and HTP rejects BOOL entirely —
    so even after killing the attention_mask path, 28 BOOL tensors
    (one per layer) remain just for this guard. For a decode-only draft
    model with a full causal cache, softmax never produces NaN (the
    attention window is always non-empty), so the guard is safely
    removable. Returns the number of guards elided.
    """
    nodes = list(model.graph.node)
    producer = {o: n for n in nodes for o in n.output}
    renames: dict[str, str] = {}
    nodes_to_drop: set[int] = set()

    for w in nodes:
        if w.op_type != "Where":
            continue
        cond, _, false_val = w.input[:3]
        isnan = producer.get(cond)
        if isnan is None or isnan.op_type != "IsNaN":
            continue
        if isnan.input[0] != false_val:
            continue
        renames[w.output[0]] = false_val
        nodes_to_drop.add(id(w))
        # only drop IsNaN if it has no other consumers
        isnan_consumers = [n for n in nodes if isnan.output[0] in n.input]
        if len(isnan_consumers) == 1:
            nodes_to_drop.add(id(isnan))

    for n in nodes:
        for i, inp in enumerate(list(n.input)):
            if inp in renames:
                n.input[i] = renames[inp]
    for out in model.graph.output:
        if out.name in renames:
            out.name = renames[out.name]

    kept = [n for n in nodes if id(n) not in nodes_to_drop]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    # Prune stale value_info for tensors no longer produced by any node.
    live_names = {vi.name for vi in model.graph.input}
    live_names |= {vi.name for vi in model.graph.output}
    live_names |= {o for n in kept for o in n.output}
    live_names |= {i.name for i in model.graph.initializer}
    surviving_vi = [vi for vi in model.graph.value_info if vi.name in live_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(surviving_vi)

    return len(renames)


def promote_attention_mask_to_constant(model: onnx.ModelProto) -> None:
    inputs_to_keep = [i for i in model.graph.input if i.name != "attention_mask"]
    assert len(inputs_to_keep) == len(model.graph.input) - 1, \
        "attention_mask was not found among graph inputs"
    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)

    arr = np.ones((1, CTX_LEN), dtype=np.int64)
    tensor = numpy_helper.from_array(arr, name="attention_mask")
    model.graph.initializer.append(tensor)


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {SRC} (external data kept as references)")
    model = onnx.load(str(SRC), load_external_data=False)

    print("promoting attention_mask from graph input -> constant initializer (all ones)")
    promote_attention_mask_to_constant(model)

    overrides = build_shape_overrides()
    print(f"overriding {len(overrides)} input shapes")

    print(f"chdir to {SRC_DIR} so onnxsim can locate {DST_DATA}")
    os.chdir(SRC_DIR)

    skip_list = ["eliminate_duplicate_initializer"]
    print(f"simplifying (skipped_optimizers={skip_list})...")
    simplified, check_ok = simplify(
        model,
        overwrite_input_shapes=overrides,
        skipped_optimizers=skip_list,
    )
    if not check_ok:
        print("WARNING: onnxsim self-check reported not OK")

    elided = elide_isnan_where_guards(simplified)
    print(f"elided {elided} Where(IsNaN) nan-safety guards")

    external_tensors = [
        i for i in simplified.graph.initializer
        if i.data_location == onnx.TensorProto.EXTERNAL
    ]
    print(f"inlining {len(external_tensors)} external tensors from source sidecar")
    for t in external_tensors:
        load_external_data_for_tensor(t, str(SRC_DIR))
        t.data_location = onnx.TensorProto.DEFAULT
        t.ClearField("external_data")

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

    print(f"\nnode count: {len(model.graph.node)} -> {len(simplified.graph.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
