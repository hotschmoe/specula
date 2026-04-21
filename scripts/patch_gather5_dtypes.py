"""Patch HTP-unfriendly dtypes on /model/Gather_5 (and similar if found).

Context: HTP's Gather op requires (data_dtype, indices_dtype) to fit
one of: all-BF16, all-FP16, all-INT16, all-INT8, or an unspecified
'OTHERS' set. Our optimum export has:
  /model/Gather_5.data    = BOOL  (from Cast(attention_mask) then Flatten)
  /model/Gather_5.indices = INT64 (from Range-based position math)
This combination is rejected. Seven compile attempts have burned on
this same node, so fix it at the ONNX level rather than asking QAIRT
for yet another flag.

Strategy -- minimal graph edit:
  1. Before Gather_5:
       data    : insert Cast BOOL   -> INT8
       indices : insert Cast INT64  -> INT32
  2. After Gather_5:
       output  : insert Cast INT8   -> BOOL   (downstream And_1 wants BOOL)

Other Gather nodes in the graph are overwhelmingly Shape->Gather which
QAIRT handles via --truncate_64bit_tensors-equivalent behaviour.
Gather_5 is the only bool-data one in the attention-mask chain, so
a targeted patch is enough. If another similar dtype bug surfaces at
AI Hub, add another entry to PATCHES.

Run:
    .venv\\Scripts\\python.exe scripts\\patch_gather5_dtypes.py
"""

import shutil
import sys
import time
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum-frozen-ortopt"
DEST_DIR = REPO_ROOT / "models" / "qwen3-0.6b-patched"
SOURCE_ONNX = SOURCE_DIR / "model.onnx"
DEST_ONNX = DEST_DIR / "model.onnx"


def patch_gather(model: onnx.ModelProto, gather_name: str) -> bool:
    """Insert dtype Cast nodes around a specific Gather to satisfy HTP."""
    gather = None
    for n in model.graph.node:
        if n.name == gather_name:
            gather = n
            break
    if gather is None:
        print(f"  WARNING: {gather_name} not found; skipping")
        return False
    if gather.op_type != "Gather":
        print(f"  WARNING: {gather_name} is {gather.op_type}, not Gather; skipping")
        return False

    prefix = gather_name.replace("/", "_").strip("_")
    data_in = gather.input[0]
    idx_in = gather.input[1]
    out = gather.output[0]

    # Intermediate tensor names.
    data_int8 = f"{prefix}_data_int8"
    idx_int32 = f"{prefix}_idx_int32"
    gather_int8_out = f"{prefix}_out_int8"

    cast_data = helper.make_node(
        "Cast",
        inputs=[data_in],
        outputs=[data_int8],
        to=TensorProto.INT8,
        name=f"{prefix}_cast_data_bool_to_int8",
    )
    cast_idx = helper.make_node(
        "Cast",
        inputs=[idx_in],
        outputs=[idx_int32],
        to=TensorProto.INT32,
        name=f"{prefix}_cast_idx_int64_to_int32",
    )
    # Rewrite Gather in place: new inputs + new output, keep attributes.
    gather.input[:] = [data_int8, idx_int32]
    gather.output[:] = [gather_int8_out]
    cast_out = helper.make_node(
        "Cast",
        inputs=[gather_int8_out],
        outputs=[out],
        to=TensorProto.BOOL,
        name=f"{prefix}_cast_out_int8_to_bool",
    )

    # Find the Gather's index in the graph and insert new nodes around it.
    nodes = list(model.graph.node)
    idx = nodes.index(gather)
    new_nodes = nodes[:idx] + [cast_data, cast_idx, gather, cast_out] + nodes[idx + 1:]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  patched {gather_name}: BOOL/INT64 -> INT8/INT32 ->(Gather)-> INT8 -> BOOL")
    return True


PATCHES = ["/model/Gather_5"]


def main() -> int:
    if not SOURCE_ONNX.exists():
        print(f"ERROR: source missing at {SOURCE_ONNX}")
        return 2
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {SOURCE_ONNX} ...")
    t0 = time.perf_counter()
    model = onnx.load(str(SOURCE_ONNX), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s")

    print(f"\napplying {len(PATCHES)} patches:")
    patched = 0
    for name in PATCHES:
        if patch_gather(model, name):
            patched += 1
    print(f"\n{patched}/{len(PATCHES)} patches applied")

    # Re-run shape inference so dtypes propagate through the new Cast nodes.
    print("re-running shape inference ...")
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model, strict_mode=False)

    print(f"saving {DEST_ONNX} ...")
    onnx.save(model, str(DEST_ONNX), save_as_external_data=False)
    print(f"  {DEST_ONNX.stat().st_size / (1024*1024):.1f} MB")

    # Bring weights + config.
    for fname in ("model.onnx_data", "config.json"):
        src = SOURCE_DIR / fname
        dst = DEST_DIR / fname
        if src.exists() and not dst.exists():
            print(f"copying {fname} ...")
            shutil.copy2(src, dst)

    print(f"\n--- verify downstream in ORT-CPU ---")
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(str(DEST_ONNX), sess_options=opts, providers=["CPUExecutionProvider"])
        print(f"  ORT loaded OK: {len(sess.get_inputs())} inputs / {len(sess.get_outputs())} outputs")
    except Exception as exc:
        print(f"  ORT load FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
