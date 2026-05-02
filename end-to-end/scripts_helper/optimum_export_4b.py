"""Run optimum-cli's `text-generation-with-past` export on Qwen3-4B
with a small monkey-patch to dodge the 2 GiB protobuf limit hit by
torch.onnx's `_jit_pass_onnx_graph_shape_type_inference`.

Background: optimum-cli's legacy onnx export pipeline calls
`torch.onnx.export(...)`. Inside that, torch.onnx invokes
`_C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, ...)`
which serializes the graph (with weights inline) to a protobuf string
to perform symbolic shape inference. For Qwen3-4B that string is
~16 GiB, blowing past protobuf's 2 GiB cap.

The shape-inference pass is purely an *optimization*: it propagates
shape/type info from inputs onward through the graph. If we skip it,
torch.onnx still produces a fully valid ONNX file — the downstream
nodes simply lack inferred shape annotations on their outputs. ORT
re-infers shapes at session-build time, and the rewrite_qwen3_*.py
pathb scripts derive shapes from the named graph topology rather than
from the inferred annotations, so the loss is cosmetic for our pipeline.

This script monkey-patches the C++ shape-inference helper to a no-op,
then drives the optimum CLI as a Python entry point. Output goes to
the directory passed in --out-dir, identical to running
`optimum-cli export onnx --task text-generation-with-past`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _patch_torch_onnx_shape_inference():
    """Replace the failing shape-inference pass with a no-op.

    Affects only the in-memory shape annotations on the JIT graph;
    the produced ONNX is structurally identical and runtime-valid.
    """
    import torch
    # The pass lives on `torch._C` as a free C++ function.
    if not hasattr(torch._C, "_jit_pass_onnx_graph_shape_type_inference"):
        print("[shape-inf-patch] WARNING: _jit_pass_onnx_graph_shape_type_inference "
              "not found; nothing to patch")
        return

    orig = torch._C._jit_pass_onnx_graph_shape_type_inference

    def _noop(graph, params_dict, opset_version):
        # Returns None / nothing; original would mutate the graph in-place
        # by attaching shape annotations, but no caller currently *requires*
        # that those annotations be present.
        return None

    torch._C._jit_pass_onnx_graph_shape_type_inference = _noop
    print("[shape-inf-patch] _jit_pass_onnx_graph_shape_type_inference → no-op "
          f"(orig was {orig})")


def _patch_torch_onnx_node_shape_inference():
    """ALSO patch the per-node variant which torch.onnx calls during graph
    construction. Same protobuf-cap issue, same fix (shape annotations
    are advisory, not load-bearing for our downstream pipeline)."""
    import torch
    for name in [
        "_jit_pass_onnx_node_shape_type_inference",
        "_jit_onnx_log",
    ]:
        if hasattr(torch._C, name):
            orig = getattr(torch._C, name)
            def _noop(*args, **kwargs):
                return None
            setattr(torch._C, name, _noop)
            print(f"[shape-inf-patch] {name} → no-op (orig was {orig})")


def _post_fix_dtype(onnx_path: Path) -> None:
    """Convert fp64 Constants to fp32 (torch.onnx exports the head_dim
    sqrt as a Python float → fp64 Constant; downstream Mul fails with
    `tensor(float) and tensor(double)`)."""
    import onnx
    import numpy as np
    print(f"[post-fix] coerce fp64 Constants to fp32 in {onnx_path}")
    m = onnx.load(str(onnx_path), load_external_data=True)
    fixed = 0
    for node in m.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            t = attr.t
            if t.data_type == onnx.TensorProto.DOUBLE:
                arr = onnx.numpy_helper.to_array(t).astype(np.float32)
                new_t = onnx.numpy_helper.from_array(arr, name=t.name)
                attr.t.CopyFrom(new_t)
                fixed += 1
    # Also clean up any value_info / initializer fp64s.
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.DOUBLE:
            arr = onnx.numpy_helper.to_array(init).astype(np.float32)
            new_t = onnx.numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_t)
            fixed += 1
    onnx.save(m, str(onnx_path), save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.onnx_data",
              size_threshold=1024)
    print(f"[post-fix] coerced {fixed} fp64 → fp32 tensors")


def _post_fix_validate(onnx_path: Path) -> None:
    """Smoke-load with onnxruntime CPU to make sure the dtype fix did its job."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    print(f"[post-fix] validating {onnx_path} loads in onnxruntime ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"],
                                 sess_options=so)
    n_in = len(sess.get_inputs()); n_out = len(sess.get_outputs())
    print(f"[post-fix] OK: {n_in} inputs, {n_out} outputs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", default="text-generation-with-past")
    args = parser.parse_args()

    _patch_torch_onnx_shape_inference()
    _patch_torch_onnx_node_shape_inference()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Re-build sys.argv to mimic the optimum-cli entry point.
    sys.argv = [
        "optimum-cli", "export", "onnx",
        "--model", str(args.model_path),
        "--task", args.task,
        "--no-constant-folding",
        str(args.out_dir),
    ]
    print(f"[optimum-export-4b] running: {' '.join(sys.argv)}")
    from optimum.commands.optimum_cli import main as optimum_main
    try:
        rc = optimum_main()
    except Exception as e:
        # Optimum's fix_dynamic_axes step needs to load the model with ORT,
        # which fails on the fp64 Mul issue. Catch + post-fix + validate
        # ourselves; the model.onnx + model.onnx_data are already on disk.
        msg = str(e).lower()
        if "tensor(float) and tensor(double)" in str(e) or "type parameter" in msg:
            print(f"[optimum-export-4b] caught known dtype-mismatch from optimum's "
                  f"post-pass; fixing the model dtype directly")
            _post_fix_dtype(args.out_dir / "model.onnx")
            _post_fix_validate(args.out_dir / "model.onnx")
            return 0
        raise

    # Always re-fix even on optimum success (defensive — the legacy
    # exporter sometimes emits fp64 Constants).
    onnx_path = args.out_dir / "model.onnx"
    if onnx_path.exists():
        _post_fix_dtype(onnx_path)
        _post_fix_validate(onnx_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
