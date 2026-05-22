"""Decompose ORT-fused ops in the Gemma 4 decoder into primitive ONNX.

The `onnx-community/gemma-4-E2B-it-ONNX` decoder was run through ONNX
Runtime's transformer optimizer, which fused subgraphs into contrib /
non-standard ops. Qualcomm AI Hub's ONNX parser rejects them — the
first compile (jgoex7kdp) died on:

    No Op registered for SimplifiedLayerNormalization (domain_version 21)

This script unfuses those ops back to standard ONNX so AI Hub (and the
QAIRT/HTP toolchain behind it) can lower the graph.

Decompositions implemented:

  SimplifiedLayerNormalization  (RMSNorm)  -> Mul/ReduceMean/Add/Sqrt/Div/Mul
      Y = X / sqrt(mean(X^2, axis) + eps) * scale

Not yet implemented (resubmit after SLN and let AI Hub report the next
blocker — `Gelu` is standard ONNX since opset 20 and may pass as-is;
`RotaryEmbedding` / `GroupQueryAttention` are com.microsoft contrib ops
that will likely also need decomposition — see scripts/README.md):

  RotaryEmbedding         (com.microsoft)
  GroupQueryAttention     (com.microsoft)

Loads the full model (~5 GB RAM), rewrites, and writes a qai-hub model
directory (`model.onnx` + single `model.data`) ready for
submit_compile_gemma4.py.

Usage:
    python scripts/rewrite_gemma4_htp.py \
        ../models/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_fp16.onnx \
        ../models/gemma-4-E2B-it-ONNX/qaihub_decoder_fp16_unfused
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _np_dtype(elem_type: int):
    return {TensorProto.FLOAT: np.float32,
            TensorProto.FLOAT16: np.float16}[elem_type]


def decompose_simplified_layernorm(graph: onnx.GraphProto) -> int:
    """Replace every SimplifiedLayerNormalization with primitive ops.

    SimplifiedLayerNormalization(X, scale) computes
        Y = X / sqrt(ReduceMean(X*X, axis) + epsilon) * scale
    (axis default -1, epsilon default 1e-5). Done in the tensor's own
    dtype — the decoder is fp16 throughout and fp16 RMSNorm is standard.
    """
    # Map every value name -> its elem_type, so we can build correctly
    # typed eps constants.
    vtype: dict[str, int] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        vtype[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        vtype[init.name] = init.data_type

    targets = [(i, n) for i, n in enumerate(graph.node)
               if n.op_type == "SimplifiedLayerNormalization"]
    if not targets:
        return 0

    new_nodes: list = []
    new_inits: list = []
    drop = {i for i, _ in targets}
    extra_by_index: dict[int, list] = {}

    for idx, node in targets:
        x, scale = node.input[0], node.input[1]
        y = node.output[0]
        if len(node.output) > 1 and node.output[1]:
            # 2nd output (inv_std_dev) is almost never consumed; bail
            # loudly rather than silently drop it.
            used = any(node.output[1] in n.input for n in graph.node)
            if used:
                raise NotImplementedError(
                    f"{node.name}: 2nd output {node.output[1]} is consumed "
                    f"— inv_std_dev decomposition not implemented.")

        axis = -1
        eps = 1e-5
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
            elif a.name == "epsilon":
                eps = a.f

        et = vtype.get(x, TensorProto.FLOAT16)
        npdt = _np_dtype(et)
        p = node.name or f"sln_{idx}"

        eps_name = f"{p}/eps"
        axes_name = f"{p}/axes"
        new_inits.append(numpy_helper.from_array(
            np.array(eps, dtype=npdt), eps_name))
        new_inits.append(numpy_helper.from_array(
            np.array([axis], dtype=np.int64), axes_name))

        sq = f"{p}/sq"
        mean = f"{p}/mean"
        plus = f"{p}/plus"
        rms = f"{p}/rms"
        norm = f"{p}/norm"
        sub = [
            helper.make_node("Mul", [x, x], [sq], name=f"{p}/Mul_sq"),
            helper.make_node("ReduceMean", [sq, axes_name], [mean],
                             name=f"{p}/ReduceMean", keepdims=1),
            helper.make_node("Add", [mean, eps_name], [plus],
                             name=f"{p}/Add_eps"),
            helper.make_node("Sqrt", [plus], [rms], name=f"{p}/Sqrt"),
            helper.make_node("Div", [x, rms], [norm], name=f"{p}/Div"),
            helper.make_node("Mul", [norm, scale], [y], name=f"{p}/Mul_scale"),
        ]
        extra_by_index[idx] = sub

    # Rebuild the node list in original order, swapping each SLN for its
    # decomposition (keeps topological validity — SLN had no forward dep
    # on its own replacements).
    for i, n in enumerate(graph.node):
        if i in drop:
            new_nodes.extend(extra_by_index[i])
        else:
            new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return len(targets)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if not src.exists():
        print(f"FATAL: not found: {src}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[load]  {src}  (+ external data, ~5 GB)")
    model = onnx.load(str(src), load_external_data=True)
    print(f"[load]  done in {time.time() - t0:.0f}s  "
          f"({len(model.graph.node)} nodes)")

    n = decompose_simplified_layernorm(model.graph)
    print(f"[rewrite]  decomposed {n} SimplifiedLayerNormalization nodes "
          f"-> {len(model.graph.node)} nodes total")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        f.unlink()
    onnx_out = out_dir / "model.onnx"
    t1 = time.time()
    print(f"[save]  {onnx_out} + model.data")
    onnx.save_model(model, str(onnx_out), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.data",
                    size_threshold=1024)
    print(f"[save]  done in {time.time() - t1:.0f}s")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<14} {f.stat().st_size / 1e9:.2f} GB")

    # Check the saved model by path (>2 GB must be checked from disk).
    onnx.checker.check_model(str(onnx_out), full_check=True)
    print("[check]  onnx.checker (full) passed")
    print(f"\nReady: submit_compile_gemma4.py --onnx {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
