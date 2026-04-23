"""Rewrite pathb split ONNX files (part2/part3/part4) to accept half-dim
cos/sin inputs ([1, 1, 64] instead of [1, 1, 128]), matching Qualcomm's
genie bundle structure where `position_ids_cos`/`position_ids_sin` are
half-dim.

Surgical change per file:
  1. Narrow graph inputs `position_ids_cos` and `position_ids_sin` from
     shape [1, 1, 128] to [1, 1, 64].
  2. Insert a Concat node that doubles them: Concat([half, half], axis=-1)
     -> [1, 1, 128]. This preserves the full-dim tensor that the existing
     Unsqueeze + Mul chain downstream expects.
  3. Rewire the existing Unsqueeze (currently consumes the 128-wide input)
     to consume the Concat output instead.

Functionally equivalent to the original full-dim export (Concat([cos,cos])
yields the same 128-wide tensor), but the graph INPUT is now half-dim so
the export matches Qualcomm's [1, 1, 1, 64] AR1 convention after their
subsequent Unsqueeze.

Input  : models/qwen3-4b-arm-pathb-ctx512-part{2,3,4}/model.onnx
Output : models/qwen3-4b-arm-pathb-ctx512-part{2,3,4}/model_halfdim.onnx

The rewritten ONNX is saved into the SAME directory as the source so
its existing external-data `model.onnx_data` file reference resolves
unchanged (no weight duplication).

Run:
    .venv/Scripts/python.exe scripts/rewrite_halfdim_cos_sin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import helper


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

HALF_DIM = 64  # head_dim // 2

# Full-dim cos/sin graph input names (unchanged across parts 2/3/4).
COS_INPUT = "position_ids_cos"
SIN_INPUT = "position_ids_sin"


def rewrite_one(src_onnx: Path, dst_onnx: Path) -> None:
    # Load without reading external data so we can rewrite freely and avoid
    # duplicating the ~5 GB external-data file that the model references
    # via relative path.
    assert src_onnx.parent == dst_onnx.parent, (
        "src/dst must be in the same directory so the external-data "
        "relative reference resolves from the new ONNX."
    )
    model = onnx.load(str(src_onnx), load_external_data=False)

    # Step 1: narrow graph inputs position_ids_cos / position_ids_sin.
    for gi in model.graph.input:
        if gi.name in (COS_INPUT, SIN_INPUT):
            dims = gi.type.tensor_type.shape.dim
            # Existing shape should be [1, 1, 128]; narrow last to HALF_DIM.
            assert len(dims) == 3 and dims[-1].dim_value == HALF_DIM * 2, (
                f"unexpected {gi.name} shape "
                f"{[d.dim_value for d in dims]}"
            )
            dims[-1].dim_value = HALF_DIM

    # Step 2+3: for each of cos/sin, find the Unsqueeze consumer whose input
    # is the cos/sin graph input. Insert a Concat node that doubles the
    # half-dim input, and rewire the Unsqueeze to consume the Concat output.
    # The doubled tensor is named <input>__doubled.
    nodes_to_add: list = []
    for full_input_name in (COS_INPUT, SIN_INPUT):
        doubled_name = f"{full_input_name}__doubled"
        concat = helper.make_node(
            "Concat",
            inputs=[full_input_name, full_input_name],
            outputs=[doubled_name],
            name=f"__rewrite_halfdim_concat_{full_input_name}",
            axis=-1,
        )
        nodes_to_add.append(concat)
        # Rewire: find the Unsqueeze that consumes full_input_name and
        # redirect its input.
        rewired = 0
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == full_input_name:
                    node.input[i] = doubled_name
                    rewired += 1
        assert rewired >= 1, (
            f"no downstream consumer found for {full_input_name}"
        )

    # Insert the two new Concat nodes at the FRONT of the node list so they
    # execute before any consumers (ORT's topo-sort will also handle it,
    # but being explicit costs nothing).
    nodes = list(model.graph.node)
    del model.graph.node[:]
    for n in nodes_to_add:
        model.graph.node.append(n)
    for n in nodes:
        model.graph.node.append(n)

    onnx.save(model, str(dst_onnx), save_as_external_data=False)
    print(f"  {src_onnx} -> {dst_onnx}")


def main() -> int:
    for part in (2, 3, 4):
        pdir = MODELS / f"qwen3-4b-arm-pathb-ctx512-part{part}"
        src = pdir / "model.onnx"
        dst = pdir / "model_halfdim.onnx"
        print(f"part{part}:")
        rewrite_one(src, dst)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
