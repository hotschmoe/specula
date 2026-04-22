"""Path B rewrite: hoist rotary_emb out of the graph.

Source:  models/qwen3-0.6b-pathbmask/  (additive-mask rewrite, inline rotary)
Target:  models/qwen3-0.6b-pathb/      (additive mask + rotary hoisted)

Surgery:
  - Add `position_ids_cos`, `position_ids_sin` as graph inputs after
    `attention_bias`. Shape `[batch, sequence_length, 128]` float32.
  - Rewire `Cast_4.input[0] := position_ids_cos` and
    `Cast_5.input[0] := position_ids_sin`. Cast_4/Cast_5 (both FP32→FP32
    identity for Qwen3-0.6B) stay; their outputs continue feeding
    layer-0's `Unsqueeze_6/7` which broadcasts to every layer.
  - Drop the orphaned upstream chain (everything under /model/rotary_emb/*
    that no live tensor reads).

Note: For Qwen3-0.6B, /model/rotary_emb/Constant_7 == Constant_8 == 1.0
(attention_scaling is identity), so dropping Mul_1/Mul_2 is numerically
exact. For Qwen3.5 with non-1.0 scaling, fold the scalar into the
externally-computed cos/sin in the runtime helper.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "models" / "qwen3-0.6b-pathbmask"
DST_DIR = REPO / "models" / "qwen3-0.6b-pathb"
SRC_ONNX = SRC_DIR / "model.onnx"
DST_ONNX = DST_DIR / "model.onnx"

ROTARY_PREFIX = "/model/rotary_emb/"
# layer-0 Unsqueeze_6/7 are the only direct consumers of the rotary
# terminals. Rewiring them lets the entire rotary subgraph (including
# the FP32->FP32 identity Cast_4/5) prune cleanly, leaving zero
# rotary_emb nodes per the deliverable checklist.
COS_TERMINAL = "/model/rotary_emb/Cast_4_output_0"
SIN_TERMINAL = "/model/rotary_emb/Cast_5_output_0"
LAYER0_UNSQ_COS = "/model/layers.0/self_attn/Unsqueeze_6"
LAYER0_UNSQ_SIN = "/model/layers.0/self_attn/Unsqueeze_7"
HEAD_DIM = 128
COS_INPUT = "position_ids_cos"
SIN_INPUT = "position_ids_sin"


def main() -> None:
    print(f"loading {SRC_ONNX} (with external data)...")
    m = onnx.load(str(SRC_ONNX), load_external_data=True)
    print(f"  inputs={len(m.graph.input)} nodes={len(m.graph.node)}")

    # Sanity-check Constant_7 / Constant_8 are 1.0 — if not, we'd need to
    # fold scaling into the externally-computed cos/sin (Qwen3.5 path).
    for n in m.graph.node:
        if n.name in ("/model/rotary_emb/Constant_7", "/model/rotary_emb/Constant_8"):
            for attr in n.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    val = float(arr.flatten()[0])
                    print(f"  {n.name} = {val}")
                    if abs(val - 1.0) > 1e-6:
                        raise SystemExit(
                            f"FATAL: {n.name} = {val} (expected 1.0). "
                            "Non-identity attention_scaling — must fold into runtime cos/sin."
                        )

    # 1. Rewire layer-0 Unsqueeze_6/7 to read directly from the new graph
    #    inputs (instead of from /model/rotary_emb/Cast_4_output_0 etc.).
    #    These two Unsqueezes are the *only* direct consumers of the
    #    rotary terminals, and their 4D output broadcasts to every layer.
    rewires = 0
    for n in m.graph.node:
        if n.name == LAYER0_UNSQ_COS:
            assert n.input[0] == COS_TERMINAL, f"unexpected Unsqueeze_6 input: {n.input[0]}"
            n.input[0] = COS_INPUT
            rewires += 1
        elif n.name == LAYER0_UNSQ_SIN:
            assert n.input[0] == SIN_TERMINAL, f"unexpected Unsqueeze_7 input: {n.input[0]}"
            n.input[0] = SIN_INPUT
            rewires += 1
    print(f"  rewired {rewires} layer-0 Unsqueeze inputs (expected 2)")
    assert rewires == 2

    # 2. Add new graph inputs after attention_bias.
    cos_input = helper.make_tensor_value_info(
        COS_INPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", HEAD_DIM]
    )
    sin_input = helper.make_tensor_value_info(
        SIN_INPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", HEAD_DIM]
    )
    m.graph.input.append(cos_input)
    m.graph.input.append(sin_input)
    print(f"  added inputs: {COS_INPUT}, {SIN_INPUT}  shape=[batch_size, sequence_length, {HEAD_DIM}]")

    # 3. Iteratively prune dead nodes. A node is dead if all of its outputs
    #    have zero consumers among live nodes (and aren't graph outputs).
    graph_output_names = {o.name for o in m.graph.output}

    def prune_pass(nodes):
        # Build consumer set
        consumed = set()
        for n in nodes:
            for inp in n.input:
                consumed.add(inp)
        kept, dropped = [], []
        for n in nodes:
            outputs_used = any(
                (out in consumed) or (out in graph_output_names) for out in n.output
            )
            if outputs_used:
                kept.append(n)
            else:
                dropped.append(n)
        return kept, dropped

    nodes = list(m.graph.node)
    total_dropped = 0
    iter_idx = 0
    while True:
        iter_idx += 1
        nodes, dropped = prune_pass(nodes)
        if not dropped:
            break
        # Only count rotary-namespace drops for sanity (other dead nodes shouldn't exist)
        rotary_drops = [n for n in dropped if n.name.startswith(ROTARY_PREFIX)]
        non_rotary = [n for n in dropped if not n.name.startswith(ROTARY_PREFIX)]
        if non_rotary:
            print(f"  WARNING: pruning non-rotary nodes (iter {iter_idx}):")
            for n in non_rotary[:10]:
                print(f"    {n.name} ({n.op_type})")
        total_dropped += len(dropped)
        print(f"  iter {iter_idx}: dropped {len(dropped)} nodes ({len(rotary_drops)} rotary)")

    remaining_rotary = [n for n in nodes if n.name.startswith(ROTARY_PREFIX)]
    print(f"  total nodes dropped: {total_dropped}")
    print(f"  remaining rotary_emb nodes: {len(remaining_rotary)} (should be 0)")
    if remaining_rotary:
        for n in remaining_rotary:
            print(f"    LIVE: {n.name} ({n.op_type})")

    del m.graph.node[:]
    m.graph.node.extend(nodes)
    print(f"  final graph: inputs={len(m.graph.input)} nodes={len(m.graph.node)}")

    # 4. Drop unused initializers (the orphaned Constant_5 etc. produced
    #    .weight values may exist; rotary_emb has 'onnx::Expand_460' and
    #    similar). Keep only initializers that some live node reads.
    used_inits = set()
    for n in m.graph.node:
        for inp in n.input:
            used_inits.add(inp)
    before = len(m.graph.initializer)
    kept_inits = [init for init in m.graph.initializer if init.name in used_inits]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept_inits)
    print(f"  initializers: {before} -> {len(kept_inits)}")

    # 5. Save with consolidated external data.
    DST_DIR.mkdir(parents=True, exist_ok=True)
    # If a stale model.data exists, remove so save_as_external_data gets a clean target
    for f in DST_DIR.iterdir():
        if f.suffix in {".onnx", ".data"} or f.name.endswith(".onnx_data"):
            f.unlink()

    print(f"saving to {DST_ONNX}...")
    onnx.save(
        m,
        str(DST_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    print(f"  done.")
    sz = (DST_DIR / "model.data").stat().st_size
    print(f"  model.data size: {sz / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
