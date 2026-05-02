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

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[1]

# Defaults for the original Qwen3-0.6B pipeline. Override via --model-stem
# (e.g. `--model-stem qwen3-4b-optimum-arm` -> `<stem>-pathbmask` and
# `<stem>-pathb`).
DEFAULT_STEM = "qwen3-0.6b"

ROTARY_PREFIX = "/model/rotary_emb/"
HEAD_DIM = 128
COS_INPUT = "position_ids_cos"
SIN_INPUT = "position_ids_sin"


def _detect_rotary_cos_sin_terminals(model: onnx.ModelProto) -> tuple[str, str]:
    """Walk the rotary_emb subgraph to find the cos and sin output tensor
    names (whatever the optimum exporter named them).

    Pattern: there's exactly one Cos op and one Sin op under
    /model/rotary_emb/. Each feeds an identity Mul (* attention_scaling=1.0)
    via Constant_7/Constant_8, then a Cast (the terminal). We return the
    names of the two terminal Cast outputs.

    Falls back to walking just downstream of Cos/Sin if the structure
    differs slightly (no Mul, or no terminal Cast). Numeric Constant_7
    values are validated separately in main().
    """
    out_to_node = {}
    for n in model.graph.node:
        for o in n.output:
            if o:
                out_to_node[o] = n

    cos_node = sin_node = None
    for n in model.graph.node:
        if not n.name.startswith(ROTARY_PREFIX):
            continue
        if n.op_type == "Cos":
            cos_node = n
        elif n.op_type == "Sin":
            sin_node = n
    if cos_node is None or sin_node is None:
        raise RuntimeError(
            f"could not locate Cos/Sin nodes in rotary subgraph; "
            f"is the export shape correct?"
        )

    def walk_terminal(start_out: str) -> str:
        """Walk downstream as long as the next node is a Mul (scaling)
        or a Cast and stays in the rotary namespace. Return the last
        rotary-namespace tensor name reachable in this chain."""
        cur_out = start_out
        # Build consumer index lazily.
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for n in model.graph.node:
            for inp in n.input:
                consumers.setdefault(inp, []).append(n)
        while True:
            # Find the *one* rotary-namespace consumer along the chain.
            rotary_consumers = [
                c for c in consumers.get(cur_out, [])
                if c.name.startswith(ROTARY_PREFIX)
                and c.op_type in ("Mul", "Cast")
            ]
            if len(rotary_consumers) != 1:
                return cur_out
            nxt = rotary_consumers[0]
            cur_out = nxt.output[0]

    cos_term = walk_terminal(cos_node.output[0])
    sin_term = walk_terminal(sin_node.output[0])
    return cos_term, sin_term


def _detect_layer0_unsqueeze_consumers(
    model: onnx.ModelProto, cos_term: str, sin_term: str,
) -> tuple[onnx.NodeProto, onnx.NodeProto]:
    """The two consumers of cos_term / sin_term should both be Unsqueeze
    ops in layer-0's self_attn (their outputs broadcast through every
    layer). Locate them by tensor-flow rather than hardcoded name."""
    cos_node = sin_node = None
    for n in model.graph.node:
        if cos_term in n.input:
            if n.op_type != "Unsqueeze":
                raise RuntimeError(
                    f"cos terminal {cos_term!r} consumed by {n.op_type} "
                    f"({n.name}); expected Unsqueeze"
                )
            if cos_node is not None:
                raise RuntimeError(
                    f"multiple consumers of cos terminal {cos_term!r}; "
                    f"unexpected"
                )
            cos_node = n
        if sin_term in n.input:
            if n.op_type != "Unsqueeze":
                raise RuntimeError(
                    f"sin terminal {sin_term!r} consumed by {n.op_type} "
                    f"({n.name}); expected Unsqueeze"
                )
            if sin_node is not None:
                raise RuntimeError(
                    f"multiple consumers of sin terminal {sin_term!r}; "
                    f"unexpected"
                )
            sin_node = n
    if cos_node is None or sin_node is None:
        raise RuntimeError(
            f"could not locate Unsqueeze consumers of cos/sin terminals "
            f"({cos_term!r}, {sin_term!r})"
        )
    return cos_node, sin_node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-stem", default=DEFAULT_STEM,
        help=f"Model directory stem (default: {DEFAULT_STEM!r}). Used to derive "
             f"<stem>-pathbmask (input) and <stem>-pathb (output) under models/.",
    )
    parser.add_argument(
        "--src-dir", type=Path, default=None,
        help="Override the path-B-mask input directory (default: models/<stem>-pathbmask)",
    )
    parser.add_argument(
        "--dst-dir", type=Path, default=None,
        help="Override the path-B output directory (default: models/<stem>-pathb)",
    )
    args = parser.parse_args()

    src_dir = args.src_dir or (REPO / "models" / f"{args.model_stem}-pathbmask")
    dst_dir = args.dst_dir or (REPO / "models" / f"{args.model_stem}-pathb")
    src_onnx = src_dir / "model.onnx"
    dst_onnx = dst_dir / "model.onnx"

    print(f"loading {src_onnx} (with external data)...")
    m = onnx.load(str(src_onnx), load_external_data=True)
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

    # 1. Rewire the layer-0 Unsqueeze consumers of cos/sin terminals to
    #    read directly from the new graph inputs (instead of from
    #    `/model/rotary_emb/Cast_*_output_0`). These two Unsqueezes are
    #    the *only* direct consumers of the rotary terminals, and their
    #    4D output broadcasts to every layer.
    #
    #    The exact Unsqueeze node names (Unsqueeze_6/_7 in 0.6B vs
    #    _12/_13 in 4B) and Cast terminal indices vary across models due
    #    to differing pre-attention node counts (4B has q_norm/k_norm
    #    that 0.6B doesn't). Detect dynamically rather than hardcode.
    cos_terminal, sin_terminal = _detect_rotary_cos_sin_terminals(m)
    print(f"  detected cos terminal: {cos_terminal}")
    print(f"  detected sin terminal: {sin_terminal}")
    cos_unsq, sin_unsq = _detect_layer0_unsqueeze_consumers(m, cos_terminal, sin_terminal)
    print(f"  detected layer-0 Unsqueeze (cos): {cos_unsq.name}")
    print(f"  detected layer-0 Unsqueeze (sin): {sin_unsq.name}")
    cos_unsq.input[0] = COS_INPUT
    sin_unsq.input[0] = SIN_INPUT
    rewires = 2
    print(f"  rewired {rewires} layer-0 Unsqueeze inputs (expected 2)")

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
    dst_dir.mkdir(parents=True, exist_ok=True)
    # If a stale model.data exists, remove so save_as_external_data gets a clean target
    for f in dst_dir.iterdir():
        if f.suffix in {".onnx", ".data"} or f.name.endswith(".onnx_data"):
            f.unlink()

    print(f"saving to {dst_onnx}...")
    onnx.save(
        m,
        str(dst_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    print(f"  done.")
    sz = (dst_dir / "model.data").stat().st_size
    print(f"  model.data size: {sz / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
