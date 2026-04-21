"""Phase 5 step 6 - recon on the BOOL-tainted region of the staged ONNX.

Walks every BOOL-producing node + BOOL-typed tensor in a staged
Qwen3-0.6B graph, reports upstream producers (back to graph
inputs/initializers) and the downstream non-BOOL boundary (where
each BOOL eventually dies into an FP/INT op like `Where`).

Purpose: before writing either fold (Path A ConstantOfShape-BOOL, or
Path B-mask additive FP16 mask), we need to know:

    1. How many BOOL-producing ops are in the graph, broken down
       by op type. The phase5 doc predicts 3 Cast->BOOL; staged
       output confirms 3. Are there also BOOL producers from
       Less/Greater/Equal/LessOrEqual/And/Not that the doc didn't
       inventory?
    2. Do all BOOL-tainted paths trace to `attention_mask` (now a
       constant initializer) or are there other taint sources (e.g.,
       a comparison involving `position_ids`)?
    3. What's the "collapse point" -- the first non-BOOL consumer
       downstream of the BOOL region? For Path B-mask that's the
       splice site where Where(bool, scores, -inf) -> Add(scores,
       attention_bias).
    4. Does `Gather_5` exist in the graph by that exact name, and
       does its input dtype chain match the pattern the doc named?
       (`attention_mask -> Cast(BOOL) -> Flatten -> Cast(INT8) ->
       Gather_5`.)

Reads a staged ONNX (default: models/qwen3-0.6b-staged/model.onnx)
and emits:

    - stdout : human-readable report
    - --json : machine-readable report for the handoff

Does NOT rewrite the graph. Pure analysis.

Run:
    python scripts/survey_bool_region.py \\
        --model models/qwen3-0.6b-staged/model.onnx \\
        --json results/phase5_step6_bool_region.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parent.parent


# Ops whose outputs are BOOL (ONNX spec). Used to seed BOOL-tensor
# enumeration so we don't miss non-Cast producers.
BOOL_OUTPUT_OPS = {
    "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "IsNaN", "IsInf", "And", "Or", "Xor", "Not",
}


def tensor_type_name(t: int) -> str:
    # ONNX TensorProto dtype enum -> readable.
    try:
        return TensorProto.DataType.Name(t)
    except ValueError:
        return f"UNK({t})"


def collect_value_info_dtypes(model: onnx.ModelProto) -> dict[str, int]:
    """Map tensor name -> elem_type int, from graph value_info + input
    + output + initializer records. Best-effort; some edges have no
    declared dtype."""
    out: dict[str, int] = {}
    for vi_list in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in vi_list:
            if vi.type.tensor_type.elem_type:
                out[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        out[init.name] = init.data_type
    return out


def build_edge_maps(model: onnx.ModelProto) -> tuple[dict, dict]:
    producer: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers[i].append(n)
    return producer, consumers


def node_label(n: onnx.NodeProto) -> str:
    name = n.name or "<anon>"
    return f"{n.op_type}[{name}]"


def walk_up(tensor: str, producer: dict, max_depth: int = 16) -> list[str]:
    """BFS back through producers. Returns a list of lines suitable for
    printing, in order of increasing depth."""
    out = []
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    while queue:
        t, depth = queue.popleft()
        if t in seen or depth > max_depth:
            continue
        seen.add(t)
        src = producer.get(t)
        if src is None:
            out.append(f"{'  ' * depth}<<input/init>> {t}")
            continue
        out.append(f"{'  ' * depth}{node_label(src)} -> {t}")
        for inp in src.input:
            queue.append((inp, depth + 1))
    return out


def walk_down_to_non_bool(
    tensor: str,
    consumers: dict,
    producer: dict,
    dtype_map: dict,
    bool_tensors: set[str],
) -> list[str]:
    """BFS forward until we hit consumers whose *output* is non-BOOL
    or whose inputs mix BOOL+non-BOOL (the splice boundary)."""
    out = []
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    while queue:
        t, depth = queue.popleft()
        if t in seen or depth > 8:
            continue
        seen.add(t)
        for c in consumers.get(t, []):
            # Is this consumer a "BOOL sink" -- consumes BOOL but
            # produces non-BOOL? That's the boundary.
            produces_bool = any(o in bool_tensors for o in c.output)
            label = node_label(c)
            marker = "  " * depth
            if produces_bool:
                out.append(f"{marker}-> {label} (still BOOL)")
                for o in c.output:
                    queue.append((o, depth + 1))
            else:
                # Boundary: record input dtypes so we know what the
                # non-BOOL side expects.
                input_dtypes = []
                for inp in c.input:
                    dt = dtype_map.get(inp)
                    input_dtypes.append(f"{inp}:{tensor_type_name(dt) if dt else '?'}")
                out.append(f"{marker}=> SINK {label} (inputs: {', '.join(input_dtypes)})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "qwen3-0.6b-staged" / "model.onnx",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "results" / "phase5_step6_bool_region.json",
    )
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    t0 = time.perf_counter()
    model = onnx.load(str(args.model), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s "
          f"({len(model.graph.node)} nodes, "
          f"{len(model.graph.initializer)} initializers)")

    dtype_map = collect_value_info_dtypes(model)
    producer, consumers = build_edge_maps(model)

    # Run onnx shape inference to populate dtypes that weren't
    # recorded in value_info. Without this we can't tell which Cast
    # outputs or which Equal outputs sit downstream, because ONNX
    # files don't always carry full intermediate type info.
    print("running onnx shape inference to fill dtype map ...")
    t0 = time.perf_counter()
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        for vi in inferred.graph.value_info:
            if vi.type.tensor_type.elem_type:
                dtype_map[vi.name] = vi.type.tensor_type.elem_type
        print(f"  inferred in {time.perf_counter() - t0:.2f} s "
              f"(dtype_map now has {len(dtype_map)} entries)")
    except Exception as exc:
        print(f"  shape inference failed ({exc}); continuing with partial dtype_map")

    # Identify BOOL tensors.
    bool_tensors: set[str] = set()
    cast_to_bool_nodes: list[onnx.NodeProto] = []
    bool_op_producers: dict[str, list[onnx.NodeProto]] = defaultdict(list)

    for n in model.graph.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.BOOL:
                    cast_to_bool_nodes.append(n)
                    for o in n.output:
                        bool_tensors.add(o)
        if n.op_type in BOOL_OUTPUT_OPS:
            bool_op_producers[n.op_type].append(n)
            for o in n.output:
                bool_tensors.add(o)

    # Also mark BOOL-typed initializers + inputs.
    bool_inits = [i for i in model.graph.initializer
                  if i.data_type == TensorProto.BOOL]
    for init in bool_inits:
        bool_tensors.add(init.name)

    # Propagate: any op whose input is a BOOL tensor AND whose output
    # is declared BOOL in dtype_map is also in the region.
    # (Reshape, Identity, Concat etc. pass BOOL through.)
    changed = True
    while changed:
        changed = False
        for n in model.graph.node:
            if any(i in bool_tensors for i in n.input):
                for o in n.output:
                    if dtype_map.get(o) == TensorProto.BOOL and o not in bool_tensors:
                        bool_tensors.add(o)
                        changed = True

    print(f"\n=== BOOL-producing ops ===")
    print(f"  Cast(to=BOOL):          {len(cast_to_bool_nodes)}")
    for op, nodes in sorted(bool_op_producers.items()):
        print(f"  {op:24s} {len(nodes)}")
    print(f"  BOOL initializers:       {len(bool_inits)}")
    print(f"  total BOOL tensors:     {len(bool_tensors)}")

    # For each BOOL-producing node, report upstream lineage + downstream sink.
    print(f"\n=== Cast(to=BOOL) detail ===")
    for i, n in enumerate(cast_to_bool_nodes):
        label = node_label(n)
        cast_in = n.input[0] if n.input else "<none>"
        cast_out = n.output[0] if n.output else "<none>"
        in_dtype = tensor_type_name(dtype_map.get(cast_in, 0))
        print(f"\n[{i}] {label}")
        print(f"    input : {cast_in} ({in_dtype})")
        print(f"    output: {cast_out} (BOOL)")
        print(f"    upstream lineage (producer -> tensor):")
        for line in walk_up(cast_in, producer)[:10]:
            print(f"      {line}")
        print(f"    downstream until non-BOOL sink:")
        for line in walk_down_to_non_bool(
            cast_out, consumers, producer, dtype_map, bool_tensors
        )[:20]:
            print(f"      {line}")

    print(f"\n=== Other BOOL producers ===")
    for op, nodes in sorted(bool_op_producers.items()):
        for n in nodes:
            label = node_label(n)
            outs = ", ".join(n.output)
            print(f"\n  {label} -> {outs}")
            print(f"    upstream lineage (producer -> tensor):")
            for line in walk_up(n.input[0], producer)[:10]:
                print(f"      {line}")
            print(f"    downstream until non-BOOL sink:")
            for o in n.output:
                for line in walk_down_to_non_bool(
                    o, consumers, producer, dtype_map, bool_tensors
                )[:10]:
                    print(f"      {line}")

    # Specifically inspect /model/Gather_5 or whatever optimum 2.1 renames it to.
    print(f"\n=== Gather nodes whose data is BOOL or traces to attention_mask ===")
    for n in model.graph.node:
        if n.op_type != "Gather":
            continue
        data_name = n.input[0] if n.input else ""
        data_dtype = dtype_map.get(data_name, 0)
        # Trace back and see if attention_mask is an ancestor.
        ancestors = set()
        queue = deque([data_name])
        while queue:
            t = queue.popleft()
            if t in ancestors or len(ancestors) > 60:
                continue
            ancestors.add(t)
            src = producer.get(t)
            if src is not None:
                for inp in src.input:
                    queue.append(inp)
        if "attention_mask" in ancestors or data_dtype == TensorProto.BOOL or \
           any(tensor_type_name(dtype_map.get(a, 0)) == "BOOL" for a in ancestors):
            name = n.name or "<anon>"
            print(f"  Gather[{name}]  data={data_name}({tensor_type_name(data_dtype)}) "
                  f"indices={n.input[1] if len(n.input) > 1 else '?'}  "
                  f"attention_mask-in-ancestors={'yes' if 'attention_mask' in ancestors else 'no'}")

    # Collect structural summary for the JSON report.
    summary = {
        "model": str(args.model),
        "node_count": len(model.graph.node),
        "cast_to_bool_count": len(cast_to_bool_nodes),
        "cast_to_bool_nodes": [
            {
                "name": n.name or "",
                "op_type": n.op_type,
                "input": n.input[0] if n.input else "",
                "input_dtype": tensor_type_name(dtype_map.get(n.input[0], 0))
                    if n.input else "",
                "output": n.output[0] if n.output else "",
            }
            for n in cast_to_bool_nodes
        ],
        "bool_op_counts": {op: len(ns) for op, ns in bool_op_producers.items()},
        "total_bool_tensors": len(bool_tensors),
        "bool_initializers": len(bool_inits),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote JSON summary to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
