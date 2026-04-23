"""Split the Qwen3-4B pathb ctx=512 graph into 4 sub-ONNX files at the
same seams Qualcomm uses in their shipping bundle:

  part1 : input_ids -> embed.Gather
  part2 : embed_hidden + attn_bias + cos/sin + past_kv[0..11]
             -> layer11.hidden + present_kv[0..11]
  part3 : layer11.hidden + attn_bias + cos/sin + past_kv[12..23]
             -> layer23.hidden + present_kv[12..23]
  part4 : layer23.hidden + attn_bias + cos/sin + past_kv[24..35]
             -> logits + present_kv[24..35]

Algorithm: backward BFS from each part's declared output tensors. We
stop descending when we hit a declared part input (boundary tensor)
or an initializer. Every node we traverse lands in this part, every
initializer we consume lands in this part's external-data file. The
input model's external data is loaded only for the initializers each
part claims, so peak memory stays near one part's weights (~4.5 GB
fp32) rather than the whole 17.6 GB graph.

Run:
    .venv/Scripts/python.exe scripts/split_qwen3_4b_pathb.py

Output: models/qwen3-4b-arm-pathb-ctx512-part{1..4}/ each with
        model.onnx + model.data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import load_external_data_for_tensor


REPO = Path(__file__).resolve().parents[1]

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
VOCAB = 151936
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128


def hidden_shape(seq_q: int = 1) -> list[int]:
    return [1, seq_q, HIDDEN]


def build_part_specs() -> list[dict]:
    """For each part, declare inputs (boundary tensors coming in) and
    outputs (tensors this part must produce). Names match the source
    pathb-ctx512 graph verbatim."""
    specs: list[dict] = []

    # Part 1 - embed only.
    specs.append({
        "name": "part1",
        "inputs": [
            ("input_ids", TensorProto.INT64, [1, 1]),
        ],
        "outputs": [
            ("/model/embed_tokens/Gather_output_0", TensorProto.FLOAT, hidden_shape()),
        ],
    })

    # Helper for middle/last parts: shared non-KV inputs.
    def decode_inputs(layer_start: int, layer_end: int) -> list[tuple]:
        items = [
            ("attention_bias", TensorProto.FLOAT, [1, 1, 1, CTX]),
            ("position_ids_cos", TensorProto.FLOAT, [1, 1, HEAD_DIM]),
            ("position_ids_sin", TensorProto.FLOAT, [1, 1, HEAD_DIM]),
        ]
        for li in range(layer_start, layer_end + 1):
            items.append((f"past_key_values.{li}.key", TensorProto.FLOAT,
                          [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
            items.append((f"past_key_values.{li}.value", TensorProto.FLOAT,
                          [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
        return items

    def decode_outputs(layer_start: int, layer_end: int) -> list[tuple]:
        items = []
        for li in range(layer_start, layer_end + 1):
            items.append((f"present.{li}.key", TensorProto.FLOAT,
                          [1, NUM_KV_HEADS, CTX, HEAD_DIM]))
            items.append((f"present.{li}.value", TensorProto.FLOAT,
                          [1, NUM_KV_HEADS, CTX, HEAD_DIM]))
        return items

    # Part 2 - layers 0..11.
    specs.append({
        "name": "part2",
        "inputs": [
            ("/model/embed_tokens/Gather_output_0", TensorProto.FLOAT, hidden_shape()),
        ] + decode_inputs(0, 11),
        "outputs": [
            ("/model/layers.11/Add_1_output_0", TensorProto.FLOAT, hidden_shape()),
        ] + decode_outputs(0, 11),
    })

    # Part 3 - layers 12..23.
    specs.append({
        "name": "part3",
        "inputs": [
            ("/model/layers.11/Add_1_output_0", TensorProto.FLOAT, hidden_shape()),
        ] + decode_inputs(12, 23),
        "outputs": [
            ("/model/layers.23/Add_1_output_0", TensorProto.FLOAT, hidden_shape()),
        ] + decode_outputs(12, 23),
    })

    # Part 4 - layers 24..35 + norm + lm_head.
    specs.append({
        "name": "part4",
        "inputs": [
            ("/model/layers.23/Add_1_output_0", TensorProto.FLOAT, hidden_shape()),
        ] + decode_inputs(24, 35),
        "outputs": [
            ("logits", TensorProto.FLOAT, [1, 1, VOCAB]),
        ] + decode_outputs(24, 35),
    })

    return specs


def extract_part(
    model: onnx.ModelProto,
    part: dict,
    src_dir: Path,
    dst_dir: Path,
) -> None:
    """Extract `part` from `model` (loaded WITHOUT external data) and
    save to dst_dir/model.onnx with its own model.data. Only the
    initializers this part needs get loaded into memory."""
    t0 = time.perf_counter()
    input_names = {n for n, _, _ in part["inputs"]}
    output_names = {n for n, _, _ in part["outputs"]}

    # Producer map: tensor_name -> Node
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    init_map: dict[str, onnx.TensorProto] = {init.name: init for init in model.graph.initializer}

    # Backward BFS from outputs to collect nodes + initializers we need.
    selected_node_set: set[int] = set()   # id() keys, since NodeProto not hashable by content
    selected_init_names: set[str] = set()
    boundary_seen: set[str] = set()

    open_tensors: list[str] = list(output_names)
    while open_tensors:
        t = open_tensors.pop()
        if t in input_names:
            boundary_seen.add(t)
            continue
        if t in init_map:
            selected_init_names.add(t)
            continue
        node = producer.get(t)
        if node is None:
            raise RuntimeError(
                f"[{part['name']}] tensor '{t}' has no producer and is "
                f"neither a declared input nor an initializer"
            )
        if id(node) in selected_node_set:
            continue
        selected_node_set.add(id(node))
        for inp in node.input:
            if inp:
                open_tensors.append(inp)

    missing = input_names - boundary_seen
    if missing:
        # Not fatal, but the caller may have declared inputs that aren't
        # actually consumed by this part (e.g. unused past_kv). Warn so we
        # know to prune them from the sub-graph's declared input list.
        print(f"  [{part['name']}] WARNING: declared inputs not reached: {sorted(missing)}")

    # Preserve topological order from the source graph.
    selected_nodes: list[onnx.NodeProto] = [
        n for n in model.graph.node if id(n) in selected_node_set
    ]

    # Load external data ONLY for selected initializers.
    selected_inits: list[onnx.TensorProto] = []
    loaded_bytes = 0
    for init in model.graph.initializer:
        if init.name not in selected_init_names:
            continue
        if init.data_location == onnx.TensorProto.EXTERNAL:
            load_external_data_for_tensor(init, str(src_dir))
            loaded_bytes += len(init.raw_data)
        selected_inits.append(init)

    # Build sub-graph I/O ValueInfos.
    def make_vi(name: str, elem: int, shape: list[int]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, elem, shape)

    graph_inputs = [make_vi(n, e, s) for n, e, s in part["inputs"]]
    graph_outputs = [make_vi(n, e, s) for n, e, s in part["outputs"]]

    sub_graph = helper.make_graph(
        nodes=selected_nodes,
        name=f"qwen3_4b_pathb_ctx512_{part['name']}",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=selected_inits,
    )

    sub_model = helper.make_model(
        sub_graph,
        opset_imports=list(model.opset_import),
        producer_name="specula-split",
    )
    sub_model.ir_version = model.ir_version

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in dst_dir.iterdir():
        if f.suffix in {".onnx", ".data"} or f.name.endswith(".onnx_data"):
            f.unlink()
    dst_onnx = dst_dir / "model.onnx"
    print(f"  [{part['name']}] writing {dst_onnx} ({len(selected_nodes)} nodes, "
          f"{len(selected_inits)} initializers, {loaded_bytes / 1e9:.2f} GB weights)")
    onnx.save(
        sub_model,
        str(dst_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    data_path = dst_dir / "model.data"
    data_size = data_path.stat().st_size if data_path.exists() else 0
    print(f"  [{part['name']}] done in {time.perf_counter() - t0:.1f}s; "
          f"model.data = {data_size / 1e9:.2f} GB")

    # Release initializer bytes back to the allocator - next part call
    # re-loads what it needs and we don't want peaks compounding.
    for init in selected_inits:
        init.ClearField("raw_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir", type=Path,
        default=REPO / "models" / "qwen3-4b-arm-pathb-ctx512",
        help="Source pathb ctx=512 directory (model.onnx + model.data).",
    )
    parser.add_argument(
        "--dst-root", type=Path,
        default=REPO / "models",
        help="Parent directory for the 4 part dirs. Each lands at "
             "<dst_root>/qwen3-4b-arm-pathb-ctx512-part{1..4}/.",
    )
    parser.add_argument(
        "--parts", type=str, default="1,2,3,4",
        help="Comma-separated subset of parts to emit. Useful when iterating.",
    )
    args = parser.parse_args()

    src_onnx = args.src_dir / "model.onnx"
    src_data = args.src_dir / "model.data"
    if not src_onnx.exists() or not src_data.exists():
        print(f"FATAL: source not found at {args.src_dir} "
              f"(need model.onnx + model.data)")
        return 2

    print(f"loading graph {src_onnx} (external data NOT yet loaded) ...")
    t0 = time.perf_counter()
    m = onnx.load(str(src_onnx), load_external_data=False)
    print(f"  loaded graph: {len(m.graph.node)} nodes, "
          f"{len(m.graph.initializer)} initializers, "
          f"{len(m.graph.input)} inputs, {len(m.graph.output)} outputs "
          f"in {time.perf_counter() - t0:.1f}s")

    wanted = {int(p) for p in args.parts.split(",")}
    specs = build_part_specs()
    for idx, spec in enumerate(specs, start=1):
        if idx not in wanted:
            continue
        dst_dir = args.dst_root / f"qwen3-4b-arm-pathb-ctx512-{spec['name']}"
        print(f"\n--- extracting {spec['name']} -> {dst_dir} ---")
        extract_part(m, spec, args.src_dir, dst_dir)

    print("\nall requested parts extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
