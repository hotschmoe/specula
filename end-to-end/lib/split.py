"""Split a pathb-pinned (or AIMET-emitted post-pathb) ONNX into N
weight-sharing-friendly sub-ONNXs at the same seams Qualcomm uses in
their shipping multi-part bundles.

The default split for an N-layer transformer:

  part 1            : input_ids → embed.Gather
  parts 2..N-1      : layer_start..layer_end (12 layers each)
  last part         : last_layers + final_norm + lm_head → logits

Each sub-graph is independent (own initializers, own external-data
file). The seam between consecutive parts is the residual-stream
hidden tensor (`/model/layers.<L-1>/Add_1_output_0` for layer L).

This generalizes scripts/split_qwen3_4b_pathb.py. The only thing
that changes between models is the layer count, hidden size, vocab
size, num_kv_heads, head_dim — all read from ModelInfo.

Encodings: AIMET emits a single JSON with `activation_encodings` and
`param_encodings` keyed by tensor name. After splitting the graph,
each part carries the subset of encodings whose tensor names appear
in that part's nodes/initializers/inputs/outputs. We split the
encodings JSON the same way so qairt-converter
`--quantization_overrides` works per part.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import load_external_data_for_tensor


@dataclass
class PartSpec:
    name: str
    inputs: list[tuple]   # [(tensor_name, elem_dtype, shape), ...]
    outputs: list[tuple]


def build_part_specs(
    *, num_layers: int, hidden_size: int, vocab_size: int,
    num_kv_heads: int, head_dim: int, ctx: int,
    num_parts: int = 4,
) -> list[PartSpec]:
    """Layout: part 1 is embed-only; the remaining (num_parts - 1) parts
    each take an even slice of layers; the last part absorbs the residual
    layers + norm + lm_head → logits."""
    if num_parts < 2:
        raise ValueError("num_parts must be >= 2 (embed + at least one decoder part)")

    layers_per = num_layers // (num_parts - 1)
    extra = num_layers - layers_per * (num_parts - 1)

    past = ctx - 1
    hidden_shape = [1, 1, hidden_size]

    def decode_inputs(start: int, end: int) -> list[tuple]:
        items = [
            ("attention_bias", TensorProto.FLOAT, [1, 1, 1, ctx]),
            ("position_ids_cos", TensorProto.FLOAT, [1, 1, head_dim]),
            ("position_ids_sin", TensorProto.FLOAT, [1, 1, head_dim]),
        ]
        for li in range(start, end + 1):
            items.append((f"past_key_values.{li}.key", TensorProto.FLOAT,
                          [1, num_kv_heads, past, head_dim]))
            items.append((f"past_key_values.{li}.value", TensorProto.FLOAT,
                          [1, num_kv_heads, past, head_dim]))
        return items

    def decode_outputs(start: int, end: int) -> list[tuple]:
        items = []
        for li in range(start, end + 1):
            items.append((f"present.{li}.key", TensorProto.FLOAT,
                          [1, num_kv_heads, ctx, head_dim]))
            items.append((f"present.{li}.value", TensorProto.FLOAT,
                          [1, num_kv_heads, ctx, head_dim]))
        return items

    specs: list[PartSpec] = []
    # Part 1 — embed only.
    specs.append(PartSpec(
        name="part1",
        inputs=[("input_ids", TensorProto.INT64, [1, 1])],
        outputs=[("/model/embed_tokens/Gather_output_0", TensorProto.FLOAT, hidden_shape)],
    ))

    # Parts 2..num_parts. The last part absorbs `extra` extra layers + norm + lm_head.
    layer_cursor = 0
    for pi in range(2, num_parts + 1):
        is_last = (pi == num_parts)
        n_layers_here = layers_per + (extra if is_last else 0)
        layer_start = layer_cursor
        layer_end = layer_cursor + n_layers_here - 1
        layer_cursor = layer_end + 1

        # Input seam: previous part's last layer's Add_1, OR the embed.
        if pi == 2:
            seam_in = "/model/embed_tokens/Gather_output_0"
        else:
            seam_in = f"/model/layers.{layer_start - 1}/Add_1_output_0"

        # Output seam: this part's last layer's Add_1, except last part outputs logits.
        if is_last:
            seam_out = ("logits", TensorProto.FLOAT, [1, 1, vocab_size])
        else:
            seam_out = (f"/model/layers.{layer_end}/Add_1_output_0",
                        TensorProto.FLOAT, hidden_shape)

        specs.append(PartSpec(
            name=f"part{pi}",
            inputs=[(seam_in, TensorProto.FLOAT, hidden_shape)] + decode_inputs(layer_start, layer_end),
            outputs=[seam_out] + decode_outputs(layer_start, layer_end),
        ))

    if layer_cursor != num_layers:
        raise RuntimeError(
            f"layer accounting bug: cursor {layer_cursor} != num_layers {num_layers}"
        )

    return specs


def extract_part(
    model: onnx.ModelProto,
    part: PartSpec,
    src_dir: Path,
    dst_dir: Path,
) -> dict:
    """Extract `part` from `model` (loaded WITHOUT external data) and
    save to dst_dir/model.onnx + model.onnx_data. Only the initializers
    this part needs are loaded into memory."""
    t0 = time.perf_counter()
    input_names = {n for n, _, _ in part.inputs}
    output_names = {n for n, _, _ in part.outputs}

    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producer[out] = node

    init_map: dict[str, onnx.TensorProto] = {init.name: init for init in model.graph.initializer}

    selected_node_set: set[int] = set()
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
                f"[{part.name}] tensor '{t}' has no producer and is "
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
        print(f"  [{part.name}] WARNING: declared inputs not reached: {sorted(missing)}")

    selected_nodes: list[onnx.NodeProto] = [
        n for n in model.graph.node if id(n) in selected_node_set
    ]

    selected_inits: list[onnx.TensorProto] = []
    loaded_bytes = 0
    for init in model.graph.initializer:
        if init.name not in selected_init_names:
            continue
        if init.data_location == onnx.TensorProto.EXTERNAL:
            load_external_data_for_tensor(init, str(src_dir))
            loaded_bytes += len(init.raw_data)
        selected_inits.append(init)

    def make_vi(name: str, elem: int, shape: list[int]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, elem, shape)

    graph_inputs = [make_vi(n, e, s) for n, e, s in part.inputs]
    graph_outputs = [make_vi(n, e, s) for n, e, s in part.outputs]

    sub_graph = helper.make_graph(
        nodes=selected_nodes,
        name=f"specula_pathb_{part.name}",
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
    onnx.save(
        sub_model, str(dst_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )
    data_path = dst_dir / "model.onnx_data"
    data_size = data_path.stat().st_size if data_path.exists() else 0
    info = {
        "part": part.name,
        "wall_s": time.perf_counter() - t0,
        "n_nodes": len(selected_nodes),
        "n_initializers": len(selected_inits),
        "weights_loaded_gb": loaded_bytes / 1e9,
        "data_size_gb": data_size / 1e9,
        "model_onnx_path": str(dst_onnx),
    }
    print(f"  [{part.name}] {info['n_nodes']} nodes, {info['n_initializers']} inits, "
          f"data {info['data_size_gb']:.2f} GB ({info['wall_s']:.1f}s)")

    # Release initializer bytes back to allocator so peaks don't compound.
    for init in selected_inits:
        init.ClearField("raw_data")

    # Capture the per-part tensor name set for encoding subsetting.
    tensor_names = set()
    for n in selected_nodes:
        tensor_names.update(n.input)
        tensor_names.update(n.output)
    for init in selected_inits:
        tensor_names.add(init.name)
    for n, _, _ in part.inputs:
        tensor_names.add(n)
    for n, _, _ in part.outputs:
        tensor_names.add(n)
    info["tensor_names"] = tensor_names
    return info


def split_encodings(
    *, src_encodings: Path, parts_info: list[dict],
    dst_dirs: list[Path],
) -> list[Path]:
    """Subset src_encodings into per-part encodings JSONs based on each
    part's tensor name set. Tensors absent from a part are dropped from
    that part's encoding file. Returns list of written encoding paths."""
    raw = json.loads(src_encodings.read_text())
    # AIMET emits either {activation_encodings, param_encodings} (v1)
    # or just a flat dict. Handle both.
    sections = ["activation_encodings", "param_encodings"]
    if not any(k in raw for k in sections):
        # Flat dict — wrap as a single section for uniform handling.
        flat_in = raw
        encs_root = {"_flat": flat_in}
        sections = ["_flat"]
    else:
        encs_root = raw

    written: list[Path] = []
    for part_info, dst_dir in zip(parts_info, dst_dirs):
        names = part_info["tensor_names"]
        out_root = {}
        for sec in sections:
            sec_in = encs_root.get(sec, {})
            sub = {k: v for k, v in sec_in.items() if k in names}
            out_root[sec] = sub
        # If wrapped, unwrap.
        if "_flat" in out_root:
            out_root = out_root["_flat"]
        # Carry over any sibling metadata (version, quantizer_args, etc).
        for k, v in raw.items():
            if k not in sections and k not in out_root:
                out_root[k] = v

        dst_path = dst_dir / "model.encodings"
        dst_path.write_text(json.dumps(out_root, indent=2))
        n_act = sum(1 for sec in ("activation_encodings",) if sec in out_root for _ in out_root[sec])
        n_par = sum(1 for sec in ("param_encodings",) if sec in out_root for _ in out_root[sec])
        print(f"  [{part_info['part']}] encodings: act={n_act}, param={n_par} → {dst_path.name}")
        written.append(dst_path)

    return written


def split_aimet_output(
    *, aimet_dir: Path, export_prefix: str, model_info, ctx: int,
    out_root: Path, num_parts: int = 4,
) -> dict:
    """Drive the full split: load AIMET-emitted ONNX + encodings, write
    `num_parts` sub-onnx + sub-encodings under out_root/part{N}/.

    AIMET's `sim.export(prefix=PREFIX)` writes
        {prefix}.onnx              — the pathb graph (QDQs stripped)
        {prefix}.encodings         — per-tensor scale/offset/bw
        {prefix}.data              — external weight data
    We consume those and produce
        out_root/part{1..num_parts}/{model.onnx, model.onnx_data, model.encodings}
    """
    src_onnx = aimet_dir / f"{export_prefix}.onnx"
    src_enc = aimet_dir / f"{export_prefix}.encodings"
    if not src_onnx.exists() or not src_enc.exists():
        raise FileNotFoundError(
            f"missing AIMET output: {src_onnx} or {src_enc}"
        )

    print(f"[split] loading AIMET ONNX (graph only, external data deferred): {src_onnx}")
    t0 = time.perf_counter()
    m = onnx.load(str(src_onnx), load_external_data=False)
    print(f"  graph: {len(m.graph.node)} nodes, {len(m.graph.initializer)} inits "
          f"({time.perf_counter() - t0:.1f}s)")

    specs = build_part_specs(
        num_layers=model_info.num_hidden_layers,
        hidden_size=model_info.hidden_size,
        vocab_size=model_info.vocab_size,
        num_kv_heads=model_info.num_key_value_heads,
        head_dim=model_info.head_dim,
        ctx=ctx, num_parts=num_parts,
    )

    parts_info: list[dict] = []
    dst_dirs: list[Path] = []
    for spec in specs:
        dst_dir = out_root / spec.name
        print(f"\n--- extracting {spec.name} → {dst_dir} ---")
        info = extract_part(m, spec, aimet_dir, dst_dir)
        parts_info.append(info)
        dst_dirs.append(dst_dir)

    print(f"\n--- splitting encodings JSON ---")
    enc_paths = split_encodings(
        src_encodings=src_enc, parts_info=parts_info, dst_dirs=dst_dirs,
    )

    overall = {
        "num_parts": num_parts,
        "parts": [
            {k: v for k, v in p.items() if k != "tensor_names"}  # drop the set
            for p in parts_info
        ],
        "encoding_paths": [str(p) for p in enc_paths],
    }
    return overall


if __name__ == "__main__":
    # CLI entry for ad-hoc use, mirroring scripts/split_qwen3_4b_pathb.py.
    import argparse, sys
    from .model_config import load_model_info
    p = argparse.ArgumentParser()
    p.add_argument("--aimet-dir", type=Path, required=True)
    p.add_argument("--export-prefix", type=str, required=True)
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--num-parts", type=int, default=4)
    p.add_argument("--out-root", type=Path, required=True)
    args = p.parse_args()
    info = load_model_info(model_id=args.model_id, model_path=args.model_path)
    res = split_aimet_output(
        aimet_dir=args.aimet_dir, export_prefix=args.export_prefix,
        model_info=info, ctx=args.ctx, out_root=args.out_root,
        num_parts=args.num_parts,
    )
    print(json.dumps(res, indent=2))
