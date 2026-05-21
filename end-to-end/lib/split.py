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
import re
import time
from collections import defaultdict
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


def detect_shared_attn_mask(model: onnx.ModelProto) -> Optional[str]:
    """Return the pathb internal attention-mask tensor name, or None.

    transformers 4.51 exports the causal mask as an internal subgraph
    (a `ScatterND`); the pathb `fold-pathbmask` rewrite folds that in and
    leaves the original `attention_bias` graph input dead. The folded mask
    is built ONCE in the model preamble and then consumed by a `Slice`
    inside every decoder layer's `self_attn`.

    In a single-bin model that is fine. In an N-part split the preamble
    lands wholly inside the first decoder part, so the mask becomes a
    genuine cross-part tensor — it must be threaded from the part that
    produces it into every later part that consumes it. We identify it
    structurally: the one tensor fed to a `Slice` in >=2 distinct layers'
    `self_attn` (no false positives — per-layer Slices touch per-layer
    tensors; only the shared mask spans layers).
    """
    layers_for_tensor: dict[str, set[int]] = defaultdict(set)
    for n in model.graph.node:
        if n.op_type != "Slice":
            continue
        mm = re.match(r"/model/layers\.(\d+)/self_attn/", n.name or "")
        if not mm:
            continue
        li = int(mm.group(1))
        for inp in n.input:
            if inp:
                layers_for_tensor[inp].add(li)
    shared = sorted(t for t, ls in layers_for_tensor.items() if len(ls) >= 2)
    if not shared:
        return None
    if len(shared) > 1:
        raise RuntimeError(
            "split: expected exactly one cross-layer self_attn mask tensor, "
            f"found {len(shared)}: {shared}. The cross-part threading logic "
            "assumes a single shared mask — extend it before proceeding."
        )
    return shared[0]


def build_part_specs(
    *, num_layers: int, hidden_size: int, vocab_size: int,
    num_kv_heads: int, head_dim: int, ctx: int,
    num_parts: int = 4, shared_mask: Optional[str] = None,
) -> list[PartSpec]:
    """Layout: part 1 is embed-only; the remaining (num_parts - 1) parts
    each take an even slice of layers; the last part absorbs the residual
    layers + norm + lm_head → logits.

    `shared_mask`, if given, is the pathb internal attention-mask tensor
    (see `detect_shared_attn_mask`). It is threaded as cross-part I/O: the
    first decoder part produces and exports it, the last part imports it,
    and middle parts pass it through (import + re-export) so it reaches
    every decoder part even under a strict consecutive-wiring runtime.
    The dead `attention_bias` graph input is dropped from every part.
    """
    if num_parts < 2:
        raise ValueError("num_parts must be >= 2 (embed + at least one decoder part)")

    layers_per = num_layers // (num_parts - 1)
    extra = num_layers - layers_per * (num_parts - 1)

    past = ctx - 1
    hidden_shape = [1, 1, hidden_size]
    mask_shape = [1, 1, 1, ctx]

    def decode_inputs(start: int, end: int) -> list[tuple]:
        # `attention_bias` is intentionally absent — the pathb rewrite folds
        # the mask into the graph (see `shared_mask` threading below) and
        # leaves the original `attention_bias` input dead.
        items = [
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

        part_inputs = ([(seam_in, TensorProto.FLOAT, hidden_shape)]
                       + decode_inputs(layer_start, layer_end))
        part_outputs = [seam_out] + decode_outputs(layer_start, layer_end)

        # Thread the pathb internal attention mask across parts. The first
        # decoder part (pi == 2) produces it natively in the preamble it
        # absorbs, so it only re-exports it. Every later part imports it.
        # Any part with a later consumer also exports it — part2 produces +
        # exports; middle parts import + re-export (pass-through); the last
        # part only imports. With num_parts == 2 the mask stays internal.
        if shared_mask is not None:
            produces_mask = (pi == 2)
            has_later_consumer = (pi < num_parts)
            if not produces_mask:
                part_inputs.append((shared_mask, TensorProto.FLOAT, mask_shape))
            if has_later_consumer:
                part_outputs.append((shared_mask, TensorProto.FLOAT, mask_shape))

        specs.append(PartSpec(
            name=f"part{pi}",
            inputs=part_inputs,
            outputs=part_outputs,
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


def _subset_encoding_section(sec_value, names: set) -> tuple:
    """Subset one encodings section to entries whose tensor name is in
    `names`. Handles both AIMET schemas:
      * 1.0.0  — section is a list of {"name": <tensor>, ...} objects.
      * legacy — section is a dict keyed by tensor name.
    Returns (subset, kept_count)."""
    if isinstance(sec_value, list):
        sub = [e for e in sec_value
               if isinstance(e, dict) and e.get("name") in names]
        return sub, len(sub)
    if isinstance(sec_value, dict):
        sub = {k: v for k, v in sec_value.items() if k in names}
        return sub, len(sub)
    # Unknown shape — leave untouched rather than corrupt it.
    return sec_value, 0


def split_encodings(
    *, src_encodings: Path, parts_info: list[dict],
    dst_dirs: list[Path],
) -> list[Path]:
    """Subset src_encodings into per-part encodings JSONs based on each
    part's tensor name set. Tensors absent from a part are dropped from
    that part's encoding file. Returns list of written encoding paths.

    AIMET 2.x emits the 1.0.0 schema — `activation_encodings` and
    `param_encodings` are lists of per-tensor objects carrying a `name`
    field, alongside sibling metadata (`version`, `quantizer_args`). The
    legacy name-keyed-dict schema and a bare flat dict are also handled.
    """
    raw = json.loads(src_encodings.read_text())
    sections = ["activation_encodings", "param_encodings"]
    is_sectioned = any(k in raw for k in sections)

    written: list[Path] = []
    for part_info, dst_dir in zip(parts_info, dst_dirs):
        names = part_info["tensor_names"]
        if is_sectioned:
            # Carry version / quantizer_args / etc. through unchanged,
            # overwrite only the two encoding sections with their subsets.
            out_root = dict(raw)
            counts = {}
            for sec in sections:
                if sec not in raw:
                    continue
                out_root[sec], counts[sec] = _subset_encoding_section(
                    raw[sec], names)
            n_act = counts.get("activation_encodings", 0)
            n_par = counts.get("param_encodings", 0)
        else:
            # Bare flat dict — the whole document is a name-keyed map.
            out_root, n_act = _subset_encoding_section(raw, names)
            n_par = 0

        dst_path = dst_dir / "model.encodings"
        dst_path.write_text(json.dumps(out_root, indent=2))
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

    shared_mask = detect_shared_attn_mask(m)
    if shared_mask is not None:
        print(f"[split] cross-part attention mask: {shared_mask} "
              f"(threaded part2 → parts 3..{num_parts})")
    else:
        print("[split] no shared cross-part attention mask detected")

    specs = build_part_specs(
        num_layers=model_info.num_hidden_layers,
        hidden_size=model_info.hidden_size,
        vocab_size=model_info.vocab_size,
        num_kv_heads=model_info.num_key_value_heads,
        head_dim=model_info.head_dim,
        ctx=ctx, num_parts=num_parts, shared_mask=shared_mask,
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
