"""Pin symbolic dims on the Qwen3-4B pathb ONNX for QAIRT compile.

Optimum exports leave the decode-shape dims symbolic (batch_size=?,
sequence_length=?, past_sequence_length=?). QAIRT's qairt-converter
needs concrete dims to lower the graph to DLC.

Pins to match the Qualcomm Qwen3-4B Genie reference bundle:
  CL=512 (past_sequence_length=511, attention window seq_k=512)
  AR=1  (sequence_length=1 = single-token decode step)

Output goes to models/qwen3-4b-arm-pathb-ctx512/ with model.onnx +
model.data (the QAIRT-friendly external-data filename — qai-hub also
expects this, and qairt-converter handles either name).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[1]


# torch.onnx auto-generates dim_param names like "Concatpresent.5.key_dim_1"
# when it can't propagate inferred shapes (e.g. shape inference disabled,
# as we do for the >2 GiB protobuf-cap workaround in optimum_export_4b.py).
# Match each pattern and resolve to a concrete value derived from the
# transformer config so the graph fully pins.
def _resolve_auto_dim(name: str, *, ctx: int, num_kv_heads: int,
                      head_dim: int, vocab_size: int) -> int | None:
    """Returns concrete int for known auto-dim patterns, else None."""
    # Concatpresent.<L>.{key,value}_dim_{1,2,3}: present_kv at this layer.
    # Shape (B, num_kv_heads, ctx, head_dim).
    m = re.match(r"Concatpresent\.\d+\.(key|value)_dim_(\d+)$", name)
    if m:
        idx = int(m.group(2))
        return {1: num_kv_heads, 2: ctx, 3: head_dim}.get(idx)
    # MatMullogits_dim_2: logits matmul output third dim = vocab_size.
    if name == "MatMullogits_dim_2":
        return vocab_size
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path,
                        default=REPO / "models" / "qwen3-4b-arm-pathb")
    parser.add_argument("--dst-dir", type=Path,
                        default=REPO / "models" / "qwen3-4b-arm-pathb-ctx512")
    parser.add_argument("--ctx", type=int, default=512,
                        help="Total attention window (past + current). "
                             "past_sequence_length is pinned to ctx-1.")
    parser.add_argument("--seq-q", type=int, default=1,
                        help="Decode-step query length (AR=1 for single-token).")
    # Auto-dim resolution context — only consulted when the bare dim_map
    # below misses the dim_param (i.e. for torch.onnx auto-generated names).
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="Number of KV heads (8 for Qwen3-{0.6B,4B,14B}, "
                             "consult config.json for new model families).")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=151936)
    args = parser.parse_args()

    src_onnx = args.src_dir / "model.onnx"
    dst_onnx = args.dst_dir / "model.onnx"
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    # past sequence holds ctx-1 slots; current step adds one to make ctx.
    past_len = args.ctx - 1

    dim_map = {
        "batch_size": 1,
        "sequence_length": args.seq_q,
        "past_sequence_length": past_len,
        "past_sequence_length + sequence_length": args.ctx,
        "seq_q": args.seq_q,
        "seq_k": args.ctx,
    }

    print(f"loading {src_onnx} (graph only — external data stays put) ...")
    m = onnx.load(str(src_onnx), load_external_data=False)

    # Repoint external_data location from model.onnx_data -> model.data
    # (matches the AI Hub / Qualcomm convention; both qairt-converter and
    # the existing pipeline expect this name).
    init_patched = 0
    for tensor in m.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value == "model.onnx_data":
                entry.value = "model.data"
                init_patched += 1
                break
    node_patched = 0
    for node in m.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.external_data:
                for e in attr.t.external_data:
                    if e.key == "location" and e.value == "model.onnx_data":
                        e.value = "model.data"
                        node_patched += 1
                        break
    print(f"  external_data refs repointed: {init_patched} initializers + {node_patched} node attrs")

    # Pin every symbolic dim across input / output / value_info collections.
    substituted = 0
    auto_resolved = 0
    unresolved: set[str] = set()
    for collection in (m.graph.input, m.graph.output, m.graph.value_info):
        for vi in collection:
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_param"):
                    name = d.dim_param
                    if name in dim_map:
                        d.dim_value = int(dim_map[name])
                        substituted += 1
                    else:
                        # Try the auto-dim resolver (handles Concatpresent.N.*
                        # / MatMullogits names that torch.onnx emits when
                        # shape inference is disabled).
                        v = _resolve_auto_dim(
                            name, ctx=args.ctx, num_kv_heads=args.num_kv_heads,
                            head_dim=args.head_dim, vocab_size=args.vocab_size,
                        )
                        if v is not None:
                            d.dim_value = int(v)
                            auto_resolved += 1
                        else:
                            unresolved.add(name)
    print(f"  pinned {substituted} symbolic dims; auto-resolved {auto_resolved}; "
          f"unresolved: {sorted(unresolved)}")
    if unresolved:
        print(f"FATAL: {len(unresolved)} unknown dim_params; refusing to write")
        return 2

    # Sanity: report pinned-input shapes
    print(f"  graph after pin:")
    print(f"    inputs={len(m.graph.input)} outputs={len(m.graph.output)}")
    for inp in m.graph.input[:3]:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}  shape={shape}")
    for inp in m.graph.input[-3:]:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}  shape={shape}")

    # Save graph; copy weights file under the new name.
    print(f"saving graph to {dst_onnx} ...")
    onnx.save(m, str(dst_onnx))

    src_data = args.src_dir / "model.data"
    if not src_data.exists():
        # fallback: optimum's name
        alt = args.src_dir / "model.onnx_data"
        if alt.exists():
            src_data = alt
        else:
            print(f"FATAL: no external-data file at {src_data} or {alt}")
            return 2
    dst_data = args.dst_dir / "model.data"
    if dst_data.exists():
        dst_data.unlink()
    print(f"copying weights {src_data.name} -> {dst_data} ...")
    shutil.copy2(src_data, dst_data)
    print(f"  done. weights: {dst_data.stat().st_size / 1e9:.2f} GB")

    # Mirror sidecars for tokenizer use later.
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "added_tokens.json", "special_tokens_map.json", "merges.txt",
                 "vocab.json", "generation_config.json", "chat_template.jinja"):
        src = args.src_dir / name
        if src.exists():
            shutil.copy2(src, args.dst_dir / name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
