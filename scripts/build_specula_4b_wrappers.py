"""Build 4 EPContext wrapper ONNX files pointing at our single merged
4-graph context binary (`qwen3_4b_4part_w4a16.bin`).

ctx-bin-gen with comma-separated `--dlc_path` packs all 4 graphs into
ONE .bin; graph selection at run time is by IO-name matching (each
wrapper declares the IO signature of exactly one graph, and ORT-QNN
routes it to that graph). Tensor names must match the .bin's
underscored form (leading `/` stripped, `/` -> `_`, `.` -> `_`), not
our source ONNX slash/dot form. Dtypes come from the actual compiled
binary: `input_ids` int32, everything else uint16 at this stage
(Phase 4 levers can push KV to uint8 and cos/sin to half-dim later).

Run AFTER ctx-bin-gen:
    .venv/Scripts/python.exe scripts/build_specula_4b_wrappers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
VOCAB = 151936
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Underscored tensor names (match the .bin's internal layout).
EMBED_HIDDEN = "_model_embed_tokens_Gather_output_0"
L11_HIDDEN = "_model_layers_11_Add_1_output_0"
L23_HIDDEN = "_model_layers_23_Add_1_output_0"


def past_name(kind: str, layer: int) -> str:
    return f"past_key_values_{layer}_{kind}"


def present_name(kind: str, layer: int) -> str:
    return f"present_{layer}_{kind}"


def part_io_spec(part: int) -> tuple[list[tuple], list[tuple], str]:
    """Returns (inputs, outputs, internal_graph_name). Names here match
    the compiled .bin's per-graph IO exactly."""
    def decode_inputs(s: int, e: int) -> list[tuple]:
        items = [
            ("attention_bias", "uint16", [1, 1, 1, CTX]),
            # Phase 5o: half-dim cos/sin [1, 1, 64] matching Qualcomm's
            # genie bundle. Graph internally concats to [1,1,128] before
            # the existing Unsqueeze + rotary Muls.
            ("position_ids_cos", "uint16", [1, 1, HEAD_DIM // 2]),
            ("position_ids_sin", "uint16", [1, 1, HEAD_DIM // 2]),
        ]
        # Phase 5n: KV now uint8 (matching Qualcomm's genie bundle). Per-layer
        # scales applied via --quantization_overrides at convert time.
        for li in range(s, e + 1):
            items.append((past_name("key", li), "uint8",
                          [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
            items.append((past_name("value", li), "uint8",
                          [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
        return items

    def decode_outputs(s: int, e: int) -> list[tuple]:
        return [
            (present_name(kind, li), "uint8",
             [1, NUM_KV_HEADS, CTX, HEAD_DIM])
            for li in range(s, e + 1) for kind in ("key", "value")
        ]

    if part == 1:
        return (
            [("input_ids", "int32", [1, 1])],
            [(EMBED_HIDDEN, "uint16", [1, 1, HIDDEN])],
            "qwen3_4b_part1_fp32",
        )
    if part == 2:
        return (
            [(EMBED_HIDDEN, "uint16", [1, 1, HIDDEN])] + decode_inputs(0, 11),
            [(L11_HIDDEN, "uint16", [1, 1, HIDDEN])] + decode_outputs(0, 11),
            "qwen3_4b_part2_fp32",
        )
    if part == 3:
        return (
            [(L11_HIDDEN, "uint16", [1, 1, HIDDEN])] + decode_inputs(12, 23),
            [(L23_HIDDEN, "uint16", [1, 1, HIDDEN])] + decode_outputs(12, 23),
            "qwen3_4b_part3_fp32",
        )
    if part == 4:
        return (
            [(L23_HIDDEN, "uint16", [1, 1, HIDDEN])] + decode_inputs(24, 35),
            [("logits", "uint16", [1, 1, VOCAB])] + decode_outputs(24, 35),
            "qwen3_4b_part4_fp32",
        )
    raise ValueError(f"bad part {part}")


_DTYPE_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
}


def build_wrapper(part: int, bin_path: Path, dst: Path) -> None:
    inputs, outputs, graph_name = part_io_spec(part)
    inputs_decl = [helper.make_tensor_value_info(n, _DTYPE_PROTO[dt], s)
                   for n, dt, s in inputs]
    outputs_decl = [helper.make_tensor_value_info(n, _DTYPE_PROTO[dt], s)
                    for n, dt, s in outputs]
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name=graph_name,
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_path.name,
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"specula_qwen3_4b_wrapper_part{part}",
        inputs=inputs_decl,
        outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-4part-bundle",
    )
    model.ir_version = 10
    onnx.save(model, str(dst))
    print(f"wrote {dst.name} -> graph={graph_name}, "
          f"{len(inputs_decl)} inputs / {len(outputs_decl)} outputs, "
          f"ep_cache_context={bin_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=RESULTS,
                        help="Directory holding per-part .bin files.")
    parser.add_argument("--binary-basename", type=str,
                        default="qwen3_4b_4part_w4a16",
                        help="Per-part bin stem: <basename>_part{N}.bin.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {int(p) for p in args.parts.split(",")}
    for part in (1, 2, 3, 4):
        if part not in wanted:
            continue
        bin_path = args.bin_dir / f"{args.binary_basename}_part{part}.bin"
        if not bin_path.exists():
            print(f"WARNING: .bin not found at {bin_path}")
        dst = args.out_dir / f"specula_qwen3_4b_part{part}.wrapper.onnx"
        build_wrapper(part, bin_path, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
