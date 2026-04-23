"""Build 4 EPContext wrapper ONNX files pointing at our 4-part bundle.

Each wrapper declares the part's graph I/O and embeds the .bin path
via the EPContext `com.microsoft` op. ORT-QNN 2.1 (QAIRT 2.45) loads
the wrapper, follows the embedded context-binary reference, and runs
the graph on HTP.

Mirrors the layout of scripts/qualcomm_qwen3_4b_oracle.py but targets
our produced bundle (no quant params from metadata.yaml — we read
scales/offsets from each part's .dlc via qairt-dlc-to-json).

Run (needs scale/offset info, so run AFTER qairt-quantizer):
    .venv/Scripts/python.exe scripts/build_specula_4b_wrappers.py \\
        --dlc-dir results/phase5_qwen3_4b_bundle \\
        --bin-dir results/phase5_qwen3_4b_bundle \\
        --out-dir results/phase5_qwen3_4b_bundle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
VOCAB = 151936
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128

EMBED_HIDDEN = "/model/embed_tokens/Gather_output_0"
L11_HIDDEN = "/model/layers.11/Add_1_output_0"
L23_HIDDEN = "/model/layers.23/Add_1_output_0"


# Per-part I/O contract in declared order. Matches the split sub-ONNXs exactly.
def part_io_spec(part: int) -> tuple[list[tuple], list[tuple]]:
    def decode_inputs(s: int, e: int) -> list[tuple]:
        items = [
            ("attention_bias", "float32", [1, 1, 1, CTX]),
            ("position_ids_cos", "float32", [1, 1, HEAD_DIM]),
            ("position_ids_sin", "float32", [1, 1, HEAD_DIM]),
        ]
        for li in range(s, e + 1):
            items.append((f"past_key_values.{li}.key", "float32", [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
            items.append((f"past_key_values.{li}.value", "float32", [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
        return items

    def decode_outputs(s: int, e: int) -> list[tuple]:
        items = []
        for li in range(s, e + 1):
            items.append((f"present.{li}.key", "float32", [1, NUM_KV_HEADS, CTX, HEAD_DIM]))
            items.append((f"present.{li}.value", "float32", [1, NUM_KV_HEADS, CTX, HEAD_DIM]))
        return items

    if part == 1:
        return ([("input_ids", "int64", [1, 1])],
                [(EMBED_HIDDEN, "float32", [1, 1, HIDDEN])])
    if part == 2:
        return ([(EMBED_HIDDEN, "float32", [1, 1, HIDDEN])] + decode_inputs(0, 11),
                [(L11_HIDDEN, "float32", [1, 1, HIDDEN])] + decode_outputs(0, 11))
    if part == 3:
        return ([(L11_HIDDEN, "float32", [1, 1, HIDDEN])] + decode_inputs(12, 23),
                [(L23_HIDDEN, "float32", [1, 1, HIDDEN])] + decode_outputs(12, 23))
    if part == 4:
        return ([(L23_HIDDEN, "float32", [1, 1, HIDDEN])] + decode_inputs(24, 35),
                [("logits", "float32", [1, 1, VOCAB])] + decode_outputs(24, 35))
    raise ValueError(f"bad part {part}")


_DTYPE_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
}


def dlc_io_info(dlc_path: Path) -> dict[str, dict]:
    """Use qairt-dlc-to-json to read the per-IO scale/offset/dtype for
    the quantized DLC. Returns {tensor_name: {scale, offset, dtype}}."""
    cmd = ["qairt-dlc-to-json", "--input_dlc", str(dlc_path),
           "--output_json", str(dlc_path.with_suffix(".json"))]
    subprocess.run(cmd, check=True, capture_output=True)
    data = json.loads(dlc_path.with_suffix(".json").read_text(encoding="utf-8"))
    info: dict[str, dict] = {}
    # The DLC JSON nests tensors under different keys depending on version;
    # walk it robustly.
    for tensors_key in ("tensors", "graph_tensors", "input_tensors", "output_tensors"):
        tensors = data.get(tensors_key)
        if isinstance(tensors, dict):
            for name, meta in tensors.items():
                info[name] = meta
        elif isinstance(tensors, list):
            for meta in tensors:
                if "name" in meta:
                    info[meta["name"]] = meta
    return info


def build_wrapper(part: int, bin_path: Path, dst: Path,
                  dlc_io: dict[str, dict] | None = None) -> None:
    """Emit wrapper.onnx declaring the part's I/O and referencing bin_path
    via EPContext. Dtypes here use the fp32/int64 "pre-quant" declaration
    so CPU testers can feed dequantized fp32 values; if the DLC's IO is
    quantized, callers must still match the DLC's port dtypes at QNN
    session run time. (The wrapper's declared dtype at the ORT boundary
    is what ORT binds; QNN EP handles the dequant marshaling when the
    wrapper declares fp32 on a uint16/uint8 port.)
    """
    inputs, outputs = part_io_spec(part)
    inputs_decl = [helper.make_tensor_value_info(n, _DTYPE_PROTO[dt], s) for n, dt, s in inputs]
    outputs_decl = [helper.make_tensor_value_info(n, _DTYPE_PROTO[dt], s) for n, dt, s in outputs]
    graph_name = f"specula_qwen3_4b_ar1_cl512_part{part}"
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
    print(f"wrote {dst} -> ep_cache_context={bin_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True,
                        help="Directory holding the .bin files (used for ep_cache_context).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write the 4 wrapper.onnx files into.")
    parser.add_argument("--binary-basename", type=str,
                        default="qwen3_4b_4part_w4a16",
                        help="ctx-bin-gen basename. Produces <basename>_{1..4}.bin.")
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {int(p) for p in args.parts.split(",")}
    for part in (1, 2, 3, 4):
        if part not in wanted:
            continue
        bin_path = args.bin_dir / f"{args.binary_basename}_{part}.bin"
        if not bin_path.exists():
            # ctx-bin-gen may also emit <basename>_ar1_cl512_part{N}.bin or
            # just <basename>.bin if single-part. Try a couple fallbacks.
            alt1 = args.bin_dir / f"{args.binary_basename}_part{part}.bin"
            if alt1.exists():
                bin_path = alt1
            else:
                print(f"WARNING: no .bin found for part{part} at {bin_path}; "
                      f"wrapper still emitted but will fail at load time")
        dst = args.out_dir / f"specula_qwen3_4b_part{part}.wrapper.onnx"
        build_wrapper(part, bin_path, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
