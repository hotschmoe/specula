"""Phase 5.5 Lever C side-quest — probe Qualcomm Qwen3-4B Genie w4a16 bundle.

Question this answers: does ORT-QNN 1.24.4 (same QAIRT 2.42 bundle as
AI Hub compiles with) load a binary whose IO is fully quantized to
uint8 (past_kv) and uint16 (attention_mask, position_ids_*, logits)?

Verified so far:
  - Our AI Hub pathb w4a16 bin and Qualcomm's qwen3_4b Genie bin share
    the same QNN context binary magic (0000 0002 0000 0003 ...).
  - Qualcomm's metadata.yaml spells out scale/offset per input.
  - Same SoC (88) + dsp_arch (v81) per htp_backend_ext_config.json.

Strategy:
  1. Try the simplest partition first: `ar1_cl512_1_of_4.bin` — a
     single-step decode embedding lookup. Input: input_ids [1,1] int32.
     Output: embedding [1,1,2560] uint16.
  2. Build an EPContext wrapper declaring input/output dtypes per
     metadata.yaml (UINT16 for the embedding output, not FLOAT).
  3. Load via legacy QNN EP. Report what session.get_inputs() /
     get_outputs() say about dtypes.
  4. Run a forward pass with a zero input_ids tensor. Measure latency.

If step 3-4 succeed for part 1, promote to part 2 (which has
uint8 past_kv + uint16 attn_mask + uint16 cos/sin inputs) to validate
the full quantized-IO surface.

Run:
    .venv\\Scripts\\python.exe scripts\\probe_qualcomm_qwen3_4b.py --part 1
    .venv\\Scripts\\python.exe scripts\\probe_qualcomm_qwen3_4b.py --part 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
SOC_MODEL = "88"
HTP_ARCH = "81"

# Per Qualcomm metadata.yaml for ar1_cl512 (decode AR=1, ctx=512).
# Graph 1 of 4 is the embedding lookup only: input_ids -> embedded.
# Graph 2 of 4 is layers 0..11. Graph 3 is layers 12..23. Graph 4 is
# layers 24..35 + final norm + logits. Full decode = 4 sequential runs.
#
# For the probe we only need graph 1 to answer "does ORT-QNN accept
# a fully-quantized-IO binary?" Promoted to graph 2 once graph 1
# confirms; graph 2 exercises the full quant-IO surface (uint8 past_kv,
# uint16 attn_mask/cos/sin, uint16 hidden input).
PARTS = {
    1: {
        "bin": "qwen3_4b_part_1_of_4.bin",
        # Each .bin is a multi-graph weight-shared context binary carrying
        # 10 graphs (5 ctx × 2 AR). Pick `token_ar1_cl512_<N>_of_4` for
        # single-token decode at ctx=512 — the target regime for our
        # draft-model use case. Strings probe confirmed the exact name.
        "graph_name": "token_ar1_cl512_1_of_4",
        "inputs": [
            ("input_ids", [1, 1], TensorProto.INT32),
        ],
        "outputs": [
            ("_model_model_embed_tokens_Gather_output_0", [1, 1, 2560], TensorProto.UINT16),
        ],
    },
    2: {
        "bin": "qwen3_4b_part_2_of_4.bin",
        "graph_name": "token_ar1_cl512_2_of_4",
        # Partition 2 IO — 12 layers worth of past_kv (0..11) plus the
        # shared graph inputs. Scales/offsets from metadata ar1_cl512_2_of_4.
        # Tensor names in the binary use underscored form (slashes in
        # metadata.yaml are converted to underscores at compile time) —
        # verified by grepping the .bin (see strings probe in probe session).
        "inputs": [
            ("_model_model_embed_tokens_Gather_output_0", [1, 1, 2560], TensorProto.UINT16),
            ("attention_mask", [1, 1, 1, 512], TensorProto.UINT16),
            ("position_ids_cos", [1, 1, 1, 64], TensorProto.UINT16),
            ("position_ids_sin", [1, 1, 1, 64], TensorProto.UINT16),
            # 12 layers × {key [8,1,128,511], value [8,1,511,128]}  — all uint8.
            *[(f"past_key_{i}_in", [8, 1, 128, 511], TensorProto.UINT8) for i in range(12)],
            *[(f"past_value_{i}_in", [8, 1, 511, 128], TensorProto.UINT8) for i in range(12)],
        ],
        "outputs": [
            # Hidden state into partition 3.
            ("_model_model_layers_11_Add_1_output_0", [1, 1, 2560], TensorProto.UINT16),
            # Incremental single-token K/V slice for layers 0..11 (the current
            # step's contribution; Genie stitches them into the persistent
            # past_kv on its side).
            *[(f"past_key_{i}_out", [8, 1, 128, 1], TensorProto.UINT8) for i in range(12)],
            *[(f"past_value_{i}_out", [8, 1, 1, 128], TensorProto.UINT8) for i in range(12)],
        ],
    },
}


def build_wrapper(bin_path: Path, wrapper_path: Path, part_cfg: dict) -> None:
    """Build a tiny EPContext wrapper ONNX pointing at the .bin."""
    inputs_decl = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape, dtype in part_cfg["inputs"]
    ]
    if part_cfg.get("outputs"):
        outputs_decl = [
            helper.make_tensor_value_info(name, dtype, shape)
            for name, shape, dtype in part_cfg["outputs"]
        ]
    else:
        # Placeholder output — let ORT-QNN tell us what's inside.
        outputs_decl = [
            helper.make_tensor_value_info("_introspect_marker", TensorProto.FLOAT, [1])
        ]

    # node name MUST match one of the graph names embedded in the .bin
    # (`strings <bin>` reveals e.g. token_ar1_cl512_1_of_4). ORT-QNN
    # looks up partition_name → qnn_models[partition_name] at load time.
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name=part_cfg["graph_name"],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_path.name,
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"qualcomm_qwen3_4b_probe_{bin_path.stem}",
        inputs=inputs_decl,
        outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-sidequest",
    )
    model.ir_version = 10
    onnx.save(model, str(wrapper_path))


def load_session(wrapper_path: Path) -> ort.InferenceSession:
    backend = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
    provider_options = {
        "backend_path": str(backend),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    return ort.InferenceSession(
        str(wrapper_path),
        sess_options=sess_opts,
        providers=[("QNNExecutionProvider", provider_options)],
    )


def summarize_io(sess: ort.InferenceSession) -> None:
    print(f"  inputs ({len(sess.get_inputs())}):")
    for x in sess.get_inputs()[:10]:
        print(f"    {x.name:55s} shape={x.shape}  type={x.type}")
    if len(sess.get_inputs()) > 10:
        print(f"    ... +{len(sess.get_inputs()) - 10} more")
    print(f"  outputs ({len(sess.get_outputs())}):")
    for x in sess.get_outputs()[:10]:
        print(f"    {x.name:55s} shape={x.shape}  type={x.type}")
    if len(sess.get_outputs()) > 10:
        print(f"    ... +{len(sess.get_outputs()) - 10} more")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=(1, 2), default=1,
                        help="partition to probe (default: 1 = embedding only)")
    args = parser.parse_args()

    part_cfg = PARTS[args.part]
    bin_path = BUNDLE_DIR / part_cfg["bin"]
    if not bin_path.exists():
        print(f"ERROR: {bin_path} missing")
        return 2

    # Wrapper must live next to the .bin — ep_cache_context is resolved
    # relative to the wrapper ONNX's directory by ORT-QNN.
    wrapper_path = BUNDLE_DIR / f"part{args.part}.wrapper.onnx"
    print(f"=== Qualcomm Qwen3-4B Genie w4a16 probe (part {args.part}) ===\n")
    print(f"bin           : {bin_path} ({bin_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"wrapper onnx  : {wrapper_path}")

    print("\n--- build EPContext wrapper ---")
    try:
        build_wrapper(bin_path, wrapper_path, part_cfg)
        print(f"  wrote {wrapper_path.stat().st_size} bytes")
    except Exception:
        traceback.print_exc()
        return 1

    print("\n--- load via legacy QNN EP (ORT 1.24.4, QAIRT 2.42) ---")
    t0 = time.perf_counter()
    try:
        sess = load_session(wrapper_path)
    except Exception:
        traceback.print_exc()
        print("\n=== STATUS: load FAILED — see traceback above ===")
        return 1
    t_load = time.perf_counter() - t0
    print(f"  loaded in {t_load:.1f} s")
    print(f"  providers: {sess.get_providers()}")
    if sess.get_providers()[0] != "QNNExecutionProvider":
        print("ERROR: session silently fell back to CPU — QNN did not claim the graph")
        return 1

    print("\n--- IO signature as reported by ORT-QNN ---")
    summarize_io(sess)

    print("\n--- forward pass: zero inputs at wrapper-declared dtypes ---")
    dtype_map = {
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }
    feed = {}
    for x in sess.get_inputs():
        np_dtype = dtype_map.get(x.type)
        if np_dtype is None:
            print(f"ERROR: unknown dtype {x.type} for {x.name}")
            return 1
        shape = tuple(d if isinstance(d, int) else 1 for d in x.shape)
        if x.name == "input_ids":
            feed[x.name] = np.array([[151643]], dtype=np_dtype)  # BOS
        else:
            # Use the dtype's zero-equivalent; for uint8 offset=-128 that means 128
            # decodes back to 0.0. We pick fill=128 for uint8, fill=0 for others
            # (uint16 logits offset=-30800 so zero integer = small negative, not
            # critical for a structural smoke test).
            fill = 128 if np_dtype == np.uint8 else 0
            feed[x.name] = np.full(shape, fill, dtype=np_dtype)

    try:
        t0 = time.perf_counter()
        outputs = sess.run(None, feed)
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"  forward FAILED: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        return 1

    print(f"  run latency : {ms:.2f} ms")
    for name, arr in zip((o.name for o in sess.get_outputs()), outputs):
        finite_frac = (
            float(np.isfinite(arr.astype(np.float32)).mean())
            if arr.dtype != np.uint16 and arr.dtype != np.uint8
            else 1.0
        )
        print(f"  {name:55s} shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min()} max={arr.max()} finite={finite_frac:.2f}")

    print("\n=== STATUS: ok — ORT-QNN accepted a Qualcomm-reference full-quant-IO binary ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
