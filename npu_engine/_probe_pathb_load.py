"""Minimal load probe: can ORT-QNN load a specula pathb .bin part?

The pathb bundles were compiled with QAIRT 2.45 (bin_info buildId
v2.45.40.260406); the venv's ORT-QNN 1.24.4 bundles QAIRT 2.42. This
probe wraps part1 (the tiny 1-in/1-out embed graph) in an EPContext
ONNX and tries to create a session + run one inference, first with the
venv-bundled QnnHtp.dll, then with the system QAIRT 2.45 DLL.

Exit 0 if either path produces a sane embedding output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "models" / "specula-qwen3-4b-ref" / "qwen3-4b_w4a16_pathb_ctx512_x2e_v81"
BIN = "qwen3-4b_w4a16_pathb_ctx512_x2e_v81_part_1_of_4.bin"
GRAPH = "qwen3_4b_pathb_w4a16_part1"
SYS_QNN = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\QnnHtp.dll")


def build_wrapper(dst: Path) -> None:
    node = helper.make_node(
        "EPContext", inputs=["input_ids"],
        outputs=["_model_embed_tokens_Gather_output_0"],
        name=GRAPH, domain="com.microsoft",
        embed_mode=0, ep_cache_context=BIN, source="Qnn",
    )
    g = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1])],
        [helper.make_tensor_value_info(
            "_model_embed_tokens_Gather_output_0", TensorProto.FLOAT, [1, 1, 2560])],
    )
    m = helper.make_model(g, opset_imports=[
        helper.make_operatorsetid("", 17),
        helper.make_operatorsetid("com.microsoft", 1)])
    m.ir_version = 10
    onnx.save(m, str(dst))


def try_load(wrapper: Path, backend: Path | None) -> bool:
    label = "venv-bundled QnnHtp.dll" if backend is None else f"system {backend}"
    print(f"\n--- attempt: {label} ---", flush=True)
    opts = {
        "htp_performance_mode": "burst",
        "soc_model": "88",
        "htp_arch": "81",
        "enable_htp_fp16_precision": "1",
    }
    if backend is not None:
        opts["backend_path"] = str(backend)
    so = ort.SessionOptions()
    so.log_severity_level = 2
    try:
        sess = ort.InferenceSession(
            str(wrapper), sess_options=so,
            providers=[("QNNExecutionProvider", opts)])
    except Exception as e:
        print(f"  CreateSession FAILED: {type(e).__name__}: {e}")
        return False
    prov = sess.get_providers()[0]
    print(f"  provider: {prov}")
    if prov != "QNNExecutionProvider":
        print("  FAILED: fell back off QNN")
        return False
    out = sess.run(None, {"input_ids": np.array([[9707]], dtype=np.int64)})[0]
    print(f"  inference OK: shape={out.shape} dtype={out.dtype} "
          f"mean={out.mean():.4f} std={out.std():.4f} "
          f"finite={np.isfinite(out).all()}")
    return bool(np.isfinite(out).all() and out.std() > 1e-6)


def main() -> int:
    wrapper = BUNDLE / "_probe_part1.wrapper.onnx"
    build_wrapper(wrapper)
    print(f"wrapper: {wrapper}")
    print(f"bin    : {(BUNDLE / BIN)}  exists={(BUNDLE / BIN).exists()}")
    ok_venv = try_load(wrapper, None)
    ok_sys = False
    if not ok_venv and SYS_QNN.exists():
        ok_sys = try_load(wrapper, SYS_QNN)
    print(f"\n=== RESULT: venv={ok_venv}  system2.45={ok_sys} ===")
    return 0 if (ok_venv or ok_sys) else 1


if __name__ == "__main__":
    sys.exit(main())
