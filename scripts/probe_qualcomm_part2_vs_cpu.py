"""Run Qualcomm's shipping Part 2 HTP with pos=0 BOS + empty past_kv
and compare L11 to CPU-ORT fp32. Single decisive question: does
Qualcomm's w4a16 HTP math produce the full L11 activation magnitude
(matching CPU-ORT ±16000), or does it compress internally like ours
does (±1400)?

If Qualcomm's L11 is wide: their HTP math is fine, our pipeline's
issue is purely calibration quality (cascading clip) and AIMET /
AI Hub / iterative paths are the fix.

If Qualcomm's L11 is also compressed: w4a16 fundamentally can't
represent BOS-extreme activations, and Qualcomm's coherent output
comes from something else (different normalization, bias correction,
their prompt distribution never hitting ±16000 at L11, etc.).

Run:
    PYTHONIOENCODING=utf-8 .venv-ort21/Scripts/python.exe \\
        scripts/probe_qualcomm_part2_vs_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn  # noqa: F401
import yaml
from onnx import TensorProto, helper


REPO = Path(__file__).resolve().parents[1]
BUNDLE = (REPO / "models" / "qualcomm-qwen3-4b-ref"
          / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite")
PATHB_PART = REPO / "models" / "qwen3-4b-arm-pathb-ctx512-part2"

NUM_LAYERS_IN_PART = 12
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
CTX_LEN = 512
PAST_LEN = CTX_LEN - 1
ROPE_THETA = 1_000_000.0

BOS_ID = 151644  # <|im_start|>

_DTYPE_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "float32": TensorProto.FLOAT,
}


def to_underscore(name: str) -> str:
    if name.startswith("/"):
        return name.replace("/", "_").replace(".", "_")
    return name


def quant_u16(x: np.ndarray, scale: float, offset: int) -> np.ndarray:
    q = np.round(x.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_u16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.int32) + offset).astype(np.float32) * scale


def build_qualcomm_wrapper(part_meta: dict, part_idx: int, dst: Path) -> None:
    inputs_decl = [
        helper.make_tensor_value_info(
            to_underscore(name), _DTYPE_PROTO[spec["dtype"]], list(spec["shape"])
        )
        for name, spec in part_meta["inputs"].items()
    ]
    outputs_decl = [
        helper.make_tensor_value_info(
            to_underscore(name), _DTYPE_PROTO[spec["dtype"]], list(spec["shape"])
        )
        for name, spec in part_meta["outputs"].items()
    ]
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name=f"token_ar1_cl512_{part_idx}_of_4",
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=f"qwen3_4b_part_{part_idx}_of_4.bin",
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"qualcomm_probe_part{part_idx}",
        inputs=inputs_decl, outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17),
                       helper.make_operatorsetid("com.microsoft", 1)],
        producer_name="specula-probe",
    )
    model.ir_version = 10
    onnx.save(model, str(dst))


def load_qnn_session(wrapper_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    qnn_devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not qnn_devs:
        raise RuntimeError("no QNN devices visible")
    htp_dll = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / "QnnHtp.dll"
    so.add_provider_for_devices(qnn_devs, {
        "backend_path": str(htp_dll),
        "htp_performance_mode": "burst",
        "soc_model": "88",
        "htp_arch": "81",
        "enable_htp_fp16_precision": "1",
    })
    return ort.InferenceSession(str(wrapper_path), sess_options=so)


def main() -> int:
    print("=== Qualcomm Part 2 HTP vs CPU-ORT L11 probe ===")
    print(f"bundle: {BUNDLE}")
    meta = yaml.safe_load((BUNDLE / "metadata.yaml").read_text())
    p1_meta = meta["components"]["ar1_cl512_1_of_4"]
    p2_meta = meta["components"]["ar1_cl512_2_of_4"]

    # Build wrappers in bundle dir (so the .bin path resolves relative).
    w1 = BUNDLE / "probe_p1.wrapper.onnx"
    w2 = BUNDLE / "probe_p2.wrapper.onnx"
    build_qualcomm_wrapper(p1_meta, 1, w1)
    build_qualcomm_wrapper(p2_meta, 2, w2)

    # Register QNN EP.
    ort.register_execution_provider_library("QNNExecutionProvider",
                                            onnxruntime_qnn.get_library_path())

    print("loading Qualcomm Part 1 + Part 2 HTP sessions ...")
    s1 = load_qnn_session(w1)
    s2 = load_qnn_session(w2)

    # === Part 1 forward: BOS -> embed ===
    embed_out_name = to_underscore("/model/model/embed_tokens/Gather_output_0")
    s1_out = s1.run([embed_out_name],
                    {"input_ids": np.array([[BOS_ID]], dtype=np.int32)})[0]
    embed_s = p1_meta["outputs"]["/model/model/embed_tokens/Gather_output_0"][
        "quantization_parameters"]["scale"]
    embed_o = p1_meta["outputs"]["/model/model/embed_tokens/Gather_output_0"][
        "quantization_parameters"]["offset"]
    embed_fp_qualcomm = dequant_u16(s1_out, embed_s, embed_o)
    print(f"  part1 embed: uint16 shape={s1_out.shape} dtype={s1_out.dtype}")
    print(f"  dequant embed range: [{embed_fp_qualcomm.min():.4f}, {embed_fp_qualcomm.max():.4f}]")

    # === CPU-ORT reference: run our pathb part 2 with the SAME embed ===
    # (The fp32 embed of a single token is literally the embedding table row,
    # which is the same for optimum and pathb — cos ~= 1.0 vs embed_fp_qualcomm.)
    print("\nloading CPU-ORT pathb part 1 + part 2 ...")
    s1_cpu = ort.InferenceSession(
        str(REPO / "models" / "qwen3-4b-arm-pathb-ctx512-part1" / "model.onnx"),
        providers=["CPUExecutionProvider"])
    s2_cpu = ort.InferenceSession(str(PATHB_PART / "model.onnx"),
                                  providers=["CPUExecutionProvider"])
    embed_cpu = s1_cpu.run(["/model/embed_tokens/Gather_output_0"],
                           {"input_ids": np.array([[BOS_ID]], dtype=np.int64)})[0]
    af = embed_cpu.reshape(-1).astype(np.float64)
    bf = embed_fp_qualcomm.reshape(-1).astype(np.float64)
    embed_cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    print(f"  cos(CPU pathb embed, Qualcomm HTP embed) = {embed_cos:.6f}")
    print(f"  CPU embed range: [{embed_cpu.min():.4f}, {embed_cpu.max():.4f}]")

    # === Part 2 feed ===
    # Quantize inputs to Qualcomm's encoding.
    p2_in = p2_meta["inputs"]

    def qparam(name: str) -> tuple[float, int]:
        q = p2_in[name]["quantization_parameters"]
        return float(q["scale"]), int(q["offset"])

    mask_s, mask_o = qparam("attention_mask")
    cos_s, cos_o = qparam("position_ids_cos")
    sin_s, sin_o = qparam("position_ids_sin")

    # Build mask: pos=0, only current slot (511) attends.
    # Qualcomm convention (from oracle): q=65535 -> 0 (attend), q=0 -> mask.
    mask_q = np.zeros((1, 1, 1, CTX_LEN), dtype=np.uint16)
    mask_q[..., -1] = 65535

    # Build half-dim cos/sin at pos=0. cos(0)=1, sin(0)=0.
    cos_h = np.ones((1, 1, 1, HALF_HEAD_DIM), dtype=np.float32)
    sin_h = np.zeros((1, 1, 1, HALF_HEAD_DIM), dtype=np.float32)
    cos_u = quant_u16(cos_h, cos_s, cos_o)
    sin_u = quant_u16(sin_h, sin_s, sin_o)

    # Build past_kv: Qualcomm uses uint8 with "zero" at q=128 (offset=-128, s=*).
    # Shapes per metadata: past_key [NUM_KV_HEADS, 1, HEAD_DIM, PAST_LEN],
    #                     past_value [NUM_KV_HEADS, 1, PAST_LEN, HEAD_DIM].
    feed2 = {
        to_underscore("/model/model/embed_tokens/Gather_output_0"): s1_out,
        "attention_mask": mask_q,
        "position_ids_cos": cos_u,
        "position_ids_sin": sin_u,
    }
    for li in range(0, NUM_LAYERS_IN_PART):
        feed2[f"past_key_{li}_in"] = np.full(
            (NUM_KV_HEADS, 1, HEAD_DIM, PAST_LEN), 128, dtype=np.uint8)
        feed2[f"past_value_{li}_in"] = np.full(
            (NUM_KV_HEADS, 1, PAST_LEN, HEAD_DIM), 128, dtype=np.uint8)

    # Run Qualcomm Part 2.
    l11_out_name = to_underscore("/model/model/layers.11/Add_1_output_0")
    s2_outs = s2.run([l11_out_name], feed2)
    l11_u16_qualcomm = s2_outs[0]
    l11_s, l11_o = p2_meta["outputs"]["/model/model/layers.11/Add_1_output_0"][
        "quantization_parameters"]["scale"], \
        p2_meta["outputs"]["/model/model/layers.11/Add_1_output_0"][
        "quantization_parameters"]["offset"]
    l11_fp_qualcomm = dequant_u16(l11_u16_qualcomm, l11_s, l11_o)

    # CPU-ORT pathb Part 2 forward with the SAME fp32 inputs (full-dim cos/sin,
    # fp32 mask, fp32 zero past_kv). The math is equivalent to Qualcomm's
    # half-dim rotary — produces the same L11.
    cos_full = np.concatenate([cos_h.reshape(1, 1, HALF_HEAD_DIM),
                               cos_h.reshape(1, 1, HALF_HEAD_DIM)], axis=-1)
    sin_full = np.concatenate([sin_h.reshape(1, 1, HALF_HEAD_DIM),
                               sin_h.reshape(1, 1, HALF_HEAD_DIM)], axis=-1)
    mask_fp = np.full((1, 1, 1, CTX_LEN), -65504.0, dtype=np.float32)
    mask_fp[..., -1] = 0.0

    cpu_feed = {
        "/model/embed_tokens/Gather_output_0": embed_cpu,
        "attention_bias": mask_fp,
        "position_ids_cos": cos_full.astype(np.float32),
        "position_ids_sin": sin_full.astype(np.float32),
    }
    for li in range(NUM_LAYERS_IN_PART):
        cpu_feed[f"past_key_values.{li}.key"] = np.zeros(
            (1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), dtype=np.float32)
        cpu_feed[f"past_key_values.{li}.value"] = np.zeros(
            (1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), dtype=np.float32)
    l11_fp_cpu = s2_cpu.run(["/model/layers.11/Add_1_output_0"], cpu_feed)[0]

    # === Comparison ===
    print("\n== RESULT ==")
    print(f"Qualcomm encoding:  scale={l11_s:.4e}  offset={l11_o}  "
          f"range=[{l11_o*l11_s:.2f}, {(65535+l11_o)*l11_s:.2f}]")
    af = l11_fp_cpu.reshape(-1).astype(np.float64)
    bf = l11_fp_qualcomm.reshape(-1).astype(np.float64)
    cos_val = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    print(f"CPU-ORT L11 range:  [{l11_fp_cpu.min():9.2f}, {l11_fp_cpu.max():9.2f}]")
    print(f"Qualcomm HTP L11:   [{l11_fp_qualcomm.min():9.2f}, {l11_fp_qualcomm.max():9.2f}]")
    print(f"cos(CPU, Qualcomm HTP) = {cos_val:+.6f}")
    # Saturation check on Qualcomm's uint16 output.
    under = float((l11_u16_qualcomm == 0).mean())
    over = float((l11_u16_qualcomm == 65535).mean())
    print(f"Qualcomm HTP uint16 saturation: q==0 {under*100:.3f}%, "
          f"q==65535 {over*100:.3f}%")
    print()
    print("Interpretation:")
    if abs(l11_fp_qualcomm).max() > 5000:
        print("  Qualcomm HTP L11 is WIDE (>5k) -> w4a16 HTP math CAN produce full")
        print("  activation magnitude. Our pipeline issue is calibration quality.")
    else:
        print("  Qualcomm HTP L11 is compressed -> w4a16 HTP math fundamentally limits")
        print("  BOS-extreme activation, and Qualcomm's coherence comes from other")
        print("  means (calibration dist, bias correction, etc.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
