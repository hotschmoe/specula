"""Quick probe: load AI Hub's w4a16-AIMET Part 2 bin (fp32 IO boundary),
feed it the same test input as our per-part probe (pos=0 BOS + empty
past_kv), compare L11 output to CPU-ORT.

Decisive signal: does AI Hub's AIMET calibration produce a Part 2 with
materially better cos vs CPU-ORT than our qairt-quantizer w8+CLE build
(cos 0.9996)? If yes, we extend to parts 3/4 and re-run the full oracle.

Run:
    PYTHONIOENCODING=utf-8 .venv-ort21/Scripts/python.exe \\
        scripts/probe_aihub_part2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn  # noqa: F401
from onnx import TensorProto, helper


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "results" / "phase5_qwen3_4b_bundle"
MODELS = REPO / "models"

NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
CTX = 512
PAST = CTX - 1
HIDDEN = 2560
ROPE_THETA = 1_000_000.0
BOS_ID = 151644

EMBED_HIDDEN = "_model_embed_tokens_Gather_output_0"


def build_aihub_part2_wrapper(bin_path: Path, wrapper_path: Path) -> None:
    """AI Hub's bin has 28 fp32 inputs (named like ours) + 25 fp32 outputs
    named output_0..output_24. output_0 is L11 hidden; outputs 1..24 alternate
    present_N_key, present_N_value across 12 layers."""
    inputs = [
        helper.make_tensor_value_info(EMBED_HIDDEN, TensorProto.FLOAT, [1, 1, HIDDEN]),
        helper.make_tensor_value_info("attention_bias", TensorProto.FLOAT, [1, 1, 1, CTX]),
        helper.make_tensor_value_info("position_ids_cos", TensorProto.FLOAT, [1, 1, HALF_HEAD_DIM]),
        helper.make_tensor_value_info("position_ids_sin", TensorProto.FLOAT, [1, 1, HALF_HEAD_DIM]),
    ]
    for li in range(12):
        inputs.append(helper.make_tensor_value_info(
            f"past_key_values_{li}_key", TensorProto.FLOAT, [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
        inputs.append(helper.make_tensor_value_info(
            f"past_key_values_{li}_value", TensorProto.FLOAT, [1, NUM_KV_HEADS, PAST, HEAD_DIM]))
    outputs = [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [1, 1, HIDDEN])]
    for i in range(1, 25):
        outputs.append(helper.make_tensor_value_info(
            f"output_{i}", TensorProto.FLOAT, [1, NUM_KV_HEADS, CTX, HEAD_DIM]))

    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs],
        outputs=[v.name for v in outputs],
        name="aihub_part2",
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_path.name,
        source="Qnn",
    )
    graph = helper.make_graph([node], "aihub_part2_wrapper", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17),
                       helper.make_operatorsetid("com.microsoft", 1)],
        producer_name="specula-aihub-probe",
    )
    model.ir_version = 10
    onnx.save(model, str(wrapper_path))


def rope_half_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    cos = np.cos(freqs).astype(np.float32).reshape(1, 1, HALF_HEAD_DIM)
    sin = np.sin(freqs).astype(np.float32).reshape(1, 1, HALF_HEAD_DIM)
    return cos, sin


def main() -> int:
    print("=== AI Hub Part 2 (AIMET w4a16) vs CPU-ORT probe ===")
    bin_path = BUNDLE / "qwen3_4b_4part_w4a16_aihub_part2.bin"
    wrapper = BUNDLE / "aihub_part2.wrapper.onnx"
    build_aihub_part2_wrapper(bin_path, wrapper)

    # Init QNN EP
    ort.register_execution_provider_library("QNNExecutionProvider",
                                            onnxruntime_qnn.get_library_path())
    qnn_devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    htp_dll = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / "QnnHtp.dll"

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_provider_for_devices(qnn_devs, {
        "backend_path": str(htp_dll),
        "htp_performance_mode": "burst",
        "soc_model": "88", "htp_arch": "81",
        "enable_htp_fp16_precision": "1",
    })
    print(f"loading AI Hub Part 2 HTP session ({bin_path.stat().st_size / 1e6:.1f} MB) ...")
    sess = ort.InferenceSession(str(wrapper), sess_options=so)
    print("  loaded")

    # Build test input: pos=0 BOS, empty past_kv.
    print("\nrunning CPU-ORT part1+part2 reference ...")
    so_cpu = ort.SessionOptions(); so_cpu.log_severity_level = 3
    p1 = ort.InferenceSession(str(MODELS / "qwen3-4b-arm-pathb-ctx512-part1" / "model.onnx"),
                              sess_options=so_cpu, providers=["CPUExecutionProvider"])
    p2 = ort.InferenceSession(str(MODELS / "qwen3-4b-arm-pathb-ctx512-part2" / "model_halfdim.onnx"),
                              sess_options=so_cpu, providers=["CPUExecutionProvider"])
    embed_cpu = p1.run(["/model/embed_tokens/Gather_output_0"],
                       {"input_ids": np.array([[BOS_ID]], dtype=np.int64)})[0]
    cos_h, sin_h = rope_half_dim(0)
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float32)
    mask[..., -1] = 0.0
    past_k = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)
    past_v = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)

    cpu_feed = {
        "/model/embed_tokens/Gather_output_0": embed_cpu,
        "attention_bias": mask, "position_ids_cos": cos_h, "position_ids_sin": sin_h,
    }
    for li in range(12):
        cpu_feed[f"past_key_values.{li}.key"] = past_k
        cpu_feed[f"past_key_values.{li}.value"] = past_v
    l11_cpu = p2.run(["/model/layers.11/Add_1_output_0"], cpu_feed)[0]
    print(f"  CPU-ORT L11 range: [{l11_cpu.min():.2f}, {l11_cpu.max():.2f}]")

    # Run AI Hub part 2 with same fp32 inputs.
    print("\nrunning AI Hub Part 2 HTP ...")
    ah_feed = {
        EMBED_HIDDEN: embed_cpu,
        "attention_bias": mask, "position_ids_cos": cos_h, "position_ids_sin": sin_h,
    }
    for li in range(12):
        ah_feed[f"past_key_values_{li}_key"] = past_k
        ah_feed[f"past_key_values_{li}_value"] = past_v
    l11_ah = sess.run(["output_0"], ah_feed)[0]
    print(f"  AI Hub L11 range:  [{l11_ah.min():.2f}, {l11_ah.max():.2f}]")

    af = l11_cpu.reshape(-1).astype(np.float64)
    bf = l11_ah.reshape(-1).astype(np.float64)
    cos_val = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    diff = np.abs(l11_cpu - l11_ah)
    print(f"\ncos(CPU, AI Hub) = {cos_val:+.6f}")
    print(f"max_abs_diff     = {diff.max():.3f}")
    print(f"mean_abs_diff    = {diff.mean():.4f}")
    print("\nReference (our local builds):")
    print("  Phase 5k (w4 per-channel+CLE): cos=+0.999628  range [-4551, +16148]")
    print("  Phase 5m (w8 per-channel+CLE): cos=+0.999972  range [-4566, +16065]")
    print("  Phase 5n (u8 KV):              cos=+0.999971")
    print("  Phase 5o (halfdim):            cos=+0.999971")
    return 0


if __name__ == "__main__":
    sys.exit(main())
