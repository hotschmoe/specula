"""Run AI Hub's submit_quantize_job QDQ ONNX on CPU-ORT and compare L11
output to CPU-ORT fp32 reference. Decouples AIMET quantization quality
from HTP execution quirks.

If AIMET's QDQ ONNX reproduces CPU-ORT fp32's L11 at ±16000 (cos ~1.0),
we know to invest in the conversion-to-DLC + bin-gen path. If the QDQ
ONNX already shows the 10× magnitude compression signature we saw with
submit_compile_job, AIMET's cloud path has the same limitation and we
need Path B (local AIMET on Linux+CUDA).

Run:
    .venv/Scripts/python.exe scripts/probe_aihub_qdq_part2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
QDQ = REPO / "results" / "phase5_qwen3_4b_bundle" / "aihub_quantize_part2_qdq" / "job_j5wdjmkjg_qdq_onnx" / "model.onnx"

NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
CTX = 512
PAST = CTX - 1
HIDDEN = 2560
ROPE_THETA = 1_000_000.0
BOS_ID = 151644


def rope_half_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    cos = np.cos(freqs).astype(np.float32).reshape(1, 1, HALF_HEAD_DIM)
    sin = np.sin(freqs).astype(np.float32).reshape(1, 1, HALF_HEAD_DIM)
    return cos, sin


def main() -> int:
    print("=== AI Hub submit_quantize_job QDQ ONNX CPU-ORT probe ===")
    print(f"QDQ ONNX: {QDQ}")
    if not QDQ.exists():
        print(f"FATAL: {QDQ} not found")
        return 2

    so = ort.SessionOptions(); so.log_severity_level = 3

    print("loading CPU-ORT reference (fp32 pathb halfdim part2) ...")
    p1 = ort.InferenceSession(str(MODELS / "qwen3-4b-arm-pathb-ctx512-part1" / "model.onnx"),
                              sess_options=so, providers=["CPUExecutionProvider"])
    p2_fp = ort.InferenceSession(str(MODELS / "qwen3-4b-arm-pathb-ctx512-part2" / "model_halfdim.onnx"),
                                 sess_options=so, providers=["CPUExecutionProvider"])
    print("loading AI Hub QDQ part2 ...")
    p2_qdq = ort.InferenceSession(str(QDQ), sess_options=so, providers=["CPUExecutionProvider"])
    print(f"  QDQ inputs/outputs: {len(p2_qdq.get_inputs())} / {len(p2_qdq.get_outputs())}")
    for o in p2_qdq.get_outputs()[:3]:
        print(f"    output {o.name}: shape={o.shape} dtype={o.type}")

    # Build same test input as our per-part probe.
    embed = p1.run(["/model/embed_tokens/Gather_output_0"],
                   {"input_ids": np.array([[BOS_ID]], dtype=np.int64)})[0]
    cos_h, sin_h = rope_half_dim(0)
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float32)
    mask[..., -1] = 0.0
    past_k = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)
    past_v = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)

    # Feed both sessions.
    # Note: QDQ ONNX's input names use slash form per the AI Hub upload
    # (same as our halfdim ONNX — AIMET preserves names).
    feed = {
        "/model/embed_tokens/Gather_output_0": embed,
        "attention_bias": mask,
        "position_ids_cos": cos_h,
        "position_ids_sin": sin_h,
    }
    for li in range(12):
        feed[f"past_key_values.{li}.key"] = past_k
        feed[f"past_key_values.{li}.value"] = past_v

    print("\nrunning fp32 reference ...")
    l11_fp = p2_fp.run(["/model/layers.11/Add_1_output_0"], feed)[0]
    print(f"  fp32 L11 range: [{l11_fp.min():.2f}, {l11_fp.max():.2f}]")

    print("running AI Hub QDQ ...")
    # The QDQ ONNX's outputs are the same names as our fp32 ONNX.
    l11_qdq = p2_qdq.run(["/model/layers.11/Add_1_output_0"], feed)[0]
    print(f"  QDQ  L11 range: [{l11_qdq.min():.2f}, {l11_qdq.max():.2f}]")

    af = l11_fp.reshape(-1).astype(np.float64)
    bf = l11_qdq.reshape(-1).astype(np.float64)
    cos_val = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    diff = np.abs(l11_fp - l11_qdq)
    print(f"\ncos(fp32, QDQ)   = {cos_val:+.6f}")
    print(f"max_abs_diff     = {diff.max():.3f}")
    print(f"mean_abs_diff    = {diff.mean():.4f}")

    print("\nReference comparisons:")
    print("  Phase 5k qairt-quant w4 per-channel+CLE HTP: cos=+0.999628  range [-4551, +16148]")
    print("  Phase 5m qairt-quant w8+CLE HTP:             cos=+0.999972  range [-4566, +16065]")
    print("  Phase 5q AI Hub submit_compile_job w4a16 HTP: cos=+0.997644 range [-401,  +1443]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
