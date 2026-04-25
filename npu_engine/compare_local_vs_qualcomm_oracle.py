"""Phase 3 gate: compare a locally-built single-part Qwen3-4B w4a16 binary
against the recorded Qualcomm oracle.

Inputs:
  - models/qwen3_4b_arm_pathb_ctx512.w4a16-local.bin  (our compile)
  - models/qwen3_4b_arm_pathb_ctx512.w4a16-local.encodings.json  (per-tensor q/o)
  - results/qualcomm_qwen3_4b_oracle.npz              (recorded oracle logits)

Strategy:
  Drive our binary through the SAME prompt + decode positions as the
  oracle. At each step, dequantize logits to fp32, then compute cos vs
  the oracle's logits at that step. Report per-step cos and the overall
  generation token agreement.

This is a single-part binary (whole graph in one .bin), so the wrapper
ONNX has 76 inputs (input_ids + attention_bias + cos + sin + 36 layers
of past_kv) and 73 outputs (logits + 36 layers of present_kv).
Per-tensor scale/offset come from the encodings.json that
qairt-dlc-to-json wrote.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \\
        npu_engine/compare_local_vs_qualcomm_oracle.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from tokenizers import Tokenizer


REPO = Path(__file__).resolve().parents[1]
SOC_MODEL = "88"
HTP_ARCH = "81"

NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAST_LEN = 511
CTX_LEN = 512
ROPE_THETA = 1_000_000.0


def rope_full_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    sin = np.sin(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    return cos, sin


def attention_bias_at(position: int) -> np.ndarray:
    bias = np.full((1, 1, 1, CTX_LEN), -65504.0, dtype=np.float32)
    bias[..., :position] = 0.0
    bias[..., -1] = 0.0
    return bias


_ELEM_TYPE_TO_TENSOR_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
}
_DTYPE_NUMPY = {
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}


def quantize(arr: np.ndarray, scale: float, offset: int, q_dtype: type) -> np.ndarray:
    """Quantize fp32 -> q_dtype using f = (q + offset) * scale convention."""
    q = np.round(arr / scale) - offset
    q_max = np.iinfo(q_dtype).max
    return np.clip(q, 0, q_max).astype(q_dtype)


def dequantize(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.int32) + offset).astype(np.float32) * scale


def load_encodings(path: Path) -> dict:
    """Parse the encodings.json produced by qairt-dlc-to-json into
    {tensor_name: {scale, offset, dtype}} for IO tensors only."""
    raw = json.loads(path.read_text())
    # qairt-dlc-to-json schema: top-level "tensors" or "graph": ... with
    # "encodings" per tensor. Accept multiple shapes; fall back to scanning.
    out: dict[str, dict] = {}
    def walk(obj, path=""):
        if isinstance(obj, dict):
            if "scale" in obj and "offset" in obj:
                # We don't always know the tensor name from the position;
                # let caller resolve.
                return obj
            for k, v in obj.items():
                walk(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(raw)
    # Realistically the schema varies; we'll fish out tensor encodings
    # at runtime when the script is connected to a real encodings.json.
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path,
                        default=REPO / "models" / "qwen3_4b_arm_pathb_ctx512.w4a16-local.bin")
    parser.add_argument("--encodings", type=Path,
                        default=REPO / "models" / "qwen3_4b_arm_pathb_ctx512.w4a16-local.encodings.json")
    parser.add_argument("--wrapper", type=Path,
                        default=REPO / "models" / "qwen3_4b_arm_pathb_ctx512.w4a16-local.wrapper.onnx")
    parser.add_argument("--oracle", type=Path,
                        default=REPO / "results" / "qualcomm_qwen3_4b_oracle.npz")
    parser.add_argument("--tokenizer", type=Path,
                        default=REPO / "models" / "qualcomm-qwen3-4b-ref"
                                / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
                                / "tokenizer.json")
    parser.add_argument("--prompt-file", type=Path,
                        default=REPO / "models" / "qualcomm-qwen3-4b-ref"
                                / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
                                / "sample_prompt.txt")
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument("--out-report", type=Path,
                        default=REPO / "results" / "phase3_local_vs_oracle.md")
    args = parser.parse_args()

    print(f"=== Phase 3 gate: local w4a16 vs Qualcomm oracle ===")
    print(f"binary    : {args.bin}")
    print(f"encodings : {args.encodings}")
    print(f"oracle    : {args.oracle}")

    # 1) Load oracle (full-vocab fp32 logits per step).
    if not args.oracle.exists():
        print(f"FATAL: oracle missing at {args.oracle} — run Phase 0 first.")
        return 2
    oracle = np.load(str(args.oracle))
    oracle_logits_fp32 = oracle["logits_fp32"]  # [steps, vocab]
    oracle_argmax = oracle["argmax_tokens"]
    oracle_step_tokens = oracle["step_tokens"]
    oracle_prompt_ids = oracle["prompt_ids"]
    n_oracle_steps = len(oracle_argmax)
    n_prompt = len(oracle_prompt_ids)
    n_gen = n_oracle_steps - n_prompt
    print(f"oracle steps: {n_oracle_steps} ({n_prompt} prefill + {n_gen} gen)")

    # 2) Build wrapper.onnx pointing at the .bin. Read the encodings to
    #    get IO dtypes + scale/offset.
    if not args.encodings.exists():
        print(f"FATAL: encodings missing at {args.encodings} — run qairt-dlc-to-json.")
        return 2
    enc = json.loads(args.encodings.read_text())
    print(f"encodings root keys: {list(enc.keys())[:8]}")

    # 3) Build wrapper based on encoded IO.
    # Reading the actual schema from the encodings.json — once we can run
    # qairt-dlc-to-json, we'll fill this in concretely. Skeleton for now.
    print(f"NOTE: wrapper construction needs the live encodings schema; "
          f"this script will be completed once the qairt-quantizer + "
          f"qairt-dlc-to-json produce the actual files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
