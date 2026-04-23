"""Per-part CPU-ORT vs HTP divergence probe for the 4-part Qwen3-4B bundle.

For each of the 4 parts, feeds the IDENTICAL fp32 input (CPU-ORT derived)
through both the HTP bin (via its EPContext wrapper) and the source pathb
split ONNX (CPU-ORT). Reports per-output cosine and max-abs-diff so we can
localize which part's internal quantization is leaking.

Test input: first-decode step (position 0, BOS token 151644, empty past_kv,
full-dim cos/sin, attention_bias allowing only slot 511). Matches the
oracle's step 0.

Key invariant: each part is fed its ideal fp32 inputs (derived from upstream
CPU-ORT), so divergence between HTP and CPU outputs is bounded to that
part's internal quantization error only. No cumulative drift.

Run:
    PYTHONIOENCODING=utf-8 .venv-ort21/Scripts/python.exe \\
        scripts/probe_4b_per_part_htp_vs_cpu.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn  # noqa: F401  # registers the QNN provider


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
MODELS = REPO / "models"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
VOCAB = 151936
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0

EMBED_HIDDEN = "/model/embed_tokens/Gather_output_0"
L11_HIDDEN = "/model/layers.11/Add_1_output_0"
L23_HIDDEN = "/model/layers.23/Add_1_output_0"


def rope_full_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    sin = np.sin(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    return cos, sin


def attention_bias_at(position: int) -> np.ndarray:
    bias = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float32)
    bias[..., :position] = 0.0
    bias[..., -1] = 0.0
    return bias


def quant_u16(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    q = np.round(x_fp32.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_u16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.float32) + float(offset)) * float(scale)


def load_quant_maps(results_dir: Path) -> dict[int, dict[str, tuple[float, int]]]:
    out: dict[int, dict[str, tuple[float, int]]] = {}
    for part in (1, 2, 3, 4):
        j = json.loads((results_dir / f"qwen3_4b_part{part}.w4a16-local.dlc.json").read_text())
        part_map: dict[str, tuple[float, int]] = {}
        for name, t in j["graph"]["tensors"].items():
            q = t.get("quant_params", {}).get("scale_offset", {})
            if q.get("is_fixed_point") and q.get("bitwidth"):
                part_map[name] = (float(q["scale"]), int(q["offset"]))
        out[part] = part_map
    return out


def name_to_wrapper(name: str) -> str:
    if name.startswith("/"):
        return name.replace("/", "_").replace(".", "_")
    return name.replace(".", "_")


def mk_htp_session(wrapper: Path, qnn_devs: list, htp_dll: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_provider_for_devices(qnn_devs, {
        "backend_path": str(htp_dll),
        "htp_performance_mode": "burst",
        "soc_model": "88",
        "htp_arch": "81",
        "enable_htp_fp16_precision": "1",
    })
    return ort.InferenceSession(str(wrapper), sess_options=so)


def mk_cpu_session(onnx_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).reshape(-1)
    bf = b.astype(np.float64).reshape(-1)
    n = np.linalg.norm(af) * np.linalg.norm(bf)
    if n == 0:
        return float("nan")
    return float(np.dot(af, bf) / n)


def saturation(q: np.ndarray) -> tuple[float, float]:
    """Fraction of uint16 values at 0 (under-flow) and 65535 (over-flow)."""
    under = float((q == 0).mean())
    over = float((q == 65535).mean())
    return under, over


def report(tag: str, cpu_fp: np.ndarray, htp_fp: np.ndarray, htp_u16: np.ndarray | None = None) -> None:
    c = cos(cpu_fp, htp_fp)
    d = np.abs(cpu_fp - htp_fp)
    print(f"  {tag:<36s} cos={c:+.6f}  maxdiff={d.max():9.3f}  meanabs={d.mean():7.4f}"
          f"  | cpu range=[{cpu_fp.min():9.2f}, {cpu_fp.max():9.2f}]"
          f"  htp range=[{htp_fp.min():9.2f}, {htp_fp.max():9.2f}]",
          end="")
    if htp_u16 is not None:
        u, o = saturation(htp_u16)
        print(f"  sat@0={u*100:.2f}%  sat@65535={o*100:.2f}%")
    else:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=RESULTS)
    parser.add_argument("--parts-root", type=Path, default=MODELS)
    parser.add_argument("--position", type=int, default=0,
                        help="Decode position to probe. Default=0 (BOS + empty past_kv).")
    parser.add_argument("--bos-id", type=int, default=151644,
                        help="Token ID to feed. Default=151644 (<|im_start|>).")
    args = parser.parse_args()

    print("=== Per-part HTP vs CPU-ORT divergence probe ===")
    print(f"bundle: {args.bundle_dir}")
    print(f"inputs: pos={args.position}, token_id={args.bos_id}, past_kv=zeros\n")

    qmaps = load_quant_maps(args.bundle_dir)

    # Initialize QNN EP.
    ort.register_execution_provider_library("QNNExecutionProvider",
                                            onnxruntime_qnn.get_library_path())
    qnn_devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not qnn_devs:
        print("FATAL: no QNN devices visible")
        return 2
    htp_dll = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / "QnnHtp.dll"

    print("loading HTP sessions ...")
    htp = [mk_htp_session(args.bundle_dir / f"specula_qwen3_4b_part{i}.wrapper.onnx",
                          qnn_devs, htp_dll)
           for i in (1, 2, 3, 4)]
    print("loading CPU-ORT sessions ...")
    cpu = [mk_cpu_session(args.parts_root / f"qwen3-4b-arm-pathb-ctx512-part{i}" / "model.onnx")
           for i in (1, 2, 3, 4)]

    # ---- Fixed fp32 test inputs (same as oracle step 0) ----
    input_ids_i32 = np.array([[args.bos_id]], dtype=np.int32)
    input_ids_i64 = input_ids_i32.astype(np.int64)
    cos_fp32, sin_fp32 = rope_full_dim(args.position)
    mask_fp32 = attention_bias_at(args.position)
    past_k_fp32 = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)
    past_v_fp32 = np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)

    # ==================== PART 1 ====================
    print("\n-- PART 1 (embed lookup) --")
    # HTP
    embed_out_wrap = name_to_wrapper(EMBED_HIDDEN)
    embed_u16_htp = htp[0].run([embed_out_wrap], {"input_ids": input_ids_i32})[0]
    s, o = qmaps[1][EMBED_HIDDEN]
    embed_fp_htp = dequant_u16(embed_u16_htp, s, o)
    # CPU
    embed_fp_cpu = cpu[0].run([EMBED_HIDDEN], {"input_ids": input_ids_i64})[0]
    report("embed", embed_fp_cpu, embed_fp_htp, embed_u16_htp)

    # ==================== PART 2 ====================
    print("\n-- PART 2 (layers 0-11) --")
    # Build HTP feed (quantize each fp32 input to that part's IN encoding)
    p2_embed_in_s, p2_embed_in_o = qmaps[2][EMBED_HIDDEN]
    p2_mask_s, p2_mask_o = qmaps[2]["attention_bias"]
    p2_cos_s, p2_cos_o = qmaps[2]["position_ids_cos"]
    p2_sin_s, p2_sin_o = qmaps[2]["position_ids_sin"]

    htp_feed_p2 = {
        name_to_wrapper(EMBED_HIDDEN): quant_u16(embed_fp_cpu, p2_embed_in_s, p2_embed_in_o),
        "attention_bias": quant_u16(mask_fp32, p2_mask_s, p2_mask_o),
        "position_ids_cos": quant_u16(cos_fp32, p2_cos_s, p2_cos_o),
        "position_ids_sin": quant_u16(sin_fp32, p2_sin_s, p2_sin_o),
    }
    for li in range(0, LAYERS_PER_PART):
        ks, ko = qmaps[2][f"past_key_values.{li}.key"]
        vs, vo = qmaps[2][f"past_key_values.{li}.value"]
        htp_feed_p2[name_to_wrapper(f"past_key_values.{li}.key")] = quant_u16(past_k_fp32, ks, ko)
        htp_feed_p2[name_to_wrapper(f"past_key_values.{li}.value")] = quant_u16(past_v_fp32, vs, vo)

    l11_wrap = name_to_wrapper(L11_HIDDEN)
    outs_p2 = htp[1].run(None, htp_feed_p2)
    out_names_p2 = [o.name for o in htp[1].get_outputs()]
    out_map_p2 = dict(zip(out_names_p2, outs_p2))
    l11_u16_htp = out_map_p2[l11_wrap]
    p2_l11_s, p2_l11_o = qmaps[2][L11_HIDDEN]
    l11_fp_htp = dequant_u16(l11_u16_htp, p2_l11_s, p2_l11_o)

    # CPU
    cpu_feed_p2 = {
        EMBED_HIDDEN: embed_fp_cpu,
        "attention_bias": mask_fp32,
        "position_ids_cos": cos_fp32,
        "position_ids_sin": sin_fp32,
    }
    for li in range(0, LAYERS_PER_PART):
        cpu_feed_p2[f"past_key_values.{li}.key"] = past_k_fp32
        cpu_feed_p2[f"past_key_values.{li}.value"] = past_v_fp32
    l11_fp_cpu = cpu[1].run([L11_HIDDEN], cpu_feed_p2)[0]
    report("L11 hidden (part2 OUT)", l11_fp_cpu, l11_fp_htp, l11_u16_htp)

    # ==================== PART 3 ====================
    print("\n-- PART 3 (layers 12-23) --")
    p3_l11_in_s, p3_l11_in_o = qmaps[3][L11_HIDDEN]
    p3_mask_s, p3_mask_o = qmaps[3]["attention_bias"]
    p3_cos_s, p3_cos_o = qmaps[3]["position_ids_cos"]
    p3_sin_s, p3_sin_o = qmaps[3]["position_ids_sin"]

    htp_feed_p3 = {
        name_to_wrapper(L11_HIDDEN): quant_u16(l11_fp_cpu, p3_l11_in_s, p3_l11_in_o),
        "attention_bias": quant_u16(mask_fp32, p3_mask_s, p3_mask_o),
        "position_ids_cos": quant_u16(cos_fp32, p3_cos_s, p3_cos_o),
        "position_ids_sin": quant_u16(sin_fp32, p3_sin_s, p3_sin_o),
    }
    for li in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
        ks, ko = qmaps[3][f"past_key_values.{li}.key"]
        vs, vo = qmaps[3][f"past_key_values.{li}.value"]
        htp_feed_p3[name_to_wrapper(f"past_key_values.{li}.key")] = quant_u16(past_k_fp32, ks, ko)
        htp_feed_p3[name_to_wrapper(f"past_key_values.{li}.value")] = quant_u16(past_v_fp32, vs, vo)

    l23_wrap = name_to_wrapper(L23_HIDDEN)
    outs_p3 = htp[2].run(None, htp_feed_p3)
    out_names_p3 = [o.name for o in htp[2].get_outputs()]
    out_map_p3 = dict(zip(out_names_p3, outs_p3))
    l23_u16_htp = out_map_p3[l23_wrap]
    p3_l23_s, p3_l23_o = qmaps[3][L23_HIDDEN]
    l23_fp_htp = dequant_u16(l23_u16_htp, p3_l23_s, p3_l23_o)

    cpu_feed_p3 = {
        L11_HIDDEN: l11_fp_cpu,
        "attention_bias": mask_fp32,
        "position_ids_cos": cos_fp32,
        "position_ids_sin": sin_fp32,
    }
    for li in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
        cpu_feed_p3[f"past_key_values.{li}.key"] = past_k_fp32
        cpu_feed_p3[f"past_key_values.{li}.value"] = past_v_fp32
    l23_fp_cpu = cpu[2].run([L23_HIDDEN], cpu_feed_p3)[0]
    report("L23 hidden (part3 OUT)", l23_fp_cpu, l23_fp_htp, l23_u16_htp)

    # ==================== PART 4 ====================
    print("\n-- PART 4 (layers 24-35 + norm + lm_head) --")
    p4_l23_in_s, p4_l23_in_o = qmaps[4][L23_HIDDEN]
    p4_mask_s, p4_mask_o = qmaps[4]["attention_bias"]
    p4_cos_s, p4_cos_o = qmaps[4]["position_ids_cos"]
    p4_sin_s, p4_sin_o = qmaps[4]["position_ids_sin"]

    htp_feed_p4 = {
        name_to_wrapper(L23_HIDDEN): quant_u16(l23_fp_cpu, p4_l23_in_s, p4_l23_in_o),
        "attention_bias": quant_u16(mask_fp32, p4_mask_s, p4_mask_o),
        "position_ids_cos": quant_u16(cos_fp32, p4_cos_s, p4_cos_o),
        "position_ids_sin": quant_u16(sin_fp32, p4_sin_s, p4_sin_o),
    }
    for li in range(2 * LAYERS_PER_PART, NUM_LAYERS):
        ks, ko = qmaps[4][f"past_key_values.{li}.key"]
        vs, vo = qmaps[4][f"past_key_values.{li}.value"]
        htp_feed_p4[name_to_wrapper(f"past_key_values.{li}.key")] = quant_u16(past_k_fp32, ks, ko)
        htp_feed_p4[name_to_wrapper(f"past_key_values.{li}.value")] = quant_u16(past_v_fp32, vs, vo)

    outs_p4 = htp[3].run(None, htp_feed_p4)
    out_names_p4 = [o.name for o in htp[3].get_outputs()]
    out_map_p4 = dict(zip(out_names_p4, outs_p4))
    logits_u16_htp = out_map_p4["logits"]
    p4_logits_s, p4_logits_o = qmaps[4]["logits"]
    logits_fp_htp = dequant_u16(logits_u16_htp, p4_logits_s, p4_logits_o)

    cpu_feed_p4 = {
        L23_HIDDEN: l23_fp_cpu,
        "attention_bias": mask_fp32,
        "position_ids_cos": cos_fp32,
        "position_ids_sin": sin_fp32,
    }
    for li in range(2 * LAYERS_PER_PART, NUM_LAYERS):
        cpu_feed_p4[f"past_key_values.{li}.key"] = past_k_fp32
        cpu_feed_p4[f"past_key_values.{li}.value"] = past_v_fp32
    logits_fp_cpu = cpu[3].run(["logits"], cpu_feed_p4)[0]
    report("logits (part4 OUT)", logits_fp_cpu, logits_fp_htp, logits_u16_htp)

    # ==================== SUMMARY ====================
    print("\n-- summary --")
    print("Each row feeds that part its IDEAL fp32 upstream input (CPU-ORT-derived).")
    print("So divergence is bounded to THAT part's internal quantization only.")
    print()
    print("Interpretation cheat-sheet:")
    print("  cos ≈ 1.0  : part is clean")
    print("  cos 0.7-0.95 : some saturation/quantization loss, recoverable")
    print("  cos < 0.5  : major internal clipping or numerical divergence")
    print("  sat@0/65535 > 1% : encoding range is too narrow for that tensor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
