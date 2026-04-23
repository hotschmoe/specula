"""Iterative per-part recalibration using upstream's quantized HTP output.

The initial Phase 5d calibration wrote each downstream part's seam
input as CPU-ORT fp32 (clean) values. But qairt-quantizer's forward
on the upstream part uses w4 weights, producing a WIDER activation
range at the seam. Downstream, calibrated against clean fp32, ends
up with a narrow dynamic range that clips the real (w4) upstream
output at runtime — which is exactly what destroyed our oracle cos.

This script walks part 1 -> 2 -> 3 -> 4, using each upstream's
compiled HTP bin to produce realistic uint16 seam outputs, dequants
them to fp32, and overwrites the downstream's seam raw calibration
file. After each overwrite, the downstream part's qairt-quantizer
sees the actual runtime distribution.

Run AFTER Phase 5 initial pass (all 4 w4a16 DLCs + bins exist):
    PYTHONIOENCODING=utf-8 \\
    .venv-ort21/Scripts/python.exe scripts/recalibrate_4b_iterative.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
CALIB_ROOT = REPO / "models" / "calibration"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128

EMBED_HIDDEN = "_model_embed_tokens_Gather_output_0"
L11_HIDDEN = "_model_layers_11_Add_1_output_0"
L23_HIDDEN = "_model_layers_23_Add_1_output_0"

# The slash-form name used in the DLC JSONs and raw-file key names.
EMBED_SLASH = "/model/embed_tokens/Gather_output_0"
L11_SLASH = "/model/layers.11/Add_1_output_0"
L23_SLASH = "/model/layers.23/Add_1_output_0"


def quant_u16(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    q = np.round(x_fp32.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_u16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return ((q.astype(np.int64) + int(offset)).astype(np.float64) * scale).astype(np.float32)


def load_quant_for(part: int, tensor_name_slash: str) -> tuple[float, int]:
    j = json.loads((RESULTS / f"qwen3_4b_part{part}.w4a16-local.dlc.json").read_text())
    t = j["graph"]["tensors"][tensor_name_slash]
    q = t["quant_params"]["scale_offset"]
    return float(q["scale"]), int(q["offset"])


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").lstrip("_")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-npz", type=Path,
                        default=CALIB_ROOT / "qwen3_4b_ctx512_a.npz")
    parser.add_argument("--target-part", type=int, required=True, choices=[2, 3, 4],
                        help="Which downstream part's seam input to refresh. "
                             "2: regenerate embed_hidden via part1 HTP. "
                             "3: regenerate L11_hidden via parts 1+2 HTP. "
                             "4: regenerate L23_hidden via parts 1+2+3 HTP.")
    args = parser.parse_args()

    print(f"loading calibration npz: {args.calib_npz}")
    data = np.load(str(args.calib_npz))
    n_samples = data[data.files[0]].shape[0]
    print(f"  {n_samples} samples")

    # Quant params for inputs we need to build feeds.
    # Shared across parts 2/3/4, so read once from part 2.
    bias_scale, bias_offset = load_quant_for(2, "attention_bias")
    cos_scale, cos_offset = load_quant_for(2, "position_ids_cos")
    sin_scale, sin_offset = load_quant_for(2, "position_ids_sin")

    ort.register_execution_provider_library("QNNExecutionProvider",
                                            onnxruntime_qnn.get_library_path())
    qnn_devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    htp_dll = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / "QnnHtp.dll"

    # Load whichever upstream HTP sessions we need.
    upstream_parts = list(range(1, args.target_part))
    sessions: dict[int, ort.InferenceSession] = {}
    for p in upstream_parts:
        wrapper = RESULTS / f"specula_qwen3_4b_part{p}.wrapper.onnx"
        print(f"loading part{p} HTP session ...")
        t0 = time.perf_counter()
        sessions[p] = mk_htp_session(wrapper, qnn_devs, htp_dll)
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Target (seam name, slash form, dst raw dir, dst sanitized key).
    if args.target_part == 2:
        seam_name = "part2_embed_hidden"
        seam_slash = EMBED_SLASH
        seam_underscore = EMBED_HIDDEN
    elif args.target_part == 3:
        seam_name = "part3_L11_hidden"
        seam_slash = L11_SLASH
        seam_underscore = L11_HIDDEN
    else:
        seam_name = "part4_L23_hidden"
        seam_slash = L23_SLASH
        seam_underscore = L23_HIDDEN

    dst_raw_dir = CALIB_ROOT / f"qwen3_4b_ctx512_part{args.target_part}_raw"
    if not dst_raw_dir.exists():
        print(f"FATAL: raw dir {dst_raw_dir} missing")
        return 2

    # quant params for the seam OUT at the PRODUCING upstream (for dequant)
    upstream_part = args.target_part - 1
    out_scale, out_offset = load_quant_for(upstream_part, seam_slash)
    print(f"seam {seam_slash}: part{upstream_part} out scale={out_scale:.4e} offset={out_offset}")

    # Per-sample: chain parts 1..upstream_part using THIS CALIBRATION
    # SAMPLE's inputs; capture uint16 seam output at the final upstream;
    # dequant to fp32; overwrite the downstream's seam raw file with
    # those fp32 bytes.
    print(f"\nwalking {n_samples} samples through parts {upstream_parts} ...")
    for s in range(n_samples):
        t0 = time.perf_counter()
        input_ids = data["input_ids"][s].astype(np.int32)  # [1, 1]
        attention_bias_fp32 = data["attention_bias"][s]
        cos_fp32 = data["position_ids_cos"][s]
        sin_fp32 = data["position_ids_sin"][s]
        bias_u16 = quant_u16(attention_bias_fp32, bias_scale, bias_offset)
        cos_u16 = quant_u16(cos_fp32, cos_scale, cos_offset)
        sin_u16 = quant_u16(sin_fp32, sin_scale, sin_offset)

        # Part 1 (always run if target_part >= 2).
        p1_feed = {"input_ids": input_ids}
        embed_u16 = sessions[1].run([EMBED_HIDDEN], p1_feed)[0]

        if args.target_part == 2:
            seam_fp32 = dequant_u16(embed_u16, out_scale, out_offset)
        else:
            # Need to run part 2. Requant embed into part 2's input scale.
            p2_in_scale, p2_in_off = load_quant_for(2, EMBED_SLASH)
            p1_out_scale, p1_out_off = load_quant_for(1, EMBED_SLASH)
            embed_fp32 = dequant_u16(embed_u16, p1_out_scale, p1_out_off)
            embed_u16_p2 = quant_u16(embed_fp32, p2_in_scale, p2_in_off)
            feed2: dict[str, np.ndarray] = {
                EMBED_HIDDEN: embed_u16_p2,
                "attention_bias": bias_u16,
                "position_ids_cos": cos_u16,
                "position_ids_sin": sin_u16,
            }
            for li in range(0, LAYERS_PER_PART):
                k_scale, k_off = load_quant_for(2, f"past_key_values.{li}.key")
                v_scale, v_off = load_quant_for(2, f"past_key_values.{li}.value")
                feed2[f"past_key_values_{li}_key"] = quant_u16(
                    data[f"past_key_values.{li}.key"][s], k_scale, k_off)
                feed2[f"past_key_values_{li}_value"] = quant_u16(
                    data[f"past_key_values.{li}.value"][s], v_scale, v_off)
            out_names_2 = [L11_HIDDEN]
            l11_u16 = sessions[2].run(out_names_2, feed2)[0]

            if args.target_part == 3:
                l11_out_scale, l11_out_off = load_quant_for(2, L11_SLASH)
                seam_fp32 = dequant_u16(l11_u16, l11_out_scale, l11_out_off)
            else:
                # target_part == 4: also run part 3.
                p3_in_scale, p3_in_off = load_quant_for(3, L11_SLASH)
                p2_out_scale, p2_out_off = load_quant_for(2, L11_SLASH)
                l11_fp32 = dequant_u16(l11_u16, p2_out_scale, p2_out_off)
                l11_u16_p3 = quant_u16(l11_fp32, p3_in_scale, p3_in_off)
                feed3: dict[str, np.ndarray] = {
                    L11_HIDDEN: l11_u16_p3,
                    "attention_bias": bias_u16,
                    "position_ids_cos": cos_u16,
                    "position_ids_sin": sin_u16,
                }
                for li in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
                    k_scale, k_off = load_quant_for(3, f"past_key_values.{li}.key")
                    v_scale, v_off = load_quant_for(3, f"past_key_values.{li}.value")
                    feed3[f"past_key_values_{li}_key"] = quant_u16(
                        data[f"past_key_values.{li}.key"][s], k_scale, k_off)
                    feed3[f"past_key_values_{li}_value"] = quant_u16(
                        data[f"past_key_values.{li}.value"][s], v_scale, v_off)
                l23_u16 = sessions[3].run([L23_HIDDEN], feed3)[0]
                l23_out_scale, l23_out_off = load_quant_for(3, L23_SLASH)
                seam_fp32 = dequant_u16(l23_u16, l23_out_scale, l23_out_off)

        # Overwrite the downstream's seam raw file.
        raw_path = dst_raw_dir / f"sample_{s:03d}" / f"{sanitize(seam_slash)}.raw"
        # ensure contiguous fp32
        arr = np.ascontiguousarray(seam_fp32.astype(np.float32))
        arr.tofile(str(raw_path))
        elapsed = time.perf_counter() - t0
        print(f"  sample {s:2d}: {seam_name} min={arr.min():.4f} max={arr.max():.4f} "
              f"mean={arr.mean():.4f} std={arr.std():.4f} ({elapsed*1000:.0f}ms)")

    print(f"\nOverwrote {n_samples} seam raw files in {dst_raw_dir}")
    print(f"Next steps: rerun qairt-quantizer + qnn-context-binary-generator + "
          f"build_specula_4b_wrappers for part {args.target_part}:")
    print(f"  .venv-qairt/Scripts/python.exe scripts/qairt_quantize_4b_parts.py --parts {args.target_part}")
    print(f"  .venv-qairt/Scripts/python.exe scripts/compile_4b_bundle_ctx_bin_gen.py --parts {args.target_part}")
    # wrapper and dlc.json refresh
    print(f"  # refresh DLC json:")
    print(f"  qairt-dlc-to-json --input_dlc qwen3_4b_part{args.target_part}.w4a16-local.dlc ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
