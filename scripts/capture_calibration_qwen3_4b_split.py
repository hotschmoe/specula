"""Derive per-part calibration inputs for the 4-part pathb split.

Consumes the monolithic calibration npz produced by
`capture_calibration_qwen3_4b.py` (same 10 humaneval samples, captured
at decode position 10 of the source optimum graph). Runs the split
sub-ONNX files on CPU-ORT to derive the hidden-state inputs that each
middle/last part consumes across its seam. Emits 4 per-part raw
calibration directories in the layout qairt-quantizer expects.

Output directories:
  models/calibration/qwen3_4b_ctx512_part1_raw/
  models/calibration/qwen3_4b_ctx512_part2_raw/
  models/calibration/qwen3_4b_ctx512_part3_raw/
  models/calibration/qwen3_4b_ctx512_part4_raw/

Each contains sample_NNN/*.raw plus input_list.txt.

Run:
    .venv/Scripts/python.exe scripts/capture_calibration_qwen3_4b_split.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

NUM_LAYERS = 36
LAYERS_PER_PART = 12

EMBED_HIDDEN = "/model/embed_tokens/Gather_output_0"
L11_HIDDEN = "/model/layers.11/Add_1_output_0"
L23_HIDDEN = "/model/layers.23/Add_1_output_0"


def sanitize(name: str) -> str:
    """Filesystem-safe filename from an ONNX tensor name. The input
    list stores the ORIGINAL name with this sanitized path, so the
    quantizer still gets the real tensor identity."""
    return name.replace("/", "_").replace(".", "_").lstrip("_")


def load_session(part_dir: Path, onnx_name: str = "model.onnx") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(
        str(part_dir / onnx_name), sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def graph_input_names_from(part_dir: Path, onnx_name: str) -> list[str]:
    m = onnx.load(str(part_dir / onnx_name), load_external_data=False)
    return [i.name for i in m.graph.input]


def write_sample(out_root: Path, sample_idx: int, feed: dict[str, np.ndarray],
                 graph_input_order: list[str]) -> str:
    sample_dir = out_root / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for name in graph_input_order:
        arr = np.ascontiguousarray(feed[name])
        raw_path = sample_dir / f"{sanitize(name)}.raw"
        arr.tofile(str(raw_path))
        parts.append(f"{name}:={raw_path}")
    return " ".join(parts)


def graph_input_names(part_dir: Path) -> list[str]:
    m = onnx.load(str(part_dir / "model.onnx"), load_external_data=False)
    return [i.name for i in m.graph.input]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", type=Path,
        default=MODELS / "calibration" / "qwen3_4b_ctx512_a.npz",
        help="Monolithic calibration npz produced by capture_calibration_qwen3_4b.py.",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=MODELS / "calibration",
        help="Per-part raw dirs land here as qwen3_4b_ctx512_part{1..4}_raw/.",
    )
    parser.add_argument(
        "--parts-root", type=Path,
        default=MODELS,
        help="Where the split part ONNXs live: qwen3-4b-arm-pathb-ctx512-part{1..4}/.",
    )
    parser.add_argument(
        "--halfdim", action="store_true",
        help="Load model_halfdim.onnx (half-dim cos/sin input) for parts 2/3/4 "
             "and write cos/sin raws as [1,1,64] half-dim. Matches Qualcomm's "
             "genie bundle rotary convention. See Phase 5o.",
    )
    args = parser.parse_args()

    part_dirs = [args.parts_root / f"qwen3-4b-arm-pathb-ctx512-part{i}" for i in (1, 2, 3, 4)]
    # Part 1 always uses model.onnx (no cos/sin involvement). Parts 2/3/4 can
    # use model_halfdim.onnx when --halfdim is set.
    part_onnx_names = ["model.onnx"] + (
        ["model_halfdim.onnx"] * 3 if args.halfdim else ["model.onnx"] * 3
    )
    for p, onnx_name in zip(part_dirs, part_onnx_names):
        if not (p / onnx_name).exists():
            print(f"FATAL: missing split part at {p / onnx_name}")
            return 2
    if args.halfdim:
        print("HALFDIM mode: parts 2/3/4 loaded from model_halfdim.onnx; "
              "cos/sin raws will be [1,1,64].")

    print(f"loading calibration npz: {args.npz}")
    data = np.load(str(args.npz))
    n_samples = data[data.files[0]].shape[0]
    print(f"  {len(data.files)} tensors, {n_samples} samples")

    print("reading part graph input orders ...")
    part_inputs = [graph_input_names_from(p, name)
                   for p, name in zip(part_dirs, part_onnx_names)]
    for i, names in enumerate(part_inputs, start=1):
        print(f"  part{i}: {len(names)} inputs")

    print("loading CPU sessions ...")
    t0 = time.perf_counter()
    sessions = [load_session(p, name)
                for p, name in zip(part_dirs, part_onnx_names)]
    print(f"  all 4 sessions loaded in {time.perf_counter() - t0:.1f}s")

    # Per-part raw dir + open list files.
    out_dirs = [args.out_root / f"qwen3_4b_ctx512_part{i}_raw" for i in (1, 2, 3, 4)]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    list_handles = [(d / "input_list.txt").open("w", encoding="utf-8") for d in out_dirs]

    try:
        for s in range(n_samples):
            t0 = time.perf_counter()
            input_ids = data["input_ids"][s]
            attention_bias = data["attention_bias"][s]
            cos = data["position_ids_cos"][s]
            sin = data["position_ids_sin"][s]
            if args.halfdim:
                # Full-dim cos/sin is [cos_half; cos_half] — first 64 elements
                # are the unique half. Truncate for the halfdim graph input.
                cos = cos[..., : cos.shape[-1] // 2]
                sin = sin[..., : sin.shape[-1] // 2]

            # Part 1 forward.
            p1_feed = {"input_ids": input_ids}
            embed_hidden = sessions[0].run([EMBED_HIDDEN], p1_feed)[0]

            # Part 2 feed + forward (consume past_kv[0..11]).
            p2_feed: dict[str, np.ndarray] = {
                EMBED_HIDDEN: embed_hidden,
                "attention_bias": attention_bias,
                "position_ids_cos": cos,
                "position_ids_sin": sin,
            }
            for li in range(0, LAYERS_PER_PART):
                p2_feed[f"past_key_values.{li}.key"] = data[f"past_key_values.{li}.key"][s]
                p2_feed[f"past_key_values.{li}.value"] = data[f"past_key_values.{li}.value"][s]
            l11_hidden = sessions[1].run([L11_HIDDEN], p2_feed)[0]

            # Part 3 feed + forward (consume past_kv[12..23]).
            p3_feed: dict[str, np.ndarray] = {
                L11_HIDDEN: l11_hidden,
                "attention_bias": attention_bias,
                "position_ids_cos": cos,
                "position_ids_sin": sin,
            }
            for li in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
                p3_feed[f"past_key_values.{li}.key"] = data[f"past_key_values.{li}.key"][s]
                p3_feed[f"past_key_values.{li}.value"] = data[f"past_key_values.{li}.value"][s]
            l23_hidden = sessions[2].run([L23_HIDDEN], p3_feed)[0]

            # Part 4 feed (not executed; we just need its inputs serialized).
            p4_feed: dict[str, np.ndarray] = {
                L23_HIDDEN: l23_hidden,
                "attention_bias": attention_bias,
                "position_ids_cos": cos,
                "position_ids_sin": sin,
            }
            for li in range(2 * LAYERS_PER_PART, NUM_LAYERS):
                p4_feed[f"past_key_values.{li}.key"] = data[f"past_key_values.{li}.key"][s]
                p4_feed[f"past_key_values.{li}.value"] = data[f"past_key_values.{li}.value"][s]

            # Emit per-part sample files in the declared-input order.
            feeds_per_part = [p1_feed, p2_feed, p3_feed, p4_feed]
            for i, (feed, names, handle, out_dir) in enumerate(
                zip(feeds_per_part, part_inputs, list_handles, out_dirs), start=1
            ):
                line = write_sample(out_dir, s, feed, names)
                handle.write(line + "\n")

            elapsed = time.perf_counter() - t0
            print(f"  sample {s:3d}: p1/p2/p3 forward + 4 part dumps in {elapsed:.1f}s")
    finally:
        for h in list_handles:
            h.close()

    # Report bytes per part.
    print("\n-- summary --")
    for i, d in enumerate(out_dirs, start=1):
        nbytes = sum(p.stat().st_size for p in d.rglob("*.raw"))
        print(f"  part{i}: {nbytes / 1e6:.1f} MB across {n_samples} samples "
              f"({d / 'input_list.txt'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
