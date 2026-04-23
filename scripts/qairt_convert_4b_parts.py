"""Drive qairt-converter over the 4 split pathb parts (fp32 DLCs).

One qairt-converter invocation per part. Produces .fp32.dlc files in
results/phase5_qwen3_4b_bundle/ (parallel to the phase3 results layout).

Requires: .venv-qairt activated + QAIRT 2.45 on PATH.

Run:
    .venv-qairt/Scripts/python.exe scripts/qairt_convert_4b_parts.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --preserve_onnx_output_order: keep present.N.key/value ordering
    # aligned with the ONNX graph output order so input_list + quantizer
    # stay positionally consistent with the raw calibration files.
    # --remove_unused_inputs: drops position_ids if it leaks back in
    # (shouldn't in the split graphs, but matches Phase 3 convention).
    common_flags = ["--preserve_onnx_output_order", "--remove_unused_inputs"]

    wanted = {int(p) for p in args.parts.split(",")}
    for idx in (1, 2, 3, 4):
        if idx not in wanted:
            continue
        src = MODELS / f"qwen3-4b-arm-pathb-ctx512-part{idx}" / "model.onnx"
        dst = args.out_dir / f"qwen3_4b_part{idx}.fp32.dlc"
        log = args.out_dir / f"qwen3_4b_part{idx}.qairt_converter.log"
        if not src.exists():
            print(f"FATAL: missing split ONNX at {src}")
            return 2
        cmd = [
            "qairt-converter",
            "--input_network", str(src),
            "--output_path", str(dst),
            *common_flags,
        ]
        print(f"\n=== part{idx}: qairt-converter ===")
        print(" ".join(cmd))
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"FAIL after {elapsed:.1f}s - see {log}")
            return proc.returncode
        size_mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        print(f"ok: {elapsed:.1f}s, {dst.name} = {size_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
