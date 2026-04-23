"""Drive qairt-quantizer over the 4 fp32 DLCs to produce w4a16 DLCs.

One qairt-quantizer invocation per part, consuming that part's raw
calibration list. Produces .w4a16-local.dlc files under
results/phase5_qwen3_4b_bundle/.

IMPORTANT - int64→int32 input_ids patch (per Phase 3b finding):
The pinned split ONNX declares input_ids as int64 but the DLC's
APP_WRITE port quantizes it to int32. We rewrite the raw file in
models/calibration/qwen3_4b_ctx512_part1_raw/sample_*/input_ids.raw
in-place (int64 -> int32) before handing to the quantizer. The source
.npz is left intact.

Requires: .venv-qairt activated + QAIRT 2.45 on PATH.

Run:
    .venv-qairt/Scripts/python.exe scripts/qairt_quantize_4b_parts.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
CALIB_ROOT = MODELS / "calibration"
QAIRT_BIN_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\x86_64-windows-msvc")
QAIRT_LIB_PYTHON_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\python")


def run_tool(tool: Path, args: list[str], log: Path) -> int:
    env = os.environ.copy()
    qairt_pypath = str(QAIRT_LIB_PYTHON_DEFAULT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = qairt_pypath + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, str(tool), *args]
    with log.open("w", encoding="utf-8") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def fix_input_ids_dtype(part1_raw_root: Path) -> int:
    """In-place rewrite of input_ids.raw files: int64 -> int32. Idempotent
    (if already 4 bytes per element we short-circuit). Returns number of
    files rewritten."""
    rewritten = 0
    for raw in part1_raw_root.glob("sample_*/input_ids.raw"):
        nbytes = raw.stat().st_size
        # Shape is [1, 1] = 1 element.
        if nbytes == 4:
            continue  # already int32
        if nbytes != 8:
            raise RuntimeError(f"unexpected input_ids.raw size {nbytes} at {raw}")
        arr64 = np.fromfile(raw, dtype=np.int64)
        arr32 = arr64.astype(np.int32)
        arr32.tofile(str(raw))
        rewritten += 1
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    parser.add_argument("--fp32-dir", type=Path, default=RESULTS)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--weights-bitwidth", type=int, default=4,
                        help="Default w4. Phase 5l finds Part 4 needs w8 because "
                             "the lm_head 2560x151936 matrix loses too much precision "
                             "at w4 (isolated part4 cos 0.671 at w4 -> 0.988 at w8). "
                             "Override per part via --weights-bitwidth 8 with --parts 4.")
    parser.add_argument("--act-bitwidth", type=int, default=16)
    parser.add_argument("--qairt-bin", type=Path, default=QAIRT_BIN_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tool = args.qairt_bin / "qairt-quantizer"
    if not tool.exists():
        print(f"FATAL: qairt-quantizer not found at {tool}")
        return 2

    # One-time patch: coerce input_ids.raw to int32 for part1 only.
    p1_raw = CALIB_ROOT / "qwen3_4b_ctx512_part1_raw"
    if p1_raw.exists():
        n = fix_input_ids_dtype(p1_raw)
        if n:
            print(f"patched {n} input_ids.raw files (int64 -> int32)")

    wanted = {int(p) for p in args.parts.split(",")}
    for idx in (1, 2, 3, 4):
        if idx not in wanted:
            continue
        fp32 = args.fp32_dir / f"qwen3_4b_part{idx}.fp32.dlc"
        dst = args.out_dir / f"qwen3_4b_part{idx}.w4a16-local.dlc"
        list_file = CALIB_ROOT / f"qwen3_4b_ctx512_part{idx}_raw" / "input_list.txt"
        log = args.out_dir / f"qwen3_4b_part{idx}.qairt_quantizer.log"
        if not fp32.exists():
            print(f"FATAL: missing fp32 DLC at {fp32}")
            return 2
        if not list_file.exists():
            print(f"FATAL: missing calibration list at {list_file}")
            return 2
        # Part 4 contains the lm_head (2560x151936 = 389M params). At w4 that
        # matrix has only 16 distinct weight levels, which is too coarse even
        # with per-channel scaling — isolated Part 4 cos drops to 0.67. At w8
        # it recovers to 0.988. So default Part 4 to w8 while parts 1-3 stay
        # w4. Overridable via --weights-bitwidth.
        part_weights_bw = args.weights_bitwidth
        if idx == 4 and args.weights_bitwidth == 4:
            part_weights_bw = 8
            print(f"[part4 auto-override] weights_bitwidth 4 -> 8 (lm_head fidelity)")
        tool_args = [
            "--input_dlc", str(fp32),
            "--output_dlc", str(dst),
            "--input_list", str(list_file),
            "--weights_bitwidth", str(part_weights_bw),
            "--act_bitwidth", str(args.act_bitwidth),
            # The bare default silently applies aggressive outlier rejection
            # despite help text claiming min-max. Explicit min-max asymmetric
            # restores true min-max. See Phase 5h.
            "--act_quantizer_calibration", "min-max",
            "--act_quantizer_schema", "asymmetric",
            # Per-channel (weight scale per output channel) and per-row
            # (per-Matmul-row scale) weight quantization plus Cross-Layer
            # Equalization redistribute weight magnitudes so HTP internal
            # math runs at full fp32-equivalent magnitude instead of
            # cascade-compressing. Without these flags, Part 2 HTP produces
            # L11 at ±1400 vs CPU-ORT ±16000 (10× compression); with them,
            # HTP produces ±16000 matching CPU-ORT (cos 0.9996). Also
            # brings per-part bin size in line with Qualcomm's shipping
            # bundle (Part 2: 1220 MB → 615 MB vs Qualcomm's 669 MB).
            # See Phase 5k.
            "--use_per_channel_quantization",
            "--use_per_row_quantization",
            "--apply_algorithms", "cle",
        ]
        print(f"\n=== part{idx}: qairt-quantizer ===")
        print(f"  {sys.executable} {tool} {' '.join(tool_args)}")
        t0 = time.perf_counter()
        rc = run_tool(tool, tool_args, log)
        elapsed = time.perf_counter() - t0
        if rc != 0:
            print(f"FAIL after {elapsed:.1f}s - see {log}")
            return rc
        size_mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        print(f"ok: {elapsed:.1f}s, {dst.name} = {size_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
