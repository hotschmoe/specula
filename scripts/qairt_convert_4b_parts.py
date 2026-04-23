"""Drive qairt-converter over the 4 split pathb parts (fp32 DLCs).

One qairt-converter invocation per part. Produces .fp32.dlc files in
results/phase5_qwen3_4b_bundle/ (parallel to the phase3 results layout).

Requires: .venv-qairt activated + QAIRT 2.45 on PATH.

Run:
    .venv-qairt/Scripts/python.exe scripts/qairt_convert_4b_parts.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
QAIRT_BIN_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\x86_64-windows-msvc")
QAIRT_LIB_PYTHON_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\python")


def run_tool(tool: Path, args: list[str], log: Path) -> int:
    """Windows QAIRT tools are Python scripts with no .exe wrapper, so
    invoke them via the current interpreter. Also prepend QAIRT's
    lib/python to PYTHONPATH so `qti.aisw.*` imports resolve."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--qairt-bin", type=Path, default=QAIRT_BIN_DEFAULT,
                        help="Directory containing qairt-converter (Python script).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tool = args.qairt_bin / "qairt-converter"
    if not tool.exists():
        print(f"FATAL: qairt-converter not found at {tool}")
        return 2

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
        tool_args = [
            "--input_network", str(src),
            "--output_path", str(dst),
            *common_flags,
        ]
        print(f"\n=== part{idx}: qairt-converter ===")
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
