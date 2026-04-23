"""Drive qnn-context-binary-generator on each of the 4 w4a16 DLCs to
produce a weight-shared 4-part bundle (parallels Qualcomm's shipping
layout).

Invokes ctx-bin-gen ONCE PER DLC (not a single multi-DLC call) — the
multi-DLC form packs all graphs into ONE 4.8 GB .bin, which
ORT-QNN 2.1 rejects at session load with QNN_COMMON_ERROR_MEM_ALLOC
because it tries to allocate the full 4.8 GB up front. Separate per-
DLC .bins let ORT-QNN create one session per part and keep each
part's allocation bounded. `weight_sharing_enabled: true` in the HTP
backend-extensions config still applies — at runtime the four
sessions share weight memory when the HTP driver recognizes them as
a compilation group.

Requires: .venv-qairt activated + QAIRT 2.45 on PATH.

Run:
    .venv-qairt/Scripts/python.exe scripts/compile_4b_bundle_ctx_bin_gen.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
PHASE3_RESULTS = REPO / "results" / "phase3_qwen3_4b_compile"
QAIRT_BIN_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\x86_64-windows-msvc")
QAIRT_LIB_DEFAULT = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\x86_64-windows-msvc")


def ensure_configs(dst: Path) -> tuple[Path, Path]:
    """Copy the Phase-3 compile_config.json + htp_backend_ext_config.json
    into `dst`. They already set weight_sharing_enabled=true which is
    the required flag for multi-DLC bundling."""
    compile_cfg = dst / "compile_config.json"
    htp_cfg = dst / "htp_backend_ext_config.json"
    if not compile_cfg.exists():
        src = PHASE3_RESULTS / "compile_config.json"
        if src.exists():
            shutil.copy2(src, compile_cfg)
        else:
            compile_cfg.write_text(json.dumps({
                "backend_extensions": {
                    "shared_library_path": "QnnHtpNetRunExtensions.dll",
                    "config_file_path": "htp_backend_ext_config.json",
                }
            }, indent=4))
    if not htp_cfg.exists():
        src = PHASE3_RESULTS / "htp_backend_ext_config.json"
        if src.exists():
            shutil.copy2(src, htp_cfg)
        else:
            htp_cfg.write_text(json.dumps({
                "devices": [{
                    "soc_model": 88, "dsp_arch": "v81",
                    "cores": [{"core_id": 0, "perf_profile": "burst",
                               "rpc_control_latency": 100}],
                }],
                "memory": {"mem_type": "shared_buffer"},
                "context": {"weight_sharing_enabled": True},
            }))
    return compile_cfg, htp_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc-dir", type=Path, default=RESULTS)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument(
        "--binary-basename", type=str,
        default="qwen3_4b_4part_w4a16",
        help="Per-part basename. Each part lands at <basename>_part{N}.bin.",
    )
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    parser.add_argument("--qairt-bin", type=Path, default=QAIRT_BIN_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compile_cfg, _ = ensure_configs(args.out_dir)
    tool = args.qairt_bin / "qnn-context-binary-generator.exe"
    if not tool.exists():
        print(f"FATAL: qnn-context-binary-generator not found at {tool}")
        return 2

    # Windows DLL search for QnnHtp.dll + QnnHtpNetRunExtensions.dll.
    env = os.environ.copy()
    env["PATH"] = str(QAIRT_LIB_DEFAULT) + os.pathsep + env.get("PATH", "")

    wanted = {int(p) for p in args.parts.split(",")}
    for idx in (1, 2, 3, 4):
        if idx not in wanted:
            continue
        dlc = args.dlc_dir / f"qwen3_4b_part{idx}.w4a16-local.dlc"
        if not dlc.exists():
            print(f"FATAL: missing w4a16 DLC at {dlc}")
            return 2
        out_basename = args.out_dir / f"{args.binary_basename}_part{idx}"
        log = args.out_dir / f"{args.binary_basename}_part{idx}.qnn_ctx_bin_gen.log"
        cmd = [
            str(tool),
            "--backend", "QnnHtp.dll",
            "--dlc_path", str(dlc),
            "--binary_file", str(out_basename),
            "--config_file", str(compile_cfg),
        ]
        print(f"\n=== part{idx}: qnn-context-binary-generator ===")
        print(" ".join(cmd))
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  cwd=str(args.out_dir), env=env)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"FAIL after {elapsed:.1f}s - see {log}")
            return proc.returncode
        bin_file = Path(f"{out_basename}.bin")
        size_mb = bin_file.stat().st_size / 1e6 if bin_file.exists() else 0
        print(f"ok: {elapsed:.1f}s, {bin_file.name} = {size_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
