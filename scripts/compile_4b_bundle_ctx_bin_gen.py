"""Drive qnn-context-binary-generator on the 4 w4a16 DLCs to produce a
weight-shared 4-part bundle (parallels Qualcomm's shipping layout).

The plan per docs/qualcomm_reproduction_4b.md Phase 5: pass all 4
DLCs as a comma-separated --dlc_path, keep weight_sharing_enabled in
the backend-extensions config. HTP memory per part should stay well
under the 3.67 GB serializer ceiling that Phase 3c hit with the
single 4B DLC (~4.86 GB).

Requires: .venv-qairt activated + QAIRT 2.45 on PATH.

Run:
    .venv-qairt/Scripts/python.exe scripts/compile_4b_bundle_ctx_bin_gen.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
PHASE3_RESULTS = REPO / "results" / "phase3_qwen3_4b_compile"


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
        help="Basename for the emitted bundle. ctx-bin-gen appends "
             "_1, _2, ... per DLC (one .bin per DLC when multi-DLC).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compile_cfg, _ = ensure_configs(args.out_dir)

    dlcs = [args.dlc_dir / f"qwen3_4b_part{i}.w4a16-local.dlc" for i in (1, 2, 3, 4)]
    for d in dlcs:
        if not d.exists():
            print(f"FATAL: missing w4a16 DLC at {d}")
            return 2

    dlc_arg = ",".join(str(d) for d in dlcs)
    out_basename = args.out_dir / args.binary_basename
    log = args.out_dir / f"{args.binary_basename}.qnn_ctx_bin_gen.log"

    # cwd = out_dir so the compile_config's relative reference to
    # htp_backend_ext_config.json resolves.
    cmd = [
        "qnn-context-binary-generator",
        "--backend", "QnnHtp.dll",
        "--dlc_path", dlc_arg,
        "--binary_file", str(out_basename),
        "--config_file", str(compile_cfg),
    ]
    print("=== qnn-context-binary-generator (4-part weight-shared) ===")
    print(" ".join(cmd))
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(args.out_dir))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAIL after {elapsed:.1f}s - see {log}")
        return proc.returncode
    print(f"ok: {elapsed:.1f}s")
    for bin_file in sorted(args.out_dir.glob(f"{args.binary_basename}*.bin")):
        size_mb = bin_file.stat().st_size / 1e6
        print(f"  {bin_file.name}: {size_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
