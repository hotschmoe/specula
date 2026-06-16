"""Assemble the calibrated 14B bundle on the X2E from mixed parts.

Pulls the re-quantized uint16 decoder bins (parts 2-9) from the Threadripper
(`09b_bin_calib/part{k}.bin`) and combines them with the original fp16
part1/part10 bins (which already load) into a new bundle, then regenerates
`bin_info/` for all 10 parts with qnn-context-binary-utility.

    BOX_PASS=... MSYS_NO_PATHCONV=1 \
    .venv-qairt/Scripts/python.exe end-to-end/build_server/assemble_calib_bundle.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "models" / "qwen3_14b-w8a16-specula-x2e"
DST = REPO / "models" / "qwen3_14b-w8a16-specula-x2e-calib"
BOX_BIN = "/mnt/vm_8tb/specula-build/runs/qwen3_14b_w8a16/09b_bin_calib"
UTIL = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\aarch64-windows-msvc"
            r"\qnn-context-binary-utility.exe")
BOXSSH = REPO / "end-to-end" / "build_server" / "boxssh.py"
BIN = "qwen3_14b_w8a16_part_{}_of_10.bin"


def box_get(remote: str, local: Path) -> None:
    subprocess.run([sys.executable, str(BOXSSH), "get", remote, str(local)], check=True)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    # 1. configs + tokenizer from the original bundle
    for n in ("genie_config.json", "config.json", "metadata.json",
              "htp_backend_ext_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "generation_config.json", "sample_prompt.txt"):
        if (SRC / n).exists():
            shutil.copy2(SRC / n, DST / n)
    # 2. keep original fp16 part1 + part10 (they load)
    for k in (1, 10):
        shutil.copy2(SRC / BIN.format(k), DST / BIN.format(k))
        print(f"  part{k}: reused fp16 ({(DST / BIN.format(k)).stat().st_size/1e9:.2f} GB)")
    # 3. pull new uint16 decoder bins 2..9
    for k in range(2, 10):
        dst = DST / BIN.format(k)
        box_get(f"{BOX_BIN}/part{k}.bin", dst)
        print(f"  part{k}: pulled calib int8 ({dst.stat().st_size/1e9:.2f} GB)")
    # 4. regenerate bin_info for all 10
    (DST / "bin_info").mkdir(exist_ok=True)
    for k in range(1, 11):
        out = DST / "bin_info" / f"part_{k}_of_10.json"
        subprocess.run([str(UTIL), "--context_binary", str(DST / BIN.format(k)),
                        "--json_file", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  bin_info regenerated for 10 parts")
    print(f"\nassembled: {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
