"""Repack an HF-exported ONNX into a qai-hub-compatible model directory.

qai-hub's ONNX uploader (client.py `_determine_model_type_from_dir`)
accepts a model *directory* containing exactly one `.onnx` plus at most
one `.data` external-weights file. The onnx-community Gemma 4 export
ships its weights as THREE files (`*.onnx_data`, `*.onnx_data_1`,
`*.onnx_data_2`) — qai-hub rejects that.

This consolidates the multi-file external data into a single `.data`
file and writes a clean `<out>/model.onnx` + `<out>/model.data`
directory ready to hand to `submit_compile_job(model=<out>)`.

Loads the full model into RAM (~5 GB for the fp16 decoder) — make sure
the data files are present alongside the input `.onnx`.

Usage:
    python scripts/repack_onnx.py \
        ../models/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_fp16.onnx \
        ../models/gemma-4-E2B-it-ONNX/qaihub_decoder_fp16
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import onnx


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if not src.exists():
        print(f"FATAL: not found: {src}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    # qai-hub requires the dir to hold ONLY .onnx + .data — start clean.
    for f in out_dir.iterdir():
        f.unlink()

    t0 = time.time()
    print(f"[load]  {src}  (+ external data — ~5 GB into RAM)")
    model = onnx.load(str(src), load_external_data=True)
    print(f"[load]  done in {time.time() - t0:.0f}s")

    onnx_out = out_dir / "model.onnx"
    t1 = time.time()
    print(f"[save]  {onnx_out}  + model.data (single consolidated file)")
    onnx.save_model(
        model, str(onnx_out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    print(f"[save]  done in {time.time() - t1:.0f}s")

    files = sorted(out_dir.iterdir())
    print(f"\n[out]   {out_dir}")
    for f in files:
        print(f"  {f.name:<16} {f.stat().st_size / 1e9:.2f} GB")
    exts = {f.suffix for f in files}
    if exts - {".onnx", ".data"}:
        print(f"WARNING: dir has unexpected files {exts} — qai-hub wants "
              f"only .onnx + .data", file=sys.stderr)
        return 3
    print(f"\nReady: submit_compile_gemma4.py --onnx {onnx_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
