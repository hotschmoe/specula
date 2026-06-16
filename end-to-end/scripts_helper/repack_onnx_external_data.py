"""Repack an ONNX model's external data into a tight single file.

The optimum 14B export (via optimum_export_4b.py) left a `model.onnx_data`
that is ~2x the referenced weight bytes — the post-fix `onnx.save` leaves
dead space (only ~59 GB of a 118 GB file is referenced by initializers).
This script copies *only* the referenced bytes into a new tight file,
streaming one tensor at a time (≤ one tensor in RAM, ~3 GB max), so it runs
on a 48 GB box without OOM and roughly halves the on-disk footprint —
enough to make the pathb rewrite chain fit on fast local disk.

Usage:
  python repack_onnx_external_data.py --src <dir>/model.onnx --dst <dir2>/model.onnx
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnx
from onnx import TensorProto

CHUNK = 64 * 1024 * 1024
DATA_NAME = "model.onnx_data"


def _ext(init) -> dict:
    return {kv.key: kv.value for kv in init.external_data}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="source model.onnx")
    ap.add_argument("--dst", required=True, type=Path, help="dest model.onnx")
    args = ap.parse_args()

    src_dir = args.src.parent
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    dst_data = args.dst.parent / DATA_NAME

    t0 = time.time()
    m = onnx.load(str(args.src), load_external_data=False)
    n_ext = sum(1 for i in m.graph.initializer
                if i.data_location == TensorProto.EXTERNAL)
    print(f"[repack] {n_ext} external initializers; streaming -> {dst_data}")

    pos = 0
    open_src: dict[str, object] = {}
    try:
        with open(dst_data, "wb") as out:
            for k, init in enumerate(m.graph.initializer):
                if init.data_location != TensorProto.EXTERNAL:
                    continue
                d = _ext(init)
                loc = d["location"]
                off = int(d.get("offset", 0))
                ln = int(d["length"])
                f = open_src.get(loc)
                if f is None:
                    f = open(src_dir / loc, "rb")
                    open_src[loc] = f
                f.seek(off)
                remaining = ln
                while remaining:
                    b = f.read(min(CHUNK, remaining))
                    if not b:
                        raise IOError(f"short read on {init.name} "
                                      f"({loc} @ {off}+{ln})")
                    out.write(b)
                    remaining -= len(b)
                # rewrite external ref to the new tight file
                del init.external_data[:]
                for key, val in (("location", DATA_NAME),
                                 ("offset", str(pos)),
                                 ("length", str(ln))):
                    e = init.external_data.add()
                    e.key, e.value = key, val
                pos += ln
                if (k + 1) % 100 == 0:
                    print(f"[repack]   {k+1} tensors, {pos/1e9:.1f} GB")
    finally:
        for f in open_src.values():
            f.close()

    # Write the proto only (external refs already point at the new file).
    args.dst.write_bytes(m.SerializeToString())
    print(f"[repack] done: {pos/1e9:.1f} GB data, proto {args.dst} "
          f"({args.dst.stat().st_size/1e6:.1f} MB), {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
