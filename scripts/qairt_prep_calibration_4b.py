"""Expand a Qwen3-4B calibration .npz into the on-disk layout qairt-quantizer
expects: per-sample raw binaries + an input_list.txt indexing them.

The list-file convention used: one line per sample, space-separated
`TENSOR_NAME:=PATH` entries. The order of names within a line must
match the ONNX graph's input order (qairt-quantizer treats positional
order as authoritative when names collide).

Run:
    .venv-arm-export/Scripts/python.exe scripts/qairt_prep_calibration_4b.py \\
        --npz models/calibration/qwen3_4b_ctx512_a.npz \\
        --out-dir models/calibration/qwen3_4b_ctx512_a_raw \\
        --onnx models/qwen3-4b-arm-pathb-ctx512/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True,
                        help="The pinned-shape pathb ONNX. Used only to read "
                             "graph-input ordering — qairt-quantizer is positional.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, default=None,
                        help="Defaults to <out-dir>/input_list.txt")
    args = parser.parse_args()

    list_path = args.input_list or (args.out_dir / "input_list.txt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading npz: {args.npz}")
    data = np.load(str(args.npz))
    n_samples = data[data.files[0]].shape[0]
    print(f"  {len(data.files)} tensors, {n_samples} samples")

    print(f"reading graph-input order from: {args.onnx}")
    m = onnx.load(str(args.onnx), load_external_data=False)
    graph_input_order = [i.name for i in m.graph.input]
    print(f"  graph has {len(graph_input_order)} inputs")

    # Sanity: every npz key must be a graph input.
    npz_keys = set(data.files)
    extra = npz_keys - set(graph_input_order)
    missing = set(graph_input_order) - npz_keys
    if extra:
        print(f"  WARNING: npz has {len(extra)} tensors not in graph: {sorted(extra)[:5]}...")
    if missing:
        print(f"  FATAL: graph expects inputs missing from npz: {sorted(missing)}")
        return 2

    with list_path.open("w", encoding="utf-8") as f_list:
        for s in range(n_samples):
            sample_dir = args.out_dir / f"sample_{s:03d}"
            sample_dir.mkdir(exist_ok=True)
            parts: list[str] = []
            for name in graph_input_order:
                arr = data[name][s]
                arr = np.ascontiguousarray(arr)
                # Sanitize filename: keep dots replaced with underscores
                # so file-system tools don't get confused, but keep the
                # original tensor name in the list.
                safe = name.replace(".", "_")
                raw_path = sample_dir / f"{safe}.raw"
                arr.tofile(str(raw_path))
                # Write list with the ONNX tensor name -> path
                parts.append(f"{name}:={raw_path}")
            f_list.write(" ".join(parts) + "\n")
            print(f"  sample {s}: {len(parts)} tensors written to {sample_dir.name}/")

    total_bytes = sum(p.stat().st_size for p in args.out_dir.rglob("*.raw"))
    print(f"\nwrote {n_samples} samples, {total_bytes / 1e9:.2f} GB total")
    print(f"input_list: {list_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
