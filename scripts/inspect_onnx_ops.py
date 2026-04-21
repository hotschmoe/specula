"""Inventory ONNX op types + domains. Needed to scope the QAIRT op-
lowering fix after the first AI Hub compile failed with
`No Op registered for SimplifiedLayerNormalization with domain_version of 14`.

Reports all (domain, op_type) pairs and flags any op not in the default
onnx domain as a potential QAIRT incompatibility. Use after a new ONNX
export (e.g. the x86-produced Qwen3 ONNX) to confirm the file will
make it past the AI Hub OPTIMIZING_MODEL phase.

Run:
    .venv\\Scripts\\python.exe scripts\\inspect_onnx_ops.py
    .venv\\Scripts\\python.exe scripts\\inspect_onnx_ops.py --model path/to/model.onnx
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import onnx


DEFAULT_ONNX = (
    Path(__file__).resolve().parent.parent
    / "models" / "qwen3-0.6b-onnx" / "onnx" / "model.onnx"
)


def inspect(model_path: Path) -> int:
    if not model_path.exists():
        print(f"ERROR: missing {model_path}")
        return 2
    print(f"inspecting: {model_path}")
    model = onnx.load(str(model_path), load_external_data=False)

    # Opset imports.
    print("opset imports:")
    for imp in model.opset_import:
        domain = imp.domain if imp.domain else "<default>"
        print(f"  {domain:20s} version {imp.version}")

    # Op-type histogram keyed by (domain, op_type).
    hist: Counter = Counter()
    for node in model.graph.node:
        domain = node.domain if node.domain else ""
        hist[(domain, node.op_type)] += 1

    print(f"\ntotal nodes: {sum(hist.values())}")
    print("\nop histogram (sorted by count):")
    print(f"  {'domain':22s} {'op_type':36s} {'count':>6s}")
    for (domain, op), count in hist.most_common():
        d = domain if domain else "<default>"
        print(f"  {d:22s} {op:36s} {count:6d}")

    # Flag anything NOT in the default ("") domain.
    non_standard = [(d, o, c) for (d, o), c in hist.items() if d]
    if non_standard:
        print("\nNON-STANDARD ops (will need decomposition or QAIRT custom op):")
        for d, o, c in sorted(non_standard, key=lambda t: -t[2]):
            print(f"  {d}::{o}  ({c} instances)")
    else:
        print("\nall ops are in the default onnx domain")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ONNX,
        help=f"path to .onnx file (default: {DEFAULT_ONNX})",
    )
    args = parser.parse_args()
    return inspect(args.model)


if __name__ == "__main__":
    sys.exit(main())
