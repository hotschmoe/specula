"""Replace dynamic dim_params in the ONNX with concrete values.

The optimum --no-post-process export exposes every batch / seq_len /
past_len dimension as a dim_param (symbolic). This prevents ORT's
constant-folder from eliminating the Range/Shape/Gather subgraph that
HTP rejects at /model/Gather_5. input_specs at compile time only
tells QAIRT the shapes at the IO boundary -- internal shape-inference
is still symbolic.

Pinning the dim_params directly in the graph, then running shape
inference, makes the Range ops foldable in a subsequent ORT BASIC
pass (scripts/ort_optimize_onnx.py).

Pin values align with the decode-only compile plan
(compile_qwen3_ai_hub.py: ctx 512, seq_len=1, past=511).

Run:
    .venv\\Scripts\\python.exe scripts\\freeze_onnx_dims.py
"""

import sys
import time
from pathlib import Path

import onnx
from onnx import shape_inference


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "model.onnx"
DEST_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum-frozen"
DEST_ONNX = DEST_DIR / "model.onnx"

DIM_OVERRIDES = {
    "batch_size": 1,
    "sequence_length": 1,
    "total_sequence_length": 512,
    "past_sequence_length": 511,
}


def pin_dims_on_value_info(vi: onnx.ValueInfoProto) -> int:
    patched = 0
    if not vi.type.HasField("tensor_type"):
        return 0
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return 0
    for dim in tt.shape.dim:
        if dim.HasField("dim_param") and dim.dim_param in DIM_OVERRIDES:
            value = DIM_OVERRIDES[dim.dim_param]
            dim.Clear()
            dim.dim_value = value
            patched += 1
    return patched


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: source ONNX missing at {SOURCE}")
        return 2

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {SOURCE} (graph only) ...")
    t0 = time.perf_counter()
    model = onnx.load(str(SOURCE), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s")

    total = 0
    for vi in model.graph.input:
        total += pin_dims_on_value_info(vi)
    for vi in model.graph.output:
        total += pin_dims_on_value_info(vi)
    for vi in model.graph.value_info:
        total += pin_dims_on_value_info(vi)
    print(f"pinned {total} dim_params to concrete values: {DIM_OVERRIDES}")

    # Re-run shape inference so the concrete dims propagate through the
    # graph (so ORT BASIC can then constant-fold Range/Shape/Gather).
    print("running shape inference ...")
    t0 = time.perf_counter()
    model = shape_inference.infer_shapes(model, strict_mode=False)
    print(f"  done in {time.perf_counter() - t0:.2f} s; value_info now {len(model.graph.value_info)}")

    print(f"saving {DEST_ONNX} ...")
    t0 = time.perf_counter()
    onnx.save(model, str(DEST_ONNX), save_as_external_data=False)
    print(f"  saved in {time.perf_counter() - t0:.2f} s, {DEST_ONNX.stat().st_size / (1024*1024):.1f} MB")

    # Also hardlink / copy the external data + config for downstream.
    source_data = SOURCE.with_name("model.onnx_data")
    dest_data = DEST_ONNX.with_name("model.onnx_data")
    if not dest_data.exists():
        print(f"copying {source_data.name} alongside ...")
        import shutil
        shutil.copy2(source_data, dest_data)
        print(f"  copy: {dest_data.stat().st_size / (1024*1024):.1f} MB")

    source_cfg = SOURCE.parent / "config.json"
    if source_cfg.exists():
        import shutil
        shutil.copy2(source_cfg, DEST_DIR / "config.json")

    print("\n--- next steps ---")
    print(f"1. Inspect: python scripts/inspect_onnx_ops.py --model {DEST_ONNX}")
    print(f"2. Run ORT BASIC on this frozen model (point ort_optimize_onnx.py at it)")
    print(f"3. Re-stage + compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
