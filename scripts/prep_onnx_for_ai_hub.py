"""Phase 5 step 4a - stage a Qwen3 ONNX for qai-hub upload.

qai-hub requires model directories to contain only .onnx / .data /
.encodings / .bin files. The optimum export names its external-data
file `model.onnx_data`, which qai-hub rejects. Fix is a two-step:

    (1) patch every TensorProto.external_data entry that points at
        `model.onnx_data` to point at `model.data` instead.
    (2) copy the original weights file under the new name. (We used
        to hardlink for zero-duplication, but ORT 1.24+ refuses to
        read external-data files with >1 hard links.)

A third transform pins every graph input's declared shape to the
static dims AI Hub expects. Session 11 failed both Path A and Path
B-mask because the optimum export keeps batch + sequence dims as
symbolic `-1`; AI Hub's compiler pipeline has a pass
(`_op_identity.py:30`) that calls `np.broadcast_shapes` on dims and
chokes on SymbolicDim. See `docs/phase5_step6_compile_retro.md`.
The pin is a pure protobuf edit on `ValueInfoProto` and preserves
graph numerics bit-for-bit (cos=1.0 per session 10 bisection).

Multiple source variants are supported via --path; each comes from
`docs/phase5_step6_export_report.md` (x86 handoff):

    patha      BOOL casts folded out (ConstantOfShape replaces Gather_5),
               BOOL tensors remain downstream. Drops attention_mask input.
    pathbmask  entire BOOL mask subgraph deleted + new FP32 attention_bias
               input spliced into 28 Add_2 nodes. Zero BOOL tensors.

The session 7-9 `nomask` variant is quarantined (onnxsim corruption) and
not selectable here.

Run:
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py --path patha
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py --path pathbmask
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import onnx


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_qwen3_ai_hub import PATHS as COMPILE_PATHS, build_input_specs  # noqa: E402

# path_key -> (source_dir_name, staging_dir_name)
PATHS = {
    "patha": ("qwen3-0.6b-patha", "qwen3-0.6b-patha-ai-hub"),
    "pathbmask": ("qwen3-0.6b-pathbmask", "qwen3-0.6b-pathbmask-ai-hub"),
}

OLD_LOCATION = "model.onnx_data"
NEW_LOCATION = "model.data"


def _patch_tensor(tensor: onnx.TensorProto) -> bool:
    for entry in tensor.external_data:
        if entry.key == "location" and entry.value == OLD_LOCATION:
            entry.value = NEW_LOCATION
            return True
    return False


def patch_external_data_refs(model: onnx.ModelProto) -> tuple[int, int]:
    """Rewrite every `location` external_data entry in the model.

    Covers two places external tensors can live:
    - graph.initializer (the common case)
    - node.attribute.t for Constant / Constant-like ops with tensor
      attributes (optimum's --no-post-process export keeps a lot of
      constants here instead of promoting them to initializers).

    Returns (initializer_patches, node_attribute_patches).
    """
    init_patched = 0
    for tensor in model.graph.initializer:
        if _patch_tensor(tensor):
            init_patched += 1

    node_patched = 0
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.external_data:
                if _patch_tensor(attr.t):
                    node_patched += 1
            if attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.external_data and _patch_tensor(t):
                        node_patched += 1
    return init_patched, node_patched


def materialize(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return "copy"


def pin_input_shapes(model: onnx.ModelProto, specs: dict) -> tuple[int, int]:
    """Rewrite each matching graph input's TensorShapeProto to static dims.

    For each input listed in `specs`, overwrite its `dim` entries so every
    `dim_value` is the concrete integer from the spec. Inputs not in
    `specs` are left alone (spec-vs-ONNX parity is enforced at compile
    `--check` time).

    Returns (pinned_count, already_static_count).
    """
    pinned = 0
    already_static = 0
    for inp in model.graph.input:
        spec = specs.get(inp.name)
        if spec is None:
            continue
        target_shape, _dtype = spec
        t_shape = inp.type.tensor_type.shape
        current = tuple(
            (d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in t_shape.dim
        )
        if current == tuple(int(x) for x in target_shape):
            already_static += 1
            continue
        del t_shape.dim[:]
        for dim_value in target_shape:
            t_shape.dim.add().dim_value = int(dim_value)
        pinned += 1
    return pinned, already_static


def stage(path_key: str) -> int:
    source_dir_name, staging_dir_name = PATHS[path_key]
    source_dir = REPO_ROOT / "models" / source_dir_name
    staging = REPO_ROOT / "models" / staging_dir_name
    source_onnx = source_dir / "model.onnx"
    source_data = source_dir / "model.onnx_data"
    staged_onnx = staging / "model.onnx"
    staged_data = staging / "model.data"

    if not source_onnx.exists():
        print(f"ERROR: source ONNX missing at {source_onnx}")
        return 2
    if not source_data.exists():
        print(f"ERROR: external-data file missing at {source_data}")
        return 2

    staging.mkdir(parents=True, exist_ok=True)
    print(f"path key            : {path_key}")
    print(f"source dir          : {source_dir}")
    print(f"staging dir         : {staging}")

    print(f"loading graph-only (no weights into memory) from {source_onnx.name} ...")
    t0 = time.perf_counter()
    model = onnx.load(str(source_onnx), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s, {len(model.graph.initializer)} initializers")

    init_patched, node_patched = patch_external_data_refs(model)
    total = init_patched + node_patched
    print(f"patched {total} external_data references: '{OLD_LOCATION}' -> '{NEW_LOCATION}'")
    print(f"  initializer tensors        : {init_patched}")
    print(f"  node attribute tensors     : {node_patched}")
    if total == 0:
        print("WARNING: no external_data references found to patch.")

    with (source_dir / "config.json").open() as f:
        cfg = json.load(f)
    compile_extra = COMPILE_PATHS[path_key]["extra_input_specs"]
    specs = build_input_specs(cfg, compile_extra)
    pinned, already_static = pin_input_shapes(model, specs)
    print(f"pinned {pinned} graph input shapes to static dims "
          f"({already_static} already static, "
          f"{len(specs) - pinned - already_static} not declared on graph)")

    print(f"saving patched graph to {staged_onnx.name} (graph-only, no data rewrite) ...")
    t0 = time.perf_counter()
    onnx.save(model, str(staged_onnx), save_as_external_data=False)
    print(f"  saved in {time.perf_counter() - t0:.2f} s, {staged_onnx.stat().st_size / (1024*1024):.1f} MB")

    print(f"copying weights {source_data.name} -> {staged_data.name} ...")
    t0 = time.perf_counter()
    mode = materialize(source_data, staged_data)
    elapsed = time.perf_counter() - t0
    data_mb = staged_data.stat().st_size / (1024 * 1024)
    print(f"  {mode} complete in {elapsed:.2f} s, {data_mb:.1f} MB")

    print("\nstaging directory contents:")
    for p in sorted(staging.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0
        print(f"  {p.name:20s} {size_mb:8.1f} MB")

    print(f"\n=== STATUS: ok; AI Hub upload can target {staging} ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=sorted(PATHS), required=True)
    args = parser.parse_args()
    return stage(args.path)


if __name__ == "__main__":
    sys.exit(main())
