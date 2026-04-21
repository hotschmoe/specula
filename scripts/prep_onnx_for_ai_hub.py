"""Phase 5 step 4a - stage the Qwen3 ONNX for qai-hub upload.

qai-hub requires model directories to contain only .onnx / .data /
.encodings / .bin files. onnx-community's export names its external-data
file `model.onnx_data`, which qai-hub rejects. Fix is a two-step:

    (1) patch every TensorProto.external_data entry that points at
        `model.onnx_data` to point at `model.data` instead.
    (2) hardlink the original weights file to the new name so no
        data is duplicated (or fall back to a copy on filesystems
        without hardlink support).

Output directory: models/qwen3-0.6b-ai-hub/ with:
    model.onnx  (the patched graph)
    model.data  (hardlink to original model.onnx_data, 1993 MB shared)

Idempotent: overwrites model.onnx and re-links model.data on re-run.

Run:
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

import onnx


REPO_ROOT = Path(__file__).resolve().parent.parent
# Source = ORT-BASIC-optimized version of the optimum --no-post-process
# export. The raw optimum output ran aground on HTP compile at
# /model/Gather_5 (dynamic attention-mask subgraph). ORT's basic graph
# optimizer constant-folds the mask-construction Range/Shape chain
# without fusing ops back into com.microsoft. Produced by
# scripts/ort_optimize_onnx.py.
SOURCE_ONNX = REPO_ROOT / "models" / "qwen3-0.6b-patched" / "model.onnx"
SOURCE_DATA = REPO_ROOT / "models" / "qwen3-0.6b-patched" / "model.onnx_data"
STAGING = REPO_ROOT / "models" / "qwen3-0.6b-patched-ai-hub"
STAGED_ONNX = STAGING / "model.onnx"
STAGED_DATA = STAGING / "model.data"

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
            # Single tensor attribute (e.g. Constant's `value`).
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.external_data:
                if _patch_tensor(attr.t):
                    node_patched += 1
            # Repeated-tensor attributes are rare but handle them too.
            if attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.external_data and _patch_tensor(t):
                        node_patched += 1
    return init_patched, node_patched


def materialize(src: Path, dst: Path) -> str:
    """Copy src -> dst, replacing any existing dst.

    Originally used os.link() for zero-disk-duplication, but ORT 1.24+
    ships an onnx C++ loader with a 'hardlink attack' check that refuses
    to read any external-data file that has >1 hard links. Since we both
    (a) validate the staged ONNX locally with ORT-CPU and (b) want
    qai-hub to see a clean single-reference file, we just copy. 3 GB of
    disk is cheap and this is the least-surprising behaviour.
    """
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    if not SOURCE_ONNX.exists():
        print(f"ERROR: source ONNX missing at {SOURCE_ONNX}")
        print("  run scripts/download_qwen3_onnx.py first")
        return 2
    if not SOURCE_DATA.exists():
        print(f"ERROR: external-data file missing at {SOURCE_DATA}")
        return 2

    STAGING.mkdir(parents=True, exist_ok=True)
    print(f"staging dir         : {STAGING}")

    print(f"loading graph-only (no weights into memory) from {SOURCE_ONNX.name} ...")
    t0 = time.perf_counter()
    model = onnx.load(str(SOURCE_ONNX), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s, {len(model.graph.initializer)} initializers")

    init_patched, node_patched = patch_external_data_refs(model)
    total = init_patched + node_patched
    print(f"patched {total} external_data references: '{OLD_LOCATION}' -> '{NEW_LOCATION}'")
    print(f"  initializer tensors        : {init_patched}")
    print(f"  node attribute tensors     : {node_patched}")
    if total == 0:
        print("WARNING: no external_data references found to patch.")
        print("  if weights file exists separately, qai-hub may still reject.")

    print(f"saving patched graph to {STAGED_ONNX.name} (graph-only, no data rewrite) ...")
    t0 = time.perf_counter()
    # save_as_external_data=False means "don't re-consolidate" here, since
    # the graph already has external_data refs and the actual weights
    # live in a separate file we're about to link up.
    onnx.save(model, str(STAGED_ONNX), save_as_external_data=False)
    print(f"  saved in {time.perf_counter() - t0:.2f} s, {STAGED_ONNX.stat().st_size / (1024*1024):.1f} MB")

    print(f"copying weights {SOURCE_DATA.name} -> {STAGED_DATA.name} ...")
    t0 = time.perf_counter()
    mode = materialize(SOURCE_DATA, STAGED_DATA)
    elapsed = time.perf_counter() - t0
    data_mb = STAGED_DATA.stat().st_size / (1024 * 1024)
    print(f"  {mode} complete in {elapsed:.2f} s, {data_mb:.1f} MB")

    print("\nstaging directory contents:")
    for p in sorted(STAGING.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0
        print(f"  {p.name:20s} {size_mb:8.1f} MB")

    print(f"\n=== STATUS: ok; AI Hub upload can target {STAGING} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
