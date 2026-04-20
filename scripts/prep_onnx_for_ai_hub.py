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
SOURCE_ONNX = REPO_ROOT / "models" / "qwen3-0.6b-onnx" / "onnx" / "model.onnx"
SOURCE_DATA = REPO_ROOT / "models" / "qwen3-0.6b-onnx" / "onnx" / "model.onnx_data"
STAGING = REPO_ROOT / "models" / "qwen3-0.6b-ai-hub"
STAGED_ONNX = STAGING / "model.onnx"
STAGED_DATA = STAGING / "model.data"

OLD_LOCATION = "model.onnx_data"
NEW_LOCATION = "model.data"


def patch_external_data_refs(model: onnx.ModelProto) -> int:
    """Rewrite every `location` external_data entry in initializers.

    Returns the number of TensorProto entries modified.
    """
    patched = 0
    for tensor in model.graph.initializer:
        if not tensor.external_data:
            continue
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value == OLD_LOCATION:
                entry.value = NEW_LOCATION
                patched += 1
                break
    return patched


def try_hardlink(src: Path, dst: Path) -> str:
    """Hardlink if possible, else copy. Returns 'hardlink' or 'copy'."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
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

    patched = patch_external_data_refs(model)
    print(f"patched {patched} external_data references: '{OLD_LOCATION}' -> '{NEW_LOCATION}'")
    if patched == 0:
        print("WARNING: no external_data references found to patch.")
        print("  this is unexpected for an onnx-community model with external weights;")
        print("  qai-hub will still reject if data file is missing.")

    print(f"saving patched graph to {STAGED_ONNX.name} (graph-only, no data rewrite) ...")
    t0 = time.perf_counter()
    # save_as_external_data=False means "don't re-consolidate" here, since
    # the graph already has external_data refs and the actual weights
    # live in a separate file we're about to link up.
    onnx.save(model, str(STAGED_ONNX), save_as_external_data=False)
    print(f"  saved in {time.perf_counter() - t0:.2f} s, {STAGED_ONNX.stat().st_size / (1024*1024):.1f} MB")

    print(f"linking weights {SOURCE_DATA.name} -> {STAGED_DATA.name} ...")
    t0 = time.perf_counter()
    mode = try_hardlink(SOURCE_DATA, STAGED_DATA)
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
