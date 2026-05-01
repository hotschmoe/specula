"""Final bundle assembly: copy artifacts into a target dir, write
metadata.json with sha256s + provenance, tar for transport."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assemble_bundle(
    *,
    bin_path: Path,
    encodings_path: Path,
    bin_info_path: Path,
    tokenizer_dir: Path,
    bundle_dir: Path,
    bundle_name: str,
    metadata: dict,
    tar_out: Path,
) -> dict:
    """Stage files into bundle_dir, write metadata.json with
    sha256+sizes, then tar the dir into tar_out.

    metadata is the recipe-level info dict (model_id, precision,
    pipeline params) — we extend it with files{} and write to
    bundle_dir/metadata.json.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts.
    shutil.copy2(bin_path, bundle_dir / bin_path.name)
    shutil.copy2(encodings_path, bundle_dir / encodings_path.name)
    shutil.copy2(bin_info_path, bundle_dir / "bin_info.json")
    for n in ("tokenizer.json", "tokenizer_config.json", "config.json",
              "generation_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / n
        if src.exists():
            shutil.copy2(src, bundle_dir / n)

    # Hash everything.
    files: dict[str, dict] = {}
    for entry in sorted(bundle_dir.iterdir()):
        if entry.is_file() and entry.name != "metadata.json":
            files[entry.name] = {
                "size": entry.stat().st_size,
                "sha256": _sha256(entry),
            }

    metadata = {**metadata, "bundle_name": bundle_name,
                "files": files,
                "total_bytes": sum(v["size"] for v in files.values())}
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # tar (no gzip — .bin is mostly already-binary weights, ~0% compress).
    print(f"[bundle] tar {tar_out} ...")
    t0 = time.time()
    rc = subprocess.run(
        ["tar", "-cf", str(tar_out), "-C", str(bundle_dir.parent), bundle_dir.name],
    ).returncode
    if rc != 0:
        raise RuntimeError("tar failed")
    print(f"[bundle] tar wall {time.time() - t0:.1f}s, "
          f"size {tar_out.stat().st_size / 1e9:.2f} GB")
    return {
        "bundle_dir": str(bundle_dir),
        "tar_path": str(tar_out),
        "tar_size": tar_out.stat().st_size,
        "files": files,
    }
