"""Final bundle assembly: copy artifacts into a target dir, write
metadata.json with sha256s + provenance, tar for transport.

Two layouts supported:

1. Single-bin specula bundle (what the original e2e wrote): one .bin
   + AIMET encodings + tokenizer + metadata, intended for ORT-QNN
   sidecar consumption.

2. Multi-part Qualcomm-genie-compatible bundle: N .bin files +
   genie_config.json + htp_backend_ext_config.json + tokenizer +
   metadata + sample_prompt.txt, byte-for-byte structurally similar
   to Qualcomm's shipping `qwen3_4b-genie-w4a16-...` directory.
   This is what `genie-t2t-run` consumes on the X2E.
"""
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


def assemble_genie_bundle(
    *,
    bin_paths: list[Path],          # ordered list of part_1, part_2, ... .bin files
    bin_info_paths: list[Path],     # ordered list of bin_info.json per part (optional)
    encodings_paths: list[Path],    # ordered list of per-part AIMET encodings
    tokenizer_dir: Path,
    bundle_dir: Path,
    bundle_name: str,
    metadata: dict,
    tar_out: Path,
    model_info,                     # lib.model_config.ModelInfo
    ctx: int,
    dsp_arch: str = "v81",
    soc_model: int = 88,
) -> dict:
    """Assemble a Qualcomm-genie-compatible multi-part bundle, mirroring
    the layout of `qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`.

    Writes:
      <bundle_dir>/
        <prefix>_part_1_of_N.bin      (renamed from bin_paths[0])
        <prefix>_part_2_of_N.bin
        ...
        <prefix>_part_N_of_N.bin
        genie_config.json             (engine + tokenizer + ctx-bins)
        htp_backend_ext_config.json
        tokenizer.json
        tokenizer_config.json
        config.json                   (HF config)
        generation_config.json
        sample_prompt.txt
        metadata.json                 (specula provenance + sha256 of every file)
        encodings/                    (per-part AIMET encodings, for re-derivation)
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rename + copy the part bins.
    n_parts = len(bin_paths)
    bundle_bins: list[Path] = []
    for i, src in enumerate(bin_paths, start=1):
        dst_name = f"{bundle_name}_part_{i}_of_{n_parts}.bin"
        dst = bundle_dir / dst_name
        shutil.copy2(src, dst)
        bundle_bins.append(dst)

    # 2. Per-part AIMET encodings under encodings/
    enc_dir = bundle_dir / "encodings"
    enc_dir.mkdir(exist_ok=True)
    for i, src in enumerate(encodings_paths, start=1):
        shutil.copy2(src, enc_dir / f"part_{i}_of_{n_parts}.encodings")

    # 3. Per-part bin_info.json (optional; helps debugging IO shapes).
    bi_dir = bundle_dir / "bin_info"
    bi_dir.mkdir(exist_ok=True)
    for i, src in enumerate(bin_info_paths, start=1):
        if src and src.exists():
            shutil.copy2(src, bi_dir / f"part_{i}_of_{n_parts}.json")

    # 4. Tokenizer + HF configs.
    for n in ("tokenizer.json", "tokenizer_config.json", "config.json",
              "generation_config.json", "special_tokens_map.json",
              "vocab.json", "merges.txt"):
        src = tokenizer_dir / n
        if src.exists():
            shutil.copy2(src, bundle_dir / n)

    # 5. genie_config.json (mirrors Qualcomm's shape).
    bos_token = 151643
    eos_token = 151645
    # Try to read from generation_config.json if present.
    gc_path = tokenizer_dir / "generation_config.json"
    if gc_path.exists():
        gc = json.loads(gc_path.read_text())
        bos_token = int(gc.get("bos_token_id", bos_token))
        # Genie wants a single eos; HF often gives a list.
        eos_v = gc.get("eos_token_id", eos_token)
        if isinstance(eos_v, list) and eos_v:
            eos_token = int(eos_v[0])
        elif isinstance(eos_v, int):
            eos_token = eos_v

    genie_cfg = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": ctx,
                "n-vocab": int(model_info.vocab_size),
                "bos-token": bos_token,
                "eos-token": eos_token,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {
                "version": 1,
                "path": "tokenizer.json",
            },
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": int(model_info.head_dim),
                        "pos-id-dim": int(model_info.head_dim) // 2,
                        "allow-async-init": False,
                        "rope-theta": int(model_info.rope_theta),
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": [b.name for b in bundle_bins],
                    },
                },
            },
        }
    }
    (bundle_dir / "genie_config.json").write_text(json.dumps(genie_cfg, indent=4))

    # 6. htp_backend_ext_config.json (Qualcomm's shape).
    htp_cfg = {
        "devices": [{
            "soc_model": int(soc_model),
            "dsp_arch": dsp_arch,
            "cores": [{
                "core_id": 0,
                "perf_profile": "burst",
                "rpc_control_latency": 100,
            }],
        }],
        "memory": {"mem_type": "shared_buffer"},
        "context": {"weight_sharing_enabled": True},
    }
    (bundle_dir / "htp_backend_ext_config.json").write_text(
        json.dumps(htp_cfg)
    )

    # 7. sample_prompt.txt (matches Qualcomm's chat template).
    (bundle_dir / "sample_prompt.txt").write_text(
        "<|im_start|>system\nYou are a helpful AI assistant<|im_end|>\n"
        "<|im_start|>user\nWhat is gravity? Keep the answer under ten words.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 8. metadata.json (specula provenance + sha256 manifest).
    files: dict[str, dict] = {}
    for entry in sorted(bundle_dir.rglob("*")):
        if entry.is_file() and entry.name != "metadata.json":
            rel = str(entry.relative_to(bundle_dir))
            files[rel] = {
                "size": entry.stat().st_size,
                "sha256": _sha256(entry),
            }

    metadata = {
        **metadata,
        "bundle_name": bundle_name,
        "num_parts": n_parts,
        "dsp_arch": dsp_arch,
        "soc_model": soc_model,
        "ctx": ctx,
        "files": files,
        "total_bytes": sum(v["size"] for v in files.values()),
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # 9. tar.
    print(f"[bundle-genie] tar {tar_out} ...")
    t0 = time.time()
    rc = subprocess.run(
        ["tar", "-cf", str(tar_out), "-C", str(bundle_dir.parent), bundle_dir.name],
    ).returncode
    if rc != 0:
        raise RuntimeError("tar failed")
    print(f"[bundle-genie] tar wall {time.time() - t0:.1f}s, "
          f"size {tar_out.stat().st_size / 1e9:.2f} GB")

    return {
        "bundle_dir": str(bundle_dir),
        "tar_path": str(tar_out),
        "tar_size": tar_out.stat().st_size,
        "num_parts": n_parts,
        "total_bin_bytes": sum(b.stat().st_size for b in bundle_bins),
        "files": files,
    }
