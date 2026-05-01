"""qairt-converter + qnn-context-binary-generator wrappers.

Both run as subprocess; we capture logs and inspect the output for
known failure modes (Cast op-config validation at v68, etc).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def env_with_qairt(qairt_root: Path, venv_root: Path) -> dict[str, str]:
    """PATH/LD_LIBRARY_PATH/PYTHONPATH set up so qairt-converter +
    qnn-context-binary-generator find their bundled libs and the
    venv's torch/numpy/onnx."""
    env = os.environ.copy()
    qbin = str(qairt_root / "bin" / "x86_64-linux-clang")
    qlib = str(qairt_root / "lib" / "x86_64-linux-clang")
    qpy = str(qairt_root / "lib" / "python")
    vbin = str(venv_root / "bin")
    env["PATH"] = f"{qbin}:{vbin}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{qlib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{qpy}:{env.get('PYTHONPATH', '')}"
    env["VIRTUAL_ENV"] = str(venv_root)
    return env


def run_qairt_converter(
    *,
    onnx_path: Path,
    encodings_path: Path,
    dlc_path: Path,
    qairt_root: Path,
    venv_root: Path,
    log_path: Path,
) -> dict:
    """qairt-converter → DLC. Returns {wall_s, log_path}.

    No --target_soc_model: the SDK 2.45 BackendInfo allow-list only
    accepts {SM8845, SM8850, SM8850L} which don't map to v75 anyway;
    HTP arch gets pinned at the binary-generator step instead.
    """
    env = env_with_qairt(qairt_root, venv_root)
    cmd = [
        "qairt-converter",
        "--input_network", str(onnx_path),
        "--output_path", str(dlc_path),
        "--quantization_overrides", str(encodings_path),
        "--preserve_io_datatype",
    ]
    t0 = time.time()
    print(f"[qairt-converter] {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0 or not dlc_path.exists():
        raise RuntimeError(
            f"qairt-converter failed (rc={rc}); see {log_path}"
        )
    print(f"[qairt-converter] DLC produced ({wall:.1f}s, {dlc_path.stat().st_size/1e6:.1f} MB)")
    return {"wall_s": wall, "log_path": str(log_path), "dlc_size": dlc_path.stat().st_size}


def run_qnn_context_binary_generator(
    *,
    dlc_path: Path,
    output_dir: Path,
    bin_prefix: str,
    qairt_root: Path,
    venv_root: Path,
    config_path: Path,
    log_path: Path,
) -> dict:
    """qnn-context-binary-generator → .bin.

    The config_path must be the OUTER backend_extensions wrapper that
    references an inner config setting devices[].dsp_arch="v75". A flat
    {"devices":[{"dsp_arch":"v75"}]} is rejected with
    `Unknown Key = devices/0/dsp_arch passed in config`.
    """
    env = env_with_qairt(qairt_root, venv_root)
    backend = qairt_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    cmd = [
        "qnn-context-binary-generator",
        "--backend", str(backend),
        "--dlc_path", str(dlc_path),
        "--binary_file", bin_prefix,
        "--output_dir", str(output_dir),
        "--config_file", str(config_path),
    ]
    t0 = time.time()
    print(f"[qnn-context-binary-generator] {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    bin_path = output_dir / f"{bin_prefix}.bin"
    if rc != 0 or not bin_path.exists():
        # Look for the recognizable arch-mismatch error in the log.
        try:
            tail = log_path.read_text()
        except Exception:
            tail = ""
        hint = ""
        if "Value 68, expected >= 73" in tail or "0xc26" in tail:
            hint = ("\nLikely cause: HTP arch mismatch. The default libQnnHtp.so backend "
                    "targets v68 but the encodings have ops needing >=v73. Make sure "
                    "qnn_v75_config.json points to qnn_v75_inner.json with dsp_arch=v75.")
        elif "Unknown Key = devices/0/dsp_arch" in tail:
            hint = ("\nLikely cause: dsp_arch needs the two-file backend_extensions wrap. "
                    "The OUTER config must have backend_extensions{shared_library_path,"
                    "config_file_path} and the INNER (referenced) file holds devices[].dsp_arch.")
        raise RuntimeError(f"qnn-context-binary-generator failed (rc={rc}); see {log_path}{hint}")
    print(f"[qnn-context-binary-generator] .bin produced ({wall:.1f}s, "
          f"{bin_path.stat().st_size/1e6:.1f} MB)")
    return {
        "wall_s": wall,
        "log_path": str(log_path),
        "bin_path": str(bin_path),
        "bin_size": bin_path.stat().st_size,
    }


def inspect_bin(*, bin_path: Path, qairt_root: Path, venv_root: Path,
                json_out: Path) -> dict:
    """Run qnn-context-binary-utility to extract the .bin's metadata
    (dspArch, graph IO shapes). Returns the parsed JSON."""
    env = env_with_qairt(qairt_root, venv_root)
    cmd = [
        "qnn-context-binary-utility",
        f"--context_binary={bin_path}",
        f"--json_file={json_out}",
        "--unified_qairt_format",
    ]
    rc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE).returncode
    if rc != 0 or not json_out.exists():
        raise RuntimeError(f"qnn-context-binary-utility failed (rc={rc})")
    import json
    return json.loads(json_out.read_text())
