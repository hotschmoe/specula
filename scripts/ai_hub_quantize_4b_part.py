"""Submit a single Qwen3-4B pathb split part (2/3/4) to Qualcomm AI Hub's
submit_compile_job with AIMET-grade w4a16 PTQ calibration.

Goal: AI Hub runs AIMET with Sequential MSE (per their llama3 tutorial) on
our halfdim pathb part ONNX + our 50 chat calibration samples. Returns a
compiled QNN context binary (.bin) ready to drop into our 4-part bundle.
If AI Hub's AIMET calibration produces a part bin with better per-token
agreement to Qualcomm's oracle than our qairt-quantizer w8 build, we
have the missing piece for structural + quality match.

First test: Part 2 only, with a REDUCED 10-sample calibration set (1
sample per position per 2 prompts, to cover the pos=0 BOS edge case
without hitting AI Hub upload-size limits or excessive cloud spend).

Prereqs:
    - .venv has qai_hub installed and `hub.get_devices()` works (auth set).
    - models/qwen3-4b-arm-pathb-ctx512-part{N}/model_halfdim.onnx exists.
    - models/calibration/qwen3_4b_ctx512_part{N}_raw/ has fresh halfdim raws.
    - models/calibration/qwen3_4b_ctx512_a.npz has matching sample metadata.

Run (dry-run validation — no upload, no compute):
    .venv/Scripts/python.exe scripts/ai_hub_quantize_4b_part.py --part 2 --check

Run (submit to cloud, poll, download .bin):
    .venv/Scripts/python.exe scripts/ai_hub_quantize_4b_part.py --part 2 --submit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
CALIB_ROOT = MODELS / "calibration"
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"

DEVICE = "Snapdragon X2 Elite CRD"
QAIRT_VERSION = "2.45"  # match our local toolchain

NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
PAST_LEN = 511
CTX = 512
HIDDEN = 2560
LAYERS_PER_PART = 12

EMBED_HIDDEN = "/model/embed_tokens/Gather_output_0"
L11_HIDDEN = "/model/layers.11/Add_1_output_0"
L23_HIDDEN = "/model/layers.23/Add_1_output_0"


def part_specs(part: int) -> tuple[str, dict[str, tuple[tuple, str]], Path]:
    """Returns (hidden_input_name, input_specs_dict, onnx_path)."""
    if part == 2:
        hidden_in = EMBED_HIDDEN
        layer_range = range(0, LAYERS_PER_PART)
    elif part == 3:
        hidden_in = L11_HIDDEN
        layer_range = range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)
    elif part == 4:
        hidden_in = L23_HIDDEN
        layer_range = range(2 * LAYERS_PER_PART, 3 * LAYERS_PER_PART)
    else:
        raise ValueError(f"part must be 2, 3, or 4 (part 1 is pure embed)")

    specs: dict[str, tuple[tuple, str]] = {
        hidden_in: ((1, 1, HIDDEN), "float32"),
        "attention_bias": ((1, 1, 1, CTX), "float32"),
        "position_ids_cos": ((1, 1, HALF_HEAD_DIM), "float32"),
        "position_ids_sin": ((1, 1, HALF_HEAD_DIM), "float32"),
    }
    for li in layer_range:
        specs[f"past_key_values.{li}.key"] = ((1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), "float32")
        specs[f"past_key_values.{li}.value"] = ((1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM), "float32")

    onnx_path = MODELS / f"qwen3-4b-arm-pathb-ctx512-part{part}" / "model_halfdim.onnx"
    return hidden_in, specs, onnx_path


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").lstrip("_")


def _raw_shape_dtype(spec: tuple) -> tuple[tuple, np.dtype]:
    shape, dtype_s = spec
    dtype = {"float32": np.float32, "int32": np.int32, "int64": np.int64}[dtype_s]
    return shape, dtype


def load_calibration_entries(
    raw_dir: Path,
    specs: dict[str, tuple[tuple, str]],
    sample_indices: list[int],
) -> dict[str, list[np.ndarray]]:
    """Load per-sample raw files for the given sample indices, return
    DatasetEntries dict in ONNX graph-input order."""
    entries: dict[str, list[np.ndarray]] = {n: [] for n in specs}
    for idx in sample_indices:
        sample_dir = raw_dir / f"sample_{idx:03d}"
        if not sample_dir.exists():
            raise FileNotFoundError(f"missing {sample_dir}")
        for name, spec in specs.items():
            shape, dtype = _raw_shape_dtype(spec)
            raw = sample_dir / f"{sanitize(name)}.raw"
            if not raw.exists():
                raise FileNotFoundError(f"missing raw {raw}")
            arr = np.fromfile(str(raw), dtype=dtype).reshape(shape)
            entries[name].append(arr)
    return entries


def pick_sample_indices(calib_npz: Path, n_samples: int) -> list[int]:
    """Pick a small diverse sample set. Default strategy: take one sample
    per (position × prompt) bucket, covering positions {0, 1, 5, 10, 20}.
    With 10 prompts × 5 positions in the npz, we have 50 total samples in
    order: [p0-pos0, p0-pos1, p0-pos5, p0-pos10, p0-pos20, p1-pos0, ...].
    For n_samples=10 we take 2 prompts (10 samples) covering all 5 positions."""
    data = np.load(str(calib_npz))
    n_total = data["input_ids"].shape[0]
    if n_samples >= n_total:
        return list(range(n_total))
    # Positions 0..4 are prompt 0's 5 positions, 5..9 are prompt 1's, etc.
    # For n_samples=10: take samples [0..9] = prompts 0 and 1 × 5 positions.
    return list(range(n_samples))


def build_compile_options(quant: str) -> str:
    return (
        "--target_runtime qnn_context_binary "
        "--compute_unit npu "
        "--truncate_64bit_io "
        f"--quantize_full_type {quant} "
        f"--qairt_version {QAIRT_VERSION}"
    )


def format_size(n_bytes: int) -> str:
    return f"{n_bytes / (1024**3):.2f} GB"


def check_mode(part: int, n_samples: int, calib_npz: Path) -> int:
    hidden_in, specs, onnx_path = part_specs(part)
    print("=== AI Hub 4B compile dry-run ===\n")
    print(f"part                : {part}")
    print(f"onnx                : {onnx_path}")
    print(f"hidden input        : {hidden_in}")
    print(f"input count         : {len(specs)}")

    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found. Run rewrite_halfdim_cos_sin.py first.")
        return 2
    # External data may be named model.data (pathb convention) or
    # model.onnx_data (onnx default). Find whichever exists.
    ext_data_candidates = [
        onnx_path.parent / "model.data",
        onnx_path.parent / "model.onnx_data",
    ]
    ext_data = next((p for p in ext_data_candidates if p.exists()), None)
    if ext_data is None:
        tried = [p.name for p in ext_data_candidates]
        print(f"ERROR: external-data file not found. Tried: {tried}")
        return 2

    onnx_mb = onnx_path.stat().st_size / (1024**2)
    ext_mb = ext_data.stat().st_size / (1024**2)
    print(f"onnx graph size     : {onnx_mb:.1f} MB  (protobuf)")
    print(f"onnx weights size   : {ext_mb:.1f} MB  (external-data)")
    print(f"upload size (model) : {(onnx_mb + ext_mb) / 1024:.2f} GB")

    raw_dir = CALIB_ROOT / f"qwen3_4b_ctx512_part{part}_raw"
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} missing")
        return 2
    sample_indices = pick_sample_indices(calib_npz, n_samples)
    print(f"calibration raw dir : {raw_dir}")
    print(f"calibration samples : {len(sample_indices)}  (indices {sample_indices})")

    # Measure calibration upload size.
    calib_bytes_per_sample = 0
    for name, spec in specs.items():
        shape, dtype = _raw_shape_dtype(spec)
        n_el = int(np.prod(shape))
        calib_bytes_per_sample += n_el * np.dtype(dtype).itemsize
    calib_total_mb = calib_bytes_per_sample * len(sample_indices) / (1024**2)
    print(f"calibration size    : {calib_total_mb:.1f} MB  "
          f"({calib_bytes_per_sample / (1024**2):.2f} MB × {len(sample_indices)})")

    print(f"total upload        : {(onnx_mb + ext_mb + calib_total_mb) / 1024:.2f} GB")

    compile_options = build_compile_options("w4a16")
    print(f"\ncompile options     : {compile_options}")

    print("\n--- AI Hub auth probe ---")
    import qai_hub as hub
    try:
        devices = [d for d in hub.get_devices() if d.name == DEVICE]
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: hub.get_devices() failed: {exc}")
        return 2
    if not devices:
        print(f"ERROR: device {DEVICE!r} not in catalogue")
        return 2
    print(f"device {DEVICE!r} available (os={devices[0].os})")

    print("\n--- calibration integrity probe ---")
    try:
        entries = load_calibration_entries(raw_dir, specs, sample_indices[:1])
        for name in list(entries)[:4]:
            arr = entries[name][0]
            print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  ... ({len(entries)} inputs total)")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR loading calibration: {exc}")
        return 2

    print("\n=== dry-run ok; pass --submit to actually run ===")
    return 0


def stage_upload_dir(onnx_path: Path, staging_dir: Path) -> None:
    """Prepare a clean dir with just `model.onnx` (renamed from our halfdim
    variant) + its `model.data` external-data file (hardlinked to avoid
    duplicating 4 GB on disk). AI Hub's upload_model resolves the external
    reference relative to the ONNX file's directory."""
    import os
    staging_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = staging_dir / "model.onnx"
    dst_data = staging_dir / "model.data"
    src_data = onnx_path.parent / "model.data"
    if not src_data.exists():
        raise FileNotFoundError(f"source data file {src_data} missing")
    if dst_onnx.exists():
        dst_onnx.unlink()
    # ONNX file is small (< 1 MB). Regular copy is fine.
    dst_onnx.write_bytes(onnx_path.read_bytes())
    if dst_data.exists():
        # Already staged; assume up to date.
        return
    try:
        os.link(str(src_data), str(dst_data))
    except OSError:
        # Fall back to copy (slow — ~4 GB).
        import shutil
        shutil.copy2(src_data, dst_data)


def submit_quantize_mode(part: int, n_samples: int, calib_npz: Path) -> int:
    """Submit a `submit_quantize_job` instead of `submit_compile_job`. Takes the
    fp32 ONNX + calibration, returns a quantized ONNX in QDQ format with AIMET
    encodings baked in. We then convert that QDQ ONNX to DLC locally with
    qairt-converter for compile + bin-gen via our existing pipeline.

    Phase 5q showed submit_compile_job's internal PTQ gives worse results than
    our qairt-quantizer w4+per-channel+CLE. submit_quantize_job is a different
    API — worth testing whether its quantization pipeline is AIMET-grade.
    """
    import qai_hub as hub

    hidden_in, specs, onnx_path = part_specs(part)
    raw_dir = CALIB_ROOT / f"qwen3_4b_ctx512_part{part}_raw"
    sample_indices = pick_sample_indices(calib_npz, n_samples)

    print(f"loading {len(sample_indices)} calibration samples from {raw_dir} ...")
    t0 = time.perf_counter()
    calibration_data = load_calibration_entries(raw_dir, specs, sample_indices)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    staging_dir = MODELS / f"staging-qwen3-4b-halfdim-part{part}-aihub"
    print(f"\nstaging at {staging_dir}")
    stage_upload_dir(onnx_path, staging_dir)
    upload_bytes = sum(p.stat().st_size for p in staging_dir.iterdir() if p.is_file())
    print(f"  {upload_bytes / (1024**3):.2f} GB")

    print(f"\nuploading directory to AI Hub ...")
    t0 = time.perf_counter()
    model_handle = hub.upload_model(str(staging_dir))
    print(f"  uploaded in {time.perf_counter() - t0:.1f}s  model_id={model_handle.model_id}")

    job_name = f"qwen3-4b-part{part}-halfdim-quantize-w4a16-aimet"
    print(f"\nsubmitting quantize job '{job_name}' (w=INT4, a=INT16) ...")
    job = hub.submit_quantize_job(
        model=model_handle,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT4,
        activations_dtype=hub.QuantizeDtype.INT16,
        name=job_name,
    )
    print(f"job submitted: id={job.job_id}  url={job.url}")

    print("\npolling ...")
    poll_secs = 20
    elapsed = 0
    while True:
        status = job.get_status()
        state = getattr(status, "code", str(status))
        print(f"  [{elapsed:5d}s] {state}")
        if state in ("SUCCESS", "FAILED", "RESULTS_READY"):
            break
        time.sleep(poll_secs)
        elapsed += poll_secs

    if state == "FAILED":
        print(f"\nFAILED. inspect at {job.url}")
        try:
            print(f"message: {status.message}")
        except Exception:
            pass
        return 1

    out_onnx_dir = RESULTS / f"aihub_quantize_part{part}_qdq"
    out_onnx_dir.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading QDQ ONNX to {out_onnx_dir} ...")
    target_model = job.get_target_model()
    target_model.download(str(out_onnx_dir))
    total_bytes = sum(p.stat().st_size for p in out_onnx_dir.rglob("*") if p.is_file())
    print(f"  downloaded {total_bytes / (1024**2):.1f} MB")
    for f in sorted(out_onnx_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out_onnx_dir)}: {f.stat().st_size / (1024**2):.1f} MB")
    print(f"\ntotal wall: {elapsed}s")
    print(f"=== ok ===")
    return 0


def submit_mode(part: int, n_samples: int, calib_npz: Path) -> int:
    import qai_hub as hub

    hidden_in, specs, onnx_path = part_specs(part)
    raw_dir = CALIB_ROOT / f"qwen3_4b_ctx512_part{part}_raw"
    sample_indices = pick_sample_indices(calib_npz, n_samples)

    print(f"loading {len(sample_indices)} calibration samples "
          f"({len(specs)} inputs each) from {raw_dir} ...")
    t0 = time.perf_counter()
    calibration_data = load_calibration_entries(raw_dir, specs, sample_indices)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # Stage to a clean dir so AI Hub's upload_model sees exactly one ONNX
    # + its external-data file.
    staging_dir = MODELS / f"staging-qwen3-4b-halfdim-part{part}-aihub"
    print(f"\nstaging upload dir at {staging_dir}")
    stage_upload_dir(onnx_path, staging_dir)
    staged_onnx = staging_dir / "model.onnx"
    upload_bytes = sum(p.stat().st_size for p in staging_dir.iterdir() if p.is_file())
    print(f"staged contents ({upload_bytes / (1024**3):.2f} GB):")
    for f in sorted(staging_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / (1024**2):.1f} MB")

    # AI Hub requires the DIRECTORY (contains model.onnx + model.data) not
    # the .onnx file alone — uploading just the protobuf fails the compile
    # job with "ONNX model is missing its external weights" after a ~2 min
    # OPTIMIZING_MODEL attempt. See job j563kqdn5 for the reference failure.
    print(f"\nuploading directory {staging_dir} ...")
    t0 = time.perf_counter()
    model_handle = hub.upload_model(str(staging_dir))
    print(f"  uploaded in {time.perf_counter() - t0:.1f}s  model_id={model_handle.model_id}")

    job_name = f"qwen3-4b-part{part}-halfdim-w4a16-aihub-aimet"
    print(f"\nsubmitting compile job '{job_name}' ...")
    job = hub.submit_compile_job(
        model=model_handle,
        name=job_name,
        device=hub.Device(DEVICE),
        input_specs=specs,
        options=build_compile_options("w4a16"),
        calibration_data=calibration_data,
    )
    print(f"job submitted: id={job.job_id}  url={job.url}")

    print("\npolling for completion ...")
    poll_secs = 15
    elapsed = 0
    while True:
        status = job.get_status()
        state = getattr(status, "code", str(status))
        print(f"  [{elapsed:5d}s] {state}")
        if state in ("SUCCESS", "FAILED", "RESULTS_READY"):
            break
        time.sleep(poll_secs)
        elapsed += poll_secs

    if state == "FAILED":
        print(f"\nFAILED. inspect at {job.url}")
        try:
            print(f"message: {status.message}")
        except Exception:
            pass
        return 1

    out_bin = RESULTS / f"qwen3_4b_4part_w4a16_aihub_part{part}.bin"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading to {out_bin} ...")
    target_model = job.get_target_model()
    target_model.download(str(out_bin))
    print(f"  downloaded {out_bin.stat().st_size / (1024**2):.1f} MB")
    print(f"\ntotal wall: {elapsed}s")
    print(f"=== ok ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Calibration sample count. Default 10 (2 prompts × 5 positions). "
                             "Max 50 (our npz size).")
    parser.add_argument("--calib-npz", type=Path,
                        default=CALIB_ROOT / "qwen3_4b_ctx512_a.npz")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--check", action="store_true", help="Dry run.")
    grp.add_argument("--submit", action="store_true",
                     help="submit_compile_job (cloud PTQ + compile -> .bin). Phase 5q.")
    grp.add_argument("--submit-quantize", action="store_true",
                     help="submit_quantize_job (cloud PTQ -> QDQ ONNX). Phase 5r.")
    args = parser.parse_args()

    if args.check:
        return check_mode(args.part, args.n_samples, args.calib_npz)
    if args.submit_quantize:
        return submit_quantize_mode(args.part, args.n_samples, args.calib_npz)
    return submit_mode(args.part, args.n_samples, args.calib_npz)


if __name__ == "__main__":
    sys.exit(main())
