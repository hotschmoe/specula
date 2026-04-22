"""Phase 5 step 4 - AI Hub compile of Qwen3-0.6B ONNX to QNN v81 context binary.

Two-mode script:
    --check   validate auth, ONNX, input_specs, print the exact compile
              invocation that would be submitted. No upload, no compile,
              zero AI Hub compute burned.
    --submit  upload the model + submit a compile job + wait for result
              + download the .bin.

Two source variants supported (session 11 handoff from x86):

    --path patha       58 inputs (input_ids, position_ids, 56 past_kv).
                       attention_mask promoted to initializer; BOOL casts
                       folded out. BOOL tensors remain downstream.
                       Hypothesis: HTP rejects Cast-to-BOOL, not BOOL
                       tensors in general.
    --path pathbmask   59 inputs (input_ids, position_ids, attention_bias,
                       56 past_kv). Entire BOOL mask subgraph deleted;
                       additive FP32 bias spliced into 28 Add_2 nodes.
                       Zero BOOL tensors anywhere. Matches Qualcomm zoo
                       Qwen3-4B production pattern.

Shape decisions (both paths, CONTEXT_MAX=512, decode-only):
    input_ids               [1, 1]            int64
    position_ids            [1, 1]            int64
    attention_bias          [1, 1, 1, 512]    float32   (pathbmask only)
    past_key_values.N.key   [1, 8, 511, 128]  float32   (N = 0..27)
    past_key_values.N.value [1, 8, 511, 128]  float32

Prefill-on-CPU stays on the host; NPU only does single-token draft
steps. See docs/npu_scoping.md section 6.4.

Run:
    .venv\\Scripts\\python.exe scripts\\compile_qwen3_ai_hub.py --path patha --check
    .venv\\Scripts\\python.exe scripts\\compile_qwen3_ai_hub.py --path patha --submit
    .venv\\Scripts\\python.exe scripts\\compile_qwen3_ai_hub.py --path pathbmask --submit
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONTEXT_MAX = 512
DEVICE = "Snapdragon X2 Elite CRD"
# Compile options learned across attempts 3-5 + session 9. Keep them
# comment-tagged so future sessions know what each flag buys:
#   --target_runtime qnn_context_binary  emit a preloaded HTP binary
#   --compute_unit npu                   Hexagon v81 only
#   --truncate_64bit_io                  attempts 4 error: int64 input
#                                        dtypes not accepted on HTP IO
#   --quantize_full_type <T>             attempt 5 error: HTP Gather op
#                                        rejected FP32 dtype. Force the
#                                        whole graph to T at compile.
#                                        T in {float16, w4a16, w8a16}.
#                                        w4/w8 require calibration data.
#   --qairt_version 2.42                 session-9 finding: AI Hub
#                                        defaults to QAIRT 2.45, but the
#                                        only ORT-QNN version that loads
#                                        2.45 binaries on this hardware
#                                        (2.1.0) has loader bugs. Pin to
#                                        2.42 so the binary matches
#                                        onnxruntime-qnn 1.24.4's bundle.
#                                        See docs/npu_ort_qnn_version_match.md
QUANT_CHOICES = ("float16", "w4a16", "w8a16")

QAIRT_VERSION = "2.42"


def build_compile_options(quant: str) -> str:
    if quant not in QUANT_CHOICES:
        raise ValueError(f"unknown quant {quant!r}; expected one of {QUANT_CHOICES}")
    return (
        "--target_runtime qnn_context_binary "
        "--compute_unit npu "
        "--truncate_64bit_io "
        f"--quantize_full_type {quant} "
        f"--qairt_version {QAIRT_VERSION}"
    )


# Legacy alias. Phase 5 baseline compiles used the fp16 string directly.
COMPILE_OPTIONS = build_compile_options("float16")


def build_paths(path_key: str, ctx: int, quant_tag: str = "") -> dict:
    """Return the ctx-specific source / staging / output paths for a path_key.

    Source dir is ctx-independent (the unpinned ONNX is shared between
    ctx tiers — `prep_onnx_for_ai_hub.py` pins dims per-ctx downstream).
    Staging dir, output bin, and job name are ctx-specific so parallel
    tiers can coexist on disk (e.g. ctx=512 baseline + ctx=256 Lever B).

    `quant_tag` disambiguates output filenames when multiple quant flavors
    share the same ctx + path (e.g. 'w4a16-a' vs 'w4a16-b' for Lever C's
    two calibration bundles). Empty tag preserves the Phase-5/5.5-Lever-B
    naming pattern for the fp16 baseline binaries.
    """
    if path_key not in ("patha", "pathbmask"):
        raise ValueError(f"unknown path_key {path_key!r}")
    ctx_suffix = f"ctx{ctx}"
    # ctx=512 keeps legacy naming to preserve existing artifacts + docs.
    staging_suffix = "-ai-hub" if ctx == DEFAULT_CONTEXT_MAX else f"-ai-hub-{ctx_suffix}"
    extra_input_specs: dict = {}
    if path_key == "pathbmask":
        extra_input_specs = {
            "attention_bias": ((1, 1, 1, ctx), "float32"),
        }
    tag_suffix = f".{quant_tag}" if quant_tag else ""
    job_quant_suffix = f"-{quant_tag}" if quant_tag else "-fp16"
    return {
        "source_dir": REPO_ROOT / "models" / f"qwen3-0.6b-{path_key}",
        "staging_dir": REPO_ROOT / "models" / f"qwen3-0.6b-{path_key}{staging_suffix}",
        "output_bin": REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_{ctx_suffix}.{path_key}{tag_suffix}.bin",
        "job_name": f"qwen3-0.6b-draft-v81-{ctx_suffix}-{path_key}{job_quant_suffix}",
        "extra_input_specs": extra_input_specs,
        "ctx": ctx,
    }


# Legacy alias: module-level PATHS stays for backward compatibility with
# any caller that still imports it. Points at the default (ctx=512) tier.
PATHS: dict[str, dict] = {k: build_paths(k, DEFAULT_CONTEXT_MAX) for k in ("patha", "pathbmask")}


def build_input_specs(cfg: dict, extra_input_specs: dict, ctx: int = DEFAULT_CONTEXT_MAX) -> dict:
    """Static shapes + dtypes for every ONNX input, keyed by input name.

    Follows the qai_hub convention: {name: ((shape...), "dtype_str")}.

    Order matters: AI Hub validates that its iteration order over
    input_specs matches the order of graph.input in the uploaded ONNX.
    For pathbmask, the x86-side rewrite appends `attention_bias` as the
    LAST graph input (after 56 past_kv entries), so we have to append
    `extra_input_specs` at the end, not insert it near the top.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])

    past_len = ctx - 1  # decode step adds 1 new token to reach ctx

    specs: dict[str, tuple] = {
        "input_ids": ((1, 1), "int64"),
        "position_ids": ((1, 1), "int64"),
    }
    for i in range(n_layers):
        shape = (1, n_kv, past_len, head_dim)
        specs[f"past_key_values.{i}.key"] = (shape, "float32")
        specs[f"past_key_values.{i}.value"] = (shape, "float32")
    specs.update(extra_input_specs)
    return specs


def validate_specs_vs_onnx(specs: dict, model_onnx: Path) -> list:
    """Load the ONNX header, compare input names to specs. Return any issues."""
    sess = ort.InferenceSession(str(model_onnx), providers=["CPUExecutionProvider"])
    onnx_inputs = {x.name: x for x in sess.get_inputs()}
    issues = []
    spec_names = set(specs.keys())
    onnx_names = set(onnx_inputs.keys())
    missing_from_spec = onnx_names - spec_names
    extra_in_spec = spec_names - onnx_names
    if missing_from_spec:
        issues.append(f"onnx inputs missing from specs: {sorted(missing_from_spec)[:3]} ... "
                      f"({len(missing_from_spec)} total)")
    if extra_in_spec:
        issues.append(f"specs names not in onnx: {sorted(extra_in_spec)[:3]} ... "
                      f"({len(extra_in_spec)} total)")

    dtype_map = {
        "tensor(int64)": "int64",
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
    }
    for name in spec_names & onnx_names:
        onnx_type = onnx_inputs[name].type
        spec_dtype = specs[name][1]
        if dtype_map.get(onnx_type) != spec_dtype:
            issues.append(f"dtype mismatch on '{name}': onnx={onnx_type} spec={spec_dtype}")
    return issues


def summarize_specs(specs: dict) -> None:
    """Compact summary: first few, then KV cache envelope."""
    print(f"\n--- input_specs summary ({len(specs)} entries) ---")
    for key in list(specs.keys()):
        if key.startswith("past_key_values."):
            continue
        shape, dtype = specs[key]
        print(f"  {key:40s} {shape}  {dtype}")

    kv_keys = [k for k in specs if k.startswith("past_key_values.")]
    if kv_keys:
        first = kv_keys[0]
        shape, dtype = specs[first]
        n_layers = len({k.split(".")[1] for k in kv_keys})
        per_tensor_bytes = int(shape[0] * shape[1] * shape[2] * shape[3]) * 4
        total_kv_mb = per_tensor_bytes * len(kv_keys) / (1024 * 1024)
        print(f"  past_key_values.*.{{key,value}}         {shape}  {dtype}")
        print(f"    layers={n_layers}, entries={len(kv_keys)}, total past-KV IO = {total_kv_mb:.1f} MB/call")


def validate_calibration_npz(
    npz_path: Path,
    specs: dict,
) -> tuple[int, list[str]]:
    """Load an npz saved by capture_calibration_samples.py and check its
    schema matches compile input_specs. Returns (n_samples, issues).
    """
    if not npz_path.exists():
        return 0, [f"calibration npz not found at {npz_path}"]

    issues: list[str] = []
    loaded = np.load(str(npz_path))
    npz_keys = set(loaded.files)
    spec_keys = set(specs.keys())

    missing = spec_keys - npz_keys
    extra = npz_keys - spec_keys
    if missing:
        issues.append(
            f"npz missing inputs required by specs: {sorted(missing)[:3]} "
            f"({len(missing)} total)"
        )
    if extra:
        issues.append(
            f"npz has inputs not in specs: {sorted(extra)[:3]} "
            f"({len(extra)} total)"
        )

    n_samples_set: set[int] = set()
    dtype_map = {"int64": "int64", "float32": "float32", "float16": "float16"}
    for key in spec_keys & npz_keys:
        arr = loaded[key]
        n_samples_set.add(arr.shape[0])
        # Per-sample shape = arr.shape[1:], compare with spec shape.
        spec_shape, spec_dtype = specs[key]
        if tuple(arr.shape[1:]) != tuple(spec_shape):
            issues.append(
                f"shape mismatch '{key}': npz per-sample={arr.shape[1:]} "
                f"spec={spec_shape}"
            )
        if dtype_map.get(str(arr.dtype), str(arr.dtype)) != spec_dtype:
            issues.append(
                f"dtype mismatch '{key}': npz={arr.dtype} spec={spec_dtype}"
            )

    if len(n_samples_set) > 1:
        issues.append(f"inconsistent n_samples across inputs: {sorted(n_samples_set)}")

    n_samples = n_samples_set.pop() if n_samples_set else 0
    return n_samples, issues


def check_mode(
    path_cfg: dict,
    quant: str,
    calibration_npz: Path | None = None,
    calibration_dataset_id: str | None = None,
) -> int:
    print("=== AI Hub compile dry-run ===\n")
    print(f"quant               : {quant}")

    source_dir = path_cfg["source_dir"]
    staging_dir = path_cfg["staging_dir"]
    model_onnx = staging_dir / "model.onnx"
    model_data = staging_dir / "model.data"
    config_json = source_dir / "config.json"

    if not model_onnx.exists():
        print(f"ERROR: staged ONNX not found at {model_onnx}")
        print("  run scripts/prep_onnx_for_ai_hub.py --path ... first")
        return 2
    if not model_data.exists():
        print(f"ERROR: staged external-data missing at {model_data}")
        return 2

    model_mb = model_onnx.stat().st_size / (1024 * 1024)
    data_mb = model_data.stat().st_size / (1024 * 1024)
    print(f"staged ONNX graph   : {model_onnx} ({model_mb:.1f} MB)")
    print(f"staged ONNX weights : {model_data.name} ({data_mb:.1f} MB)")
    print(f"total upload size   : {(model_mb + data_mb) / 1024:.2f} GB")

    with config_json.open() as f:
        cfg = json.load(f)
    print(f"model_type          : {cfg.get('model_type')}")
    print(f"num_hidden_layers   : {cfg['num_hidden_layers']}")

    specs = build_input_specs(cfg, path_cfg["extra_input_specs"], ctx=path_cfg["ctx"])
    summarize_specs(specs)

    print("\n--- validating specs against ONNX header ---")
    issues = validate_specs_vs_onnx(specs, model_onnx)
    if issues:
        print("issues found:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("all onnx inputs have matching specs, dtypes align")

    compile_options = build_compile_options(quant)

    if quant != "float16":
        print(f"\n--- validating calibration data ({quant} requires PTQ) ---")
        if calibration_npz is None and calibration_dataset_id is None:
            print("ERROR: --quant w4a16/w8a16 requires --calibration-npz "
                  "or --calibration-dataset-id")
            return 2
        if calibration_npz is not None:
            n_samples, cal_issues = validate_calibration_npz(calibration_npz, specs)
            if cal_issues:
                print("calibration npz issues:")
                for i in cal_issues:
                    print(f"  - {i}")
                return 1
            npz_gb = calibration_npz.stat().st_size / (1024**3)
            print(f"  npz path         : {calibration_npz}")
            print(f"  npz size         : {npz_gb:.2f} GB")
            print(f"  n_samples        : {n_samples}")
            print(f"  schema match     : ok (all {len(specs)} inputs present, "
                  f"shapes + dtypes align)")
        if calibration_dataset_id is not None:
            print(f"  dataset_id       : {calibration_dataset_id}")

    print("\n--- AI Hub auth probe ---")
    import qai_hub as hub

    try:
        device_match = [d for d in hub.get_devices() if d.name == DEVICE]
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: hub.get_devices() failed: {exc}")
        return 2
    if not device_match:
        print(f"ERROR: device '{DEVICE}' not in AI Hub catalogue")
        return 2
    print(f"device '{DEVICE}' available (os={device_match[0].os})")

    print("\n--- compile invocation that would run on --submit ---")
    print(f"  hub.upload_model(path={model_onnx.parent})")
    print(f"  hub.submit_compile_job(")
    print(f"      model=<uploaded>,")
    print(f"      name='{path_cfg['job_name']}',")
    print(f"      device=hub.Device('{DEVICE}'),")
    print(f"      input_specs=<{len(specs)} entries>,")
    print(f"      options='{compile_options}',")
    if quant != "float16":
        src = calibration_dataset_id or f"<DatasetEntries from {calibration_npz.name}>"
        print(f"      calibration_data={src},")
    print(f"  )")
    print(f"  job.wait()")
    print(f"  job.get_target_model().download('{path_cfg['output_bin'].name}')")

    print("\n=== STATUS: dry-run ok; re-run with --submit to actually compile ===")
    return 0


def _load_calibration_entries(
    npz_path: Path,
    specs: dict,
) -> dict[str, list[np.ndarray]]:
    """Read a calibration .npz (written by capture_calibration_samples.py),
    return a DatasetEntries dict in ONNX-graph-input order.

    AI Hub's PTQ validator iterates calibration_data positionally and
    checks each slot's name against the graph's input order. If order
    drifts — e.g. attention_bias appears before past_key_values.0.key —
    compile fails with "Calibration data set has input 'attention_bias'
    but expected 'past_key_values.0.key'." Rebuilding the dict keyed by
    `specs` (which is already in the right order via build_input_specs)
    locks correctness regardless of how the npz was saved.
    """
    loaded = np.load(str(npz_path))
    entries: dict[str, list[np.ndarray]] = {}
    for key in specs:
        if key not in loaded.files:
            raise ValueError(f"calibration npz missing required input '{key}'")
        arr = loaded[key]
        entries[key] = [arr[i] for i in range(arr.shape[0])]
    return entries


def submit_mode(
    path_cfg: dict,
    quant: str,
    reuse_upload: str | None = None,
    calibration_npz: Path | None = None,
    calibration_dataset_id: str | None = None,
) -> int:
    import qai_hub as hub

    source_dir = path_cfg["source_dir"]
    staging_dir = path_cfg["staging_dir"]
    model_onnx = staging_dir / "model.onnx"
    config_json = source_dir / "config.json"
    output_bin = path_cfg["output_bin"]
    job_name = path_cfg["job_name"]

    if not model_onnx.exists():
        print(f"ERROR: ONNX not found at {model_onnx}")
        return 2

    if quant != "float16" and calibration_npz is None and calibration_dataset_id is None:
        print(f"ERROR: quant={quant} requires --calibration-npz or --calibration-dataset-id")
        return 2

    with config_json.open() as f:
        cfg = json.load(f)
    specs = build_input_specs(cfg, path_cfg["extra_input_specs"], ctx=path_cfg["ctx"])
    compile_options = build_compile_options(quant)

    calibration_data = None
    if calibration_dataset_id is not None:
        print(f"reusing AI Hub dataset_id={calibration_dataset_id}")
        calibration_data = calibration_dataset_id
    elif calibration_npz is not None:
        npz_gb = calibration_npz.stat().st_size / (1024**3)
        print(f"loading calibration npz {calibration_npz} ({npz_gb:.2f} GB) ...")
        t0 = time.perf_counter()
        calibration_data = _load_calibration_entries(calibration_npz, specs)
        n_samples = len(next(iter(calibration_data.values())))
        print(f"  loaded {n_samples} samples × {len(calibration_data)} inputs "
              f"in {time.perf_counter() - t0:.1f} s (reordered to match "
              f"ONNX graph input order)")

    if reuse_upload:
        print(f"reusing uploaded model_id={reuse_upload} (skipping ~15 min upload)")
        model_handle = hub.get_model(reuse_upload)
    else:
        upload_path = model_onnx.parent
        upload_bytes = sum(p.stat().st_size for p in upload_path.iterdir() if p.is_file())
        print(f"uploading {upload_path} ({upload_bytes / (1024**3):.2f} GB) to AI Hub ...")
        t0 = time.perf_counter()
        model_handle = hub.upload_model(str(upload_path))
        t_upload = time.perf_counter() - t0
        print(f"uploaded in {t_upload:.1f} s, model_id={model_handle.model_id}")

    print(f"\nsubmitting compile job '{job_name}' (quant={quant}) ...")
    submit_kwargs = dict(
        model=model_handle,
        name=job_name,
        device=hub.Device(DEVICE),
        input_specs=specs,
        options=compile_options,
    )
    if calibration_data is not None:
        submit_kwargs["calibration_data"] = calibration_data
    job = hub.submit_compile_job(**submit_kwargs)
    print(f"job submitted: id={job.job_id} url={job.url}")

    print("\npolling for completion ...")
    poll_secs = 10
    elapsed = 0
    while True:
        status = job.get_status()
        state = getattr(status, "code", str(status))
        print(f"  [{elapsed:4d}s] {state}")
        if state in ("SUCCESS", "FAILED", "RESULTS_READY"):
            break
        time.sleep(poll_secs)
        elapsed += poll_secs

    if state == "FAILED":
        print(f"\nCOMPILE FAILED.")
        try:
            print(f"failure reason: {status.message}")
        except Exception:
            pass
        print(f"inspect at     : {job.url}")
        return 1

    print(f"\ncompile succeeded in ~{elapsed}s total")
    target_model = job.get_target_model()
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading to : {output_bin}")
    target_model.download(str(output_bin))
    print(f"downloaded {output_bin.stat().st_size / (1024*1024):.1f} MB")
    print(f"\n=== STATUS: ok ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask"), required=True)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT_MAX,
                        help=f"compiled context window size (default: {DEFAULT_CONTEXT_MAX})")
    parser.add_argument(
        "--quant", choices=QUANT_CHOICES, default="float16",
        help="quantization mode. w4a16/w8a16 require --calibration-npz or "
             "--calibration-dataset-id. Default: float16 (no calibration).",
    )
    parser.add_argument(
        "--quant-tag", default=None,
        help="override output-binary suffix (e.g. 'w4a16-a'). Default: "
             "'' for float16, otherwise matches --quant value.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate + print plan, no upload")
    group.add_argument("--submit", action="store_true", help="upload + compile + download")
    parser.add_argument(
        "--reuse-upload",
        metavar="MODEL_ID",
        default=None,
        help="skip upload, reuse an already-uploaded AI Hub model_id (e.g. mn1goxe4n)",
    )
    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument(
        "--calibration-npz", type=Path, default=None,
        help="path to .npz from capture_calibration_samples.py",
    )
    cal_group.add_argument(
        "--calibration-dataset-id", default=None,
        help="reuse a pre-uploaded AI Hub dataset_id instead of the local npz",
    )
    args = parser.parse_args()

    if args.quant_tag is not None:
        quant_tag = args.quant_tag
    elif args.quant == "float16":
        quant_tag = ""
    else:
        quant_tag = args.quant

    path_cfg = build_paths(args.path, args.ctx, quant_tag=quant_tag)
    try:
        if args.check:
            return check_mode(
                path_cfg,
                quant=args.quant,
                calibration_npz=args.calibration_npz,
                calibration_dataset_id=args.calibration_dataset_id,
            )
        if args.submit:
            return submit_mode(
                path_cfg,
                quant=args.quant,
                reuse_upload=args.reuse_upload,
                calibration_npz=args.calibration_npz,
                calibration_dataset_id=args.calibration_dataset_id,
            )
    except Exception:
        traceback.print_exc()
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
