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

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent

CONTEXT_MAX = 512
DEVICE = "Snapdragon X2 Elite CRD"
# Compile options learned across attempts 3-5 + session 9. Keep them
# comment-tagged so future sessions know what each flag buys:
#   --target_runtime qnn_context_binary  emit a preloaded HTP binary
#   --compute_unit npu                   Hexagon v81 only
#   --truncate_64bit_io                  attempts 4 error: int64 input
#                                        dtypes not accepted on HTP IO
#   --quantize_full_type float16         attempt 5 error: HTP Gather op
#                                        rejected FP32 dtype. Force the
#                                        whole graph to FP16 at compile.
#   --qairt_version 2.42                 session-9 finding: AI Hub
#                                        defaults to QAIRT 2.45, but the
#                                        only ORT-QNN version that loads
#                                        2.45 binaries on this hardware
#                                        (2.1.0) has loader bugs. Pin to
#                                        2.42 so the binary matches
#                                        onnxruntime-qnn 1.24.4's bundle.
#                                        See docs/npu_ort_qnn_version_match.md
COMPILE_OPTIONS = (
    "--target_runtime qnn_context_binary "
    "--compute_unit npu "
    "--truncate_64bit_io "
    "--quantize_full_type float16 "
    "--qairt_version 2.42"
)


# path_key -> config for this path's source + output naming.
PATHS: dict[str, dict] = {
    "patha": {
        "source_dir": REPO_ROOT / "models" / "qwen3-0.6b-patha",
        "staging_dir": REPO_ROOT / "models" / "qwen3-0.6b-patha-ai-hub",
        "output_bin": REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.patha.bin",
        "job_name": f"qwen3-0.6b-draft-v81-ctx{CONTEXT_MAX}-patha-fp16",
        "extra_input_specs": {},
    },
    "pathbmask": {
        "source_dir": REPO_ROOT / "models" / "qwen3-0.6b-pathbmask",
        "staging_dir": REPO_ROOT / "models" / "qwen3-0.6b-pathbmask-ai-hub",
        "output_bin": REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.pathbmask.bin",
        "job_name": f"qwen3-0.6b-draft-v81-ctx{CONTEXT_MAX}-pathbmask-fp16",
        # Path B-mask's additive attention bias. All-zeros at runtime for
        # the decode-only regime (past=511 + seq_q=1 = fully-valid window).
        "extra_input_specs": {
            "attention_bias": ((1, 1, 1, CONTEXT_MAX), "float32"),
        },
    },
}


def build_input_specs(cfg: dict, extra_input_specs: dict) -> dict:
    """Static shapes + dtypes for every ONNX input, keyed by input name.

    Follows the qai_hub convention: {name: ((shape...), "dtype_str")}.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])

    past_len = CONTEXT_MAX - 1  # decode step adds 1 new token to reach CONTEXT_MAX

    specs: dict[str, tuple] = {
        "input_ids": ((1, 1), "int64"),
        "position_ids": ((1, 1), "int64"),
    }
    specs.update(extra_input_specs)
    for i in range(n_layers):
        shape = (1, n_kv, past_len, head_dim)
        specs[f"past_key_values.{i}.key"] = (shape, "float32")
        specs[f"past_key_values.{i}.value"] = (shape, "float32")
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


def check_mode(path_cfg: dict) -> int:
    print("=== step 4 dry-run: AI Hub compile plan ===\n")

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

    specs = build_input_specs(cfg, path_cfg["extra_input_specs"])
    summarize_specs(specs)

    print("\n--- validating specs against ONNX header ---")
    issues = validate_specs_vs_onnx(specs, model_onnx)
    if issues:
        print("issues found:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("all onnx inputs have matching specs, dtypes align")

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
    print(f"      options='{COMPILE_OPTIONS}',")
    print(f"  )")
    print(f"  job.wait()")
    print(f"  job.get_target_model().download('{path_cfg['output_bin'].name}')")

    print("\n=== STATUS: dry-run ok; re-run with --submit to actually compile ===")
    return 0


def submit_mode(path_cfg: dict, reuse_upload: str | None = None) -> int:
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

    with config_json.open() as f:
        cfg = json.load(f)
    specs = build_input_specs(cfg, path_cfg["extra_input_specs"])

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

    print(f"\nsubmitting compile job '{job_name}' ...")
    job = hub.submit_compile_job(
        model=model_handle,
        name=job_name,
        device=hub.Device(DEVICE),
        input_specs=specs,
        options=COMPILE_OPTIONS,
    )
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
    parser.add_argument("--path", choices=sorted(PATHS), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate + print plan, no upload")
    group.add_argument("--submit", action="store_true", help="upload + compile + download")
    parser.add_argument(
        "--reuse-upload",
        metavar="MODEL_ID",
        default=None,
        help="skip upload, reuse an already-uploaded AI Hub model_id (e.g. mn1goxe4n)",
    )
    args = parser.parse_args()
    path_cfg = PATHS[args.path]
    try:
        if args.check:
            return check_mode(path_cfg)
        if args.submit:
            return submit_mode(path_cfg, reuse_upload=args.reuse_upload)
    except Exception:
        traceback.print_exc()
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
