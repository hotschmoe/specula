"""Phase 5 step 4 - AI Hub compile of Qwen3-0.6B ONNX to QNN v81 context binary.

Two-mode script:
    --check   validate auth, ONNX, input_specs, print the exact compile
              invocation that would be submitted. No upload, no compile,
              zero AI Hub compute burned.
    --submit  upload the model + submit a compile job + wait for result
              + download the .bin.

Shape decisions (first-cut, CONTEXT_MAX=512, decode-only):
    input_ids               [1, 1]            int64
    attention_mask          [1, 512]          int64
    position_ids            [1, 1]            int64
    past_key_values.N.key   [1, 8, 511, 128]  float32     (N = 0..27)
    past_key_values.N.value [1, 8, 511, 128]  float32

Prefill-on-CPU stays on the host; NPU only does single-token draft
steps. See docs/npu_scoping.md section 6.4.

Run:
    .venv\\Scripts\\python.exe scripts\\compile_qwen3_ai_hub.py --check
    .venv\\Scripts\\python.exe scripts\\compile_qwen3_ai_hub.py --submit
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent
# Source = the optimum export produced on the x86 machine (all ops in
# default onnx domain, opset 18). config.json co-ships with the ONNX.
SOURCE_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
CONFIG_JSON = SOURCE_DIR / "config.json"
# AI Hub upload targets the staging dir produced by prep_onnx_for_ai_hub.py
# (qai-hub requires .onnx/.data extensions; optimum ships .onnx_data).
STAGING_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum-ai-hub"
MODEL_ONNX = STAGING_DIR / "model.onnx"
MODEL_DATA = STAGING_DIR / "model.data"

CONTEXT_MAX = 512
DEVICE = "Snapdragon X2 Elite CRD"
JOB_NAME = f"qwen3-0.6b-draft-v81-ctx{CONTEXT_MAX}-fp16"
COMPILE_OPTIONS = "--target_runtime qnn_context_binary --compute_unit npu"
OUTPUT_BIN = REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.bin"


def build_input_specs(cfg: dict) -> dict:
    """Static shapes + dtypes for every ONNX input, keyed by input name.

    Follows the qai_hub convention: {name: ((shape...), "dtype_str")}.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])

    past_len = CONTEXT_MAX - 1  # decode step adds 1 new token to reach CONTEXT_MAX
    total_len = CONTEXT_MAX

    specs: dict[str, tuple] = {
        "input_ids": ((1, 1), "int64"),
        "attention_mask": ((1, total_len), "int64"),
        "position_ids": ((1, 1), "int64"),
    }
    for i in range(n_layers):
        shape = (1, n_kv, past_len, head_dim)
        specs[f"past_key_values.{i}.key"] = (shape, "float32")
        specs[f"past_key_values.{i}.value"] = (shape, "float32")
    return specs


def validate_specs_vs_onnx(specs: dict) -> list:
    """Load the ONNX header, compare input names to specs. Return any issues."""
    sess = ort.InferenceSession(
        str(MODEL_ONNX),
        providers=["CPUExecutionProvider"],
    )
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

    # Dtype alignment for overlapping names.
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
    for key in ("input_ids", "attention_mask", "position_ids"):
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


def check_mode() -> int:
    print("=== step 4 dry-run: AI Hub compile plan ===\n")

    if not MODEL_ONNX.exists():
        print(f"ERROR: staged ONNX not found at {MODEL_ONNX}")
        print("  run scripts/prep_onnx_for_ai_hub.py first")
        return 2
    if not MODEL_DATA.exists():
        print(f"ERROR: staged external-data missing at {MODEL_DATA}")
        return 2

    model_mb = MODEL_ONNX.stat().st_size / (1024 * 1024)
    data_mb = MODEL_DATA.stat().st_size / (1024 * 1024)
    print(f"staged ONNX graph   : {MODEL_ONNX} ({model_mb:.1f} MB)")
    print(f"staged ONNX weights : {MODEL_DATA.name} ({data_mb:.1f} MB)")
    print(f"total upload size   : {(model_mb + data_mb) / 1024:.2f} GB")

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    print(f"model_type          : {cfg.get('model_type')}")
    print(f"num_hidden_layers   : {cfg['num_hidden_layers']}")

    specs = build_input_specs(cfg)
    summarize_specs(specs)

    print("\n--- validating specs against ONNX header ---")
    issues = validate_specs_vs_onnx(specs)
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
    print(f"  hub.upload_model(path={MODEL_ONNX})")
    print(f"  hub.submit_compile_job(")
    print(f"      model=<uploaded>,")
    print(f"      name='{JOB_NAME}',")
    print(f"      device=hub.Device('{DEVICE}'),")
    print(f"      input_specs=<{len(specs)} entries>,")
    print(f"      options='{COMPILE_OPTIONS}',")
    print(f"  )")
    print(f"  job.wait()")
    print(f"  job.get_target_model().download('{OUTPUT_BIN.name}')")

    print("\n--- expected outcomes (first attempt) ---")
    print("  compile time   : ~5-20 min for a 28-layer 0.6B model at ctx 512")
    print("  binary size    : ~500-800 MB for FP16 HTP context binary")
    print("  failure modes  : QAIRT converter op-lowering (scoping doc 3.4),")
    print("                   input-name mismatch, dynamic-dim leftover")

    print("\n=== STATUS: dry-run ok; re-run with --submit to actually compile ===")
    return 0


def submit_mode() -> int:
    import qai_hub as hub

    if not MODEL_ONNX.exists():
        print(f"ERROR: ONNX not found at {MODEL_ONNX}")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    specs = build_input_specs(cfg)

    # ONNX with external data: upload the DIRECTORY, not the single .onnx
    # file. qai-hub picks up both model.onnx and model.onnx_data this way.
    # The earlier --submit attempt failed with "missing its external
    # weights" when we passed only the .onnx path.
    upload_path = MODEL_ONNX.parent
    upload_bytes = sum(p.stat().st_size for p in upload_path.iterdir() if p.is_file())
    print(f"uploading {upload_path} ({upload_bytes / (1024**3):.2f} GB) to AI Hub ...")
    t0 = time.perf_counter()
    model_handle = hub.upload_model(str(upload_path))
    t_upload = time.perf_counter() - t0
    print(f"uploaded in {t_upload:.1f} s, model_id={model_handle.model_id}")

    print(f"\nsubmitting compile job '{JOB_NAME}' ...")
    job = hub.submit_compile_job(
        model=model_handle,
        name=JOB_NAME,
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
    OUTPUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading to : {OUTPUT_BIN}")
    target_model.download(str(OUTPUT_BIN))
    print(f"downloaded {OUTPUT_BIN.stat().st_size / (1024*1024):.1f} MB")
    print(f"\n=== STATUS: ok ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate + print plan, no upload")
    group.add_argument("--submit", action="store_true", help="upload + compile + download")
    args = parser.parse_args()
    try:
        if args.check:
            return check_mode()
        if args.submit:
            return submit_mode()
    except Exception:
        traceback.print_exc()
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
