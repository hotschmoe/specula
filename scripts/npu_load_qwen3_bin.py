"""Phase 5 step 5 - load compiled HTP context binary, shape-check, smoke.

Goal: prove a `.bin` produced by AI Hub loads on the X2E's Hexagon
v81 via ORT-QNN, the IO signature matches what was compiled for,
and a single forward pass completes without DMA/signing errors.
Correctness vs CPU is step 6; this is structural only.

The .bin AI Hub downloaded is a RAW QNN context binary (header bytes
`00 00 00 02 00 00 00 03 ...`), not an ONNX-EPContext wrapper. We
hand-build a tiny ONNX whose single node is an EPContext referencing
the .bin, then load that.

Two source variants supported (matches scripts/compile_qwen3_ai_hub.py):

    --path patha       58 inputs: input_ids + position_ids + 56 past_kv.
    --path pathbmask   59 inputs: above + attention_bias (FP32 additive).

ORT-QNN version dance (see docs/npu_ort_qnn_version_match.md):

  * 1.24.4 bundles QAIRT 2.42 + uses the LEGACY built-in EP path
    (`providers=[("QNNExecutionProvider", opts)]`). This is what we
    use here, because 2.1.0 (which bundles 2.45) has loader bugs on
    this driver that crash the interpreter with no traceback.
  * AI Hub must compile with `--qairt_version 2.42` so the binary
    matches what 1.24.4 can read. Without that flag, AI Hub defaults
    to 2.45 and `LoadCachedQnnContextFromBuffer` errors with code 5000.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_load_qwen3_bin.py --path patha
    .venv\\Scripts\\python.exe scripts\\npu_load_qwen3_bin.py --path pathbmask
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parent.parent

SOC_MODEL = "88"
HTP_ARCH = "81"
CONTEXT_MAX = 512

# Logits live on the first compiled output ('output_0'). Capture this so
# downstream analysis code stays robust to the qairt naming convention.
LOGITS_OUTPUT_NAME = "output_0"


def _bin_path(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}.bin"


def _wrapper_path(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}.wrapper.onnx"


def _config_json(path_key: str) -> Path:
    source_dir = REPO_ROOT / "models" / f"qwen3-0.6b-{path_key}"
    return source_dir / "config.json"


def describe_inputs(cfg: dict, path_key: str) -> list[tuple[str, list[int], int]]:
    """Return (name, shape, elem_type) for each input.

    Names + dtypes verified against the compile-time input_specs in
    scripts/compile_qwen3_ai_hub.py:
      * the qairt-converter normalises dotted names to underscored;
      * `--preserve_io_datatype` keeps past_key_values at FLOAT_32 even
        when `--quantize_full_type float16` is set for the interior;
      * `--truncate_64bit_io` casts INT64 IO to INT32.

    Path B-mask additionally has `attention_bias` (FP32, additive)
    as an input.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_len = CONTEXT_MAX - 1

    inputs: list[tuple[str, list[int], int]] = []
    inputs.append(("input_ids", [1, 1], TensorProto.INT32))
    inputs.append(("position_ids", [1, 1], TensorProto.INT32))
    for i in range(n_layers):
        inputs.append((f"past_key_values_{i}_key", [1, n_kv, past_len, head_dim], TensorProto.FLOAT))
        inputs.append((f"past_key_values_{i}_value", [1, n_kv, past_len, head_dim], TensorProto.FLOAT))
    # Order matters for the EPContext wrapper's declared IO to align with the
    # compiled binary. Path B-mask's ONNX has `attention_bias` as the last
    # graph input (after all past_kv entries) — see compile_qwen3_ai_hub's
    # build_input_specs comment.
    if path_key == "pathbmask":
        inputs.append(("attention_bias", [1, 1, 1, CONTEXT_MAX], TensorProto.FLOAT))
    return inputs


def describe_outputs(cfg: dict) -> list[tuple[str, list[int], int]]:
    """Return (name, shape, elem_type) per output.

    The qairt-converter renames all outputs to `output_0..output_N` in
    declaration order from the source ONNX:
      output_0  = logits           [1, 1, vocab]
      output_1  = present.0.key    [1, n_kv, ctx, head_dim]
      output_2  = present.0.value
      ...
      output_56 = present.27.value
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    vocab = cfg["vocab_size"]
    total_len = CONTEXT_MAX

    outputs: list[tuple[str, list[int], int]] = []
    outputs.append(("output_0", [1, 1, vocab], TensorProto.FLOAT))
    idx = 1
    for _ in range(n_layers):
        outputs.append((f"output_{idx}", [1, n_kv, total_len, head_dim], TensorProto.FLOAT))
        idx += 1
        outputs.append((f"output_{idx}", [1, n_kv, total_len, head_dim], TensorProto.FLOAT))
        idx += 1
    return outputs


def build_ep_context_wrapper(cfg: dict, bin_path: Path, out_path: Path, path_key: str) -> None:
    """Write a small ONNX whose single node is an EPContext referencing the .bin.

    The EPContext op is ORT-QNN's documented mechanism for wrapping a
    pre-compiled HTP context binary into an ONNX model so InferenceSession
    can ingest it. embed_mode=0 means the binary lives in a sidecar file
    next to the wrapper ONNX; ep_cache_context is the binary's filename
    relative to the wrapper.
    """
    inputs_decl = [
        helper.make_tensor_value_info(n, dt, shape)
        for n, shape, dt in describe_inputs(cfg, path_key)
    ]
    outputs_decl = [
        helper.make_tensor_value_info(n, dt, shape)
        for n, shape, dt in describe_outputs(cfg)
    ]

    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name="qnn_ctx",
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_path.name,
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}_{path_key}_wrapper",
        inputs=inputs_decl,
        outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-phase5",
    )
    model.ir_version = 10
    onnx.save(model, str(out_path))


def load_wrapper(wrapper_onnx: Path) -> ort.InferenceSession:
    """Load the EPContext-wrapper ONNX via legacy QNN provider (1.24.4)."""
    backend = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
    if not backend.exists():
        raise FileNotFoundError(f"QnnHtp.dll missing at {backend}")
    provider_options = {
        "backend_path": str(backend),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    print(f"  loading wrapper ONNX {wrapper_onnx.name} via legacy QNN EP ...")
    sess = ort.InferenceSession(
        str(wrapper_onnx),
        sess_options=sess_opts,
        providers=[("QNNExecutionProvider", provider_options)],
    )
    return sess


def summarize_io(sess: ort.InferenceSession) -> dict:
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  inputs ({len(inputs)}):")
    for x in inputs[:5]:
        print(f"    {x.name:38s} {str(x.shape):24s} {x.type}")
    if len(inputs) > 5:
        print(f"    ... ({len(inputs) - 5} more)")
    print(f"  outputs ({len(outputs)}):")
    for x in outputs[:3]:
        print(f"    {x.name:38s} {str(x.shape):24s} {x.type}")
    if len(outputs) > 3:
        print(f"    ... ({len(outputs) - 3} more)")
    return {"inputs": inputs, "outputs": outputs}


def build_zero_feed(sess: ort.InferenceSession) -> dict:
    """Build a feed of zeros matching the session's actual input spec."""
    dtype_map = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }
    feed = {}
    for x in sess.get_inputs():
        np_dtype = dtype_map.get(x.type)
        if np_dtype is None:
            raise RuntimeError(f"unknown onnx dtype {x.type} for input {x.name}")
        shape = tuple(d if isinstance(d, int) else 1 for d in x.shape)
        if x.name == "input_ids":
            feed[x.name] = np.ones(shape, dtype=np_dtype)
        elif x.name == "position_ids":
            feed[x.name] = np.full(shape, CONTEXT_MAX - 1, dtype=np_dtype)
        else:
            feed[x.name] = np.zeros(shape, dtype=np_dtype)
    return feed


def main() -> int:
    import functools
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask"), required=True)
    args = parser.parse_args()

    bin_path = _bin_path(args.path)
    wrapper_onnx = _wrapper_path(args.path)
    config_json = _config_json(args.path)

    print(f"=== step 5 - load compiled HTP context binary ({args.path}) ===\n")
    if not bin_path.exists():
        print(f"ERROR: {bin_path} missing")
        return 2
    if not config_json.exists():
        print(f"ERROR: {config_json} missing (need shapes for the EPContext wrapper)")
        return 2
    with config_json.open() as f:
        cfg = json.load(f)
    print(f"bin size            : {bin_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"model layers/kv_heads/head_dim : "
          f"{cfg['num_hidden_layers']}/{cfg['num_key_value_heads']}/"
          f"{cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])}")

    print("\n--- build EPContext wrapper ONNX ---")
    build_ep_context_wrapper(cfg, bin_path, wrapper_onnx, args.path)
    print(f"  wrote {wrapper_onnx.name} ({wrapper_onnx.stat().st_size} bytes)")

    print("\n--- load wrapper via legacy QNN EP (ORT 1.24.4) ---")
    try:
        sess = load_wrapper(wrapper_onnx)
    except Exception:
        print("\nload FAILED:")
        traceback.print_exc()
        return 2

    providers = sess.get_providers()
    print(f"\nsession providers   : {providers}")
    if not providers or providers[0] != "QNNExecutionProvider":
        print(f"ERROR: session silently fell back — providers={providers}")
        return 1

    print("\n--- IO signature ---")
    summarize_io(sess)

    print("\n--- single forward pass with zero KV + BOS-ish token ---")
    feed = build_zero_feed(sess)
    t0 = time.perf_counter_ns()
    outputs = sess.run(None, feed)
    t1 = time.perf_counter_ns()
    ms = (t1 - t0) / 1e6
    names = [o.name for o in sess.get_outputs()]
    logits_idx = names.index(LOGITS_OUTPUT_NAME)
    logits = outputs[logits_idx]
    print(f"  run latency       : {ms:.2f} ms")
    print(f"  logits shape      : {logits.shape}")
    print(f"  logits dtype      : {logits.dtype}")
    print(f"  logits finite frac: {float(np.isfinite(logits).mean()):.4f}")
    print(f"  logits min/max    : {float(np.nanmin(logits)):.3f} / {float(np.nanmax(logits)):.3f}")
    top5 = np.argsort(-logits[0, -1].astype(np.float32))[:5]
    print(f"  argmax top 5 ids  : {top5.tolist()}")

    healthy = (
        logits.shape[-1] == cfg["vocab_size"]
        and np.isfinite(logits).all()
    )
    print(f"\n=== STATUS: {'ok' if healthy else 'degenerate output'} ===")
    return 0 if healthy else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
