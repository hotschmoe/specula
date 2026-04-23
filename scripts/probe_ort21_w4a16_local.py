"""Isolated probe: can ORT-QNN 2.1.0 load the x86's QAIRT 2.45 w4a16 binary?

Background: ORT-QNN 1.24.4 (QAIRT 2.42) rejected this binary with
`LoadCachedQnnContextFromBuffer Error 5000` (see
results/preflight_w4a16_local_load.log). 2.1.0 bundles QAIRT 2.45.40.260406,
which matches the x86 compile SDK exactly. Historical caveat in
docs/npu_ort_qnn_version_match.md: 2.1.0's EPContext loader was
documented to hit Code 1000 (file-mapping) on one prior AI-Hub-compiled
binary and crash silently on the retry path.

Strings inside onnxruntime_providers_qnn.dll suggest two escape hatches:

  * `disabl_file_mapped_weights=1`  (sic — the DLL literal is misspelled)
    — skip the buggy file-mapping codepath up front, load the binary
    through the standard buffer API instead.
  * `embed_mode=1` — inline the binary bytes into the wrapper ONNX
    itself. The DLL logs "File mapped weights feature is incompatible
    with embedded EP contexts. Feature will be disabled by default" —
    so embed_mode=1 bypasses file mapping entirely.

Runs the attempts in escalating invasiveness order:

    1. legacy providers=[...]                     (known silent fallback, sanity-only)
    2. plugin-EP + disabl_file_mapped_weights=1   (cheap escape)
    3. plugin-EP + embed_mode=1 wrapper           (heavier — rebuilds wrapper with inline bin)

Run:
    .venv-ort21\\Scripts\\python.exe scripts\\probe_ort21_w4a16_local.py
"""

import argparse
import functools
import sys
import time
import traceback
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 — unbuffered so crashes don't eat progress logs

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER_PATH = REPO_ROOT / "models" / "qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.wrapper.onnx"
WRAPPER_EMBED_PATH = REPO_ROOT / "models" / "qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.wrapper.embed.onnx"
BIN_PATH = REPO_ROOT / "models" / "qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin"

SOC_MODEL = "88"
HTP_ARCH = "81"


def _build_zero_feed(sess: ort.InferenceSession) -> dict:
    dtype_map = {
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
    }
    feed = {}
    for x in sess.get_inputs():
        shape = tuple(d if isinstance(d, int) else 1 for d in x.shape)
        np_dtype = dtype_map[x.type]
        if x.name == "input_ids":
            feed[x.name] = np.ones(shape, dtype=np_dtype)
        else:
            feed[x.name] = np.zeros(shape, dtype=np_dtype)
    return feed


def _summarize(sess: ort.InferenceSession) -> None:
    print(f"  inputs  ({len(sess.get_inputs())}):")
    for x in sess.get_inputs()[:3]:
        print(f"    {x.name:32s} shape={x.shape} type={x.type}")
    if len(sess.get_inputs()) > 3:
        print(f"    ... +{len(sess.get_inputs()) - 3} more")
    print(f"  outputs ({len(sess.get_outputs())}):")
    for x in sess.get_outputs()[:3]:
        print(f"    {x.name:32s} shape={x.shape} type={x.type}")
    if len(sess.get_outputs()) > 3:
        print(f"    ... +{len(sess.get_outputs()) - 3} more")


def _qnn_devices() -> list:
    lib_path = onnxruntime_qnn.get_library_path()
    print(f"  registering EP library: {lib_path}")
    try:
        ort.register_execution_provider_library("QNNExecutionProvider", lib_path)
    except Exception:
        # already registered in this process — safe to ignore
        pass
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    print(f"  QNN devices visible: {len(devs)}")
    return devs


def _try_session(label: str, wrapper_path: Path, provider_options: dict) -> int:
    qnn_devs = _qnn_devices()
    if not qnn_devs:
        print(f"  [{label}] FAIL: no QNN devices enumerated")
        return 1
    so = ort.SessionOptions()
    so.log_severity_level = 2  # warning-and-up; keeps log small
    try:
        so.add_provider_for_devices(qnn_devs, provider_options)
    except Exception:
        print(f"  [{label}] add_provider_for_devices FAILED:")
        traceback.print_exc()
        return 1
    try:
        sess = ort.InferenceSession(str(wrapper_path), sess_options=so)
    except Exception:
        print(f"  [{label}] InferenceSession FAILED:")
        traceback.print_exc()
        return 1
    providers = sess.get_providers()
    print(f"  [{label}] session providers: {providers}")
    if "QNNExecutionProvider" not in providers:
        print(f"  [{label}] FAIL: QNN EP did not bind")
        return 1
    return _run_and_report(sess, label)


def try_legacy_providers(wrapper_path: Path) -> int:
    print("\n--- attempt 1: legacy providers=[('QNNExecutionProvider', opts)] ---")
    qnn_htp = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / ("QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so")
    provider_options = {
        "backend_path": str(qnn_htp),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    so = ort.SessionOptions()
    so.log_severity_level = 2
    try:
        sess = ort.InferenceSession(
            str(wrapper_path),
            sess_options=so,
            providers=[("QNNExecutionProvider", provider_options)],
        )
    except Exception:
        print("  legacy load exception:")
        traceback.print_exc()
        return 1
    providers = sess.get_providers()
    print(f"  session providers: {providers}")
    if "QNNExecutionProvider" not in providers:
        print("  FAIL: silent fallback to CPU — 2.x plugin-EP not bound via legacy API")
        return 1
    return _run_and_report(sess, "legacy")


def try_plugin_disable_file_map(wrapper_path: Path) -> int:
    print("\n--- attempt 2: plugin-EP + disabl_file_mapped_weights=1 (skip buggy retry) ---")
    provider_options = {
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
        # DLL strings dump shows two spellings. The log on first probe emitted
        # "User specified disable_file_mapped_weights: 0" so the EP reads the
        # correctly-spelled key; the typo spelling goes unread. Set both to be
        # safe.
        "disable_file_mapped_weights": "1",
        "disabl_file_mapped_weights": "1",
    }
    return _try_session("disable-filemap", wrapper_path, provider_options)


def try_plugin_embed_mode(bin_path: Path, embed_wrapper_path: Path) -> int:
    print("\n--- attempt 3: plugin-EP + embed_mode=1 (inline binary in wrapper) ---")
    # Rebuild a wrapper that embeds the .bin as an inline bytes attribute.
    # Schema must mirror npu_load_qwen3_bin._describe_inputs_pathb_w4a16_local —
    # duplicated here so the probe stays self-contained.
    print(f"  building inline-binary wrapper at {embed_wrapper_path}")
    vocab, n_layers, n_kv, head_dim, past_len, total_len = 151936, 28, 8, 128, 255, 256
    inputs = [("input_ids", [1, 1], TensorProto.INT32)]
    for i in range(n_layers):
        inputs.append((f"past_key_values.{i}.key", [1, n_kv, past_len, head_dim], TensorProto.UINT16))
        inputs.append((f"past_key_values.{i}.value", [1, n_kv, past_len, head_dim], TensorProto.UINT16))
    inputs.append(("attention_bias", [1, 1, 1, total_len], TensorProto.UINT16))
    inputs.append(("position_ids_cos", [1, 1, head_dim], TensorProto.UINT16))
    inputs.append(("position_ids_sin", [1, 1, head_dim], TensorProto.UINT16))

    outputs = [("logits", [1, 1, vocab], TensorProto.UINT16)]
    for i in range(n_layers):
        outputs.append((f"present.{i}.key", [1, n_kv, total_len, head_dim], TensorProto.UINT16))
        outputs.append((f"present.{i}.value", [1, n_kv, total_len, head_dim], TensorProto.UINT16))

    inputs_decl = [helper.make_tensor_value_info(n, dt, s) for n, s, dt in inputs]
    outputs_decl = [helper.make_tensor_value_info(n, dt, s) for n, s, dt in outputs]

    bin_bytes = bin_path.read_bytes()
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name="qnn_ctx",
        domain="com.microsoft",
        embed_mode=1,
        ep_cache_context=bin_bytes,
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node], name="qwen3_w4a16_local_embed",
        inputs=inputs_decl, outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-ort21-probe",
    )
    model.ir_version = 10
    # Use external-data save so the ONNX itself stays small; the bytes still
    # live inline via ep_cache_context — save_model doesn't touch node attributes.
    onnx.save(model, str(embed_wrapper_path))
    print(f"  wrote {embed_wrapper_path.stat().st_size // (1024 * 1024)} MB wrapper "
          f"(inline embed of {len(bin_bytes) // (1024 * 1024)} MB binary)")

    provider_options = {
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    return _try_session("embed-mode-1", embed_wrapper_path, provider_options)


def _run_and_report(sess: ort.InferenceSession, label: str) -> int:
    print(f"\n  [{label}] IO summary after successful load:")
    _summarize(sess)
    print(f"\n  [{label}] single forward pass with zero-uint16 KV + BOS ...")
    feed = _build_zero_feed(sess)
    t0 = time.perf_counter_ns()
    try:
        outs = sess.run(None, feed)
    except Exception:
        print(f"  [{label}] forward pass FAILED:")
        traceback.print_exc()
        return 1
    t1 = time.perf_counter_ns()
    names = [o.name for o in sess.get_outputs()]
    logits = outs[names.index("logits")]
    print(f"  [{label}] latency          : {(t1 - t0) / 1e6:.2f} ms")
    print(f"  [{label}] logits shape     : {logits.shape}")
    print(f"  [{label}] logits dtype     : {logits.dtype}")
    print(f"  [{label}] logits min/max   : {int(logits.min())}/{int(logits.max())} (uint16 raw)")
    print(f"  [{label}] logits unique ct : {len(np.unique(logits[0, -1]))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-filemap-disable", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    print(f"ort        : {ort.__version__}")
    print(f"ort-qnn    : {onnxruntime_qnn.__version__}")
    print(f"wrapper    : {WRAPPER_PATH}  exists={WRAPPER_PATH.exists()}")
    print(f"bin        : {BIN_PATH}  exists={BIN_PATH.exists()}  "
          f"size={BIN_PATH.stat().st_size // (1024 * 1024) if BIN_PATH.exists() else 0} MB")
    if not WRAPPER_PATH.exists() or not BIN_PATH.exists():
        print("\nERROR: wrapper or bin missing.")
        return 2

    rc = 1
    if not args.skip_legacy:
        if try_legacy_providers(WRAPPER_PATH) == 0:
            rc = 0
    if rc != 0 and not args.skip_filemap_disable:
        if try_plugin_disable_file_map(WRAPPER_PATH) == 0:
            rc = 0
    if rc != 0 and not args.skip_embed:
        if try_plugin_embed_mode(BIN_PATH, WRAPPER_EMBED_PATH) == 0:
            rc = 0
    print(f"\n=== STATUS: {'ok' if rc == 0 else 'all attempts failed'} ===")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
