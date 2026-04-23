"""Phase 5 step 5 - load compiled HTP context binary, shape-check, smoke.

Goal: prove a `.bin` produced by AI Hub loads on the X2E's Hexagon
v81 via ORT-QNN, the IO signature matches what was compiled for,
and a single forward pass completes without DMA/signing errors.
Correctness vs CPU is step 6; this is structural only.

The .bin AI Hub downloaded is a RAW QNN context binary (header bytes
`00 00 00 02 00 00 00 03 ...`), not an ONNX-EPContext wrapper. We
hand-build a tiny ONNX whose single node is an EPContext referencing
the .bin, then load that.

Three source variants supported (matches scripts/compile_qwen3_ai_hub.py):

    --path patha       58 inputs: input_ids + position_ids + 56 past_kv.
    --path pathbmask   59 inputs: above + attention_bias (FP32 additive).
    --path pathb       61 inputs: pathbmask + position_ids_cos +
                        position_ids_sin (rotary hoisted out of graph).

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
    SPECULA_NPU_VARIANT=w4a16-a .venv\\Scripts\\python.exe scripts\\npu_load_qwen3_bin.py --path pathb
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parent.parent

SOC_MODEL = "88"
HTP_ARCH = "81"
# Runtime context-window tier. Defaults to 512 (Phase 5 baseline). Set
# SPECULA_NPU_CTX=256 before invoking the outer loop / sweep to target
# the Lever B ctx=256 binary. Value must match the compiled binary's
# past_len (CONTEXT_MAX - 1) or the wrapper ONNX signature won't match
# what the binary expects.
CONTEXT_MAX = int(os.environ.get("SPECULA_NPU_CTX", 512))

# Quant-variant suffix appended before ".bin" / ".wrapper.onnx". Unset (or
# empty) targets the legacy fp16 baseline binary. Set e.g. SPECULA_NPU_VARIANT=w4a16-a
# before invoking the outer loop / sweep / probe to switch to a Lever C
# W4A16 calibration-bundle binary. Value must match the suffix passed as
# --quant-tag at compile time (see scripts/compile_qwen3_ai_hub.py).
VARIANT = os.environ.get("SPECULA_NPU_VARIANT", "")
_VARIANT_SUFFIX = f".{VARIANT}" if VARIANT else ""

# Local-compile variants (x86 QAIRT pipeline) share three properties that
# AI-Hub-compiled binaries don't: `--remove_unused_inputs` drops
# position_ids, qairt-converter's output-rename pass is skipped so
# outputs stay as `logits` / `present_N_{key,value}`, and the wrapper
# must declare those names.
#
# Naming convention: every local variant carries a `-local` token
# somewhere in VARIANT. Examples we've shipped or planned:
#   fp16-local              — no PTQ, float32 IO
#   w4a16-local             — w4 weights, a16 activations, uint16 IO
#   w4a16-local-tfe         — + enhanced act calibration
#   w4a16-local-cle         — + cross-layer equalisation
#   w4a16-local-b           — + Bundle B calibration
#   w8a16-local             — w8 weights, a16 activations, uint16 IO
#
# The dtype split is PTQ-applied → UINT16 IO; no-PTQ → FLOAT IO.
# A `fp*-local` prefix identifies no-PTQ variants (qairt-quantizer
# called with `--float_bitwidth {16,32}` instead of weight/act flags).
# Anything else that's local went through PTQ and has UINT16 IO.
IS_LOCAL_COMPILE = "-local" in VARIANT
IS_LOCAL_FP_NO_PTQ = IS_LOCAL_COMPILE and VARIANT.startswith("fp")
# Retain the IS_LOCAL_W4A16 name (public across probe imports) but key
# it on "quantized local variant" semantics rather than the w4a16 prefix
# literally — w8a16-local and any future precision combo qualifies.
IS_LOCAL_W4A16 = IS_LOCAL_COMPILE and not IS_LOCAL_FP_NO_PTQ

# Logits live on the first compiled output. AI-Hub-compiled binaries go
# through qairt-converter's output-renaming pass so every output ends up
# as `output_{0..N}`; the local-compile variants did not rename and kept
# the ONNX names, so logits is literally `logits`.
LOGITS_OUTPUT_NAME = "logits" if IS_LOCAL_COMPILE else "output_0"

# Qwen3-0.6B RoPE params (config.json: head_dim=128, rope_theta=1e6,
# attention_scaling=1.0). Path B hoists the rotary_emb subgraph out of
# the compiled binary — callers feed pre-computed cos/sin per decode
# step. Formula matches the optimum export's seam precisely; verified
# cos=1.000000 vs the source graph on x86 (see status_x86.md session 2
# and scripts/probe_pathb_equivalence.py). For Qwen3.5 with non-1.0
# attention_scaling, fold the scalar into cos/sin here.
ROPE_THETA = 1_000_000.0
ROPE_HEAD_DIM = 128


def rope_tables(position_id: int,
                head_dim: int = ROPE_HEAD_DIM,
                rope_theta: float = ROPE_THETA) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin for one decode step. Shape [1, 1, head_dim] float32 each."""
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    freqs = position_id * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)[None, None, :].astype(np.float32)
    sin = np.sin(emb)[None, None, :].astype(np.float32)
    return cos, sin


def _bin_path(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}{_VARIANT_SUFFIX}.bin"


def _wrapper_path(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}{_VARIANT_SUFFIX}.wrapper.onnx"


def _encodings_path(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}{_VARIANT_SUFFIX}.encodings.json"


def _config_json(path_key: str) -> Path:
    source_dir = REPO_ROOT / "models" / f"qwen3-0.6b-{path_key}"
    return source_dir / "config.json"


# ---------------------------------------------------------------------------
# W4A16 asymmetric uint16 quantization helpers.
#
# QAIRT convention (verified against dlc_info_w4a16.txt min/max boundaries):
#     x_fp32   = (q_uint16 + offset) * scale     # dequant
#     q_uint16 = clip(round(x / scale) - offset, 0, 65535)   # quant
#
# `offset` is stored in the DLC as a negative integer (the negated
# zero-point, per QNN SDK convention). For layer-0 past_kv.0.key the
# declared encoding is scale=0.014685, offset=-33631, min=-493.88, max=468.52;
# q=0 dequants to -493.88 and q=65535 to 468.52, matching exactly.
#
# Tensor-name matching: the .bin uses underscored names
# (`past_key_values_0_key`), but `encodings.json` keeps the dotted names
# from the upstream ONNX (`past_key_values.0.key`). `load_quant_specs`
# does the translation at load time, keyed by the wrapper's runtime
# tensor names so downstream code can just do specs[sess_input.name].
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSpec:
    scale: float
    offset: int
    bitwidth: int

    @property
    def qmax(self) -> int:
        return (1 << self.bitwidth) - 1


def _dot_form(underscored: str) -> str:
    """Translate binary-internal name to encodings.json key.

    `past_key_values_0_key` <-> `past_key_values.0.key`
    `present_12_value`      <-> `present.12.value`
    Everything else (input_ids, attention_bias, position_ids_{cos,sin},
    logits) passes through unchanged.
    """
    for prefix in ("past_key_values_", "present_"):
        if underscored.startswith(prefix):
            tail = underscored[len(prefix):]
            # tail is like "12_key" / "12_value" — split on the LAST underscore
            head, _, kind = tail.rpartition("_")
            if head and kind in ("key", "value"):
                return f"{prefix.rstrip('_')}.{head}.{kind}"
    return underscored


def load_quant_specs(
    encodings_path: Path,
    runtime_names: list[str],
) -> dict[str, QuantSpec]:
    """Build {runtime_name -> QuantSpec} for each named tensor.

    Skips tensors that are not quantized (data_type 50 = int32 on input_ids;
    `quant_params.scale_offset.bitwidth == 0` marks the sentinel "no
    encoding" state seen on `input_ids`).
    """
    with encodings_path.open() as f:
        graph_tensors = json.load(f)["graph"]["tensors"]

    specs: dict[str, QuantSpec] = {}
    missing: list[str] = []
    unquantized: list[str] = []
    for name in runtime_names:
        key = _dot_form(name)
        tensor = graph_tensors.get(key)
        if tensor is None:
            missing.append(name)
            continue
        qp = tensor.get("quant_params", {}).get("scale_offset", {})
        bw = int(qp.get("bitwidth", 0) or 0)
        if bw == 0:
            unquantized.append(name)
            continue
        specs[name] = QuantSpec(
            scale=float(qp["scale"]),
            offset=int(qp["offset"]),
            bitwidth=bw,
        )
    if missing:
        raise KeyError(
            f"encodings.json lacks {len(missing)} runtime-named tensor(s); "
            f"first few: {missing[:3]}"
        )
    # `unquantized` is expected (input_ids). Don't raise.
    return specs


def quant_to_uint16(arr: np.ndarray, spec: QuantSpec) -> np.ndarray:
    if spec.bitwidth != 16:
        raise ValueError(f"quant_to_uint16 requires bitwidth=16, got {spec.bitwidth}")
    q = np.rint(arr.astype(np.float32) / spec.scale) - spec.offset
    return np.clip(q, 0, spec.qmax).astype(np.uint16)


def dequant_from_uint16(arr: np.ndarray, spec: QuantSpec) -> np.ndarray:
    if spec.bitwidth != 16:
        raise ValueError(f"dequant_from_uint16 requires bitwidth=16, got {spec.bitwidth}")
    # int32 intermediate so (q + offset) doesn't underflow uint16 when offset < 0
    return (arr.astype(np.int32) + spec.offset).astype(np.float32) * spec.scale


def quantized_zero(spec: QuantSpec) -> int:
    """The uint16 representation of fp32 == 0 under this spec. Always `-offset`
    mod qmax+1 in practice; clip for safety against pathologically-offset
    tensors like attention_bias (offset -65535 -> q_zero = 65535, which is
    the actual max)."""
    return int(np.clip(0 - spec.offset, 0, spec.qmax))


def describe_inputs(cfg: dict, path_key: str) -> list[tuple[str, list[int], int]]:
    """Return (name, shape, elem_type) for each input.

    Names + dtypes verified against the compile-time input_specs in
    scripts/compile_qwen3_ai_hub.py:
      * the qairt-converter normalises dotted names to underscored;
      * `--preserve_io_datatype` keeps past_key_values at FLOAT_32 even
        when `--quantize_full_type float16` is set for the interior;
      * `--truncate_64bit_io` casts INT64 IO to INT32.

    Path B-mask additionally has `attention_bias` (FP32, additive)
    as an input. Path B extends that with `position_ids_cos` +
    `position_ids_sin` (rotary hoisted out of the graph).

    The `w4a16-local` VARIANT (x86-compiled per docs/phase5_local_qairt_compile.md)
    deviates on three axes documented in
    docs/phase5_local_qairt_compile_findings.md: (1) `position_ids`
    dropped via `--remove_unused_inputs`, (2) dotted ONNX names
    preserved (no underscore rename), (3) uint16 quantized IO for
    every tensor except `input_ids` (int32).
    """
    if path_key == "pathb" and IS_LOCAL_COMPILE:
        return _describe_inputs_pathb_local(
            cfg,
            dtype=TensorProto.UINT16 if IS_LOCAL_W4A16 else TensorProto.FLOAT,
        )
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
    # build_input_specs comment. Path B then appends cos/sin after that.
    if path_key in ("pathbmask", "pathb"):
        inputs.append(("attention_bias", [1, 1, 1, CONTEXT_MAX], TensorProto.FLOAT))
    if path_key == "pathb":
        inputs.append(("position_ids_cos", [1, 1, head_dim], TensorProto.FLOAT))
        inputs.append(("position_ids_sin", [1, 1, head_dim], TensorProto.FLOAT))
    return inputs


def _describe_inputs_pathb_local(cfg: dict, dtype: int) -> list[tuple[str, list[int], int]]:
    # qnn-context-binary-generator normalises dotted tensor names to
    # underscored when emitting the .bin, even though the intermediate
    # DLC / encodings.json retain the dots. Verified by scanning strings
    # in models/qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-local.bin —
    # binary exposes `past_key_values_0_key`, not `past_key_values.0.key`.
    # ORT-QNN's EPContext binder matches by literal name, so the wrapper
    # must use the underscored form.
    #
    # `dtype` is UINT16 for w4a16-local (PTQ quantized IO) or FLOAT for
    # fp16-local (PTQ skipped, IO stays fp32).
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_len = CONTEXT_MAX - 1

    inputs: list[tuple[str, list[int], int]] = [
        ("input_ids", [1, 1], TensorProto.INT32),
    ]
    for i in range(n_layers):
        inputs.append((f"past_key_values_{i}_key", [1, n_kv, past_len, head_dim], dtype))
        inputs.append((f"past_key_values_{i}_value", [1, n_kv, past_len, head_dim], dtype))
    inputs.append(("attention_bias", [1, 1, 1, CONTEXT_MAX], dtype))
    inputs.append(("position_ids_cos", [1, 1, head_dim], dtype))
    inputs.append(("position_ids_sin", [1, 1, head_dim], dtype))
    return inputs


def describe_outputs(cfg: dict, path_key: str = "") -> list[tuple[str, list[int], int]]:
    """Return (name, shape, elem_type) per output.

    The qairt-converter renames all outputs to `output_0..output_N` in
    declaration order from the source ONNX:
      output_0  = logits           [1, 1, vocab]
      output_1  = present.0.key    [1, n_kv, ctx, head_dim]
      output_2  = present.0.value
      ...
      output_56 = present.27.value

    The `w4a16-local` VARIANT did not apply the rename pass, so the
    binary exposes the literal ONNX names (`logits` / `present.N.{key,value}`),
    and every output is uint16 per the PTQ.
    """
    if path_key == "pathb" and IS_LOCAL_COMPILE:
        return _describe_outputs_pathb_local(
            cfg,
            dtype=TensorProto.UINT16 if IS_LOCAL_W4A16 else TensorProto.FLOAT,
        )
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


def _describe_outputs_pathb_local(cfg: dict, dtype: int) -> list[tuple[str, list[int], int]]:
    # Same dot→underscore normalisation as inputs, per the binary-strings
    # scan. `logits` has no dots in the source so it passes through.
    # `dtype` is UINT16 for w4a16-local / FLOAT for fp16-local.
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    vocab = cfg["vocab_size"]
    total_len = CONTEXT_MAX

    outputs: list[tuple[str, list[int], int]] = [
        ("logits", [1, 1, vocab], dtype),
    ]
    for i in range(n_layers):
        outputs.append((f"present_{i}_key", [1, n_kv, total_len, head_dim], dtype))
        outputs.append((f"present_{i}_value", [1, n_kv, total_len, head_dim], dtype))
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
        for n, shape, dt in describe_outputs(cfg, path_key)
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


def build_zero_feed(
    sess: ort.InferenceSession,
    quant_specs: dict[str, QuantSpec] | None = None,
) -> dict:
    """Build a feed of zeros matching the session's actual input spec.

    For quantized uint16 inputs (w4a16-local variant), literal uint16=0
    dequantizes to the most-NEGATIVE representable fp32 value under each
    tensor's scale/offset — meaningless as a "zero KV". When `quant_specs`
    is provided (keyed by runtime input name), produces the uint16 value
    that dequantizes to fp32==0 for each quantized tensor. Non-quantized
    inputs (input_ids, position_ids) keep their previous semantics.
    """
    dtype_map = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
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
        elif quant_specs is not None and x.name in quant_specs:
            feed[x.name] = np.full(shape, quantized_zero(quant_specs[x.name]), dtype=np_dtype)
        else:
            feed[x.name] = np.zeros(shape, dtype=np_dtype)
    return feed


def main() -> int:
    import functools
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask", "pathb"), required=True)
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

    # For w4a16-local, load the encodings.json so the zero feed quantizes
    # correctly and we can dequant the uint16 logits into a float view.
    quant_specs = None
    logits_spec: QuantSpec | None = None
    if VARIANT == "w4a16-local":
        enc_path = _encodings_path(args.path)
        if not enc_path.exists():
            print(f"ERROR: {enc_path} missing (w4a16-local requires the quant encodings)")
            return 2
        runtime_in_names = [x.name for x in sess.get_inputs()]
        runtime_out_names = [x.name for x in sess.get_outputs()]
        quant_specs = load_quant_specs(enc_path, runtime_in_names + runtime_out_names)
        logits_spec = quant_specs.get(LOGITS_OUTPUT_NAME)
        print(f"  loaded {len(quant_specs)} quant specs from encodings.json")

    print("\n--- single forward pass with fp32-zero KV + BOS-ish token ---")
    feed = build_zero_feed(sess, quant_specs=quant_specs)
    t0 = time.perf_counter_ns()
    outputs = sess.run(None, feed)
    t1 = time.perf_counter_ns()
    ms = (t1 - t0) / 1e6
    names = [o.name for o in sess.get_outputs()]
    logits_idx = names.index(LOGITS_OUTPUT_NAME)
    logits_raw = outputs[logits_idx]
    logits = (
        dequant_from_uint16(logits_raw, logits_spec)
        if logits_spec is not None
        else logits_raw
    )
    print(f"  run latency       : {ms:.2f} ms")
    print(f"  logits raw dtype  : {logits_raw.dtype}  shape {logits_raw.shape}")
    print(f"  logits fp32 min/max: {float(np.nanmin(logits)):.3f} / {float(np.nanmax(logits)):.3f}")
    print(f"  logits finite frac : {float(np.isfinite(logits).mean()):.4f}")
    top5 = np.argsort(-logits[0, -1].astype(np.float32))[:5]
    print(f"  argmax top 5 ids   : {top5.tolist()}")

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
