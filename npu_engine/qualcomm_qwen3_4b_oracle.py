"""Phase 0 of the Qualcomm Qwen3-4B reproduction effort: record an oracle.

Drives the 4-part Qualcomm Qwen3-4B Genie w4a16 bundle through
prompt prefill + N generation steps using AR1 / CL512, recording
per-step logits and generated tokens. Output is the ground truth
that any future specula-built Qwen3-4B w4a16 must match (logit cos
on the same prompt) for the pipeline to be considered verified.

This is a CPU-cheap harness (no torch, no transformers) — all model
ops happen on the X2E HTP via ORT-QNN. The harness only handles
tokenization, position-id arithmetic, quantize/dequant, and
KV-cache stitching.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
        npu_engine/qualcomm_qwen3_4b_oracle.py --gen-steps 8

Output:
    results/qualcomm_qwen3_4b_oracle.npz   - per-step logits + tokens
    results/qualcomm_qwen3_4b_oracle.md    - human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
SOC_MODEL = "88"
HTP_ARCH = "81"

NUM_LAYERS = 36
NUM_PARTS = 4  # 1 embed + 3 transformer (parts 2/3/4 own 12 layers each)
LAYERS_PER_PART = 12
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
CTX_LEN = 512
PAST_LEN = CTX_LEN - 1  # 511 — buffer size for AR1 cl512
HIDDEN_DIM = 2560
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0

# 4B w4a16 bundle was AI-Hub-built against QAIRT 2.42 — matches the
# venv-bundled ORT 1.24.4. Default backend (None ⇒ venv-bundled
# QnnHtp.dll) loads it correctly. Sister 7B bundle needs an explicit
# 2.45 DLL override via this constant; see qualcomm_qwen2_5_7b_oracle.
BACKEND_PATH = None

# AR128 prefill graphs share the same .bin files as AR1 but expect a
# 128-wide query batch. The cl512 cap splits as past=384 + current=128.
AR128_BATCH = 128
PAST_LEN_AR128_CL512 = CTX_LEN - AR128_BATCH  # 384


def quant_uint16(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """dequant convention: f = (q + offset) * scale  =>  q = round(f/scale) - offset."""
    q = np.round(x_fp32 / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_uint16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.int32) + offset).astype(np.float32) * scale


def half_dim_rope_quantized(
    pos: int, scale: float, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Qualcomm-style half-dim cos/sin tables for a single position.
    Shape [1, 1, 1, 64], uint16-quantized with given scale/offset."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = pos * inv_freq  # [64]
    cos_h = np.cos(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    sin_h = np.sin(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    return quant_uint16(cos_h, scale, offset), quant_uint16(sin_h, scale, offset)


def half_dim_rope_quantized_ar128(
    p_base: int, scale: float, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """Half-dim cos/sin tables for an AR128 batch starting at p_base.
    Shape [1, 1, 128, 64], uint16-quantized."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    positions = np.arange(p_base, p_base + AR128_BATCH, dtype=np.float32).reshape(-1, 1)
    freqs = positions * inv_freq.reshape(1, -1)  # [128, 64]
    cos_h = np.cos(freqs).reshape(1, 1, AR128_BATCH, HALF_HEAD_DIM)
    sin_h = np.sin(freqs).reshape(1, 1, AR128_BATCH, HALF_HEAD_DIM)
    return quant_uint16(cos_h, scale, offset), quant_uint16(sin_h, scale, offset)


def attention_mask_quantized(
    pos: int, scale: float, offset: int, ctx_len: int = CTX_LEN
) -> np.ndarray:
    """Build the CL=N attention mask for AR1 decode at position `pos`.

    KV layout: past_kv has N-1 slots (chronological, left-aligned — slot
    0 holds position 0, slot t-1 holds position t-1, slots t..N-2 are
    zero-filled and unused). The model internally concatenates the
    current step's K/V (1 slot) onto the past, producing N attention
    slots: [past_(N-1) | current_1].

    Therefore at step t (0-indexed):
      - past slots 0..t-1   -> attend (valid history)
      - past slots t..N-2   -> block (unused, all-zero KV)
      - current slot N-1    -> attend (always, this is the live token)

    Quantized representation: q=65535 -> dequant=0.0 (attend);
    q=0 -> dequant=(0-65535)*0.0015259 ~= -100 (mask). The dequantized
    mask is added to attention scores before softmax."""
    mask = np.zeros((1, 1, 1, ctx_len), dtype=np.uint16)
    if pos > 0:
        mask[..., :pos] = 65535  # past positions 0..pos-1
    mask[..., -1] = 65535  # current slot ctx_len-1
    _ = scale, offset  # values are fixed by metadata; included for symmetry
    return mask


def attention_mask_quantized_ar128(
    p_base: int, scale: float, offset: int, ctx_len: int = CTX_LEN
) -> np.ndarray:
    """CL=N mask for an AR128 batch at base absolute position p_base.

    Layout: mask[..., 0..(N-128-1)] indexes past KV slots (chronological:
    slot k holds K/V for absolute position k); mask[..., (N-128)..N-1]
    indexes the new 128 query positions (slot (N-128)+j holds K/V for
    absolute position p_base+j).

    For query q (0..127, absolute position p_base+q):
      - past slot k (0..N-129): attend iff k < p_base (valid past data
        the previous calls wrote; later slots are zero-coded padding).
      - current slot (N-128)+j: attend iff j <= q (causal within batch).

    q=65535 -> dequant=0 (attend); q=0 -> dequant ~ -100 (mask).
    """
    past_len = ctx_len - AR128_BATCH
    mask = np.zeros((1, 1, AR128_BATCH, ctx_len), dtype=np.uint16)
    if p_base > 0:
        # All queries attend to all valid past slots — past < p_base is
        # always < p_base + q so causality is automatic.
        mask[..., :p_base] = 65535
    # Causal triangle for new tokens within the batch: row q attends
    # to current slots 0..q.
    causal = np.tri(AR128_BATCH, AR128_BATCH, k=0, dtype=np.uint16) * 65535
    mask[0, 0, :, past_len:] = causal
    _ = scale, offset
    return mask


def load_parts_cfg(ar: int = 1, ctx: int = CTX_LEN) -> dict:
    """Model-uniform helper for sidecar.py — reads the bundle's
    metadata.yaml, calls build_part_cfg, returns the cfg dict. The 7B
    oracle exposes a same-named helper that builds cfg from QNN
    introspection JSONs instead. Sidecar can call _model.load_parts_cfg
    without caring which path produced the dict."""
    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    return build_part_cfg(metadata, ar=ar, ctx=ctx)


def build_part_cfg(metadata: dict, ar: int = 1, ctx: int = CTX_LEN) -> dict:
    """Pull cl=N IO specs from the bundle's metadata.yaml.

    `ar=1` selects the single-token decode graphs (`ar1_cl{N}_*_of_4`);
    `ar=128` selects the batched prefill graphs (`ar128_cl{N}_*_of_4`).
    Both target the same .bin files — only the graph_name differs.

    `ctx` selects the context-length tier. The Qwen3-4B bundle ships
    {512, 1024, 2048, 3072, 4096}. KeyError if the bundle wasn't
    compiled for that tier.
    """
    cfg = {}
    for part in (1, 2, 3, 4):
        comp_name = f"ar{ar}_cl{ctx}_{part}_of_4"
        comp = metadata["components"][comp_name]
        # The wrapper.onnx must use UNDERSCORED tensor names — that's how
        # the QAIRT compiler stored them in the binary. metadata.yaml has
        # the SLASH form (original ONNX), so we translate.
        def to_underscore(name: str) -> str:
            if name.startswith("/"):
                # /model/model/.../foo_output_0  ->  _model_model_..._foo_output_0
                return name.replace("/", "_").replace(".", "_")
            return name
        # Qualcomm names decode-mode graphs `token_ar1_cl{N}_*` and
        # prefill-mode graphs `prompt_ar128_cl{N}_*`. Both graph names
        # live in the SAME .bin (each .bin is multi-graph: 5 ctx tiers
        # × 2 AR modes = 10 graphs). Multiple ORT sessions backed by
        # the same .bin coexist as long as they all point at the same
        # file path — QNN registers all graphs once and binds each
        # session's EPContext to its named graph. Pointing different
        # sessions at distinct paths (even byte-identical copies) trips
        # QNN error 1002 because the runtime manager indexes by path.
        prefix = "token" if ar == 1 else "prompt"
        cfg[part] = {
            "bin": f"qwen3_4b_part_{part}_of_4.bin",
            "graph_name": f"{prefix}_ar{ar}_cl{ctx}_{part}_of_4",
            "ar": ar,
            "ctx": ctx,
            "inputs": [
                {
                    "name": to_underscore(name),
                    "metaname": name,
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype"],
                    # int32 input_ids has no quant params — leave as None.
                    "scale": spec.get("quantization_parameters", {}).get("scale"),
                    "offset": spec.get("quantization_parameters", {}).get("offset"),
                }
                for name, spec in comp["inputs"].items()
            ],
            "outputs": [
                {
                    "name": to_underscore(name),
                    "metaname": name,
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype"],
                    "scale": spec.get("quantization_parameters", {}).get("scale"),
                    "offset": spec.get("quantization_parameters", {}).get("offset"),
                }
                for name, spec in comp["outputs"].items()
            ],
        }
    return cfg


def wrapper_path(bundle_dir: Path, part_idx: int, suffix: str = "", ctx: int = CTX_LEN) -> Path:
    """Standard wrapper-ONNX path for one partition.

    For ctx=512 the legacy filename is preserved (`oracle_part{N}{suffix}.wrapper.onnx`)
    so existing pre-built wrappers continue to be reused. For ctx>512 a `_cl{ctx}`
    suffix is appended to disambiguate per-tier — wrappers are tier-specific
    because they embed the QNN graph_name (`{prefix}_ar{ar}_cl{ctx}_{N}_of_4`).
    """
    if ctx == CTX_LEN:
        return bundle_dir / f"oracle_part{part_idx}{suffix}.wrapper.onnx"
    return bundle_dir / f"oracle_part{part_idx}{suffix}_cl{ctx}.wrapper.onnx"


_DTYPE_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "float32": TensorProto.FLOAT,
}
_DTYPE_NUMPY = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "float32": np.float32,
}


def build_wrapper(part_cfg: dict, wrapper_path: Path) -> None:
    inputs_decl = [
        helper.make_tensor_value_info(io["name"], _DTYPE_PROTO[io["dtype"]], io["shape"])
        for io in part_cfg["inputs"]
    ]
    outputs_decl = [
        helper.make_tensor_value_info(io["name"], _DTYPE_PROTO[io["dtype"]], io["shape"])
        for io in part_cfg["outputs"]
    ]
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name=part_cfg["graph_name"],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=part_cfg["bin"],
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"qualcomm_qwen3_4b_oracle_part{part_cfg['graph_name'][-6]}",
        inputs=inputs_decl,
        outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-oracle",
    )
    model.ir_version = 10
    onnx.save(model, str(wrapper_path))


def load_session(wrapper_path: Path, backend_path: Path | str | None = None) -> ort.InferenceSession:
    """Load an EPContext-wrapped QNN binary.

    `backend_path` overrides the QnnHtp.dll location. Default: the venv-
    bundled DLL (matches QAIRT version that built the 4B w4a16 bundle —
    QAIRT 2.42 in ORT 1.24.4). Override needed for bundles built by a
    different QAIRT (e.g. the Qwen2.5-7B w8a16 bundle was AI-Hub-built
    against QAIRT 2.45 and won't load with the venv DLL — pass the
    system QAIRT 2.45 DLL explicitly).
    """
    if backend_path is None:
        backend_path = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
    provider_options = {
        "backend_path": str(backend_path),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(
        str(wrapper_path),
        sess_options=sess_opts,
        providers=[("QNNExecutionProvider", provider_options)],
    )
    if sess.get_providers()[0] != "QNNExecutionProvider":
        raise RuntimeError(f"session fell back to {sess.get_providers()[0]}")
    return sess


class KVStore:
    """Persistent uint8 KV cache, shape [8, 1, 128, t] for keys and
    [8, 1, t, 128] for values, per layer. Grows by 1 timestep per
    decode call.

    Per-layer scale/offset are FIXED (Qualcomm chose them at compile
    time) — verified that past_kv_in scale/offset == past_kv_out
    scale/offset for every layer, so we can concat raw uint8 across
    steps without requantizing.

    `ctx_len` selects the bundle's context tier; defaults to 512 for
    backward compatibility. Larger tiers grow the master buffer linearly
    (cl=4096 ⇒ ~32 MB per layer × 36 layers × 2 KV ≈ 2.3 GB total)."""

    def __init__(
        self,
        num_layers: int,
        with_ar128_input: bool = False,
        ctx_len: int = CTX_LEN,
    ):
        self.num_layers = num_layers
        self.ctx_len = ctx_len
        self.past_len = ctx_len - 1
        self.past_len_ar128 = ctx_len - AR128_BATCH
        # Master cache: shape [8, 1, 128, past_len] / [8, 1, past_len, 128]
        # filled with the "zero" code (q=128 since offset=-128 for KV).
        # Attended slots are mask-controlled; unused slots stay zero so
        # any accidental contribution to attention is zero. This buffer
        # is what the AR1 graphs read directly (zero-copy via IOBinding).
        self.keys: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, HEAD_DIM, self.past_len), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        self.values: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, self.past_len, HEAD_DIM), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        # Optional AR128 input buffer at exactly the shape the AR128
        # graphs expect ([8,1,128,past_len_ar128] / [8,1,past_len_ar128,128]).
        # Eliminates the per-call ascontiguousarray copy that slicing the
        # master would cost. Kept in sync with the master via
        # stitch_batch's dual-write.
        self.has_ar128_in = with_ar128_input
        if with_ar128_input:
            self.keys_ar128_in: list[np.ndarray] = [
                np.full(
                    (NUM_KV_HEADS, 1, HEAD_DIM, self.past_len_ar128), 128, dtype=np.uint8
                )
                for _ in range(num_layers)
            ]
            self.values_ar128_in: list[np.ndarray] = [
                np.full(
                    (NUM_KV_HEADS, 1, self.past_len_ar128, HEAD_DIM), 128, dtype=np.uint8
                )
                for _ in range(num_layers)
            ]
        self.t = 0  # next free slot index in [0, past_len)

    def stitch_step(self, k_outs: list[np.ndarray], v_outs: list[np.ndarray]) -> None:
        if self.t >= self.past_len:
            raise RuntimeError(f"KV cache full at t={self.t}; ctx={self.ctx_len}")
        for i in range(self.num_layers):
            # k_out shape: [8, 1, 128, 1]
            self.keys[i][..., self.t : self.t + 1] = k_outs[i]
            # v_out shape: [8, 1, 1, 128]
            self.values[i][:, :, self.t : self.t + 1, :] = v_outs[i]
        # Don't bother mirroring AR1 decode steps into the AR128 input
        # buffer — once we're decoding AR1, AR128 prefill is done.
        self.t += 1

    def stitch_batch(
        self,
        p_base: int,
        k_outs: list[np.ndarray],
        v_outs: list[np.ndarray],
    ) -> None:
        """Write a batch of AR128_BATCH new K/V slots starting at p_base.

        Each k_out has shape [8, 1, 128, 128] (heads, batch, head_dim,
        time_slots) and each v_out has shape [8, 1, 128, 128] (heads,
        batch, time_slots, head_dim).

        Mirrors into the AR128 input buffer when present so the next
        AR128 call can read it zero-copy.
        """
        end = p_base + AR128_BATCH
        if end > self.past_len:
            raise RuntimeError(
                f"AR128 batch ending at {end} exceeds buffer {self.past_len}"
            )
        mirror = self.has_ar128_in and end <= self.past_len_ar128
        for i in range(self.num_layers):
            self.keys[i][..., p_base:end] = k_outs[i]
            self.values[i][:, :, p_base:end, :] = v_outs[i]
            if mirror:
                self.keys_ar128_in[i][..., p_base:end] = k_outs[i]
                self.values_ar128_in[i][:, :, p_base:end, :] = v_outs[i]
        self.t = end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-steps", type=int, default=8,
                        help="number of generation steps after prompt prefill")
    parser.add_argument("--prompt-file", type=str,
                        default=str(BUNDLE_DIR / "sample_prompt.txt"))
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "results" / "qualcomm_qwen3_4b_oracle"))
    args = parser.parse_args()

    print(f"=== Qualcomm Qwen3-4B w4a16 oracle (AR1 / CL512) ===")
    print(f"bundle: {BUNDLE_DIR}")
    print(f"prompt file: {args.prompt_file}")

    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts = build_part_cfg(metadata)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(prompt_text).ids
    print(f"prompt: {len(prompt_text)} chars, {len(prompt_ids)} tokens")
    print(f"prefill positions 0..{len(prompt_ids) - 1}, "
          f"generate positions {len(prompt_ids)}..{len(prompt_ids) + args.gen_steps - 1}")

    # Build + load 4 sessions.
    print("\n--- building wrappers and loading sessions ---")
    sessions: dict[int, ort.InferenceSession] = {}
    t0 = time.perf_counter()
    for part_idx in (1, 2, 3, 4):
        wrapper_path = BUNDLE_DIR / f"oracle_part{part_idx}.wrapper.onnx"
        build_wrapper(parts[part_idx], wrapper_path)
        t_load = time.perf_counter()
        sessions[part_idx] = load_session(wrapper_path)
        print(f"  part {part_idx}: wrapper={wrapper_path.name}, "
              f"loaded in {time.perf_counter() - t_load:.1f} s")
    print(f"all 4 parts loaded in {time.perf_counter() - t0:.1f} s")

    # Capture per-input metadata for fast quantization at run time.
    # cos/sin/mask scales are constant across parts 2/3/4 (verified earlier).
    cos_scale = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["scale"]
    cos_offset = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["offset"]
    mask_scale = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["scale"]
    mask_offset = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["offset"]
    logits_io = next(io for io in parts[4]["outputs"] if io["name"] == "logits")
    logits_scale = logits_io["scale"]
    logits_offset = logits_io["offset"]

    # Per-layer KV input names are needed to address feed entries.
    def layer_input_name(part_idx: int, kv: str, layer: int) -> str:
        return f"past_{kv}_{layer}_in"

    def layer_output_name(part_idx: int, kv: str, layer: int) -> str:
        return f"past_{kv}_{layer}_out"

    # Hidden-state handoff names (underscored).
    HIDDEN_FROM_PART1 = "_model_model_embed_tokens_Gather_output_0"
    HIDDEN_FROM_PART2 = "_model_model_layers_11_Add_1_output_0"
    HIDDEN_FROM_PART3 = "_model_model_layers_23_Add_1_output_0"

    kv = KVStore(NUM_LAYERS)

    all_logits_uint16: list[np.ndarray] = []
    all_logits_fp32: list[np.ndarray] = []
    all_argmax: list[int] = []
    all_step_tokens: list[int] = []  # what was fed in
    all_decoded: list[str] = []      # what was the next token (for gen steps)
    step_latency_ms: list[float] = []

    # Schedule: feed prompt[0..len-1] one token at a time, then for
    # generation feed argmax of previous step's logits.
    total_steps = len(prompt_ids) + args.gen_steps
    next_token = prompt_ids[0]
    print(f"\n--- decoding {total_steps} steps ---")
    for step in range(total_steps):
        is_prefill = step < len(prompt_ids)
        token_in = prompt_ids[step] if is_prefill else next_token
        all_step_tokens.append(token_in)
        position = step  # KV slot to write into for this step

        # Quantize position-dependent inputs.
        cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
        mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

        t_step = time.perf_counter()

        # --- part 1: input_ids -> embedding ---
        feed1 = {"input_ids": np.array([[token_in]], dtype=np.int32)}
        emb_out = sessions[1].run([HIDDEN_FROM_PART1], feed1)[0]

        # --- part 2: layers 0..11 ---
        feed2 = {
            HIDDEN_FROM_PART1: emb_out,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(0, LAYERS_PER_PART):
            feed2[layer_input_name(2, "key", layer)] = kv.keys[layer]
            feed2[layer_input_name(2, "value", layer)] = kv.values[layer]
        out_names_2 = [HIDDEN_FROM_PART2] + [
            layer_output_name(2, kvtype, layer)
            for layer in range(0, LAYERS_PER_PART)
            for kvtype in ("key", "value")
        ]
        out_2 = sessions[2].run(out_names_2, feed2)
        hidden_after_p2 = out_2[0]
        # Stitch parts of out_2 into KV store (12 layers * 2)
        new_keys_p2 = []
        new_vals_p2 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p2.append(out_2[1 + 2 * i])
            new_vals_p2.append(out_2[1 + 2 * i + 1])

        # --- part 3: layers 12..23 ---
        feed3 = {
            HIDDEN_FROM_PART2: hidden_after_p2,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
            feed3[layer_input_name(3, "key", layer)] = kv.keys[layer]
            feed3[layer_input_name(3, "value", layer)] = kv.values[layer]
        out_names_3 = [HIDDEN_FROM_PART3] + [
            layer_output_name(3, kvtype, layer)
            for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)
            for kvtype in ("key", "value")
        ]
        out_3 = sessions[3].run(out_names_3, feed3)
        hidden_after_p3 = out_3[0]
        new_keys_p3 = []
        new_vals_p3 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p3.append(out_3[1 + 2 * i])
            new_vals_p3.append(out_3[1 + 2 * i + 1])

        # --- part 4: layers 24..35 + lm_head -> logits ---
        feed4 = {
            HIDDEN_FROM_PART3: hidden_after_p3,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS):
            feed4[layer_input_name(4, "key", layer)] = kv.keys[layer]
            feed4[layer_input_name(4, "value", layer)] = kv.values[layer]
        out_names_4 = ["logits"] + [
            layer_output_name(4, kvtype, layer)
            for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)
            for kvtype in ("key", "value")
        ]
        out_4 = sessions[4].run(out_names_4, feed4)
        logits_uint16 = out_4[0]
        new_keys_p4 = []
        new_vals_p4 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p4.append(out_4[1 + 2 * i])
            new_vals_p4.append(out_4[1 + 2 * i + 1])

        # Stitch all 36 layers' new K/V into the persistent cache.
        kv.stitch_step(
            new_keys_p2 + new_keys_p3 + new_keys_p4,
            new_vals_p2 + new_vals_p3 + new_vals_p4,
        )

        step_latency_ms.append((time.perf_counter() - t_step) * 1000)

        logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
        argmax_id = int(np.argmax(logits_fp32))
        all_logits_uint16.append(logits_uint16.squeeze())
        all_logits_fp32.append(logits_fp32.squeeze())
        all_argmax.append(argmax_id)

        # Always seed next_token from the most recent argmax so the
        # first generation step uses the prediction from the last
        # prefill step (positions are 0-indexed, so argmax at pos N
        # predicts the token at pos N+1).
        next_token = argmax_id
        if not is_prefill:
            tok_str = tokenizer.id_to_token(argmax_id) or f"<id {argmax_id}>"
            all_decoded.append(tok_str)
            print(f"  step {step:3d} (gen)    pos={position:3d}  "
                  f"in={token_in:6d}  argmax={argmax_id:6d}  "
                  f"tok={tok_str!r}  {step_latency_ms[-1]:.1f} ms")
        else:
            print(f"  step {step:3d} (prefill) pos={position:3d}  "
                  f"in={token_in:6d}  (next argmax preview={argmax_id:6d})  "
                  f"{step_latency_ms[-1]:.1f} ms")

    # Concatenate outputs and save.
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.out).with_suffix(".npz")
    md_path = Path(args.out).with_suffix(".md")

    np.savez_compressed(
        npz_path,
        logits_uint16=np.stack(all_logits_uint16),  # [steps, vocab]
        logits_fp32=np.stack(all_logits_fp32),
        step_tokens=np.array(all_step_tokens, dtype=np.int64),
        argmax_tokens=np.array(all_argmax, dtype=np.int64),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        step_latency_ms=np.array(step_latency_ms),
        logits_scale=np.array(logits_scale),
        logits_offset=np.array(logits_offset),
    )

    decoded_text = tokenizer.decode(all_argmax[len(prompt_ids):])
    md_path.write_text(
        "# Qualcomm Qwen3-4B w4a16 oracle trace\n\n"
        f"- bundle: `{BUNDLE_DIR.relative_to(REPO_ROOT)}`\n"
        f"- mode: AR1 / CL512, greedy argmax\n"
        f"- prompt: {len(prompt_ids)} tokens, generation steps: {args.gen_steps}\n"
        f"- prefill mean latency: {np.mean(step_latency_ms[:len(prompt_ids)]):.1f} ms/step\n"
        f"- generation mean latency: "
        f"{np.mean(step_latency_ms[len(prompt_ids):]):.1f} ms/step\n"
        f"- logits quant: scale={logits_scale}, offset={logits_offset}\n\n"
        "## Generated tokens\n\n"
        + "\n".join(
            f"{i}. id={tid}  tok={tokenizer.id_to_token(tid)!r}"
            for i, tid in enumerate(all_argmax[len(prompt_ids):])
        )
        + f"\n\n## Decoded continuation\n\n```\n{decoded_text}\n```\n"
        + f"\n## Saved\n\n- {npz_path}\n- {md_path}\n",
        encoding="utf-8",
    )

    print(f"\n--- saved ---")
    print(f"  {npz_path}")
    print(f"  {md_path}")
    print(f"\ngeneration text:\n{decoded_text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
