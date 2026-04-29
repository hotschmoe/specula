"""Qwen2.5-7B NPU chain runner via ORT-QNN (chained 6-partition).

Companion to bench_qwen3_4b_ortqnn.py — same shape, different model.
The sidecar / http_server import the chain functions
(_step, _step_ar128, make_bound_chain, CTX_TIERS, PROMPT_PATH) from
this module when the user selects --model qwen2_5-7b.

This file is intentionally a *runtime-only* counterpart — the standalone
bench main() has been omitted because cross-backend benchmarking for
the 7B is already covered by `scripts/bench_qwen2_5_7b_all_backends.py`
(which drives Genie directly). What we need here is the IOBinding-
optimized chain runner that the sidecar speaks.

Bundle deltas vs 4B (see qualcomm_qwen2_5_7b_oracle.py docstring for
the full table):
  - 6 partitions (1 embed + 5 transformer; last includes lm_head)
  - 5 transformer parts: parts 2..6
  - layer split: 6/6/6/6/4
  - num_kv_heads = 4 (vs 8 for 4B)
  - hidden_dim = 3584 (vs 2560)
  - vocab = 152064
  - single ctx tier: cl=4096
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Same-package import for the model module.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from qualcomm_qwen2_5_7b_oracle import (  # noqa: E402
    AR128_BATCH,
    BUNDLE_DIR,
    CTX_LEN,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_PARTS,
    PAST_LEN,
    PAST_LEN_AR128,
    PART_HIDDEN_IN,
    PART_HIDDEN_OUT,
    PART_LAYER_RANGES,
    QAIRT_2_45_BACKEND,
    HIDDEN_FROM_PART1,
    KVStore,
    attention_mask_quantized,
    attention_mask_quantized_ar128,
    build_part_cfg,
    dequant_uint16,
    half_dim_rope_quantized,
    half_dim_rope_quantized_ar128,
    layer_input_name,
    layer_output_name,
    wrapper_path,
)
from qualcomm_qwen3_4b_oracle import build_wrapper, load_session  # noqa: E402


# Bundle ships only this tier. A tuple is kept for API parity with the
# 4B bench (which has CTX_TIERS = (512, 1024, 2048, 3072, 4096)).
CTX_TIERS = (CTX_LEN,)

REPO_ROOT = _HERE.parent
# 7B baseline driver writes a 512-token prompt to this path.
PROMPT_PATH = REPO_ROOT / "results" / "qwen2_5_7b_baseline" / "pp512_prompt.txt"


_DTYPE_NUMPY = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "float32": np.float32,
}


def make_bound_chain(sessions, parts_cfg):
    """Per-session ORT IOBinding + pre-allocated output buffers, for all
    NUM_PARTS partitions. Output buffers bound once and reused across
    every call to that session — no per-step output allocation."""
    bindings: dict[int, ort.IOBinding] = {}
    out_bufs: dict[int, dict[str, np.ndarray]] = {}
    for part_idx in range(1, NUM_PARTS + 1):
        sess = sessions[part_idx]
        binding = sess.io_binding()
        out_bufs[part_idx] = {}
        for io in parts_cfg[part_idx]["outputs"]:
            np_dtype = _DTYPE_NUMPY[io["dtype"]]
            arr = np.empty(tuple(io["shape"]), dtype=np_dtype)
            out_bufs[part_idx][io["name"]] = arr
            binding.bind_output(
                name=io["name"],
                device_type="cpu",
                device_id=0,
                element_type=np_dtype,
                shape=arr.shape,
                buffer_ptr=arr.ctypes.data,
            )
        bindings[part_idx] = binding
    return bindings, out_bufs


def _run_transformer_part(
    session, binding, out_bufs_part, *,
    part_idx, hidden_in, mask_q, cos_q, sin_q,
    kv_keys_src, kv_values_src,
):
    """Bind inputs for one transformer partition (2..NUM_PARTS) and run.

    Returns (new_keys, new_vals) — references into the part's pre-bound
    output buffers (must be consumed/copied before the next call to the
    same session reuses them).

    `kv_keys_src` / `kv_values_src` are full NUM_LAYERS-entry per-layer
    lists; only the slice owned by `part_idx` is read.
    """
    layer_lo, layer_hi = PART_LAYER_RANGES[part_idx]
    binding.clear_binding_inputs()
    binding.bind_cpu_input(PART_HIDDEN_IN[part_idx], hidden_in)
    binding.bind_cpu_input("attention_mask", mask_q)
    binding.bind_cpu_input("position_ids_cos", cos_q)
    binding.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(layer_lo, layer_hi):
        binding.bind_cpu_input(layer_input_name("key", layer), kv_keys_src[layer])
        binding.bind_cpu_input(layer_input_name("value", layer), kv_values_src[layer])
    session.run_with_iobinding(binding)
    new_keys = [out_bufs_part[layer_output_name("key", layer)]
                for layer in range(layer_lo, layer_hi)]
    new_vals = [out_bufs_part[layer_output_name("value", layer)]
                for layer in range(layer_lo, layer_hi)]
    return new_keys, new_vals


def _step(sessions, bindings, out_bufs, kv, position, token_in, scales):
    """One AR1 forward pass through all NUM_PARTS partitions."""
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
    mask_q = attention_mask_quantized(position, mask_scale, mask_offset, ctx_len=kv.ctx_len)

    t0 = time.perf_counter()

    # Part 1 — input_ids -> embedding
    b1 = bindings[1]
    b1.clear_binding_inputs()
    b1.bind_cpu_input("input_ids", np.array([[token_in]], dtype=np.int32))
    sessions[1].run_with_iobinding(b1)
    hidden = out_bufs[1][HIDDEN_FROM_PART1]

    # Parts 2..NUM_PARTS — transformer layers (and lm_head on last)
    new_keys: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    for part_idx in range(2, NUM_PARTS + 1):
        keys_p, vals_p = _run_transformer_part(
            sessions[part_idx], bindings[part_idx], out_bufs[part_idx],
            part_idx=part_idx, hidden_in=hidden,
            mask_q=mask_q, cos_q=cos_q, sin_q=sin_q,
            kv_keys_src=kv.keys, kv_values_src=kv.values,
        )
        new_keys.extend(keys_p)
        new_vals.extend(vals_p)
        if part_idx in PART_HIDDEN_OUT:
            hidden = out_bufs[part_idx][PART_HIDDEN_OUT[part_idx]]
    logits_uint16 = out_bufs[NUM_PARTS]["logits"]

    # Stitch must happen BEFORE the next step reuses out_bufs.
    kv.stitch_step(new_keys, new_vals)

    wall_ms = (time.perf_counter() - t0) * 1000
    logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
    return logits_fp32, wall_ms


def _step_ar128(sessions, bindings, out_bufs, kv, p_base, token_batch, scales):
    """One AR128 forward pass (128-wide query batch starting at p_base)."""
    if not kv.has_ar128_in:
        raise RuntimeError("_step_ar128 requires KVStore(with_ar128_input=True)")
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized_ar128(p_base, cos_scale, cos_offset)
    mask_q = attention_mask_quantized_ar128(p_base, mask_scale, mask_offset, ctx_len=kv.ctx_len)

    t0 = time.perf_counter()

    b1 = bindings[1]
    b1.clear_binding_inputs()
    b1.bind_cpu_input(
        "input_ids",
        np.asarray(token_batch, dtype=np.int32).reshape(1, AR128_BATCH),
    )
    sessions[1].run_with_iobinding(b1)
    hidden = out_bufs[1][HIDDEN_FROM_PART1]

    new_keys: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    for part_idx in range(2, NUM_PARTS + 1):
        keys_p, vals_p = _run_transformer_part(
            sessions[part_idx], bindings[part_idx], out_bufs[part_idx],
            part_idx=part_idx, hidden_in=hidden,
            mask_q=mask_q, cos_q=cos_q, sin_q=sin_q,
            kv_keys_src=kv.keys_ar128_in, kv_values_src=kv.values_ar128_in,
        )
        new_keys.extend(keys_p)
        new_vals.extend(vals_p)
        if part_idx in PART_HIDDEN_OUT:
            hidden = out_bufs[part_idx][PART_HIDDEN_OUT[part_idx]]
    logits_uint16_batch = out_bufs[NUM_PARTS]["logits"]  # [1, 128, vocab]

    kv.stitch_batch(p_base, new_keys, new_vals)

    wall_ms = (time.perf_counter() - t0) * 1000
    last_logits_uint16 = logits_uint16_batch[0, -1, :]
    last_logits_fp32 = dequant_uint16(last_logits_uint16, logits_scale, logits_offset)
    return last_logits_fp32, wall_ms
