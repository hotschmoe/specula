"""Qwen3-4B NPU bench via ORT-QNN (chained 4-partition, AR1/CL512).

Companion to `scripts/bench_qwen3_4b_all_backends.py`. Where Genie drives
the Qualcomm-shipped w4a16 bundle through the vendor runtime, this
script drives the *same* binary through **our** runtime stack
(ORT-QNN 1.24.4 + QAIRT 2.42 context binaries) — the same stack our
speculative-decode sidecar speaks. The gap between this script's
numbers and Genie's tells us how much performance we'd inherit if we
built our own heterogeneous inference engine today instead of going
through the vendor-closed Genie CLI.

Scope:
  * **AR128 prefill.** Prompt tokens are processed in 128-wide batches
    using the bundle's `ar128_cl512_*_of_4` graphs. Tail tokens (when
    `pp_tokens` is not a multiple of 128) fall back to AR1. AR128 is
    apples-to-apples comparable to Genie's PP — same graphs, same
    silicon, only the host-side dispatch and KV plumbing differ.
  * **AR1 decode.** Same `ar1_cl512_*_of_4` graphs as before. Per-step
    NPU work is identical to Genie; any delta is ORT-QNN dispatch
    overhead vs Genie's native glue.
  * **CL=512 fixed.** The oracle script also uses cl512. With AR128
    prefill, CL=512 caps total context at 512 (prefill + decode in
    contiguous KV slots). We feed 256 prompt tokens and generate 128
    → 384 slots used, under the cap.

Reuses machinery from npu_engine/qualcomm_qwen3_4b_oracle.py. The oracle
is kept as the single source of truth for the chain; this script is a
stripped-down bench wrapper that skips the per-step prints + NPZ dump
and adds PP/TG separation + optional battery J/tok sampling.

Usage:
    .venv/Scripts/python.exe npu_engine/bench_qwen3_4b_ortqnn.py \\
        --power-state {ac,bat} --tag YYYY-MM-DD_state

Outputs:
    results/csv/qwen3_4b_ortqnn_<tag>.csv
    marked_for_deletion/qwen3_4b_ortqnn_<tag>/stdout.log  (redirected)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Windows cp1252 default stdout can't encode Qwen's "Ġ" space marker.
# Force UTF-8 so tokenizer.id_to_token() output renders without crashing
# the whole bench before the CSV is written.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import onnxruntime as ort
import yaml
from tokenizers import Tokenizer

# Same-package import — Python prepends the script's directory to
# sys.path so `qualcomm_qwen3_4b_oracle` resolves without explicit
# manipulation when run as `python npu_engine/bench_qwen3_4b_ortqnn.py`.
from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    AR128_BATCH,
    BUNDLE_DIR,
    CTX_LEN,
    HIDDEN_DIM,
    LAYERS_PER_PART,
    NUM_LAYERS,
    PAST_LEN,
    PAST_LEN_AR128_CL512,
    KVStore,
    attention_mask_quantized,
    attention_mask_quantized_ar128,
    build_part_cfg,
    build_wrapper,
    dequant_uint16,
    half_dim_rope_quantized,
    half_dim_rope_quantized_ar128,
    load_session,
)

# Battery helpers live in the cross-backend driver under scripts/.
# Add scripts/ to sys.path so this import resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from bench_qwen3_4b_all_backends import (  # noqa: E402
    PowerSampler,
    sample_battery_mwh,
    sample_power_online,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "results" / "qwen3_4b_baseline" / "pp512_prompt.txt"
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"

HIDDEN_FROM_PART1 = "_model_model_embed_tokens_Gather_output_0"
HIDDEN_FROM_PART2 = "_model_model_layers_11_Add_1_output_0"
HIDDEN_FROM_PART3 = "_model_model_layers_23_Add_1_output_0"


def layer_input_name(part_idx: int, kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_in"


def layer_output_name(part_idx: int, kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_out"


_DTYPE_NUMPY = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "float32": np.float32,
}


def make_bound_chain(sessions, parts_cfg):
    """Build per-session ORT IOBinding objects + pre-allocated output
    numpy buffers. Output buffers are bound once and reused across all
    calls to that session — eliminates the per-step output-tensor
    allocation that vanilla `sess.run()` does on every call.

    Returns (bindings, out_bufs):
      bindings: dict[part_idx -> ort.IOBinding]
      out_bufs: dict[part_idx -> dict[output_name -> np.ndarray]]
    """
    bindings: dict[int, ort.IOBinding] = {}
    out_bufs: dict[int, dict[str, np.ndarray]] = {}
    for part_idx in (1, 2, 3, 4):
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


def _step(sessions, bindings, out_bufs, kv, position, token_in, scales):
    """One AR1 forward pass through all 4 partitions at `position` with
    `token_in`. Uses pre-bound IOBinding for outputs; inputs use
    bind_cpu_input each call. Stitches KV. Returns (logits_fp32, wall_ms).

    Inputs are bound from numpy buffers. KV inputs come straight from
    the persistent KVStore master buffers (zero-copy on read since the
    AR1 graph wants exactly the [8,1,128,511] / [8,1,511,128] shape).
    Outputs land in out_bufs[part]; the stitch step COPIES from there
    into the persistent KV before the next call reuses out_bufs.
    """
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
    mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

    t0 = time.perf_counter()

    # part 1 — single int32 input, embedding output
    b = bindings[1]
    b.clear_binding_inputs()
    b.bind_cpu_input("input_ids", np.array([[token_in]], dtype=np.int32))
    sessions[1].run_with_iobinding(b)
    emb_out = out_bufs[1][HIDDEN_FROM_PART1]

    # part 2 — layers 0..11
    b = bindings[2]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART1, emb_out)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(0, LAYERS_PER_PART):
        b.bind_cpu_input(layer_input_name(2, "key", layer), kv.keys[layer])
        b.bind_cpu_input(layer_input_name(2, "value", layer), kv.values[layer])
    sessions[2].run_with_iobinding(b)
    hidden_after_p2 = out_bufs[2][HIDDEN_FROM_PART2]
    new_keys_p2 = [out_bufs[2][layer_output_name(2, "key", layer)]
                   for layer in range(0, LAYERS_PER_PART)]
    new_vals_p2 = [out_bufs[2][layer_output_name(2, "value", layer)]
                   for layer in range(0, LAYERS_PER_PART)]

    # part 3 — layers 12..23
    b = bindings[3]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART2, hidden_after_p2)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
        b.bind_cpu_input(layer_input_name(3, "key", layer), kv.keys[layer])
        b.bind_cpu_input(layer_input_name(3, "value", layer), kv.values[layer])
    sessions[3].run_with_iobinding(b)
    hidden_after_p3 = out_bufs[3][HIDDEN_FROM_PART3]
    new_keys_p3 = [out_bufs[3][layer_output_name(3, "key", layer)]
                   for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)]
    new_vals_p3 = [out_bufs[3][layer_output_name(3, "value", layer)]
                   for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)]

    # part 4 — layers 24..35 + lm_head
    b = bindings[4]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART3, hidden_after_p3)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS):
        b.bind_cpu_input(layer_input_name(4, "key", layer), kv.keys[layer])
        b.bind_cpu_input(layer_input_name(4, "value", layer), kv.values[layer])
    sessions[4].run_with_iobinding(b)
    logits_uint16 = out_bufs[4]["logits"]
    new_keys_p4 = [out_bufs[4][layer_output_name(4, "key", layer)]
                   for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)]
    new_vals_p4 = [out_bufs[4][layer_output_name(4, "value", layer)]
                   for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)]

    # Stitch must happen BEFORE the next step reuses out_bufs.
    kv.stitch_step(
        new_keys_p2 + new_keys_p3 + new_keys_p4,
        new_vals_p2 + new_vals_p3 + new_vals_p4,
    )

    wall_ms = (time.perf_counter() - t0) * 1000
    logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
    return logits_fp32, wall_ms


def _step_ar128(sessions, bindings, out_bufs, kv, p_base, token_batch, scales):
    """One AR128 forward pass through all 4 partitions for a 128-wide
    query batch starting at absolute position `p_base`.

    Uses pre-bound IOBinding for outputs and binds KV inputs from the
    persistent `kv.keys_ar128_in` / `values_ar128_in` buffers (which
    `KVStore.stitch_batch` keeps in sync with the master). That removes
    the per-call ~28 MB ascontiguousarray copy that slicing a 384-slot
    prefix from the 511-slot master buffer would cost.
    """
    if not kv.has_ar128_in:
        raise RuntimeError(
            "_step_ar128 requires KVStore(with_ar128_input=True)"
        )
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized_ar128(p_base, cos_scale, cos_offset)
    mask_q = attention_mask_quantized_ar128(p_base, mask_scale, mask_offset)

    t0 = time.perf_counter()

    # part 1
    b = bindings[1]
    b.clear_binding_inputs()
    b.bind_cpu_input(
        "input_ids",
        np.asarray(token_batch, dtype=np.int32).reshape(1, AR128_BATCH),
    )
    sessions[1].run_with_iobinding(b)
    emb_out = out_bufs[1][HIDDEN_FROM_PART1]

    # part 2 — KV inputs zero-copy from persistent AR128 buffer
    b = bindings[2]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART1, emb_out)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(0, LAYERS_PER_PART):
        b.bind_cpu_input(layer_input_name(2, "key", layer), kv.keys_ar128_in[layer])
        b.bind_cpu_input(layer_input_name(2, "value", layer), kv.values_ar128_in[layer])
    sessions[2].run_with_iobinding(b)
    hidden_after_p2 = out_bufs[2][HIDDEN_FROM_PART2]
    new_keys_p2 = [out_bufs[2][layer_output_name(2, "key", layer)]
                   for layer in range(0, LAYERS_PER_PART)]
    new_vals_p2 = [out_bufs[2][layer_output_name(2, "value", layer)]
                   for layer in range(0, LAYERS_PER_PART)]

    # part 3
    b = bindings[3]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART2, hidden_after_p2)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
        b.bind_cpu_input(layer_input_name(3, "key", layer), kv.keys_ar128_in[layer])
        b.bind_cpu_input(layer_input_name(3, "value", layer), kv.values_ar128_in[layer])
    sessions[3].run_with_iobinding(b)
    hidden_after_p3 = out_bufs[3][HIDDEN_FROM_PART3]
    new_keys_p3 = [out_bufs[3][layer_output_name(3, "key", layer)]
                   for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)]
    new_vals_p3 = [out_bufs[3][layer_output_name(3, "value", layer)]
                   for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)]

    # part 4
    b = bindings[4]
    b.clear_binding_inputs()
    b.bind_cpu_input(HIDDEN_FROM_PART3, hidden_after_p3)
    b.bind_cpu_input("attention_mask", mask_q)
    b.bind_cpu_input("position_ids_cos", cos_q)
    b.bind_cpu_input("position_ids_sin", sin_q)
    for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS):
        b.bind_cpu_input(layer_input_name(4, "key", layer), kv.keys_ar128_in[layer])
        b.bind_cpu_input(layer_input_name(4, "value", layer), kv.values_ar128_in[layer])
    sessions[4].run_with_iobinding(b)
    logits_uint16_batch = out_bufs[4]["logits"]  # [1, 128, vocab]
    new_keys_p4 = [out_bufs[4][layer_output_name(4, "key", layer)]
                   for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)]
    new_vals_p4 = [out_bufs[4][layer_output_name(4, "value", layer)]
                   for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)]

    kv.stitch_batch(
        p_base,
        new_keys_p2 + new_keys_p3 + new_keys_p4,
        new_vals_p2 + new_vals_p3 + new_vals_p4,
    )

    wall_ms = (time.perf_counter() - t0) * 1000
    last_logits_uint16 = logits_uint16_batch[0, -1, :]  # last query position
    last_logits_fp32 = dequant_uint16(last_logits_uint16, logits_scale, logits_offset)
    return last_logits_fp32, wall_ms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--power-state", choices=("ac", "bat"), required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--pp-tokens", type=int, default=256,
                   help="# of prompt tokens for PP measurement (CL=512 caps prefill+decode at 511 KV slots total, so 256+128 fits)")
    p.add_argument("--tg-tokens", type=int, default=128)
    p.add_argument("--skip-power-check", action="store_true")
    args = p.parse_args()

    online = sample_power_online()
    if online is None:
        print("WARNING: PowerOnline WMI unavailable; skipping check")
    elif args.power_state == "ac" and not online and not args.skip_power_check:
        print("ERROR: --power-state ac but WMI says PowerOnline=False")
        return 2
    elif args.power_state == "bat" and online and not args.skip_power_check:
        print("ERROR: --power-state bat but WMI says PowerOnline=True")
        return 2

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    trash_dir = TRASH_ROOT / f"qwen3_4b_ortqnn_{tag}"
    trash_dir.mkdir(parents=True, exist_ok=True)
    log_path = trash_dir / "stdout.log"

    if args.pp_tokens + args.tg_tokens > PAST_LEN:
        print(f"ERROR: pp_tokens + tg_tokens = {args.pp_tokens + args.tg_tokens} "
              f"exceeds CL-512 cache capacity {PAST_LEN}")
        return 2

    print(f"=== Qwen3-4B ORT-QNN bench (AR128 prefill + AR1 decode, CL{CTX_LEN}, chained 4-part) ===")
    print(f"tag            : {tag}")
    print(f"power state    : {args.power_state}")
    print(f"pp tokens      : {args.pp_tokens}")
    print(f"tg tokens      : {args.tg_tokens}")
    print(f"prompt         : {PROMPT_PATH}")
    print(f"log (trash)    : {log_path}")

    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts_cfg_ar1 = build_part_cfg(metadata, ar=1)
    parts_cfg_ar128 = build_part_cfg(metadata, ar=AR128_BATCH)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(prompt_text).ids[: args.pp_tokens]
    print(f"prompt tokenized to {len(prompt_ids)} (target {args.pp_tokens})")

    n_ar128_calls = len(prompt_ids) // AR128_BATCH
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH

    print("\n--- loading sessions: 4 AR1 + 4 AR128 ---")
    sessions_ar1: dict[int, ort.InferenceSession] = {}
    sessions_ar128: dict[int, ort.InferenceSession] = {}
    t_load = time.perf_counter()
    for part_idx in (1, 2, 3, 4):
        # Skip rebuild if a wrapper already exists. build_wrapper is
        # deterministic given the same parts_cfg from metadata.yaml, so
        # an existing file is identical to what we'd produce. Skipping
        # also avoids a write race when multiple bench processes are
        # spawned concurrently against the same bundle.
        ar1_path = BUNDLE_DIR / f"oracle_part{part_idx}.wrapper.onnx"
        if not ar1_path.exists():
            build_wrapper(parts_cfg_ar1[part_idx], ar1_path)
        sessions_ar1[part_idx] = load_session(ar1_path)

        ar128_path = BUNDLE_DIR / f"oracle_part{part_idx}_ar128.wrapper.onnx"
        if not ar128_path.exists():
            build_wrapper(parts_cfg_ar128[part_idx], ar128_path)
        sessions_ar128[part_idx] = load_session(ar128_path)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f} s")

    # Quant scales used every step. AR1 and AR128 happen to share the
    # same cos/sin/mask scales on this bundle, but extract from each
    # config explicitly so this script doesn't break if a future bundle
    # diverges.
    def _io(part_cfg, key, side="inputs"):
        return next(io for io in part_cfg[2][side] if io["name"] == key)

    scales_ar1 = (
        _io(parts_cfg_ar1, "position_ids_cos")["scale"],
        _io(parts_cfg_ar1, "position_ids_cos")["offset"],
        _io(parts_cfg_ar1, "attention_mask")["scale"],
        _io(parts_cfg_ar1, "attention_mask")["offset"],
        next(io for io in parts_cfg_ar1[4]["outputs"] if io["name"] == "logits")["scale"],
        next(io for io in parts_cfg_ar1[4]["outputs"] if io["name"] == "logits")["offset"],
    )
    scales_ar128 = (
        _io(parts_cfg_ar128, "position_ids_cos")["scale"],
        _io(parts_cfg_ar128, "position_ids_cos")["offset"],
        _io(parts_cfg_ar128, "attention_mask")["scale"],
        _io(parts_cfg_ar128, "attention_mask")["offset"],
        next(io for io in parts_cfg_ar128[4]["outputs"] if io["name"] == "logits")["scale"],
        next(io for io in parts_cfg_ar128[4]["outputs"] if io["name"] == "logits")["offset"],
    )

    # IOBinding setup: pre-allocate output buffers + bind once per
    # session. Inputs are bound per call via bind_cpu_input. Eliminates
    # the per-step output-tensor allocation that vanilla sess.run does.
    print("\n--- building IOBinding for AR1 + AR128 chains ---")
    bindings_ar1, out_bufs_ar1 = make_bound_chain(sessions_ar1, parts_cfg_ar1)
    bindings_ar128, out_bufs_ar128 = make_bound_chain(sessions_ar128, parts_cfg_ar128)

    # Power sampling (battery only). 2 s interval for parity with the
    # llama.cpp bench driver.
    sampler = PowerSampler(interval_s=2.0) if args.power_state == "bat" else None
    mwh_before = sample_battery_mwh() if args.power_state == "bat" else None

    # AR128 path needs the matching-shape KV input buffer; the AR1 path
    # reads kv.keys / kv.values directly.
    kv = KVStore(NUM_LAYERS, with_ar128_input=(n_ar128_calls > 0))

    if sampler:
        sampler.start()

    # Warmup: throw-away calls to absorb HMX context-init for both AR1
    # and AR128 graphs. The first call in each AR mode pays a ~1s
    # init cost that would otherwise dominate the short PP/TG windows.
    print("\n--- warmup (1 AR1 + 1 AR128 step, discarded) ---")
    if n_ar128_calls > 0:
        warmup_batch = list(prompt_ids[:AR128_BATCH])
        _, _ = _step_ar128(
            sessions_ar128, bindings_ar128, out_bufs_ar128,
            kv, 0, warmup_batch, scales_ar128,
        )
        kv = KVStore(NUM_LAYERS, with_ar128_input=True)
    _, _ = _step(
        sessions_ar1, bindings_ar1, out_bufs_ar1,
        kv, 0, prompt_ids[0], scales_ar1,
    )
    kv = KVStore(NUM_LAYERS, with_ar128_input=(n_ar128_calls > 0))

    # Prefill: AR128 batches first, then AR1 for any tail tokens.
    print(
        f"\n--- prefill: {n_ar128_calls} AR128 calls"
        + (f" + {n_ar1_tail} AR1 steps" if n_ar1_tail else "")
        + f" = {len(prompt_ids)} tokens ---"
    )
    t_pp_start = time.perf_counter()
    pp_ar128_latencies_ms: list[float] = []
    pp_ar1_latencies_ms: list[float] = []
    last_logits = None
    p = 0
    for call_idx in range(n_ar128_calls):
        batch = list(prompt_ids[p : p + AR128_BATCH])
        last_logits, ms = _step_ar128(
            sessions_ar128, bindings_ar128, out_bufs_ar128,
            kv, p, batch, scales_ar128,
        )
        pp_ar128_latencies_ms.append(ms)
        print(
            f"  pp ar128 call {call_idx} (positions {p}..{p + AR128_BATCH - 1})  "
            f"{ms:.1f} ms  ({AR128_BATCH * 1000 / ms:.1f} t/s in-call)"
        )
        p += AR128_BATCH
    while p < len(prompt_ids):
        last_logits, ms = _step(
            sessions_ar1, bindings_ar1, out_bufs_ar1,
            kv, p, prompt_ids[p], scales_ar1,
        )
        pp_ar1_latencies_ms.append(ms)
        if (p - n_ar128_calls * AR128_BATCH) % 32 == 0 or p == len(prompt_ids) - 1:
            print(f"  pp ar1 step {p}  {ms:.1f} ms")
        p += 1
    pp_wall_s = time.perf_counter() - t_pp_start
    pp_tps = len(prompt_ids) / pp_wall_s
    pp_ar128_median_ms = (
        float(np.median(pp_ar128_latencies_ms)) if pp_ar128_latencies_ms else 0.0
    )
    pp_ar1_median_ms = (
        float(np.median(pp_ar1_latencies_ms)) if pp_ar1_latencies_ms else 0.0
    )
    pp_median_ms = pp_ar128_median_ms if pp_ar128_latencies_ms else pp_ar1_median_ms
    print(
        f"  PP total {pp_wall_s:.2f} s  -> {pp_tps:.2f} t/s  "
        f"(ar128 median {pp_ar128_median_ms:.1f} ms; ar1-tail median {pp_ar1_median_ms:.1f} ms)"
    )

    # Decode: generate `tg_tokens` with greedy argmax from last logits.
    print(f"\n--- decode: {args.tg_tokens} AR1 steps (greedy) ---")
    t_tg_start = time.perf_counter()
    tg_latencies_ms: list[float] = []
    next_token = int(np.argmax(last_logits))
    gen_ids: list[int] = []
    for i in range(args.tg_tokens):
        position = len(prompt_ids) + i
        logits_fp32, ms = _step(
            sessions_ar1, bindings_ar1, out_bufs_ar1,
            kv, position, next_token, scales_ar1,
        )
        tg_latencies_ms.append(ms)
        gen_ids.append(next_token)
        next_token = int(np.argmax(logits_fp32))
        if i % 16 == 0 or i == args.tg_tokens - 1:
            print(f"  tg step {i:3d}  {ms:.1f} ms  (median so far {np.median(tg_latencies_ms):.1f} ms)")
    tg_wall_s = time.perf_counter() - t_tg_start
    tg_tps = args.tg_tokens / tg_wall_s
    tg_median_ms = float(np.median(tg_latencies_ms))

    if sampler:
        sampler.stop()
    mwh_after = sample_battery_mwh() if args.power_state == "bat" else None

    print(f"\n  TG total {tg_wall_s:.2f} s  -> {tg_tps:.2f} t/s  (median step {tg_median_ms:.1f} ms)")

    total_tokens = len(prompt_ids) + args.tg_tokens
    total_wall_s = pp_wall_s + tg_wall_s
    mean_w = sampler.mean_watts if sampler else None
    mwh_drop = None
    if mwh_before is not None and mwh_after is not None:
        mwh_drop = mwh_before - mwh_after
    energy_j = mean_w * total_wall_s if mean_w is not None else (mwh_drop * 3.6 if mwh_drop else None)
    j_per_tok = energy_j / total_tokens if energy_j is not None and total_tokens > 0 else None

    print(f"\n  mean W         : {mean_w}")
    print(f"  mWh drop       : {mwh_drop}")
    print(f"  J/tok (total)  : {j_per_tok}")

    pp_mode = (
        "ar128+ar1tail" if n_ar128_calls and n_ar1_tail
        else "ar128" if n_ar128_calls
        else "ar1"
    )
    row = dict(
        backend="npu-ortqnn",
        pp_mode=pp_mode,
        pp_tokens=len(prompt_ids),
        pp_ar128_calls=n_ar128_calls,
        pp_ar1_steps=n_ar1_tail,
        tg_tokens=args.tg_tokens,
        pp_wall_s=pp_wall_s,
        tg_wall_s=tg_wall_s,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        pp_median_ms=pp_median_ms,
        pp_ar128_median_ms=pp_ar128_median_ms,
        pp_ar1_median_ms=pp_ar1_median_ms,
        tg_median_ms=tg_median_ms,
        load_s=load_s,
        mwh_before=mwh_before,
        mwh_after=mwh_after,
        mwh_drop=mwh_drop,
        mean_w=mean_w,
        j_per_tok=j_per_tok,
        power_state=args.power_state,
        tag=tag,
        ctx_tier=CTX_LEN,
        note="AR128 prefill + AR1 decode + IOBinding; same binary as Genie via ORT-QNN 1.24.4",
    )

    csv_path = CSV_DIR / f"qwen3_4b_ortqnn_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"\n=== Summary ===")
    print(f"  PP     : {pp_tps:.2f} t/s  ({pp_mode}; Genie AR128 baseline is ~1598 t/s)")
    print(f"  TG-AR1 : {tg_tps:.2f} t/s  (Genie AR1 baseline ~23.3 t/s on same binary)")
    print(f"  J/tok  : {j_per_tok}")
    print(f"  csv    : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
