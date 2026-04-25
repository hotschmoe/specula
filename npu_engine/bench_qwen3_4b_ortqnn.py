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

# Parts 2/3/4 share the same per-layer KV input/output naming scheme.
# Layer indices are global (0..35) regardless of which part owns them.
def layer_input_name(kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_in"


def layer_output_name(kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_out"


# Layer ranges owned by each transformer partition.
PART_LAYER_RANGES = {
    2: (0, LAYERS_PER_PART),
    3: (LAYERS_PER_PART, 2 * LAYERS_PER_PART),
    4: (2 * LAYERS_PER_PART, NUM_LAYERS),
}
# Each transformer part takes its hidden state from the previous part's
# output and (for parts 2/3) emits a hidden state for the next.
PART_HIDDEN_IN = {2: HIDDEN_FROM_PART1, 3: HIDDEN_FROM_PART2, 4: HIDDEN_FROM_PART3}
PART_HIDDEN_OUT = {2: HIDDEN_FROM_PART2, 3: HIDDEN_FROM_PART3}


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


def _run_transformer_part(
    session, binding, out_bufs_part, *,
    part_idx, hidden_in, mask_q, cos_q, sin_q,
    kv_keys_src, kv_values_src,
):
    """Bind inputs for one transformer partition (2/3/4) and run.

    Returns (new_keys, new_vals) — references into the part's pre-bound
    output buffers (must be consumed/copied before the next call to the
    same session reuses them).

    `kv_keys_src` / `kv_values_src` are full 36-entry per-layer lists;
    only the slice owned by `part_idx` is read, but passing the whole
    list keeps the AR1 vs AR128 swap (kv.keys vs kv.keys_ar128_in) at
    the call site clear.
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
    """One AR1 forward pass through all 4 partitions at `position` with
    `token_in`. Uses pre-bound IOBinding for outputs; inputs use
    bind_cpu_input each call. Stitches KV. Returns (logits_fp32, wall_ms).

    KV inputs come straight from the persistent KVStore master buffers
    (zero-copy on read — the AR1 graph wants exactly the master's shape).
    Outputs land in out_bufs[part]; the stitch step COPIES from there
    into the persistent KV before the next call reuses out_bufs.
    """
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
    mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

    t0 = time.perf_counter()

    # part 1 — input_ids -> embedding
    b1 = bindings[1]
    b1.clear_binding_inputs()
    b1.bind_cpu_input("input_ids", np.array([[token_in]], dtype=np.int32))
    sessions[1].run_with_iobinding(b1)
    hidden = out_bufs[1][HIDDEN_FROM_PART1]

    # parts 2/3/4 — transformer layers (and lm_head on part 4)
    new_keys: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    for part_idx in (2, 3, 4):
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
    logits_uint16 = out_bufs[4]["logits"]

    # Stitch must happen BEFORE the next step reuses out_bufs.
    kv.stitch_step(new_keys, new_vals)

    wall_ms = (time.perf_counter() - t0) * 1000
    logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
    return logits_fp32, wall_ms


def _step_ar128(sessions, bindings, out_bufs, kv, p_base, token_batch, scales):
    """One AR128 forward pass through all 4 partitions for a 128-wide
    query batch starting at absolute position `p_base`.

    KV inputs are zero-copy from `kv.keys_ar128_in` / `values_ar128_in`,
    the AR128-shaped mirror buffers `KVStore.stitch_batch` keeps in sync
    with the master. That removes the per-call ~28 MB ascontiguousarray
    copy that slicing a 384-slot prefix from the 511-slot master would
    cost.
    """
    if not kv.has_ar128_in:
        raise RuntimeError("_step_ar128 requires KVStore(with_ar128_input=True)")
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized_ar128(p_base, cos_scale, cos_offset)
    mask_q = attention_mask_quantized_ar128(p_base, mask_scale, mask_offset)

    t0 = time.perf_counter()

    # part 1 — input_ids batch -> embedding batch
    b1 = bindings[1]
    b1.clear_binding_inputs()
    b1.bind_cpu_input(
        "input_ids",
        np.asarray(token_batch, dtype=np.int32).reshape(1, AR128_BATCH),
    )
    sessions[1].run_with_iobinding(b1)
    hidden = out_bufs[1][HIDDEN_FROM_PART1]

    # parts 2/3/4 — transformer layers (and lm_head on part 4)
    new_keys: list[np.ndarray] = []
    new_vals: list[np.ndarray] = []
    for part_idx in (2, 3, 4):
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
    logits_uint16_batch = out_bufs[4]["logits"]  # [1, 128, vocab]

    kv.stitch_batch(p_base, new_keys, new_vals)

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
    p.add_argument("--no-ar128", action="store_true",
                   help="skip the AR128 prefill chain unconditionally. "
                        "Hard escape hatch / AR1-only baseline.")
    p.add_argument("--ar128-min-tokens", type=int, default=512,
                   help="prompt-token threshold for taking the AR128 swap "
                        "path (vLLM-style request-size routing). Below the "
                        "threshold the bench runs AR1-only — the ~36 s swap "
                        "would dominate end-to-end latency. The default 512 "
                        "is just below the empirical crossover (~559 tokens) "
                        "where AR128-with-swap starts winning end-to-end on "
                        "Qwen3-4B + this hardware.")
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

    want_ar128 = (
        not args.no_ar128
        and len(prompt_ids) >= AR128_BATCH
        and len(prompt_ids) >= args.ar128_min_tokens
    )
    n_ar128_calls = (len(prompt_ids) // AR128_BATCH) if want_ar128 else 0
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH

    # One-line routing summary for the log so the policy is visible.
    if args.no_ar128:
        route = "AR1-only (--no-ar128 forced)"
    elif len(prompt_ids) < AR128_BATCH:
        route = f"AR1-only (prompt {len(prompt_ids)} < AR128 batch {AR128_BATCH})"
    elif len(prompt_ids) < args.ar128_min_tokens:
        route = (f"AR1-only (prompt {len(prompt_ids)} < threshold "
                 f"{args.ar128_min_tokens}; AR1 prefill is faster end-to-end "
                 f"than AR128-with-swap below ~559 tokens)")
    else:
        route = (f"AR128 swap (prompt {len(prompt_ids)} >= threshold "
                 f"{args.ar128_min_tokens})")
    print(f"route          : {route}")

    # Quant scales used every step. AR1 and AR128 happen to share the
    # same cos/sin/mask scales on this bundle, but extract from each
    # config explicitly so this script doesn't break if a future bundle
    # diverges.
    def _scales_tuple(part_cfg):
        def find(side, part_idx, name):
            return next(io for io in part_cfg[part_idx][side] if io["name"] == name)
        cos = find("inputs", 2, "position_ids_cos")
        mask = find("inputs", 2, "attention_mask")
        logits = find("outputs", 4, "logits")
        return (
            cos["scale"], cos["offset"],
            mask["scale"], mask["offset"],
            logits["scale"], logits["offset"],
        )

    scales_ar1 = _scales_tuple(parts_cfg_ar1)
    scales_ar128 = _scales_tuple(parts_cfg_ar128) if want_ar128 else None

    def _load_chain(parts_cfg, suffix, label):
        """Load 4 ORT-QNN sessions for the parts in parts_cfg. Wrappers
        are built lazily and reused across runs.

        Per-partition timing is logged so we can see whether load cost
        scales with .bin size (~I/O dominated) or is roughly fixed per
        partition (~HTP context init dominated). Sizes for reference:
        part 1 ~742 MB, part 2/3 ~637 MB, part 4 ~1020 MB.
        """
        sessions = {}
        per_part_s: list[float] = []
        for part_idx in (1, 2, 3, 4):
            wrapper = BUNDLE_DIR / f"oracle_part{part_idx}{suffix}.wrapper.onnx"
            if not wrapper.exists():
                build_wrapper(parts_cfg[part_idx], wrapper)
            bin_path = BUNDLE_DIR / parts_cfg[part_idx]["bin"]
            bin_mb = bin_path.stat().st_size / 1024 / 1024 if bin_path.exists() else 0
            t = time.perf_counter()
            sessions[part_idx] = load_session(wrapper)
            dt = time.perf_counter() - t
            per_part_s.append(dt)
            mb_per_s = bin_mb / dt if dt > 0 else 0
            print(f"    {label} part {part_idx}: {dt:.1f} s  "
                  f"({bin_mb:.0f} MB → {mb_per_s:.0f} MB/s effective)")
        return sessions, per_part_s

    # Power sampling (battery only). Sample over the full run including
    # session swaps so mWh-based energy reflects real-world cost.
    sampler = PowerSampler(interval_s=2.0) if args.power_state == "bat" else None
    mwh_before = sample_battery_mwh() if args.power_state == "bat" else None
    if sampler:
        sampler.start()

    # AR128 path needs the matching-shape KV input buffer; the AR1 path
    # reads kv.keys / kv.values directly.
    kv = KVStore(NUM_LAYERS, with_ar128_input=want_ar128)

    last_logits = None
    pp_ar128_latencies_ms: list[float] = []
    pp_ar1_latencies_ms: list[float] = []
    ar128_load_s = ar1_load_s = ar128_teardown_s = 0.0
    ar128_per_part_s: list[float] = []
    ar1_per_part_s: list[float] = []

    # ============================================================
    # Phase A: AR128 prefill (in swap mode — only AR128 sessions live)
    # ============================================================
    if want_ar128:
        print(f"\n--- phase A: load 4 AR128 sessions for prefill ---")
        t = time.perf_counter()
        sessions_ar128, ar128_per_part_s = _load_chain(
            parts_cfg_ar128, suffix="_ar128", label="AR128",
        )
        bindings_ar128, out_bufs_ar128 = make_bound_chain(
            sessions_ar128, parts_cfg_ar128
        )
        ar128_load_s = time.perf_counter() - t
        print(f"  AR128 load total: {ar128_load_s:.1f} s")

        # warmup AR128 — first call has ~1 s HMX init cost
        print("  warmup (1 AR128 step, discarded)")
        warmup_batch = list(prompt_ids[:AR128_BATCH])
        _, _ = _step_ar128(
            sessions_ar128, bindings_ar128, out_bufs_ar128,
            kv, 0, warmup_batch, scales_ar128,
        )
        kv = KVStore(NUM_LAYERS, with_ar128_input=True)

        print(f"  prefill: {n_ar128_calls} AR128 calls = "
              f"{n_ar128_calls * AR128_BATCH} tokens")
        t_pp_ar128 = time.perf_counter()
        p = 0
        for call_idx in range(n_ar128_calls):
            batch = list(prompt_ids[p : p + AR128_BATCH])
            last_logits, ms = _step_ar128(
                sessions_ar128, bindings_ar128, out_bufs_ar128,
                kv, p, batch, scales_ar128,
            )
            pp_ar128_latencies_ms.append(ms)
            print(
                f"    ar128 call {call_idx} (positions {p}..{p + AR128_BATCH - 1})  "
                f"{ms:.1f} ms  ({AR128_BATCH * 1000 / ms:.0f} t/s in-call)"
            )
            p += AR128_BATCH
        pp_ar128_wall_s = time.perf_counter() - t_pp_ar128

        print(f"\n--- phase A.5: tear down AR128 sessions ---")
        t = time.perf_counter()
        # Drop all references → InferenceSession.__del__ releases the
        # QNN context. gc.collect() forces immediate cleanup so the
        # AR1 load below has the freed HTP memory available.
        del sessions_ar128, bindings_ar128, out_bufs_ar128
        import gc
        gc.collect()
        ar128_teardown_s = time.perf_counter() - t
        print(f"  teardown: {ar128_teardown_s:.1f} s")
    else:
        pp_ar128_wall_s = 0.0

    # ============================================================
    # Phase B: AR1 sessions (used for AR1 tail prefill if any, plus decode)
    # ============================================================
    print(f"\n--- phase B: load 4 AR1 sessions for "
          f"{'tail prefill + decode' if n_ar1_tail else 'decode'} ---")
    t = time.perf_counter()
    sessions_ar1, ar1_per_part_s = _load_chain(
        parts_cfg_ar1, suffix="", label="AR1",
    )
    bindings_ar1, out_bufs_ar1 = make_bound_chain(sessions_ar1, parts_cfg_ar1)
    ar1_load_s = time.perf_counter() - t
    print(f"  AR1 load total: {ar1_load_s:.1f} s")

    # warmup AR1 (only meaningful if we'll do many AR1 calls — but cheap
    # insurance so we always do it)
    print("  warmup (1 AR1 step, discarded)")
    if not want_ar128:
        # No prior prefill yet; warm up against a fresh KV (use prompt[0]).
        kv_w = KVStore(NUM_LAYERS)
        _, _ = _step(sessions_ar1, bindings_ar1, out_bufs_ar1, kv_w, 0, prompt_ids[0], scales_ar1)
    else:
        # KV is already at position n_ar128_calls*128; warm up at the next
        # slot using the prefill-predicted next token, then DISCARD that
        # write by snapshotting the KV state. Simpler: just warm with a
        # 0 token at a throwaway position past the cache, restoring kv.t.
        # Even simpler: skip — the per-call AR1 latency variance is small
        # (~38ms median) and the first real step's overhead is bounded.
        # We accept up to ~1 s skew on the first AR1 step.
        pass

    # AR1 tail prefill (when pp_tokens isn't a multiple of 128)
    pp_ar1_wall_s = 0.0
    if want_ar128:
        p = n_ar128_calls * AR128_BATCH
    else:
        p = 0
    if not want_ar128 or n_ar1_tail > 0:
        print(f"  AR1 prefill: {len(prompt_ids) - p} steps")
        t_pp_ar1 = time.perf_counter()
        while p < len(prompt_ids):
            last_logits, ms = _step(
                sessions_ar1, bindings_ar1, out_bufs_ar1,
                kv, p, prompt_ids[p], scales_ar1,
            )
            pp_ar1_latencies_ms.append(ms)
            if p % 32 == 0 or p == len(prompt_ids) - 1:
                print(f"    ar1 step {p}  {ms:.1f} ms")
            p += 1
        pp_ar1_wall_s = time.perf_counter() - t_pp_ar1

    pp_wall_s = pp_ar128_wall_s + pp_ar1_wall_s
    pp_tps = len(prompt_ids) / pp_wall_s if pp_wall_s > 0 else 0.0
    pp_ar128_median_ms = (
        float(np.median(pp_ar128_latencies_ms)) if pp_ar128_latencies_ms else 0.0
    )
    pp_ar1_median_ms = (
        float(np.median(pp_ar1_latencies_ms)) if pp_ar1_latencies_ms else 0.0
    )
    pp_median_ms = pp_ar128_median_ms if pp_ar128_latencies_ms else pp_ar1_median_ms
    print(
        f"\n  PP total compute {pp_wall_s:.2f} s  -> {pp_tps:.2f} t/s  "
        f"(ar128 median {pp_ar128_median_ms:.1f} ms; ar1-tail median {pp_ar1_median_ms:.1f} ms)"
    )

    # Decode: generate tg_tokens with greedy argmax from last logits.
    print(f"\n--- phase C: decode {args.tg_tokens} AR1 steps (greedy) ---")
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
    swap_wall_s = ar128_load_s + ar128_teardown_s + ar1_load_s
    compute_wall_s = pp_wall_s + tg_wall_s
    total_wall_s = compute_wall_s + swap_wall_s
    mean_w = sampler.mean_watts if sampler else None
    mwh_drop = None
    if mwh_before is not None and mwh_after is not None:
        mwh_drop = mwh_before - mwh_after
    # J/tok against pure compute (excludes session swaps) so the number
    # is comparable to Genie's run that doesn't pay swap cost.
    energy_j = mean_w * compute_wall_s if mean_w is not None else (mwh_drop * 3.6 if mwh_drop else None)
    j_per_tok = energy_j / total_tokens if energy_j is not None and total_tokens > 0 else None

    print(f"\n  mean W              : {mean_w}")
    print(f"  mWh drop            : {mwh_drop}")
    print(f"  J/tok (compute-only): {j_per_tok}")
    print(f"  swap overhead       : {swap_wall_s:.1f} s  "
          f"(AR128 load {ar128_load_s:.1f} + teardown {ar128_teardown_s:.1f} + AR1 load {ar1_load_s:.1f})")

    # Per-partition load profile: if load time tracks .bin size, I/O
    # dominates and a sidecar architecture (warm page cache) closes
    # the gap. If load time is roughly fixed per partition, HTP
    # context init dominates and only a long-lived engine helps.
    if ar128_per_part_s or ar1_per_part_s:
        print(f"  per-partition load profile (sec each):")
        if ar128_per_part_s:
            print(f"    AR128: {[round(x, 1) for x in ar128_per_part_s]}")
        print(f"    AR1  : {[round(x, 1) for x in ar1_per_part_s]}")

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
        pp_ar128_wall_s=pp_ar128_wall_s,
        pp_ar1_wall_s=pp_ar1_wall_s,
        tg_wall_s=tg_wall_s,
        compute_wall_s=compute_wall_s,
        swap_wall_s=swap_wall_s,
        total_wall_s=total_wall_s,
        ar128_load_s=ar128_load_s,
        ar128_teardown_s=ar128_teardown_s,
        ar1_load_s=ar1_load_s,
        ar128_per_part_s=";".join(f"{x:.2f}" for x in ar128_per_part_s),
        ar1_per_part_s=";".join(f"{x:.2f}" for x in ar1_per_part_s),
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        pp_median_ms=pp_median_ms,
        pp_ar128_median_ms=pp_ar128_median_ms,
        pp_ar1_median_ms=pp_ar1_median_ms,
        tg_median_ms=tg_median_ms,
        mwh_before=mwh_before,
        mwh_after=mwh_after,
        mwh_drop=mwh_drop,
        mean_w=mean_w,
        j_per_tok=j_per_tok,
        power_state=args.power_state,
        tag=tag,
        ctx_tier=CTX_LEN,
        note="AR128 swap-mode prefill + AR1 decode + IOBinding; ORT-QNN 1.24.4",
    )

    csv_path = CSV_DIR / f"qwen3_4b_ortqnn_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"\n=== Summary ===")
    print(f"  PP     : {pp_tps:.2f} t/s  ({pp_mode}; Genie AR128 baseline ~1598 t/s)")
    print(f"  TG-AR1 : {tg_tps:.2f} t/s  (Genie AR1 baseline ~23.3 t/s on same binary)")
    print(f"  J/tok  : {j_per_tok}")
    print(f"  total wall (incl swap): {total_wall_s:.1f} s   compute-only: {compute_wall_s:.1f} s")
    print(f"  csv    : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
