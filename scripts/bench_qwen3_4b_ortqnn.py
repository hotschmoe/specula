"""Qwen3-4B NPU bench via ORT-QNN (chained 4-partition, AR1/CL512).

Companion to `scripts/bench_qwen3_4b_all_backends.py`. Where Genie drives
the Qualcomm-shipped w4a16 bundle through the vendor runtime, this
script drives the *same* binary through **our** runtime stack
(ORT-QNN 1.24.4 + QAIRT 2.42 context binaries) — the same stack our
speculative-decode sidecar speaks. The gap between this script's
numbers and Genie's tells us how much performance we'd inherit if we
built our own heterogeneous inference engine today instead of going
through the vendor-closed Genie CLI.

Scope (tonight's run — see roadmap W1/W2/B20 for extensions):
  * **AR1 only.** Both prefill and decode use the single-token `ar1`
    graphs. Genie uses the batched `ar128` prefill path; supporting
    AR128 in our stack needs new wrapper ONNXs + per-partition plumbing
    and is a follow-up workstream. The AR1-prefill number reported here
    is the "naive AR1 everything" lower bound and should NOT be
    compared apples-to-apples against Genie's PP t/s.
  * **AR1 decode is apples-to-apples vs Genie TG.** Per-step NPU work
    is identical — both are doing `ar1_cl512_*_of_4` graph executions.
    Any delta is ORT-QNN dispatch overhead vs Genie's native glue.
  * **CL=512 fixed.** The oracle script also uses cl512, which caps
    prefill+decode at 511 KV slots. We feed 256 prompt tokens and
    generate 128 → 384 slots used, under the cap.

Reuses machinery from scripts/qualcomm_qwen3_4b_oracle.py. The oracle
is kept as the single source of truth for the chain; this script is a
stripped-down bench wrapper that skips the per-step prints + NPZ dump
and adds PP/TG separation + optional battery J/tok sampling.

Usage:
    .venv/Scripts/python.exe scripts/bench_qwen3_4b_ortqnn.py \\
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

# Reuse oracle machinery.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    BUNDLE_DIR,
    CTX_LEN,
    HIDDEN_DIM,
    LAYERS_PER_PART,
    NUM_LAYERS,
    PAST_LEN,
    KVStore,
    attention_mask_quantized,
    build_part_cfg,
    build_wrapper,
    dequant_uint16,
    half_dim_rope_quantized,
    load_session,
)

# Battery helpers — reuse from the llama.cpp bench driver so both
# scripts report on the same measurement basis.
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


def _step(sessions, parts_cfg, kv, position, token_in, scales):
    """One forward pass through all 4 partitions at position `position`,
    feeding `token_in`. Stitches KV. Returns (logits_fp32, wall_ms).
    """
    cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset = scales
    cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
    mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

    t0 = time.perf_counter()

    # part 1
    feed1 = {"input_ids": np.array([[token_in]], dtype=np.int32)}
    emb_out = sessions[1].run([HIDDEN_FROM_PART1], feed1)[0]

    # part 2
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
    new_keys_p2 = [out_2[1 + 2 * i] for i in range(LAYERS_PER_PART)]
    new_vals_p2 = [out_2[1 + 2 * i + 1] for i in range(LAYERS_PER_PART)]

    # part 3
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
    new_keys_p3 = [out_3[1 + 2 * i] for i in range(LAYERS_PER_PART)]
    new_vals_p3 = [out_3[1 + 2 * i + 1] for i in range(LAYERS_PER_PART)]

    # part 4
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
    new_keys_p4 = [out_4[1 + 2 * i] for i in range(LAYERS_PER_PART)]
    new_vals_p4 = [out_4[1 + 2 * i + 1] for i in range(LAYERS_PER_PART)]

    kv.stitch_step(
        new_keys_p2 + new_keys_p3 + new_keys_p4,
        new_vals_p2 + new_vals_p3 + new_vals_p4,
    )

    wall_ms = (time.perf_counter() - t0) * 1000
    logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
    return logits_fp32, wall_ms


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

    print(f"=== Qwen3-4B ORT-QNN bench (AR1/CL{CTX_LEN}, chained 4-part) ===")
    print(f"tag            : {tag}")
    print(f"power state    : {args.power_state}")
    print(f"pp tokens      : {args.pp_tokens}")
    print(f"tg tokens      : {args.tg_tokens}")
    print(f"prompt         : {PROMPT_PATH}")
    print(f"log (trash)    : {log_path}")

    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts_cfg = build_part_cfg(metadata)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(prompt_text).ids[: args.pp_tokens]
    print(f"prompt tokenized to {len(prompt_ids)} (target {args.pp_tokens})")

    print("\n--- loading 4 sessions ---")
    sessions: dict[int, ort.InferenceSession] = {}
    t_load = time.perf_counter()
    for part_idx in (1, 2, 3, 4):
        wrapper_path = BUNDLE_DIR / f"oracle_part{part_idx}.wrapper.onnx"
        # Skip rebuild if a wrapper already exists. build_wrapper is
        # deterministic given the same parts_cfg from metadata.yaml, so
        # an existing file is identical to what we'd produce. Skipping
        # also avoids a write race when multiple bench processes are
        # spawned concurrently against the same bundle.
        if not wrapper_path.exists():
            build_wrapper(parts_cfg[part_idx], wrapper_path)
        sessions[part_idx] = load_session(wrapper_path)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f} s")

    # Quant scales used every step.
    cos_scale = next(io for io in parts_cfg[2]["inputs"] if io["name"] == "position_ids_cos")["scale"]
    cos_offset = next(io for io in parts_cfg[2]["inputs"] if io["name"] == "position_ids_cos")["offset"]
    mask_scale = next(io for io in parts_cfg[2]["inputs"] if io["name"] == "attention_mask")["scale"]
    mask_offset = next(io for io in parts_cfg[2]["inputs"] if io["name"] == "attention_mask")["offset"]
    logits_io = next(io for io in parts_cfg[4]["outputs"] if io["name"] == "logits")
    logits_scale = logits_io["scale"]
    logits_offset = logits_io["offset"]
    scales = (cos_scale, cos_offset, mask_scale, mask_offset, logits_scale, logits_offset)

    # Power sampling (battery only). 2 s interval for parity with the
    # llama.cpp bench driver.
    sampler = PowerSampler(interval_s=2.0) if args.power_state == "bat" else None
    mwh_before = sample_battery_mwh() if args.power_state == "bat" else None

    kv = KVStore(NUM_LAYERS)

    if sampler:
        sampler.start()

    # Warmup: one throw-away step that the bench discards. The first
    # step takes ~1s of HMX context-init that would dominate short
    # prefill measurements; without discarding it, the 256-token PP
    # window is contaminated.
    print("\n--- warmup (1 step, discarded) ---")
    _, _ = _step(sessions, parts_cfg, kv, 0, prompt_ids[0], scales)
    kv = KVStore(NUM_LAYERS)  # fresh cache

    # Prefill: feed prompt_ids[0..N-1] one at a time.
    print(f"\n--- prefill: {len(prompt_ids)} AR1 steps ---")
    t_pp_start = time.perf_counter()
    pp_latencies_ms: list[float] = []
    last_logits = None
    for step, tok in enumerate(prompt_ids):
        last_logits, ms = _step(sessions, parts_cfg, kv, step, tok, scales)
        pp_latencies_ms.append(ms)
        if step % 32 == 0 or step == len(prompt_ids) - 1:
            print(f"  pp step {step:3d}  {ms:.1f} ms  (median so far {np.median(pp_latencies_ms):.1f} ms)")
    pp_wall_s = time.perf_counter() - t_pp_start
    pp_tps = len(prompt_ids) / pp_wall_s
    pp_median_ms = float(np.median(pp_latencies_ms))
    print(f"  PP total {pp_wall_s:.2f} s  -> {pp_tps:.2f} t/s  (median step {pp_median_ms:.1f} ms)")

    # Decode: generate `tg_tokens` with greedy argmax from last logits.
    print(f"\n--- decode: {args.tg_tokens} AR1 steps (greedy) ---")
    t_tg_start = time.perf_counter()
    tg_latencies_ms: list[float] = []
    next_token = int(np.argmax(last_logits))
    gen_ids: list[int] = []
    for i in range(args.tg_tokens):
        position = len(prompt_ids) + i
        logits_fp32, ms = _step(sessions, parts_cfg, kv, position, next_token, scales)
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

    row = dict(
        backend="npu-ortqnn-ar1",
        pp_tokens=len(prompt_ids),
        tg_tokens=args.tg_tokens,
        pp_wall_s=pp_wall_s,
        tg_wall_s=tg_wall_s,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        pp_median_ms=pp_median_ms,
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
        note="AR1 prefill + AR1 decode; same binary as Genie, driven through ORT-QNN 1.24.4",
    )

    csv_path = CSV_DIR / f"qwen3_4b_ortqnn_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"\n=== Summary ===")
    print(f"  PP-AR1 : {pp_tps:.2f} t/s  (Genie AR128 is ~1598; ours is AR1 — NOT apples to apples)")
    print(f"  TG-AR1 : {tg_tps:.2f} t/s  (Genie AR1 is ~23.3 on same binary — apples to apples)")
    print(f"  J/tok  : {j_per_tok}")
    print(f"  csv    : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
