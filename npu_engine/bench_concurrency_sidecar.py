"""Single-process N-stream NPU concurrency bench.

The architectural fix to the subprocess-fan-out concurrency benchmark
(`bench_concurrency4_npu_ortqnn.py`). That driver spawns N independent
processes, each loading its own 4 ORT-QNN sessions = 4N total sessions
on one Hexagon engine. At N=4 the 16-session resource demand exhausts
HTP scheduler resources mid-execute (QNN error 1003), and at N=2 the
"win" comes from accidental Python-overhead overlap across processes,
not from real NPU parallelism.

This bench loads ONE chain of 4 AR1 sessions and runs N logical streams
through it — each stream owns its own KVStore + position state but
shares the same NPU sessions. Decode is round-robin interleaved
(step 0 of every stream, then step 1 of every stream, …) so the NPU
sees a steady mix of work and the HTP scheduler doesn't ping-pong
across process address spaces.

What this measures:
  * Stable-N ceiling. With one set of sessions, we should be able to
    crank N as high as we want (limited only by KV memory) without the
    QNN 1003 crashes that subprocess-fan-out hits at N=4.
  * Real aggregate throughput. Single-process round-robin removes the
    cross-process Python-overhead-overlap shortcut, so aggregate ≈
    single-stream throughput (the NPU is one device — multiplexing
    work across N streams just spreads the same pie). This is the
    *honest* aggregate number; the subprocess driver's 1.12× scaling
    overstated reality.
  * Per-stream tail latency. At N streams round-robin, each stream's
    step time grows ~Nx the single-stream step time.

Scope (v0):
  * AR1-only path. Apples-to-apples with the existing concurrency-4
    baseline. AR128 batched-prefill across N streams is a follow-on.
  * Greedy argmax decode. Same as the standalone bench.
  * 256-token prefill, 128-token decode per stream by default.

Usage:
    .venv/Scripts/python.exe npu_engine/bench_concurrency_sidecar.py \\
        --power-state ac --tag 2026-04-25_ac --n-streams 4

Outputs:
    results/csv/qwen3_4b_ortqnn_sidecar_concN_<tag>.csv  — aggregate row
    marked_for_deletion/qwen3_4b_ortqnn_sidecar_concN_<tag>/stdout.log
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import yaml
from tokenizers import Tokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    BUNDLE_DIR,
    NUM_LAYERS,
    PAST_LEN,
    KVStore,
    build_part_cfg,
    dequant_uint16,
)
from bench_qwen3_4b_ortqnn import _step, PROMPT_PATH  # noqa: E402
from sidecar import EngineState, _maybe_warmup  # noqa: E402

REPO_ROOT = _HERE.parent
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"


class StreamState:
    """One logical stream's per-tenant state.

    KV is a fresh KVStore (each stream has its own decode history); the
    engine's sessions are shared. position tracks where in the decode
    sequence this stream is; next_token is what to feed at the next
    step.
    """
    __slots__ = ("idx", "kv", "position", "next_token", "decoded",
                 "step_ms_samples")

    def __init__(self, idx: int):
        self.idx = idx
        self.kv = KVStore(NUM_LAYERS)
        self.position = 0
        self.next_token = None
        self.decoded: list[int] = []
        self.step_ms_samples: list[float] = []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--power-state", choices=("ac", "bat"), required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--n-streams", type=int, required=True)
    p.add_argument("--pp-tokens", type=int, default=256)
    p.add_argument("--tg-tokens", type=int, default=128)
    p.add_argument("--skip-power-check", action="store_true")
    args = p.parse_args()

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    trash_dir = TRASH_ROOT / f"qwen3_4b_ortqnn_sidecar_conc{args.n_streams}_{tag}"
    trash_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    if args.pp_tokens + args.tg_tokens > PAST_LEN:
        print(f"ERROR: pp+tg = {args.pp_tokens + args.tg_tokens} exceeds CL-512 cap {PAST_LEN}")
        return 2

    print(f"=== Qwen3-4B sidecar-bench (single-process, {args.n_streams} streams, AR1-only) ===")
    print(f"tag       : {tag}")
    print(f"streams   : {args.n_streams}")
    print(f"pp_tokens : {args.pp_tokens}")
    print(f"tg_tokens : {args.tg_tokens}")

    # ----- engine load (paid once for all N streams) -----
    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts_cfg_ar1 = build_part_cfg(metadata, ar=1)
    parts_cfg_ar128 = build_part_cfg(metadata, ar=128)
    state = EngineState(parts_cfg_ar1, parts_cfg_ar128)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    base_tokens = tokenizer.encode(
        PROMPT_PATH.read_text(encoding="utf-8")
    ).ids[: args.pp_tokens]
    if len(base_tokens) != args.pp_tokens:
        print(f"WARNING: prompt has {len(base_tokens)} tokens, asked for {args.pp_tokens}")

    print(f"\n--- engine load (one chain, shared across all streams) ---")
    t = time.perf_counter()
    swap_s, per_part_s = state.ensure_mode("ar1")
    engine_load_s = time.perf_counter() - t
    print(f"  load total : {engine_load_s:.1f} s   per-partition: "
          f"{[round(x, 2) for x in (per_part_s or [])]}")

    print(f"  warmup (1 AR1 step)")
    _maybe_warmup(state, base_tokens)

    # ----- prefill phase: each stream's full 256-token AR1 prefill,
    # one stream at a time. Could interleave but for prefill there's
    # no benefit (steps are sequential per stream). -----
    streams = [StreamState(i) for i in range(args.n_streams)]
    print(f"\n--- prefill: {args.n_streams} × {args.pp_tokens}-token AR1 prefills ---")
    t_pp = time.perf_counter()
    for s in streams:
        for pos in range(args.pp_tokens):
            tok = base_tokens[pos]
            t0 = time.perf_counter()
            logits, _ = _step(
                state.sessions, state.bindings, state.out_bufs,
                s.kv, pos, tok, state.scales_ar1,
            )
            ms = (time.perf_counter() - t0) * 1000
            s.step_ms_samples.append(ms)
        s.position = args.pp_tokens
        s.next_token = int(np.argmax(logits))
        print(f"  stream {s.idx}: prefill done ({args.pp_tokens} steps, "
              f"median {np.median(s.step_ms_samples):.1f} ms)")
    pp_wall_s = time.perf_counter() - t_pp
    pp_total_tokens = args.pp_tokens * args.n_streams
    pp_aggregate_tps = pp_total_tokens / pp_wall_s

    # Reset per-stream samples so the prefill medians don't pollute
    # decode medians; we still report prefill medians from the
    # accumulated samples below.
    pp_step_ms_per_stream = [
        float(np.median(s.step_ms_samples)) for s in streams
    ]
    for s in streams:
        s.step_ms_samples = []

    # ----- decode phase: round-robin interleave. step 0 of every
    # stream, then step 1 of every stream, etc. NPU sees a steady mix;
    # KV stitch + argmax for stream A overlaps in time-on-CPU with
    # NPU compute for stream B (Python GIL releases inside ORT). -----
    print(f"\n--- decode: round-robin {args.n_streams} streams × {args.tg_tokens} tokens ---")
    t_tg = time.perf_counter()
    for step_idx in range(args.tg_tokens):
        for s in streams:
            t0 = time.perf_counter()
            logits, _ = _step(
                state.sessions, state.bindings, state.out_bufs,
                s.kv, s.position, s.next_token, state.scales_ar1,
            )
            ms = (time.perf_counter() - t0) * 1000
            s.step_ms_samples.append(ms)
            s.decoded.append(s.next_token)
            s.next_token = int(np.argmax(logits))
            s.position += 1
        if step_idx % 16 == 0 or step_idx == args.tg_tokens - 1:
            medians = [round(float(np.median(s.step_ms_samples)), 1) for s in streams]
            print(f"  decode step {step_idx:3d}  per-stream medians: {medians} ms")
    tg_wall_s = time.perf_counter() - t_tg

    # Per-stream + aggregate metrics
    tg_per_stream_tps = [args.tg_tokens / (np.sum(s.step_ms_samples) / 1000) for s in streams]
    tg_aggregate_tps = (args.tg_tokens * args.n_streams) / tg_wall_s
    tg_step_ms_per_stream = [float(np.median(s.step_ms_samples)) for s in streams]

    print(f"\n=== summary ===")
    print(f"  engine load  : {engine_load_s:.1f} s (one-time, amortized over all streams)")
    print(f"  prefill wall : {pp_wall_s:.1f} s    aggregate {pp_aggregate_tps:.2f} t/s "
          f"({pp_total_tokens} tokens across {args.n_streams} streams)")
    print(f"  decode wall  : {tg_wall_s:.1f} s    aggregate {tg_aggregate_tps:.2f} t/s")
    print(f"  per-stream TG (t/s) : {[round(x, 2) for x in tg_per_stream_tps]}")
    print(f"  per-stream step ms  : prefill {[round(x, 1) for x in pp_step_ms_per_stream]} | "
          f"decode {[round(x, 1) for x in tg_step_ms_per_stream]}")

    # CSV row
    row = dict(
        backend="npu-ortqnn-sidecar",
        n_streams=args.n_streams,
        pp_tokens=args.pp_tokens,
        tg_tokens=args.tg_tokens,
        engine_load_s=engine_load_s,
        pp_wall_s=pp_wall_s,
        tg_wall_s=tg_wall_s,
        pp_aggregate_tps=pp_aggregate_tps,
        tg_aggregate_tps=tg_aggregate_tps,
        tg_per_stream_avg_tps=float(np.mean(tg_per_stream_tps)),
        tg_step_ms_median=float(np.median(tg_step_ms_per_stream)),
        pp_step_ms_median=float(np.median(pp_step_ms_per_stream)),
        per_part_load_s=";".join(f"{x:.2f}" for x in (per_part_s or [])),
        power_state=args.power_state,
        tag=tag,
        note=f"single-process, shared 4 AR1 sessions, round-robin decode interleave",
    )
    csv_path = CSV_DIR / f"qwen3_4b_ortqnn_sidecar_conc{args.n_streams}_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n  csv : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
