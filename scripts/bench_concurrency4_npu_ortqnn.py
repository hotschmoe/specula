"""Concurrency-4 NPU bench via ORT-QNN (Qwen3-4B chained 4-partition).

Spawns 4 simultaneous subprocesses of scripts/bench_qwen3_4b_ortqnn.py,
each driving its own 4 ORT-QNN sessions on the X2 Elite Hexagon NPU.
The QNN HTP backend's weight_sharing_enabled=true (per the bundle's
htp_backend_ext_config.json) lets multiple contexts share the
underlying weight buffers, so memory cost is bounded; the throughput
penalty comes from HTP scheduler contention as 4 chains queue for the
single Hexagon engine.

Reads each stream's CSV after they finish, sums throughputs to compute
aggregate. Aggregate / 4 = effective per-stream throughput.

Why Qwen3-4B and not 7B: only the 4B has the wrapper ONNXs we need for
ORT-QNN chaining (oracle_part1.wrapper.onnx … oracle_part4.wrapper.onnx,
generated as part of the existing oracle pipeline). The 7B Workbench
bundle ships only raw context binaries.

Usage:
    .venv/Scripts/python.exe scripts/bench_concurrency4_npu_ortqnn.py \\
        --power-state ac --tag 2026-04-25_ac

Outputs:
    results/csv/concurrency4_npu_ortqnn_qwen3_4b_<tag>.csv  — aggregate row
    results/csv/qwen3_4b_ortqnn_npuconc4_stream{0..3}_<tag>.csv  — per-stream
    marked_for_deletion/concurrency4_npu_ortqnn_<tag>/  — stream stdout logs
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"
N_PARALLEL = 4


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--power-state", choices=("ac", "bat"), required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--pp-tokens", type=int, default=256)
    p.add_argument("--tg-tokens", type=int, default=128)
    args = p.parse_args()

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = TRASH_ROOT / f"concurrency{N_PARALLEL}_npu_ortqnn_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    bench_script = REPO_ROOT / "scripts" / "bench_qwen3_4b_ortqnn.py"
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

    print(f"=== concurrency-{N_PARALLEL} NPU ORT-QNN bench ===")
    print(f"tag        : {tag}")
    print(f"pp_tokens  : {args.pp_tokens}")
    print(f"tg_tokens  : {args.tg_tokens}")
    print(f"each stream loads 4 ORT-QNN sessions on the same Hexagon NPU")
    print(f"weight-sharing across contexts is enabled in the bundle")
    print()

    # Launch 4 in parallel.
    stream_tags = [f"npuconc{N_PARALLEL}_stream{i}_{tag}" for i in range(N_PARALLEL)]
    procs = []
    log_files = []
    t_start = time.perf_counter()
    for i in range(N_PARALLEL):
        log_path = log_dir / f"stream{i}.log"
        log_f = log_path.open("w", encoding="utf-8", errors="replace")
        log_files.append(log_f)
        cmd = [
            str(venv_python),
            str(bench_script),
            "--power-state", args.power_state,
            "--tag", stream_tags[i],
            "--pp-tokens", str(args.pp_tokens),
            "--tg-tokens", str(args.tg_tokens),
            "--skip-power-check",
        ]
        print(f"  spawning stream {i}: tag={stream_tags[i]}")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        import os
        full_env = os.environ.copy()
        full_env.update(env)
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=full_env)
        procs.append(p)

    print(f"\nwaiting for {N_PARALLEL} streams to finish...")
    exit_codes = []
    for i, p in enumerate(procs):
        p.wait()
        exit_codes.append(p.returncode)
        log_files[i].close()
        print(f"  stream {i}: exit {p.returncode}")
    aggregate_wall_s = time.perf_counter() - t_start
    print(f"\naggregate wall: {aggregate_wall_s:.1f} s")

    # Read each stream's CSV row.
    rows = []
    for i, st in enumerate(stream_tags):
        csv_path = CSV_DIR / f"qwen3_4b_ortqnn_{st}.csv"
        if not csv_path.exists():
            print(f"WARNING: stream {i} CSV missing: {csv_path}")
            continue
        with csv_path.open(newline="") as f:
            r = list(csv.DictReader(f))
            if r:
                rows.append(r[0])

    if len(rows) != N_PARALLEL:
        print(f"ERROR: only {len(rows)} of {N_PARALLEL} streams produced CSV rows")
        return 2

    pp_tps_per_stream = [float(r["pp_tps"]) for r in rows]
    tg_tps_per_stream = [float(r["tg_tps"]) for r in rows]
    pp_median_ms = [float(r["pp_median_ms"]) for r in rows]
    tg_median_ms = [float(r["tg_median_ms"]) for r in rows]

    s_pp_agg = sum(pp_tps_per_stream)
    s_tg_agg = sum(tg_tps_per_stream)
    pp_per_stream_avg = s_pp_agg / N_PARALLEL
    tg_per_stream_avg = s_tg_agg / N_PARALLEL

    print(f"\n=== aggregate ===")
    print(f"  per-stream PP t/s : {pp_tps_per_stream}")
    print(f"  per-stream TG t/s : {tg_tps_per_stream}")
    print(f"  PP-AR1 agg        : {s_pp_agg:.2f} t/s   (per stream avg {pp_per_stream_avg:.2f})")
    print(f"  TG-AR1 agg        : {s_tg_agg:.2f} t/s   (per stream avg {tg_per_stream_avg:.2f})")
    print(f"  PP step median    : range {min(pp_median_ms):.1f}-{max(pp_median_ms):.1f} ms")
    print(f"  TG step median    : range {min(tg_median_ms):.1f}-{max(tg_median_ms):.1f} ms")

    agg_csv = CSV_DIR / f"concurrency{N_PARALLEL}_npu_ortqnn_qwen3_4b_{tag}.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n_parallel", "n_pp_tokens", "n_tg_tokens",
            "s_pp_agg_tps", "s_tg_agg_tps",
            "pp_per_stream_avg", "tg_per_stream_avg",
            "pp_step_ms_min", "pp_step_ms_max",
            "tg_step_ms_min", "tg_step_ms_max",
            "aggregate_wall_s", "tag",
        ])
        w.writerow([
            N_PARALLEL, args.pp_tokens, args.tg_tokens,
            s_pp_agg, s_tg_agg,
            pp_per_stream_avg, tg_per_stream_avg,
            min(pp_median_ms), max(pp_median_ms),
            min(tg_median_ms), max(tg_median_ms),
            aggregate_wall_s, tag,
        ])

    print(f"\n  aggregate csv : {agg_csv}")
    print(f"  per-stream csvs in {CSV_DIR}/qwen3_4b_ortqnn_npuconc4_stream*.csv")
    print(f"  logs : {log_dir}")
    return 0 if all(c == 0 for c in exit_codes) else 1


if __name__ == "__main__":
    sys.exit(main())
