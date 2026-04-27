"""Phase 5: thread-count sweep on CPU at fixed (depth, KV, FA).

Snapdragon X2 Elite Extreme has 12 P-cores. The 4B/7B baselines used
-t 8; this phase asks whether 8 is right for 35B-A3B too. At long
context, TG is memory-bandwidth-bound and adding cores past the
saturation point should hurt (cache thrash + scheduling overhead);
PP is compute-bound and may benefit from more cores.

Configs:
  threads ∈ {4, 6, 8, 10, 12}, all at:
    d = 8192 (mid-depth — fast enough to bench the whole sweep
              in ~15 min total, deep enough that KV bandwidth is
              not totally negligible)
    KV q8_0, FA on (the daily-driver default)

Outputs:
  results/csv/daily_driver_phase5_threads_<tag>.csv
  marked_for_deletion/daily_driver_phase5_threads_<tag>/
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_daily_driver_longctx import run_longctx_one  # type: ignore
from bench_daily_driver import (  # type: ignore
    CSV_DIR,
    TRASH_ROOT,
    sample_power_online,
)


DEFAULT_THREADS = [4, 6, 8, 10, 12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--ctx-depth", type=int, default=8192)
    # Default config matches Phase 4 winner (f16 KV + no FA).
    parser.add_argument("--kv-type", default="f16")
    parser.add_argument("--threads-list", default=None,
                        help=f"Comma-separated thread counts. "
                             f"Default: {DEFAULT_THREADS}")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--stale-timeout-s", type=int, default=600)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    threads_list = (
        [int(x) for x in args.threads_list.split(",")]
        if args.threads_list else list(DEFAULT_THREADS)
    )
    flash_attn = args.kv_type != "f16"

    tag = args.tag or f"phase5_{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = TRASH_ROOT / f"daily_driver_phase5_threads_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"daily_driver_phase5_threads_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    rows: list[dict] = []
    for t in threads_list:
        print(f"\n=== threads={t}  (kv={args.kv_type}, fa={flash_attn}, "
              f"d={args.ctx_depth}) ===")
        log_path = log_dir / f"threads_{t}.log"
        row = run_longctx_one(
            preset="cpu",
            ctx_depth=args.ctx_depth,
            log_path=log_path,
            threads=t,
            ngl=0,
            extra_env=None,
            kv_type=args.kv_type,
            ctx_buffer=args.ctx_depth + 256,
            flash_attn=flash_attn,
            stale_timeout_s=args.stale_timeout_s,
        )
        out = {
            "threads": t,
            "ctx_depth": args.ctx_depth,
            "kv_type": args.kv_type,
            "flash_attn": int(flash_attn),
            "tg_tps": f"{row.tg_tps:.3f}" if row.tg_tps is not None else "",
            "wall_s": f"{row.wall_s:.1f}" if row.wall_s is not None else "",
            "ok": int(row.ok),
            "notes": row.notes,
        }
        rows.append(out)
        status = "OK" if row.ok else "FAIL"
        tg = f"{row.tg_tps:.2f}" if row.tg_tps else "—"
        print(f"  [{status}] TG={tg} t/s  wall={row.wall_s:.0f}s  {row.notes}")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== Summary ===")
    print(f"csv:  {csv_path}")

    # Pivot table for the doc
    print()
    print(f"Phase 5 CPU thread sweep @ d={args.ctx_depth}, KV={args.kv_type}:")
    print()
    print("| threads | TG (t/s) | wall (s) | notes |")
    print("|---:|---:|---:|---|")
    for r in rows:
        tg = r["tg_tps"] or "—"
        wall = r["wall_s"] or "—"
        print(f"| {r['threads']} | {tg} | {wall} | {r['notes']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
