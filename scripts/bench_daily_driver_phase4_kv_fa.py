"""Phase 4: disentangle FA cost from KV-quant cost on CPU at d=32k.

Phase 1 (no FA, f16 KV)  produced TG = 36.01 t/s @ d=128 (PP+TG bench).
Phase 2 v3 (FA on, q8 KV) produced TG = 13.51 t/s @ d=32768.

The 18% drop is currently attributed to "FA + q8 combined". This
phase breaks it apart by running 4 configs at the same depth (32k):

  config            | KV   | FA  | rationale
  ------------------|------|-----|-----------
  baseline          | f16  | off | what Phase 1 measured (extended to d=32k)
  fa_only           | f16  | on  | NEW — isolates FA's cost on f16 KV
  fa_q8             | q8_0 | on  | what Phase 2 v3 measured at d=32k
  fa_q4             | q4_0 | on  | the more aggressive KV-memory option

Note: KV non-f16 REQUIRES -fa 1 (verified Phase 2 attempt #2);
no-fa + q8/q4 cannot be tested.

Outputs:
  results/csv/daily_driver_phase4_kv_fa_<tag>.csv  — combined table
  marked_for_deletion/daily_driver_phase4_kv_fa_<tag>/  — per-row logs
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


CONFIGS = [
    # (label,        kv_type, flash_attn)
    ("baseline_f16_noFA",  "f16",  False),
    ("fa_f16",             "f16",  True),
    ("fa_q8",              "q8_0", True),
    ("fa_q4",              "q4_0", True),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--ctx-depth", type=int, default=32768,
                        help="Single ctx depth to bench all configs at "
                             "(default: 32768).")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--stale-timeout-s", type=int, default=1500)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    tag = args.tag or f"phase4_{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = TRASH_ROOT / f"daily_driver_phase4_kv_fa_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"daily_driver_phase4_kv_fa_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False. Plug in or "
              "pass --skip-power-check.")
        return 2

    rows: list[dict] = []
    for label, kv_type, fa in CONFIGS:
        print(f"\n=== {label}  (kv={kv_type}, fa={fa}, d={args.ctx_depth}) ===")
        log_path = log_dir / f"{label}.log"
        row = run_longctx_one(
            preset="cpu",
            ctx_depth=args.ctx_depth,
            log_path=log_path,
            threads=args.threads,
            ngl=0,
            extra_env=None,
            kv_type=kv_type,
            ctx_buffer=args.ctx_depth + 256,
            flash_attn=fa,
            stale_timeout_s=args.stale_timeout_s,
        )
        out = {
            "config": label,
            "kv_type": kv_type,
            "flash_attn": int(fa),
            "ctx_depth": args.ctx_depth,
            "tg_tps": f"{row.tg_tps:.3f}" if row.tg_tps is not None else "",
            "wall_s": f"{row.wall_s:.1f}" if row.wall_s is not None else "",
            "ok": int(row.ok),
            "notes": row.notes,
        }
        rows.append(out)
        status = "OK" if row.ok else "FAIL"
        tg = f"{row.tg_tps:.2f}" if row.tg_tps else "—"
        print(f"  [{status}] TG={tg} t/s  wall={row.wall_s:.0f}s  {row.notes}")

        # Periodic CSV flush so a mid-sweep crash doesn't lose data.
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== Summary ===")
    print(f"csv:  {csv_path}")
    print(f"logs: {log_dir}")

    # Print pivot table for the doc
    print()
    print(f"Phase 4 CPU @ d={args.ctx_depth}:")
    print()
    # ASCII-only table for Windows cp1252 stdout compatibility.
    print("| config              | KV   | FA  | TG (t/s) | delta vs baseline |")
    print("|---|---|:-:|---:|---:|")
    base_tg = next((float(r["tg_tps"]) for r in rows
                    if r["config"] == "baseline_f16_noFA" and r["tg_tps"]),
                   None)
    for r in rows:
        tg = float(r["tg_tps"]) if r["tg_tps"] else None
        delta = ""
        if tg is not None and base_tg is not None and base_tg > 0:
            delta = f"{(tg - base_tg) / base_tg * 100:+.1f}%"
        elif tg is None:
            delta = "—"
        tg_str = f"{tg:.2f}" if tg else "—"
        print(f"| {r['config']:20s} | {r['kv_type']:4s} | "
              f"{'on' if r['flash_attn'] else 'off':3s} | {tg_str} | {delta} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
