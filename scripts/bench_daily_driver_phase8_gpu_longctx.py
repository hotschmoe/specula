"""Phase 8: GPU long-context with f16 KV + no-FA path.

Phase 1 measured GPU at d=128 (PP512/TG128). Phase 2 attempt #2
showed GPU + FA + quant-KV is broken on this stack. With Phase 4
finding that f16 KV + no-FA is best on CPU anyway, the same config
on GPU might work — sidestepping the SET_ROWS issue (OpenCL) and
the FA livelock (Vulkan).

Sweeps (Vulkan, OpenCL) × (d=8k, d=32k):
  - KV f16, no FA, ngl=99, MXFP4_MOE model
  - Vulkan with the canonical knobs (DISABLE_F16=1 PREFER_HOST=1)

Outputs:
  results/csv/daily_driver_phase8_gpu_longctx_<tag>.csv
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
    VULKAN_DEFAULT_ENV,
    sample_power_online,
)


CONFIGS = [
    # (label, preset, ngl, extra_env)
    ("vulkan_f16_noFA", "vulkan", 99, dict(VULKAN_DEFAULT_ENV)),
    ("opencl_f16_noFA", "opencl", 99, None),
]
DEFAULT_DEPTHS = [8192, 32768]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--ctx-depths", default=None,
                        help=f"Comma-separated. Default: {DEFAULT_DEPTHS}")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--stale-timeout-s", type=int, default=900,
                        help="Watchdog (s). 900s covers d=32k GPU prefill "
                             "even on the slow scalar fallback path.")
    parser.add_argument("--skip-power-check", action="store_true")
    parser.add_argument("--backends", default="all",
                        help="Comma-separated subset; choices: vulkan_f16_noFA, "
                             "opencl_f16_noFA. Default: all.")
    args = parser.parse_args()

    depths = ([int(s.strip()) for s in args.ctx_depths.split(",")]
              if args.ctx_depths else list(DEFAULT_DEPTHS))

    if args.backends == "all":
        configs = CONFIGS
    else:
        wanted = set(s.strip() for s in args.backends.split(","))
        configs = [c for c in CONFIGS if c[0] in wanted]
        if {c[0] for c in configs} != wanted:
            missing = wanted - {c[0] for c in configs}
            print(f"ERROR: unknown config(s): {missing}")
            return 2

    tag = args.tag or f"phase8_{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = TRASH_ROOT / f"daily_driver_phase8_gpu_longctx_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"daily_driver_phase8_gpu_longctx_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    rows: list[dict] = []
    for label, preset, ngl, extra_env in configs:
        for d in depths:
            print(f"\n=== {label}  (d={d}) ===")
            log_path = log_dir / f"{label}_d{d}.log"
            row = run_longctx_one(
                preset=preset,
                ctx_depth=d,
                log_path=log_path,
                threads=None,
                ngl=ngl,
                extra_env=extra_env,
                kv_type="f16",
                ctx_buffer=d + 256,
                flash_attn=False,
                stale_timeout_s=args.stale_timeout_s,
            )
            out = {
                "config": label,
                "ctx_depth": d,
                "tg_tps": f"{row.tg_tps:.3f}" if row.tg_tps is not None else "",
                "wall_s": f"{row.wall_s:.1f}" if row.wall_s is not None else "",
                "ok": int(row.ok),
                "notes": row.notes,
            }
            rows.append(out)
            tg = f"{row.tg_tps:.2f}" if row.tg_tps else "-"
            status = "OK" if row.ok else "FAIL"
            print(f"  [{status}] TG={tg} t/s  wall={row.wall_s:.0f}s  {row.notes}")

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"\n=== Summary ===")
    print(f"csv:  {csv_path}")

    print()
    print(f"Phase 8 GPU long-ctx, KV f16 + no FA:")
    print()
    print("| config           | d=4k | d=8k | d=32k | notes |")
    print("|---|---:|---:|---:|---|")
    by_cfg: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {})[r["ctx_depth"]] = r
    for cfg, by_d in by_cfg.items():
        cells = [f"| {cfg} |"]
        for dval in (4096, 8192, 32768):
            r = by_d.get(dval)
            if r is None:
                cells.append(" - |")
            elif r["tg_tps"]:
                cells.append(f" {r['tg_tps']} |")
            else:
                cells.append(" FAIL |")
        notes = "; ".join(r["notes"] for r in by_d.values() if r["notes"])
        cells.append(f" {notes} |")
        print("".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
