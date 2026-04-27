"""Phase 6: memory flag toggles on CPU at fixed config.

Tests --no-mmap and --direct-io combinations to see if they affect
TG throughput on a 22 GB working-set model. mmap is the default
(virtual-mapped, pages on demand); --no-mmap forces a single-shot
load. Direct-IO bypasses the OS page cache.

Configs:
  - default               (mmap on,  direct-io off)
  - no_mmap               (mmap off, direct-io off)
  - direct_io             (mmap on,  direct-io on)
  - no_mmap_direct_io     (mmap off, direct-io on)

All at d=8192, KV f16, no FA, 8 threads (the canonical config).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_daily_driver_longctx import LLAMA_BENCH_TIMEOUT_S  # type: ignore
from bench_daily_driver import (  # type: ignore
    GGUF_CPU,
    REPO_ROOT,
    CSV_DIR,
    TRASH_ROOT,
    run_streaming,
    sample_power_online,
)


CONFIGS = [
    # (label, mmp_arg, dio_arg)
    ("default",            "1", "0"),
    ("no_mmap",            "0", "0"),
    ("direct_io",          "1", "1"),
    ("no_mmap_direct_io",  "0", "1"),
]


def run_one(label: str, mmp: str, dio: str, ctx_depth: int, threads: int,
            stale_timeout_s: int, log_path: Path) -> dict:
    import os
    bench_exe = REPO_ROOT / "llama.cpp" / "build-cpu" / "bin" / "llama-bench.exe"
    cmd = [
        str(bench_exe),
        "-m", str(GGUF_CPU),
        "-p", "0", "-n", "128",
        "-d", str(ctx_depth),
        "-r", "1",
        "-ctk", "f16", "-ctv", "f16",
        "-o", "json",
        "--progress",
        "-t", str(threads),
        "-mmp", mmp,
        "-dio", dio,
    ]
    print(f"  [{label}] cmd: {' '.join(cmd)}")
    print(f"  [{label}] log: {log_path}")
    t0 = time.perf_counter()
    rc, stdout_full, stderr_full, status = run_streaming(
        cmd, log_path, env=os.environ.copy(),
        hard_timeout_s=LLAMA_BENCH_TIMEOUT_S,
        stale_timeout_s=stale_timeout_s,
    )
    wall = time.perf_counter() - t0

    out = {"config": label, "mmap": mmp, "direct_io": dio,
           "ctx_depth": ctx_depth, "threads": threads,
           "tg_tps": "", "wall_s": f"{wall:.1f}",
           "ok": 0, "notes": status if status != "ok" else ""}

    if status != "ok" or rc != 0:
        if status == "stale_timeout":
            out["notes"] = f"stale timeout >{stale_timeout_s}s"
        elif status == "hard_timeout":
            out["notes"] = "hard timeout"
        else:
            tail = stderr_full.strip()[-200:] if stderr_full else ""
            out["notes"] = f"exit {rc}: {tail}"
        return out

    import json
    raw = stdout_full.strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        out["notes"] = f"json parse: {e}"
        return out
    for r in data:
        if r.get("n_gen", 0) == 128 and r.get("n_prompt", 0) == 0:
            ts = r.get("avg_ts")
            if ts is not None:
                out["tg_tps"] = f"{float(ts):.3f}"
                out["ok"] = 1
                break
    if not out["tg_tps"]:
        out["notes"] = "no TG row in JSON"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--ctx-depth", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--stale-timeout-s", type=int, default=600)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    tag = args.tag or f"phase6_{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = TRASH_ROOT / f"daily_driver_phase6_memflags_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"daily_driver_phase6_memflags_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, mmp, dio in CONFIGS:
        print(f"\n=== {label} (mmap={mmp}, dio={dio}) ===")
        log_path = log_dir / f"{label}.log"
        out = run_one(label, mmp, dio, args.ctx_depth, args.threads,
                      args.stale_timeout_s, log_path)
        rows.append(out)
        status = "OK" if out["ok"] else "FAIL"
        print(f"  [{status}] TG={out['tg_tps'] or '-'}  wall={out['wall_s']}s  {out['notes']}")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== Summary ===")
    print(f"csv: {csv_path}")
    print()
    print(f"Phase 6 mem-flags @ d={args.ctx_depth}, threads={args.threads}, KV f16, no FA:")
    print()
    print("| config              | mmap | direct_io | TG (t/s) |")
    print("|---|:-:|:-:|---:|")
    for r in rows:
        print(f"| {r['config']:20s} | {r['mmap']:4s} | {r['direct_io']:9s} | {r['tg_tps'] or '-'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
