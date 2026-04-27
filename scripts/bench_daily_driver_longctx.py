"""Long-context bench for the daily_driver model.

Measures TG t/s at multiple KV-cache depths to characterize the
*real* operating curve for the coding-agent workload (per
daily_driver/README.md § Primary use case: 120k+ ctx, conc=1).

Uses llama-bench's `-d N` flag which **pre-fills the KV cache
with N tokens before timing TG** — so we measure TG-at-depth
without paying the prefill cost 3 times per ctx point.

CSV layout: one row per (backend, ctx_depth), with TG t/s and
wall-clock per row. PP is measured separately at -p 512 -n 0
(once per backend) to keep wall time bounded — long-ctx PP
costs are reported via the `-p N` mode in a separate sweep
(`scripts/bench_daily_driver_longctx_pp.py`, TBD if needed).

Usage:
    .venv/Scripts/python.exe scripts/bench_daily_driver_longctx.py \\
        --power-state ac --tag 2026-04-26_longctx_ac

    # Subset of backends:
    .venv/Scripts/python.exe scripts/bench_daily_driver_longctx.py \\
        --power-state ac --backends cpu,gpu-vulkan --tag vk_vs_cpu_longctx

    # Custom ctx points (default: 4096, 32768, 131072):
    .venv/Scripts/python.exe scripts/bench_daily_driver_longctx.py \\
        --power-state ac --ctx-depths 8192,65536 --tag mid_ctx
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the model paths and Vulkan env knobs from the main runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_daily_driver import (  # type: ignore
    GGUF_CPU,
    GGUF_GPU,
    CSV_DIR,
    TRASH_ROOT,
    REPO_ROOT,
    VULKAN_DEFAULT_ENV,
    STALE_TIMEOUT_S,
    gguf_for_preset,
    run_streaming,
    sample_power_online,
)


DEFAULT_CTX_DEPTHS = [4096, 32768, 131072]
TG_TOKENS = 128
DEFAULT_KV_TYPE = "q8_0"  # see optimization.md § What we're optimizing for

# A single ctx=131072 run on CPU could spend a few minutes prefilling
# the KV cache (even though prefill is untimed, it still walks 128k
# tokens through the matmul path). With -r 1 the per-row cost is
# ~ctx/PP_t/s + (TG_TOKENS/TG_t/s). At PP=135 and TG=34 on CPU:
#   ctx=4096   →   30 s prefill +  4 s TG ≈ 35 s
#   ctx=32768  →  240 s prefill +  4 s TG ≈ 4 min
#   ctx=131072 →  970 s prefill +  4 s TG ≈ 16 min
# Per-row timeout 30 min covers the worst case with margin.
LLAMA_BENCH_TIMEOUT_S = 1800


@dataclass
class LongCtxRow:
    backend: str
    model: str
    ctx_depth: int
    tg_tps: float | None = None
    pp_tps: float | None = None  # only populated for the ctx=0 baseline row
    wall_s: float | None = None
    notes: str = ""
    ok: bool = True
    extra: dict = field(default_factory=dict)


def run_longctx_one(
    preset: str,
    ctx_depth: int,
    log_path: Path,
    threads: int | None,
    ngl: int,
    extra_env: dict | None,
    kv_type: str,
    ctx_buffer: int,
) -> LongCtxRow:
    """One llama-bench invocation, depth=ctx_depth, n=128, p=0.

    A separate PP-only baseline row (ctx_depth=0, p=512, n=0) is
    handled by the caller with a different argv shape.
    """
    name = {
        "cpu": "cpu",
        "cpu-kleidiai": "cpu-kleidiai",
        "opencl": "gpu-opencl",
        "vulkan": "gpu-vulkan",
    }[preset]
    gguf = gguf_for_preset(preset)
    row = LongCtxRow(backend=name, model=gguf.name, ctx_depth=ctx_depth)

    bench_exe = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-bench.exe"
    if not bench_exe.exists():
        row.ok = False
        row.notes = f"binary missing: {bench_exe}"
        return row
    if not gguf.exists():
        row.ok = False
        row.notes = f"gguf missing: {gguf.name}"
        return row

    cmd = [
        str(bench_exe),
        "-m", str(gguf),
        "-p", "0",
        "-n", str(TG_TOKENS),
        "-d", str(ctx_depth),
        "-r", "1",
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-o", "json",
        "--progress",  # progress lines power the stale-output watchdog
                       # AND make `tail -f log_path` show real-time
                       # progress mid-run (the per-rep prefill on a
                       # 131k-depth run can be 15+ min on CPU).
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl:
        cmd += ["-ngl", str(ngl)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
        row.extra["env"] = dict(extra_env)

    print(f"  [{name} d={ctx_depth}] cmd: {' '.join(cmd)}")
    print(f"  [{name} d={ctx_depth}] log: {log_path}  (tail -f to watch)")

    t0 = time.perf_counter()
    rc, stdout_full, stderr_full, status = run_streaming(
        cmd, log_path, env,
        hard_timeout_s=LLAMA_BENCH_TIMEOUT_S,
        stale_timeout_s=STALE_TIMEOUT_S,
    )
    row.wall_s = time.perf_counter() - t0
    row.extra["exit_code"] = rc
    row.extra["status"] = status

    if status == "stale_timeout":
        row.ok = False
        row.notes = f"stale output timeout (>{STALE_TIMEOUT_S}s) — likely GPU livelock"
        return row
    if status == "hard_timeout":
        row.ok = False
        row.notes = f"hard timeout {LLAMA_BENCH_TIMEOUT_S}s exceeded"
        return row
    if rc != 0:
        tail = stderr_full.strip()[-200:] if stderr_full else ""
        row.ok = False
        row.notes = f"llama-bench exit {rc}; stderr tail: {tail}"
        return row

    raw = stdout_full.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.rstrip(",") + "]")
        except json.JSONDecodeError as e:
            row.ok = False
            row.notes = f"json parse: {e}"
            return row

    for r in data:
        if r.get("n_gen", 0) == TG_TOKENS and r.get("n_prompt", 0) == 0:
            ts = r.get("avg_ts")
            if ts is not None:
                row.tg_tps = float(ts)
                break

    if row.tg_tps is None:
        row.ok = False
        row.notes = f"no TG row found in JSON (rows: {[(r.get('n_prompt'), r.get('n_gen'), r.get('n_depth')) for r in data]})"

    return row


def write_csv(csv_path: Path, rows: list[LongCtxRow], power_state: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "backend", "model", "ctx_depth", "tg_tps", "pp_tps",
            "wall_s", "ok", "notes", "power_state", "env",
        ])
        for r in rows:
            w.writerow([
                r.backend, r.model, r.ctx_depth,
                f"{r.tg_tps:.3f}" if r.tg_tps is not None else "",
                f"{r.pp_tps:.3f}" if r.pp_tps is not None else "",
                f"{r.wall_s:.1f}" if r.wall_s is not None else "",
                "1" if r.ok else "0",
                r.notes,
                power_state,
                json.dumps(r.extra.get("env")) if r.extra.get("env") else "",
            ])


def format_md_table(rows: list[LongCtxRow], depths: list[int], tag: str) -> str:
    """Pivot: backend rows × ctx_depth columns. TG t/s in each cell."""
    by_backend: dict[str, dict[int, LongCtxRow]] = {}
    for r in rows:
        by_backend.setdefault(r.backend, {})[r.ctx_depth] = r

    lines = [
        f"# daily_driver / Qwen3.6-35B-A3B — long-context TG sweep — {tag}",
        "",
        f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "KV at q8_0; TG measured after KV pre-fill via llama-bench `-d N`.",
        "",
    ]
    header = ["| backend |"] + [f" TG@{d} (t/s) |" for d in depths] + [" notes |"]
    sep = ["|---|"] + ["---:|" for _ in depths] + ["---|"]
    lines.append("".join(header))
    lines.append("".join(sep))
    for backend, by_d in by_backend.items():
        cells = [f"| {backend} |"]
        any_notes = []
        for d in depths:
            row = by_d.get(d)
            if row is None or not row.ok:
                cells.append(" — |")
                if row and row.notes:
                    any_notes.append(f"d={d}: {row.notes}")
            else:
                cells.append(f" {row.tg_tps:.2f} |")
        cells.append(f" {'; '.join(any_notes)} |")
        lines.append("".join(cells))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--backends", default="all",
                        help="Comma-separated subset or 'all'. "
                             "Choices: cpu, cpu-kleidiai, gpu-opencl, gpu-vulkan")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-power-check", action="store_true")
    parser.add_argument("--no-vulkan-env", action="store_true")
    parser.add_argument("--ctx-depths", default=None,
                        help=f"Comma-separated KV-cache depths. "
                             f"Default: {DEFAULT_CTX_DEPTHS}")
    parser.add_argument("--kv-type", default=DEFAULT_KV_TYPE,
                        help=f"KV cache quant type (default: {DEFAULT_KV_TYPE})")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU thread count (default: 8)")
    args = parser.parse_args()

    tag = args.tag or f"longctx_{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = TRASH_ROOT / f"daily_driver_longctx_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    online = sample_power_online()
    if online is None:
        print("WARNING: could not query PowerOnline; proceeding without check")
    elif args.power_state == "ac" and not online:
        if not args.skip_power_check:
            print("ERROR: --power-state ac but WMI reports PowerOnline=False.")
            return 2
    elif args.power_state == "bat" and online:
        if not args.skip_power_check:
            print("ERROR: --power-state bat but WMI reports PowerOnline=True.")
            return 2

    if args.ctx_depths:
        depths = [int(s.strip()) for s in args.ctx_depths.split(",")]
    else:
        depths = list(DEFAULT_CTX_DEPTHS)

    # ctx_buffer needs to be >= max depth + TG_TOKENS.
    ctx_buffer = max(depths) + 256

    vulkan_env = None if args.no_vulkan_env else dict(VULKAN_DEFAULT_ENV)
    backend_specs: list[tuple[str, dict]] = [
        ("cpu",          {"preset": "cpu",          "threads": args.threads, "ngl": 0,  "extra_env": None}),
        ("cpu-kleidiai", {"preset": "cpu-kleidiai", "threads": args.threads, "ngl": 0,  "extra_env": None}),
        ("gpu-opencl",   {"preset": "opencl",       "threads": None,         "ngl": 99, "extra_env": None}),
        ("gpu-vulkan",   {"preset": "vulkan",       "threads": None,         "ngl": 99, "extra_env": vulkan_env}),
    ]

    if args.backends == "all":
        selected = backend_specs
    else:
        wanted = set(s.strip() for s in args.backends.split(","))
        selected = [b for b in backend_specs if b[0] in wanted]
        if {b[0] for b in selected} != wanted:
            missing = wanted - {b[0] for b in selected}
            print(f"ERROR: unknown backend(s): {missing}")
            return 2

    rows: list[LongCtxRow] = []
    for name, spec in selected:
        for d in depths:
            print(f"\n=== {name} (depth={d}, kv={args.kv_type}) ===")
            log_path = log_dir / f"{name}_d{d}.log"
            row = run_longctx_one(
                preset=spec["preset"],
                ctx_depth=d,
                log_path=log_path,
                threads=spec["threads"],
                ngl=spec["ngl"],
                extra_env=spec["extra_env"],
                kv_type=args.kv_type,
                ctx_buffer=ctx_buffer,
            )
            rows.append(row)
            status = "OK" if row.ok else "FAIL"
            tg = f"{row.tg_tps:.2f} t/s" if row.tg_tps else "—"
            wall = f"{row.wall_s:.1f}s" if row.wall_s else "—"
            print(f"  [{status}] TG={tg}  wall={wall}  {row.notes}")

            # Periodic csv flush so a mid-run crash doesn't lose data.
            csv_path = CSV_DIR / f"daily_driver_longctx_{tag}.csv"
            write_csv(csv_path, rows, args.power_state)

    csv_path = CSV_DIR / f"daily_driver_longctx_{tag}.csv"
    write_csv(csv_path, rows, args.power_state)
    print()
    print(format_md_table(rows, depths, tag))
    print(f"=== Summary ===")
    print(f"csv:  {csv_path}")
    print(f"logs: {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
