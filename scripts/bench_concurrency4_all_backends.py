"""Concurrency-4 (agentic workload) bench across CPU + KleidiAI + OpenCL.

Drives llama-batched-bench with --np 4 to simulate 4 concurrent agents
each prompting with 512 tokens and decoding 128. Reports aggregate
prefill + decode throughput per backend, plus per-stream throughput.

NPU is intentionally absent: Genie is single-stream and the multi-stream
ORT-QNN sidecar path is deferred (see W4 in docs/roadmap.md). Vulkan is
skipped because its PP collapses at concurrency=1 already (3.91 t/s on
4B) — concurrency=4 won't recover.

Usage:
    .venv/Scripts/python.exe scripts/bench_concurrency4_all_backends.py \\
        --model qwen3_4b --power-state ac --tag 2026-04-25_ac

    .venv/Scripts/python.exe scripts/bench_concurrency4_all_backends.py \\
        --model qwen2_5_7b --power-state ac --tag 2026-04-25_ac

Outputs:
    results/csv/concurrency4_<model>_<tag>.csv
    marked_for_deletion/concurrency4_<model>_<tag>/<backend>.log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"

MODELS = {
    "qwen3_4b": REPO_ROOT / "models" / "Qwen3-4B-Q4_K_M.gguf",
    "qwen2_5_7b": REPO_ROOT / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

# Concurrency knobs. 4 streams × (512 prompt + 128 gen) = 2560 tokens of
# KV state. ctx 4096 is plenty.
N_PARALLEL = 4
N_PROMPT = 512
N_GEN = 128
CTX = 4096


@dataclass
class BackendResult:
    name: str
    n_parallel: int = 0
    n_prompt: int = 0
    n_gen: int = 0
    n_kv: int = 0
    t_pp_s: float | None = None        # aggregate prefill wall (s)
    s_pp_tps: float | None = None      # aggregate prefill t/s (across all streams)
    t_tg_s: float | None = None        # aggregate decode wall (s)
    s_tg_tps: float | None = None      # aggregate decode t/s (across all streams)
    t_total_s: float | None = None
    s_total_tps: float | None = None
    wall_s: float | None = None
    notes: str = ""
    ok: bool = True


def run_batched_bench(
    preset: str,
    log_path: Path,
    model_path: Path,
    threads: int | None = None,
    ngl: int = 0,
) -> BackendResult:
    """Drive llama-batched-bench, parse the markdown table from stdout.

    The tool prints a single-row table per (PP, TG, B) combo; with one
    --npp / --ntg / --npl value we get exactly one data row.
    """
    name = {"cpu": "cpu", "cpu-kleidiai": "cpu-kleidiai", "opencl": "gpu-opencl"}[preset]
    result = BackendResult(name=name, n_parallel=N_PARALLEL, n_prompt=N_PROMPT, n_gen=N_GEN)
    bench_exe = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-batched-bench.exe"
    if not bench_exe.exists():
        result.ok = False
        result.notes = f"binary missing: {bench_exe}"
        return result

    cmd = [
        str(bench_exe),
        "-m", str(model_path),
        "-c", str(CTX),
        "-npp", str(N_PROMPT),
        "-ntg", str(N_GEN),
        "-npl", str(N_PARALLEL),
        "--output-format", "md",
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl:
        cmd += ["-ngl", str(ngl)]

    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        f.write("=== stdout ===\n")
        f.write(proc.stdout)
        f.write("\n=== stderr ===\n")
        f.write(proc.stderr)
    result.wall_s = time.perf_counter() - t0

    if proc.returncode != 0:
        result.ok = False
        result.notes = f"exit {proc.returncode}"
        return result

    # Parse the markdown table. Format (one row of interest):
    #   |   PP |   TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |   T s | S t/s |
    #   |------|------|------|--------|----------|----------|----------|----------|-------|-------|
    #   |  512 |  128 |    4 |   2560 |    0.42  |  4876.19 |    5.10  |  100.39  |  5.52 | 463.77|
    #
    # The columns are separated by `|`; data row has 10 numeric fields.
    pp_re = re.compile(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"  # PP TG B N_KV
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T_PP S_PP
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T_TG S_TG
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T S
    )
    matched = False
    for line in proc.stdout.splitlines():
        m = pp_re.search(line)
        if not m:
            continue
        pp, tg, b, n_kv, t_pp, s_pp, t_tg, s_tg, t_total, s_total = m.groups()
        if int(pp) != N_PROMPT or int(tg) != N_GEN or int(b) != N_PARALLEL:
            continue
        result.n_kv = int(n_kv)
        result.t_pp_s = float(t_pp)
        result.s_pp_tps = float(s_pp)
        result.t_tg_s = float(t_tg)
        result.s_tg_tps = float(s_tg)
        result.t_total_s = float(t_total)
        result.s_total_tps = float(s_total)
        matched = True
        break

    if not matched:
        result.ok = False
        result.notes = "couldn't find PP=512,TG=128,B=4 row in stdout"

    return result


BACKENDS: list[tuple[str, dict]] = [
    ("cpu",          {"preset": "cpu", "threads": 8}),
    ("cpu-kleidiai", {"preset": "cpu-kleidiai", "threads": 8}),
    ("gpu-opencl",   {"preset": "opencl", "ngl": 99}),
]


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def format_md_table(rows: list[dict], model: str, power_state: str, tag: str) -> str:
    lines = [
        f"# Concurrency-{N_PARALLEL} bench — {model} — {power_state.upper()} — {tag}",
        "",
        f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        f"4 parallel streams × ({N_PROMPT} prompt + {N_GEN} gen) tokens.",
        "",
        "| backend | S_PP (agg t/s) | S_TG (agg t/s) | S_TG/stream | T total (s) | wall (s) | ok | notes |",
        "|---|---:|---:|---:|---:|---:|:-:|---|",
    ]
    for r in rows:
        def fmt(v, spec=".2f"):
            if v is None:
                return "—"
            try:
                return format(v, spec)
            except Exception:
                return str(v)
        s_tg_per_stream = (r["s_tg_tps"] / N_PARALLEL) if r.get("s_tg_tps") else None
        lines.append(
            f"| {r['backend']} | {fmt(r['s_pp_tps'])} | {fmt(r['s_tg_tps'])} | "
            f"{fmt(s_tg_per_stream)} | {fmt(r['t_total_s'])} | "
            f"{fmt(r['wall_s'], '.1f')} | {'✓' if r['ok'] else '✗'} | {r['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    model_path = MODELS[args.model]
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 2

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = TRASH_ROOT / f"concurrency{N_PARALLEL}_{args.model}_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, kwargs in BACKENDS:
        print(f"\n=== {name} (model={args.model}, power={args.power_state}) ===")
        log_path = log_dir / f"{name}.log"
        try:
            result = run_batched_bench(log_path=log_path, model_path=model_path, **kwargs)
        except Exception as e:
            result = BackendResult(name=name, ok=False, notes=f"exception: {type(e).__name__}: {e}")
        row = dict(
            backend=result.name,
            n_parallel=result.n_parallel,
            n_prompt=result.n_prompt,
            n_gen=result.n_gen,
            n_kv=result.n_kv,
            t_pp_s=result.t_pp_s,
            s_pp_tps=result.s_pp_tps,
            t_tg_s=result.t_tg_s,
            s_tg_tps=result.s_tg_tps,
            t_total_s=result.t_total_s,
            s_total_tps=result.s_total_tps,
            wall_s=result.wall_s,
            model=args.model,
            power_state=args.power_state,
            ok=result.ok,
            notes=result.notes,
        )
        rows.append(row)
        per_stream = (result.s_tg_tps / N_PARALLEL) if result.s_tg_tps else None
        print(f"  S_PP: {result.s_pp_tps}  S_TG: {result.s_tg_tps}  per-stream TG: {per_stream}")
        print(f"  wall: {result.wall_s:.1f}s")
        if not result.ok:
            print(f"  !! NOT OK: {result.notes}")

    csv_path = CSV_DIR / f"concurrency{N_PARALLEL}_{args.model}_{tag}.csv"
    write_csv(csv_path, rows)
    table = format_md_table(rows, args.model, args.power_state, tag)
    print()
    print(table)
    print(f"=== Summary ===")
    print(f"csv:  {csv_path}")
    print(f"logs: {log_dir}  (gitignored, soak then rm -rf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
