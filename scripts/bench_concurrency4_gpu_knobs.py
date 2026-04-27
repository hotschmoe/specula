"""Concurrency=4 bench for the 2026-04-26 GPU knob refresh.

Same shape as `scripts/bench_concurrency4_all_backends.py` (drives
llama-batched-bench with -np 4 -npp 512 -ntg 128 -npl 4) but lets us
pick the GGUF file and pass arbitrary env vars — needed because the
2026-04-26 GPU canonical configs use a Q4_0 model and (for Vulkan)
require GGML_VK_DISABLE_F16 + GGML_VK_PREFER_HOST_MEMORY. The
original driver hard-codes Q4_K_M and only supports cpu /
cpu-kleidiai / opencl presets.

Outputs CSV in the same column layout as the consolidated doc's
"Concurrency = 4" headline table.

Run:
    .venv/Scripts/python.exe scripts/bench_concurrency4_gpu_knobs.py \\
        --tag 2026-04-26_ac
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"

N_PROMPT = 512
N_GEN = 128
N_PARALLEL = 4
CTX = 4096


# (label, build-preset, gguf filename, extra_env). Mirrors the
# canonical-knob choices from the 2026-04-26 single-stream sweep,
# plus a Q4_K_M comparator per backend so the new rows can be diffed
# against the 2026-04-23 concurrency-4 baseline.
RUNS: list[tuple[str, str, str, dict[str, str]]] = [
    ("gpu-opencl-q40",     "opencl", "Qwen3-4B-Q4_0.gguf",   {}),
    ("gpu-vulkan-q40-fix", "vulkan", "Qwen3-4B-Q4_0.gguf",   {
        "GGML_VK_DISABLE_F16": "1",
        "GGML_VK_PREFER_HOST_MEMORY": "1",
    }),
    ("gpu-opencl-q4km",     "opencl", "Qwen3-4B-Q4_K_M.gguf", {}),
    ("gpu-vulkan-q4km-fix", "vulkan", "Qwen3-4B-Q4_K_M.gguf", {
        "GGML_VK_DISABLE_F16": "1",
        "GGML_VK_PREFER_HOST_MEMORY": "1",
    }),
]


PP_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"  # PP TG B N_KV
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T_PP S_PP
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T_TG S_TG
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"                          # T S
)


@dataclass
class Row:
    label: str
    preset: str
    gguf: str
    knobs: str
    pp_t_per_s_agg: float | None = None
    tg_t_per_s_agg: float | None = None
    total_t_per_s_agg: float | None = None
    t_pp_s: float | None = None
    t_tg_s: float | None = None
    t_total_s: float | None = None
    n_kv: int | None = None
    wall_s: float | None = None
    exit_code: int | None = None
    ok: bool = True
    notes: str = ""


def run_one(label: str, preset: str, gguf: Path, extra_env: dict[str, str], log_path: Path) -> Row:
    bench = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-batched-bench.exe"
    row = Row(
        label=label, preset=preset, gguf=gguf.name,
        knobs=" ".join(f"{k}={v}" for k, v in extra_env.items()) or "(none)",
    )
    if not bench.exists():
        row.ok = False
        row.notes = f"binary missing: {bench}"
        return row
    if not gguf.exists():
        row.ok = False
        row.notes = f"gguf missing: {gguf}"
        return row

    cmd = [
        str(bench),
        "-m", str(gguf),
        "-c", str(CTX),
        "-npp", str(N_PROMPT),
        "-ntg", str(N_GEN),
        "-npl", str(N_PARALLEL),
        "-ngl", "99",
        "--output-format", "md",
    ]

    env = os.environ.copy()
    env.update(extra_env)

    print(f"\n=== {label} ===", flush=True)
    print(f"  preset: {preset}  gguf: {gguf.name}", flush=True)
    print(f"  env:    {extra_env or '(none)'}", flush=True)
    print(f"  cmd:    {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    row.wall_s = round(time.perf_counter() - t0, 1)
    row.exit_code = proc.returncode

    log_path.write_text(
        f"=== cmd ===\n{' '.join(cmd)}\n"
        f"=== env extra ===\n{extra_env}\n"
        f"=== exit {proc.returncode} wall {row.wall_s:.1f}s ===\n"
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n",
        encoding="utf-8", errors="replace",
    )

    if proc.returncode != 0:
        row.ok = False
        row.notes = f"exit {proc.returncode}"
        return row

    matched = False
    for line in proc.stdout.splitlines():
        m = PP_RE.search(line)
        if not m:
            continue
        pp, tg, b, n_kv, t_pp, s_pp, t_tg, s_tg, t_total, s_total = m.groups()
        if int(pp) != N_PROMPT or int(tg) != N_GEN or int(b) != N_PARALLEL:
            continue
        row.n_kv = int(n_kv)
        row.t_pp_s = float(t_pp)
        row.pp_t_per_s_agg = float(s_pp)
        row.t_tg_s = float(t_tg)
        row.tg_t_per_s_agg = float(s_tg)
        row.t_total_s = float(t_total)
        row.total_t_per_s_agg = float(s_total)
        matched = True
        break
    if not matched:
        row.ok = False
        row.notes = "couldn't find PP=512,TG=128,B=4 row"
        return row

    print(
        f"  -> PP_agg {row.pp_t_per_s_agg:.2f}  TG_agg {row.tg_t_per_s_agg:.2f}  "
        f"total_agg {row.total_t_per_s_agg:.2f}  wall {row.wall_s:.1f}s",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=time.strftime("%Y-%m-%d_ac"),
                    help="Tag for csv + log dir (default %Y-%m-%d_ac).")
    args = ap.parse_args()

    log_dir = TRASH_ROOT / f"concurrency4_gpu_knobs_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for label, preset, gguf_name, extra_env in RUNS:
        gguf = MODELS_DIR / gguf_name
        log_path = log_dir / f"{label}.log"
        rows.append(run_one(label, preset, gguf, extra_env, log_path))

    csv_path = CSV_DIR / f"concurrency4_gpu_knobs_{args.tag}.csv"
    cols = [
        "label", "preset", "gguf", "knobs",
        "pp_t_per_s_agg", "tg_t_per_s_agg", "total_t_per_s_agg",
        "t_pp_s", "t_tg_s", "t_total_s", "n_kv",
        "wall_s", "exit_code", "ok", "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: getattr(r, c) for c in cols})
    print(f"\nCSV: {csv_path}", flush=True)
    print(f"Logs: {log_dir}", flush=True)
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
