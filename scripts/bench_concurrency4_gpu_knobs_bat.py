"""Concurrency=4 BAT bench for the 2026-04-26 GPU knob refresh.

Same matrix as `scripts/bench_concurrency4_gpu_knobs.py`
(llama-batched-bench -np 4 -npp 512 -ntg 128 -npl 4 across
OpenCL/Vulkan × Q4_0/Q4_K_M with the canonical knob combos), but
samples WMI DischargeRate during each run to compute J/tok at N=4 —
the open follow-up after the 2026-04-26 N=1 BAT and N=4 AC sweeps.

Run from a battery state (script verifies WMI PowerOnline):
    .venv/Scripts/python.exe scripts/bench_concurrency4_gpu_knobs_bat.py \\
        --tag 2026-04-26_bat
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_qwen3_4b_all_backends import (  # noqa: E402
    PowerSampler,
    sample_battery_mwh,
    sample_power_online,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"

N_PROMPT = 512
N_GEN = 128
N_PARALLEL = 4
CTX = 4096


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
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def run_one(label: str, preset: str, gguf: Path, extra_env: dict[str, str], log_path: Path) -> dict:
    bench = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-batched-bench.exe"
    row: dict = {
        "label": label, "preset": preset, "gguf": gguf.name,
        "knobs": " ".join(f"{k}={v}" for k, v in extra_env.items()) or "(none)",
        "ok": True,
    }
    if not bench.exists():
        row["ok"] = False; row["notes"] = f"binary missing: {bench}"
        return row
    if not gguf.exists():
        row["ok"] = False; row["notes"] = f"gguf missing: {gguf}"
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

    sampler = PowerSampler(interval_s=1.0)
    mwh_before = sample_battery_mwh()
    sampler.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    wall_s = time.perf_counter() - t0
    samples = sampler.stop()
    mwh_after = sample_battery_mwh()

    log_path.write_text(
        f"=== cmd ===\n{' '.join(cmd)}\n"
        f"=== env extra ===\n{extra_env}\n"
        f"=== exit {proc.returncode} wall {wall_s:.1f}s samples {len(samples)} ===\n"
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n",
        encoding="utf-8", errors="replace",
    )

    row.update({
        "wall_s": round(wall_s, 1),
        "exit_code": proc.returncode,
        "n_samples": len(samples),
        "mean_w": round(sampler.mean_watts, 2) if sampler.mean_watts else None,
        "mwh_before": mwh_before,
        "mwh_after": mwh_after,
        "mwh_drop": (mwh_before - mwh_after) if (mwh_before and mwh_after) else None,
        "ok": proc.returncode == 0,
    })

    if proc.returncode != 0:
        row["notes"] = f"exit {proc.returncode}"
        return row

    matched = False
    for line in proc.stdout.splitlines():
        m = PP_RE.search(line)
        if not m:
            continue
        pp, tg, b, n_kv, t_pp, s_pp, t_tg, s_tg, t_total, s_total = m.groups()
        if int(pp) != N_PROMPT or int(tg) != N_GEN or int(b) != N_PARALLEL:
            continue
        row["n_kv"] = int(n_kv)
        row["t_pp_s"] = float(t_pp)
        row["pp_t_per_s_agg"] = float(s_pp)
        row["t_tg_s"] = float(t_tg)
        row["tg_t_per_s_agg"] = float(s_tg)
        row["t_total_s"] = float(t_total)
        row["total_t_per_s_agg"] = float(s_total)
        matched = True
        break
    if not matched:
        row["ok"] = False
        row["notes"] = "couldn't find PP=512,TG=128,B=4 row"
        return row

    # J/tok metrics. At N=4 the "total tokens processed" is
    # N × (PP + TG) = 4 × (512 + 128) = 2560 (matches t/s definition
    # used by batched-bench's S column).
    total_tokens = N_PARALLEL * (N_PROMPT + N_GEN)
    if row["mean_w"] and total_tokens > 0:
        row["j_per_tok"] = round(row["mean_w"] * wall_s / total_tokens, 4)
    else:
        row["j_per_tok"] = None

    # J/gen-tok = (TG-window energy) / (N × TG_tokens). TG-window is
    # the t_tg_s reported by batched-bench. We assume mean_w ≈
    # constant across the run (true for GPU-bound runs where prefill
    # and decode draw similar power on the same Adreno; less true if
    # PP and TG dispatch different kernel families). Worth flagging
    # in the doc.
    n_gen_total = N_PARALLEL * N_GEN
    if row["mean_w"] and row.get("t_tg_s") and n_gen_total > 0:
        row["j_per_gen_tok"] = round(row["mean_w"] * row["t_tg_s"] / n_gen_total, 4)
    else:
        row["j_per_gen_tok"] = None

    print(
        f"  -> PP_agg {row['pp_t_per_s_agg']:.2f}  TG_agg {row['tg_t_per_s_agg']:.2f}  "
        f"total_agg {row['total_t_per_s_agg']:.2f}  mean_W {row['mean_w']}  "
        f"J/tok {row['j_per_tok']}  J/gen-tok {row['j_per_gen_tok']}  wall {wall_s:.1f}s",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=time.strftime("%Y-%m-%d_bat"))
    ap.add_argument("--skip-power-check", action="store_true")
    args = ap.parse_args()

    online = sample_power_online()
    if online and not args.skip_power_check:
        print("ERROR: WMI reports PowerOnline=True. Unplug or pass --skip-power-check.")
        return 2
    if online is None:
        print("WARNING: PowerOnline query failed; proceeding without check.")

    log_dir = TRASH_ROOT / f"concurrency4_gpu_knobs_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, preset, gguf_name, extra_env in RUNS:
        gguf = MODELS_DIR / gguf_name
        log_path = log_dir / f"{label}.log"
        rows.append(run_one(label, preset, gguf, extra_env, log_path))

    csv_path = CSV_DIR / f"concurrency4_gpu_knobs_{args.tag}.csv"
    cols = [
        "label", "preset", "gguf", "knobs",
        "pp_t_per_s_agg", "tg_t_per_s_agg", "total_t_per_s_agg",
        "mean_w", "j_per_tok", "j_per_gen_tok",
        "t_pp_s", "t_tg_s", "t_total_s", "n_kv",
        "wall_s", "n_samples", "mwh_drop",
        "mwh_before", "mwh_after", "exit_code", "ok", "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}", flush=True)
    print(f"Logs: {log_dir}", flush=True)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
