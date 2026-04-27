"""BAT-only bench for the 2026-04-26 GPU knob refresh.

Reuses the WMI DischargeRate sampler from
`scripts/bench_qwen3_4b_all_backends.py` but lets us pick the GGUF
file and pass arbitrary env vars to llama-bench — the original driver
hard-codes Q4_K_M and has no env passthrough, which is why the
2026-04-26 OpenCL-Q4_0 / Vulkan-DISABLE_F16 runs need a separate
wrapper. Outputs CSV in the same format the consolidated doc expects.

Run from a battery state (the script verifies WMI PowerOnline):
    .venv/Scripts/python.exe scripts/bench_qwen3_4b_gpu_knobs_bat.py \\
        --tag 2026-04-26_bat
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Reuse helpers from the AC driver — same WMI sampling, same JSON
# parsing semantics.
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


# (name, build-preset, gguf filename, extra_env). Each name is a row
# in the resulting CSV. Mirrors the canonical-knob choices from the
# 2026-04-26 AC sweep.
RUNS: list[tuple[str, str, str, dict[str, str]]] = [
    ("gpu-opencl-q40",     "opencl", "Qwen3-4B-Q4_0.gguf",   {}),
    ("gpu-vulkan-q40-fix", "vulkan", "Qwen3-4B-Q4_0.gguf",   {
        "GGML_VK_DISABLE_F16": "1",
        "GGML_VK_PREFER_HOST_MEMORY": "1",
    }),
    # Keep one Q4_K_M row per backend for AC↔BAT consistency check
    # vs the 2026-04-23 baseline.
    ("gpu-opencl-q4km",     "opencl", "Qwen3-4B-Q4_K_M.gguf", {}),
    ("gpu-vulkan-q4km-fix", "vulkan", "Qwen3-4B-Q4_K_M.gguf", {
        "GGML_VK_DISABLE_F16": "1",
        "GGML_VK_PREFER_HOST_MEMORY": "1",
    }),
]


def run_one(
    label: str,
    preset: str,
    gguf: Path,
    extra_env: dict[str, str],
    log_path: Path,
) -> dict:
    """Run one llama-bench config, sample power, return one CSV row."""
    bench = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-bench.exe"
    if not bench.exists():
        return {"label": label, "ok": False, "notes": f"binary missing: {bench}"}
    if not gguf.exists():
        return {"label": label, "ok": False, "notes": f"gguf missing: {gguf}"}

    cmd = [
        str(bench),
        "-m", str(gguf),
        "-p", "512", "-n", "128", "-r", "3",
        "-ngl", "99", "-o", "json",
    ]

    env = os.environ.copy()
    env.update(extra_env)

    print(f"\n=== {label} ===")
    print(f"  preset: {preset}  gguf: {gguf.name}")
    print(f"  env:    {extra_env or '(none)'}")
    print(f"  cmd:    {' '.join(cmd)}")

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
        f"=== env extra ===\n{json.dumps(extra_env)}\n"
        f"=== exit {proc.returncode} wall {wall_s:.1f}s samples {len(samples)} ===\n"
        f"=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n",
        encoding="utf-8", errors="replace",
    )

    row: dict = {
        "label": label,
        "preset": preset,
        "gguf": gguf.name,
        "knobs": " ".join(f"{k}={v}" for k, v in extra_env.items()) or "(none)",
        "wall_s": round(wall_s, 1),
        "exit_code": proc.returncode,
        "n_samples": len(samples),
        "mean_w": round(sampler.mean_watts, 2) if sampler.mean_watts else None,
        "mwh_before": mwh_before,
        "mwh_after": mwh_after,
        "mwh_drop": (mwh_before - mwh_after) if (mwh_before and mwh_after) else None,
        "ok": proc.returncode == 0,
    }

    # Parse llama-bench JSON for PP/TG t/s.
    raw = proc.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.rstrip(",") + "]")
        except json.JSONDecodeError:
            row["ok"] = False
            row["notes"] = "json parse failed"
            return row

    pp_tps = tg_tps = None
    pp_tokens = tg_tokens = 0
    for r in data:
        ts = r.get("avg_ts")
        if ts is None:
            continue
        if r.get("n_prompt") == 512 and r.get("n_gen") == 0:
            pp_tps = float(ts); pp_tokens = 512
        elif r.get("n_prompt") == 0 and r.get("n_gen") == 128:
            tg_tps = float(ts); tg_tokens = 128
    row["pp_tps"] = pp_tps
    row["tg_tps"] = tg_tps
    row["pp_tokens"] = pp_tokens
    row["tg_tokens"] = tg_tokens

    # J/tok = mean_W * wall_s / total_tokens. Same definition as
    # qwen3_4b_baseline_all_backends.md "Headline — battery" table.
    if row["mean_w"] and (pp_tokens + tg_tokens) > 0:
        row["j_per_tok"] = round(row["mean_w"] * wall_s / (pp_tokens + tg_tokens), 3)
    else:
        row["j_per_tok"] = None

    print(
        f"  -> PP {pp_tps} TG {tg_tps} mean_W {row['mean_w']} "
        f"J/tok {row['j_per_tok']} wall {wall_s:.1f}s"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=time.strftime("%Y-%m-%d_bat"),
                    help="Tag for csv + log file names. Default %Y-%m-%d_bat.")
    ap.add_argument("--skip-power-check", action="store_true",
                    help="Run even if WMI says we're plugged in.")
    args = ap.parse_args()

    online = sample_power_online()
    if online and not args.skip_power_check:
        print("ERROR: WMI reports PowerOnline=True. Unplug or pass --skip-power-check.")
        return 2
    if online is None:
        print("WARNING: PowerOnline query failed; proceeding without check.")

    log_dir = TRASH_ROOT / f"qwen3_4b_gpu_knobs_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, preset, gguf_name, extra_env in RUNS:
        gguf = MODELS_DIR / gguf_name
        log_path = log_dir / f"{label}.log"
        rows.append(run_one(label, preset, gguf, extra_env, log_path))

    csv_path = CSV_DIR / f"qwen3_4b_gpu_knobs_{args.tag}.csv"
    cols = [
        "label", "preset", "gguf", "knobs",
        "pp_tps", "tg_tps", "mean_w", "j_per_tok",
        "wall_s", "mwh_drop", "n_samples",
        "pp_tokens", "tg_tokens", "mwh_before", "mwh_after",
        "exit_code", "ok",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")
    print(f"Logs: {log_dir}")
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
