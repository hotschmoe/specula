"""One-call bench driver for the Qwen2.5-7B-Instruct all-backends side-quest.

Side-quest companion to scripts/bench_qwen3_4b_all_backends.py — same
backend set, same protocol, repointed at the Qwen2.5-7B-Instruct
artifacts: the AI Hub Workbench-compiled X2 Elite Genie bundle and the
bartowski Q4_K_M GGUF. See docs/qwen2_5_7b_baseline_all_backends.md.

Backends:
    npu-genie      — Qualcomm Genie SDK (genie-t2t-run.exe) + w4a16 bundle
    cpu            — llama.cpp build-cpu, -t 8
    cpu-kleidiai   — llama.cpp build-cpu-kleidiai, -t 8
    gpu-opencl     — llama.cpp build-opencl, -ngl 99
    gpu-vulkan     — llama.cpp build-vulkan, -ngl 99

Usage:
    .venv/Scripts/python.exe scripts/bench_qwen2_5_7b_all_backends.py \\
        --power-state ac --tag 2026-04-25_ac

    .venv/Scripts/python.exe scripts/bench_qwen2_5_7b_all_backends.py \\
        --power-state bat --tag 2026-04-25_bat

Outputs:
    results/csv/qwen2_5_7b_baseline_<tag>.csv             — machine-readable
    marked_for_deletion/qwen2_5_7b_baseline_<tag>/        — raw per-backend logs
    stdout markdown — paste into docs/qwen2_5_7b_baseline_all_backends.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen2_5-7b-ref"
    / "qwen2_5_7b_instruct-genie-w8a16-qualcomm_snapdragon_x2_elite"
)
GGUF_PATH = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
# Layout follows docs/repo_hygiene.md:
#   - prompt scaffolding is a pinned input, lives in results/ (kept).
#   - CSVs are the permanent measurement record -> results/csv/ (kept).
#   - Raw logs are chaff once findings are captured -> marked_for_deletion/
#     (gitignored; rm -rf after a soak). Log dir is per-tag so multiple
#     reruns don't step on each other.
PROMPT_DIR = REPO_ROOT / "results" / "qwen2_5_7b_baseline"
PROMPT_PATH = PROMPT_DIR / "pp512_prompt.txt"
QAIRT_ROOT = Path("C:/Qualcomm/AIStack/QAIRT/2.45.40.260406")
CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"


@dataclass
class BackendResult:
    name: str
    pp_tps: float | None = None        # prefill throughput, tokens/sec
    tg_tps: float | None = None        # decode throughput, tokens/sec
    tg_tokens: int = 0
    pp_tokens: int = 0
    pp_time_s: float | None = None
    tg_time_s: float | None = None
    wall_s: float | None = None
    notes: str = ""
    ok: bool = True
    extra: dict = field(default_factory=dict)


# ─── battery helpers ────────────────────────────────────────────────────────

def sample_battery_mwh() -> int | None:
    """Return remaining battery capacity in mWh, or None if unavailable.

    Uses WMI BatteryStatus class (namespace root\\wmi) which reports
    true remaining mWh (not the EstimatedChargeRemaining percentage,
    which rounds to 1% ≈ 700 mWh on this battery — too coarse).
    """
    try:
        ps = (
            "(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus | "
            "Select-Object -First 1).RemainingCapacity"
        )
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        v = out.stdout.strip()
        if v and v.isdigit():
            return int(v)
    except Exception:
        pass
    return None


def sample_discharge_mw() -> int | None:
    """Instantaneous battery discharge in mW (0 if on AC + full)."""
    try:
        ps = (
            "(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus | "
            "Select-Object -First 1).DischargeRate"
        )
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        v = out.stdout.strip()
        if v and v.lstrip("-").isdigit():
            return int(v)
    except Exception:
        pass
    return None


class PowerSampler:
    """Polls DischargeRate (mW) in a background thread; average × wall
    time → energy_J. RemainingCapacity has coarse update resolution
    (seconds-scale gaps) that blinds it to <30s benchmarks, so we use
    DischargeRate instead, which is instantaneous and updates sub-second.
    """
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            v = sample_discharge_mw()
            if v is not None and v > 0:
                self._samples.append(v)
            self._stop.wait(self.interval_s)

    def stop(self) -> list[int]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._samples

    @property
    def mean_watts(self) -> float | None:
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples) / 1000.0

    @property
    def n_samples(self) -> int:
        return len(self._samples)


def sample_power_online() -> bool | None:
    """True if AC power connected, False on battery, None on error."""
    try:
        ps = (
            "(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus | "
            "Select-Object -First 1).PowerOnline"
        )
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        v = out.stdout.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    except Exception:
        pass
    return None


# ─── backend: npu-genie ─────────────────────────────────────────────────────

GRAPH_START_RE = re.compile(
    r"Genie:\s+([\d.]+)ms\s+\[\s*INFO\s*\]\s+<I>\s+QnnGraph_execute started\."
)
GRAPH_DONE_RE = re.compile(
    r"Genie:\s+([\d.]+)ms\s+\[\s*INFO\s*\]\s+<I>\s+Graph\s+(\S+)\s+execution finished"
)


def run_npu_genie(log_path: Path) -> BackendResult:
    """Drive genie-t2t-run with the 512-token prompt + --log info, parse
    per-graph execution timestamps to separate PP (prompt_ar128_*) from
    TG (token_ar1_*).
    """
    result = BackendResult(name="npu-genie")
    if not PROMPT_PATH.exists():
        result.ok = False
        result.notes = f"prompt file missing: {PROMPT_PATH}"
        return result

    env = os.environ.copy()
    qairt_lib = QAIRT_ROOT / "lib" / "aarch64-windows-msvc"
    env["PATH"] = f"{qairt_lib};{env.get('PATH', '')}"

    # genie_config.json uses relative paths → must cwd into bundle dir.
    # Use absolute prompt path.
    cmd = [
        str(QAIRT_ROOT / "bin" / "aarch64-windows-msvc" / "genie-t2t-run.exe"),
        "--config", "genie_config.json",
        "--prompt_file", str(PROMPT_PATH.resolve()),
        "--log", "info",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  cwd: {BUNDLE_DIR}")

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(
            cmd, cwd=str(BUNDLE_DIR), env=env,
            stdout=f, stderr=subprocess.STDOUT, timeout=600,
        )
    result.wall_s = time.perf_counter() - t0
    result.extra["exit_code"] = proc.returncode
    # proc.returncode == 1 is expected — genie-t2t-run exits 1 on --profile
    # teardown for 2.42-compiled bundles on 2.45 runtime. The warning path
    # fires after the generation finishes cleanly, so timing is valid.

    # Parse the log for prefill vs decode graph timestamps.
    prompt_start_ms: float | None = None
    prompt_last_done_ms: float | None = None
    token_first_start_ms: float | None = None
    token_last_done_ms: float | None = None
    token_done_count = 0  # count of 4_of_4 completions = decoded tokens

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GRAPH_DONE_RE.search(line)
        if not m:
            continue
        ts_ms = float(m.group(1))
        graph = m.group(2)
        if graph.startswith("prompt_ar128_"):
            if prompt_start_ms is None:
                prompt_start_ms = ts_ms
            prompt_last_done_ms = ts_ms
        elif graph.startswith("token_ar1_"):
            if token_first_start_ms is None:
                token_first_start_ms = ts_ms
            token_last_done_ms = ts_ms
            # Count one decoded token per completion of the LAST partition's
            # token graph (e.g. _4_of_4 on the 4B bundle, _6_of_6 on this 7B
            # one). Generalize by matching `_N_of_N$`.
            tail = re.search(r"_(\d+)_of_(\d+)$", graph)
            if tail and tail.group(1) == tail.group(2):
                token_done_count += 1

    # Prefill = (end of last prompt_ar128 graph) - (start of first prompt_ar128 graph)
    # More precisely we'd use the 'started' timestamp of the first prompt
    # graph, but in Genie's trace the DONE timestamp is all we get via
    # the graph-name pattern. The DONE of last prompt graph ≈ END of
    # prefill; START of first prompt graph is a few ms before its first
    # DONE (each graph is ~15-40 ms on the attention chunks). We
    # approximate with: prefill_time = last_prompt_done - first_prompt_done
    # + per_graph_average. The inaccuracy is on the order of one-partition
    # latency (~20 ms). For a prefill of 512 tokens this is a ~2% error.
    if prompt_start_ms is not None and prompt_last_done_ms is not None:
        # use the last finish timestamp as end-of-prefill; first finish
        # is mid-prefill (1st of 4 partitions of the first ar128 chunk),
        # so we approximate prefill-start as prompt_start_ms minus one
        # median graph duration derived from the run.
        prefill_span_ms = prompt_last_done_ms - prompt_start_ms
        # Add an approximate per-graph head-of-pipeline for the first
        # graph (assume ~20 ms — partition 1 of each chunk).
        prefill_ms = prefill_span_ms + 20.0
        result.pp_time_s = prefill_ms / 1000.0
        result.pp_tokens = 512  # known from prompt file construction
        result.pp_tps = 512 / result.pp_time_s

    if token_first_start_ms is not None and token_last_done_ms is not None and token_done_count > 0:
        decode_span_ms = token_last_done_ms - token_first_start_ms
        result.tg_time_s = decode_span_ms / 1000.0
        result.tg_tokens = token_done_count
        # TG t/s counts tokens-per-sec; the first token's decode starts
        # essentially right after the last prompt graph, so span / count
        # is the right number (count includes the first token).
        result.tg_tps = token_done_count / result.tg_time_s if result.tg_time_s > 0 else None

    if result.pp_tps is None or result.tg_tps is None:
        result.ok = False
        result.notes = f"failed to parse timestamps (exit={proc.returncode}, tokens_done={token_done_count})"

    return result


# ─── backend: llama-bench ───────────────────────────────────────────────────

def run_llama_bench(
    preset: str,
    log_path: Path,
    threads: int | None = None,
    ngl: int = 0,
) -> BackendResult:
    """Drive llama.cpp's llama-bench.exe, capturing PP512 + TG128 via
    its JSON output mode.
    """
    name = {
        "cpu": "cpu",
        "cpu-kleidiai": "cpu-kleidiai",
        "opencl": "gpu-opencl",
        "vulkan": "gpu-vulkan",
    }[preset]
    result = BackendResult(name=name)
    bench_exe = REPO_ROOT / "llama.cpp" / f"build-{preset}" / "bin" / "llama-bench.exe"
    if not bench_exe.exists():
        result.ok = False
        result.notes = f"binary missing: {bench_exe}"
        return result

    cmd = [
        str(bench_exe),
        "-m", str(GGUF_PATH),
        "-p", "512",
        "-n", "128",
        "-r", "3",
        "-o", "json",
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl:
        cmd += ["-ngl", str(ngl)]

    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        f.write("=== stdout ===\n")
        f.write(proc.stdout)
        f.write("\n=== stderr ===\n")
        f.write(proc.stderr)
    result.wall_s = time.perf_counter() - t0
    result.extra["exit_code"] = proc.returncode

    if proc.returncode != 0:
        result.ok = False
        result.notes = f"llama-bench exit {proc.returncode}"
        return result

    # llama-bench emits a JSON array on stdout. The OpenCL backend on
    # Adreno sometimes fails to flush the closing `]` before teardown
    # (log messages from ggml_opencl can race the final stdout flush).
    # Try strict parse first; if it fails, retry with `]` appended.
    raw = proc.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.rstrip(",") + "]")
        except json.JSONDecodeError as e:
            result.ok = False
            result.notes = f"json parse (even with retry): {e}"
            return result

    # llama-bench JSON rows: PP rows have n_prompt>0, n_gen=0;
    # TG rows have n_prompt=0, n_gen>0. avg_ts is tokens/sec.
    for row in data:
        ts = row.get("avg_ts")
        n_prompt = row.get("n_prompt", 0)
        n_gen = row.get("n_gen", 0)
        if ts is None:
            continue
        if n_prompt == 512 and n_gen == 0:
            result.pp_tps = float(ts)
            result.pp_tokens = 512
            result.pp_time_s = row.get("avg_ns", 0) / 1e9 or None
        elif n_prompt == 0 and n_gen == 128:
            result.tg_tps = float(ts)
            result.tg_tokens = 128
            result.tg_time_s = row.get("avg_ns", 0) / 1e9 or None

    if result.pp_tps is None or result.tg_tps is None:
        result.ok = False
        rows_summary = [(r.get("n_prompt"), r.get("n_gen")) for r in data]
        result.notes = f"missing pp512 or tg128 in json (got n_prompt/n_gen pairs {rows_summary})"

    return result


# ─── driver ─────────────────────────────────────────────────────────────────

BACKENDS: list[tuple[str, callable, dict]] = [
    ("npu-genie",    run_npu_genie,   {}),
    ("cpu",          run_llama_bench, {"preset": "cpu", "threads": 8}),
    ("cpu-kleidiai", run_llama_bench, {"preset": "cpu-kleidiai", "threads": 8}),
    ("gpu-opencl",   run_llama_bench, {"preset": "opencl", "ngl": 99}),
    ("gpu-vulkan",   run_llama_bench, {"preset": "vulkan", "ngl": 99}),
]


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def format_md_table(rows: list[dict], power_state: str, tag: str) -> str:
    """Pretty markdown table. Printed to stdout at end of run; not
    written to disk — per docs/repo_hygiene.md the CSV is the permanent
    record and the consolidated human analysis lives in
    docs/qwen2_5_7b_baseline_all_backends.md. Paste this into the doc's
    update log when rolling a new run in.
    """
    lines = [
        f"# Qwen2.5-7B-Instruct baseline — {power_state.upper()} — {tag}",
        "",
        f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "| backend | PP512 (t/s) | TG (t/s) | TG tokens | mean W | mWh drop | J/tok | wall (s) | ok | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:-:|---|",
    ]
    for r in rows:
        def fmt(v, spec=".2f"):
            if v is None:
                return "—"
            try:
                return format(v, spec)
            except Exception:
                return str(v)
        lines.append(
            f"| {r['backend']} | {fmt(r['pp_tps'])} | {fmt(r['tg_tps'])} | "
            f"{r['tg_tokens']} | {fmt(r.get('mean_w'), '.1f')} | "
            f"{fmt(r['mwh_drop'], '.0f')} | "
            f"{fmt(r['j_per_tok'], '.3f')} | {fmt(r['wall_s'], '.1f')} | "
            f"{'✓' if r['ok'] else '✗'} | {r['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True,
                        help="Confirms intended power state; compared against WMI PowerOnline for safety.")
    parser.add_argument("--backends", default="all",
                        help=f"Comma-separated subset or 'all'. Choices: {[b[0] for b in BACKENDS]}")
    parser.add_argument("--tag", default=None,
                        help="Suffix for log/csv/md filenames")
    parser.add_argument("--skip-power-check", action="store_true",
                        help="Don't verify --power-state against WMI PowerOnline")
    args = parser.parse_args()

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = TRASH_ROOT / f"qwen2_5_7b_baseline_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Safety check: is reported power state what we claim?
    online = sample_power_online()
    if online is None:
        print("WARNING: could not query PowerOnline; proceeding without check")
    elif args.power_state == "ac" and not online:
        if not args.skip_power_check:
            print("ERROR: --power-state ac but WMI reports PowerOnline=False. Plug in, or pass --skip-power-check.")
            return 2
    elif args.power_state == "bat" and online:
        if not args.skip_power_check:
            print("ERROR: --power-state bat but WMI reports PowerOnline=True. Unplug, or pass --skip-power-check.")
            return 2

    # Select backends
    if args.backends == "all":
        selected = BACKENDS
    else:
        wanted = [s.strip() for s in args.backends.split(",")]
        selected = [b for b in BACKENDS if b[0] in wanted]
        if len(selected) != len(wanted):
            got = [b[0] for b in selected]
            missing = [w for w in wanted if w not in got]
            print(f"ERROR: unknown backend(s): {missing}")
            return 2

    rows: list[dict] = []
    for name, fn, kwargs in selected:
        print(f"\n=== {name} (power={args.power_state}) ===")
        log_path = log_dir / f"{name}.log"

        mwh_before = sample_battery_mwh() if args.power_state == "bat" else None
        # 2s polling keeps PowerShell subprocess overhead ≲3% of wall
        # time while still capturing 10+ samples even on the shortest
        # (~22s) CPU benchmark.
        sampler = PowerSampler(interval_s=2.0) if args.power_state == "bat" else None
        if sampler:
            sampler.start()
        t_before = time.perf_counter()
        try:
            result = fn(log_path=log_path, **kwargs) if kwargs else fn(log_path=log_path)
        except Exception as e:
            result = BackendResult(name=name, ok=False, notes=f"exception: {type(e).__name__}: {e}")
        wall = time.perf_counter() - t_before
        if sampler:
            sampler.stop()
        mwh_after = sample_battery_mwh() if args.power_state == "bat" else None

        # J/tok math: two independent estimators on battery.
        #   (a) coarse — RemainingCapacity mWh delta × 3.6 → J. Accurate
        #       for long runs (>60s) but blind to short ones because WMI
        #       updates RemainingCapacity only every few seconds.
        #   (b) fine — mean(DischargeRate_W) × wall_s → J. Works for any
        #       run length; depends on DischargeRate being steady at the
        #       backend's load (true for compute-bound steady-state).
        # We report both; J/tok prefers (b) when enough samples exist.
        mwh_drop = None
        energy_j_coarse = None
        if mwh_before is not None and mwh_after is not None:
            mwh_drop = mwh_before - mwh_after
            if mwh_drop > 0:
                energy_j_coarse = mwh_drop * 3.6

        mean_w = sampler.mean_watts if sampler else None
        n_power_samples = sampler.n_samples if sampler else 0
        energy_j_fine = mean_w * wall if mean_w is not None else None

        energy_j = energy_j_fine if energy_j_fine is not None else energy_j_coarse
        j_per_tok = None
        total_tokens = result.pp_tokens + result.tg_tokens
        if energy_j is not None and total_tokens > 0:
            j_per_tok = energy_j / total_tokens

        row = dict(
            backend=result.name,
            pp_tps=result.pp_tps,
            tg_tps=result.tg_tps,
            pp_tokens=result.pp_tokens,
            tg_tokens=result.tg_tokens,
            pp_time_s=result.pp_time_s,
            tg_time_s=result.tg_time_s,
            wall_s=wall,
            mwh_before=mwh_before,
            mwh_after=mwh_after,
            mwh_drop=mwh_drop,
            mean_w=mean_w,
            n_power_samples=n_power_samples,
            j_per_tok=j_per_tok,
            power_state=args.power_state,
            ok=result.ok,
            notes=result.notes,
        )
        rows.append(row)
        print(f"  PP512: {row['pp_tps']}  TG: {row['tg_tps']}  TG tokens: {row['tg_tokens']}")
        print(f"  wall: {wall:.1f}s  mean: {mean_w} W ({n_power_samples} samples)  mWh drop: {mwh_drop}  J/tok: {j_per_tok}")
        if not result.ok:
            print(f"  !! NOT OK: {result.notes}")

    csv_path = CSV_DIR / f"qwen2_5_7b_baseline_{tag}.csv"
    write_csv(csv_path, rows)
    table = format_md_table(rows, args.power_state, tag)
    print()
    print(table)
    print(f"=== Summary ===")
    print(f"csv:  {csv_path}")
    print(f"logs: {log_dir}  (gitignored, soak then rm -rf)")
    print(f"paste table above into docs/qwen2_5_7b_baseline_all_backends.md when promoting this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
