"""All-backends bench driver for the daily-driver model: Qwen3.6-35B-A3B.

Forked from scripts/bench_qwen2_5_7b_all_backends.py. Differences:

- **No NPU row.** Workbench compile path for a 35B MoE bundle is out
  of scope; see daily_driver/optimization.md for the deferral note.
- **Two GGUFs**, picked per backend:
    CPU / CPU+KleidiAI -> Qwen3.6-35B-A3B-Q4_K_M.gguf (lmstudio-community)
    GPU OpenCL / Vulkan -> Qwen3.6-35B-A3B-MXFP4_MOE.gguf (Unsloth)
- **Vulkan env knobs** (`GGML_VK_DISABLE_F16=1`,
  `GGML_VK_PREFER_HOST_MEMORY=1`) are applied by default — these were
  the canonical wins on Q4_0 at 7B. Whether they hold for MXFP4_MOE
  is an open question; toggle with --vulkan-default-env to A/B.
- CSVs land in results/csv/daily_driver_<tag>.csv; logs in
  marked_for_deletion/daily_driver_<tag>/ per docs/repo_hygiene.md.

Usage:
    .venv/Scripts/python.exe scripts/bench_daily_driver.py \\
        --power-state ac --tag 2026-04-XX_ac

    # Just one backend:
    .venv/Scripts/python.exe scripts/bench_daily_driver.py \\
        --power-state ac --backends gpu-vulkan --tag vk_smoketest

    # Disable the Vulkan env knobs (A/B test):
    .venv/Scripts/python.exe scripts/bench_daily_driver.py \\
        --power-state ac --backends gpu-vulkan --no-vulkan-env \\
        --tag vk_no_env
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

GGUF_CPU = REPO_ROOT / "models" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
GGUF_GPU = REPO_ROOT / "models" / "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"

CSV_DIR = REPO_ROOT / "results" / "csv"
TRASH_ROOT = REPO_ROOT / "marked_for_deletion"


@dataclass
class BackendResult:
    name: str
    pp_tps: float | None = None
    tg_tps: float | None = None
    tg_tokens: int = 0
    pp_tokens: int = 0
    pp_time_s: float | None = None
    tg_time_s: float | None = None
    wall_s: float | None = None
    notes: str = ""
    ok: bool = True
    extra: dict = field(default_factory=dict)


# ─── battery / power helpers (parity with 7B runner) ────────────────────────

def sample_battery_mwh() -> int | None:
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


def sample_power_online() -> bool | None:
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


class PowerSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
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


# ─── llama-bench driver ─────────────────────────────────────────────────────

# 35B at Q4 takes ~2-3× as long per pass as 7B, so the per-call timeout
# needs to grow. 7B used 600 s; 35B PP512/TG128 with -r 3 takes ~40s
# on CPU. Long-context bench (d=131072 prefill) on CPU can take ~16-20
# min wall, so we set the hard cap well above that for headroom.
LLAMA_BENCH_TIMEOUT_S = 3600  # 60 min

# Stale-output watchdog: if neither stdout nor stderr emits a byte for
# this long, kill the process. Catches the "100% GPU, no output"
# livelock pattern that bit us on Vulkan-Q4 at 7B and on Vulkan-MXFP4
# at 35B (Phase 3, this doc).
#
# Subtle: llama-bench's `-d N` prefill phase emits "depth run 1/1"
# THEN goes silent until N tokens are processed. At d=131k on CPU,
# this silent phase is ~16 min. So the watchdog default needs to
# tolerate that. 300s was too aggressive at long depths and killed
# legitimate runs in Phase 2 attempt #2. The runner exposes
# --stale-timeout-s for explicit control; default below is set to
# 5 min, which is fine for short-ctx (d<=8k); long-ctx callers must
# raise it explicitly.
STALE_TIMEOUT_S = 300


def gguf_for_preset(preset: str) -> Path:
    return GGUF_GPU if preset in ("opencl", "vulkan") else GGUF_CPU


def run_streaming(
    cmd: list[str],
    log_path: Path,
    env: dict,
    hard_timeout_s: int = LLAMA_BENCH_TIMEOUT_S,
    stale_timeout_s: int = STALE_TIMEOUT_S,
) -> tuple[int, str, str, str]:
    """Run cmd via Popen, stream stdout+stderr to log_path in real time,
    enforce hard timeout AND stale-output watchdog.

    Two safety mechanisms:
      - hard_timeout_s: total wall time ceiling. Fired as a backstop.
      - stale_timeout_s: max silence between any byte of output. This
        is the catch for the "GPU at 99% but no output for hours"
        livelock that we hit on Vulkan-Q4 at the 7B baseline.

    Streams output to log_path as it arrives (line-buffered + flush
    after each write), so a separate process can `tail -f` mid-run
    to see progress. stderr lines get a [err] prefix in the log.

    Returns (returncode, stdout_full, stderr_full, status). status
    is one of: "ok", "hard_timeout", "stale_timeout", "error".
    """
    last_activity_lock = threading.Lock()
    last_activity = [time.monotonic()]
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def reader(pipe, prefix: str, sink: list[str], log_f):
        try:
            for line in iter(pipe.readline, ""):
                with last_activity_lock:
                    last_activity[0] = time.monotonic()
                sink.append(line)
                try:
                    log_f.write(prefix + line)
                    log_f.flush()
                except Exception:
                    pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace", buffering=1) as log_f:
        log_f.write("=== cmd ===\n")
        log_f.write(" ".join(cmd) + "\n")
        log_f.write("=== streaming output ===\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
        except Exception as e:
            log_f.write(f"\n=== Popen failed: {e} ===\n")
            return -1, "", "", "error"

        out_thread = threading.Thread(
            target=reader, args=(proc.stdout, "", stdout_chunks, log_f),
            daemon=True,
        )
        err_thread = threading.Thread(
            target=reader, args=(proc.stderr, "[err] ", stderr_chunks, log_f),
            daemon=True,
        )
        out_thread.start()
        err_thread.start()

        t0 = time.monotonic()
        status = "ok"
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            now = time.monotonic()
            elapsed = now - t0
            with last_activity_lock:
                silence = now - last_activity[0]
            if elapsed > hard_timeout_s:
                log_f.write(f"\n=== KILLED: hard timeout {hard_timeout_s}s ===\n")
                log_f.flush()
                proc.kill()
                status = "hard_timeout"
                break
            if silence > stale_timeout_s:
                log_f.write(
                    f"\n=== KILLED: no output for {silence:.0f}s "
                    f"(stale > {stale_timeout_s}s) ===\n"
                )
                log_f.flush()
                proc.kill()
                status = "stale_timeout"
                break
            time.sleep(1.0)

        # Drain readers + collect exit code
        out_thread.join(timeout=10)
        err_thread.join(timeout=10)
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=5)

        log_f.write(f"\n=== exit={rc}  status={status}  wall={time.monotonic()-t0:.1f}s ===\n")

    return rc, "".join(stdout_chunks), "".join(stderr_chunks), status


def run_llama_bench(
    preset: str,
    log_path: Path,
    threads: int | None = None,
    ngl: int = 0,
    extra_env: dict | None = None,
) -> BackendResult:
    """Drive llama.cpp's llama-bench.exe, capturing PP512 + TG128 via JSON."""
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

    gguf = gguf_for_preset(preset)
    if not gguf.exists():
        result.ok = False
        result.notes = f"gguf missing: {gguf.name} (download not finished?)"
        return result

    cmd = [
        str(bench_exe),
        "-m", str(gguf),
        "-p", "512",
        "-n", "128",
        "-r", "3",
        "-o", "json",
        "--progress",  # emits per-test progress to stderr; powers the
                       # stale-output watchdog and lets `tail -f` show
                       # progress mid-run.
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if ngl:
        cmd += ["-ngl", str(ngl)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
        result.extra["env"] = dict(extra_env)

    print(f"  cmd: {' '.join(cmd)}")
    if extra_env:
        print(f"  env: {extra_env}")
    print(f"  log: {log_path}  (tail -f to watch progress)")

    t0 = time.perf_counter()
    rc, stdout_full, stderr_full, status = run_streaming(cmd, log_path, env)
    result.wall_s = time.perf_counter() - t0
    result.extra["exit_code"] = rc
    result.extra["status"] = status

    if status == "stale_timeout":
        result.ok = False
        result.notes = f"stale output timeout (no bytes for >{STALE_TIMEOUT_S}s) — likely GPU livelock"
        return result
    if status == "hard_timeout":
        result.ok = False
        result.notes = f"hard timeout {LLAMA_BENCH_TIMEOUT_S}s exceeded"
        return result
    if rc != 0:
        result.ok = False
        result.notes = f"llama-bench exit {rc}"
        return result

    # llama-bench emits JSON on stdout. Adreno OpenCL sometimes fails to
    # flush the closing `]` — retry-with-`]` is the same dance the 7B
    # runner does. With --progress on, stderr has progress noise but
    # stdout still contains only the JSON.
    raw = stdout_full.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.rstrip(",") + "]")
        except json.JSONDecodeError as e:
            result.ok = False
            result.notes = f"json parse (even with retry): {e}"
            return result

    for row in data:
        ts = row.get("avg_ts")
        n_prompt = row.get("n_prompt", 0)
        n_gen = row.get("n_gen", 0)
        if ts is None:
            continue
        if n_prompt == 512 and n_gen == 0:
            result.pp_tps = float(ts)
            result.pp_tokens = 512
            result.pp_time_s = (row.get("avg_ns", 0) / 1e9) or None
        elif n_prompt == 0 and n_gen == 128:
            result.tg_tps = float(ts)
            result.tg_tokens = 128
            result.tg_time_s = (row.get("avg_ns", 0) / 1e9) or None

    if result.pp_tps is None or result.tg_tps is None:
        rows_summary = [(r.get("n_prompt"), r.get("n_gen")) for r in data]
        result.ok = False
        result.notes = f"missing pp512 or tg128 in json (got {rows_summary})"

    return result


# ─── driver ─────────────────────────────────────────────────────────────────

VULKAN_DEFAULT_ENV = {
    "GGML_VK_DISABLE_F16": "1",
    "GGML_VK_PREFER_HOST_MEMORY": "1",
}


def build_backend_table(vulkan_env: dict | None) -> list[tuple[str, callable, dict]]:
    return [
        ("cpu",          run_llama_bench, {"preset": "cpu",          "threads": 8}),
        ("cpu-kleidiai", run_llama_bench, {"preset": "cpu-kleidiai", "threads": 8}),
        ("gpu-opencl",   run_llama_bench, {"preset": "opencl", "ngl": 99}),
        ("gpu-vulkan",   run_llama_bench, {"preset": "vulkan", "ngl": 99,
                                            "extra_env": vulkan_env}),
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
    lines = [
        f"# daily_driver / Qwen3.6-35B-A3B — {power_state.upper()} — {tag}",
        "",
        f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "| backend | model | PP512 (t/s) | TG (t/s) | TG tokens | mean W | mWh drop | J/tok | wall (s) | ok | notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:-:|---|",
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
            f"| {r['backend']} | {r['model']} | {fmt(r['pp_tps'])} | {fmt(r['tg_tps'])} | "
            f"{r['tg_tokens']} | {fmt(r.get('mean_w'), '.1f')} | "
            f"{fmt(r['mwh_drop'], '.0f')} | "
            f"{fmt(r['j_per_tok'], '.3f')} | {fmt(r['wall_s'], '.1f')} | "
            f"{'OK' if r['ok'] else 'X'} | {r['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--backends", default="all",
                        help="Comma-separated subset or 'all'. "
                             "Choices: cpu, cpu-kleidiai, gpu-opencl, gpu-vulkan")
    parser.add_argument("--tag", default=None,
                        help="Suffix for log/csv/md filenames")
    parser.add_argument("--skip-power-check", action="store_true")
    parser.add_argument("--no-vulkan-env", action="store_true",
                        help="Disable the Vulkan F16-off + prefer-host-memory env "
                             "knobs (default on; A/B against MXFP4_MOE).")
    args = parser.parse_args()

    tag = args.tag or f"{args.power_state}_{time.strftime('%Y%m%d_%H%M%S')}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = TRASH_ROOT / f"daily_driver_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    online = sample_power_online()
    if online is None:
        print("WARNING: could not query PowerOnline; proceeding without check")
    elif args.power_state == "ac" and not online:
        if not args.skip_power_check:
            print("ERROR: --power-state ac but WMI reports PowerOnline=False. "
                  "Plug in, or pass --skip-power-check.")
            return 2
    elif args.power_state == "bat" and online:
        if not args.skip_power_check:
            print("ERROR: --power-state bat but WMI reports PowerOnline=True. "
                  "Unplug, or pass --skip-power-check.")
            return 2

    # Pre-flight: warn if a GGUF the selected backends need is missing.
    # Doesn't abort — individual backends will report missing files
    # cleanly — but the warning surfaces it before the user waits.
    if not GGUF_CPU.exists():
        print(f"WARNING: CPU GGUF missing: {GGUF_CPU.name}")
    if not GGUF_GPU.exists():
        print(f"WARNING: GPU GGUF missing: {GGUF_GPU.name}")

    vulkan_env = None if args.no_vulkan_env else dict(VULKAN_DEFAULT_ENV)
    backends = build_backend_table(vulkan_env)

    if args.backends == "all":
        selected = backends
    else:
        wanted = [s.strip() for s in args.backends.split(",")]
        selected = [b for b in backends if b[0] in wanted]
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
        sampler = PowerSampler(interval_s=2.0) if args.power_state == "bat" else None
        if sampler:
            sampler.start()
        t_before = time.perf_counter()
        try:
            result = fn(log_path=log_path, **kwargs)
        except Exception as e:
            result = BackendResult(name=name, ok=False,
                                   notes=f"exception: {type(e).__name__}: {e}")
        wall = time.perf_counter() - t_before
        if sampler:
            sampler.stop()
        mwh_after = sample_battery_mwh() if args.power_state == "bat" else None

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

        # Which model file each row used — important here because two
        # different GGUFs are in play.
        gguf_used = gguf_for_preset(kwargs.get("preset", ""))
        row = dict(
            backend=result.name,
            model=gguf_used.name,
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
            env=json.dumps(result.extra.get("env")) if result.extra.get("env") else "",
        )
        rows.append(row)
        print(f"  PP512: {row['pp_tps']}  TG: {row['tg_tps']}  TG tokens: {row['tg_tokens']}")
        print(f"  wall: {wall:.1f}s  mean: {mean_w} W ({n_power_samples} samples)  "
              f"mWh drop: {mwh_drop}  J/tok: {j_per_tok}")
        if not result.ok:
            print(f"  !! NOT OK: {result.notes}")

    csv_path = CSV_DIR / f"daily_driver_{tag}.csv"
    write_csv(csv_path, rows)
    table = format_md_table(rows, args.power_state, tag)
    print()
    print(table)
    print(f"=== Summary ===")
    print(f"csv:  {csv_path}")
    print(f"logs: {log_dir}  (gitignored, soak then rm -rf)")
    print(f"paste table above into daily_driver/optimization.md "
          f"§ Findings when promoting this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
