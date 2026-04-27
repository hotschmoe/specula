"""Phase 9: n-gram lookup / spec-decode sanity check on the agent workload.

Confirms (or refutes) thc1006's RTX-3090 finding for our Adreno+CPU rig:
  - Qwen3.6-35B-A3B's 8/256 expert sparsity makes n-gram lookup
    bimodal: chat stays at baseline, code/JSON output collapses to
    50-70% of baseline t/s due to MoE expert-saturation.

Configs (CPU + Q4_K_M only — spec-decode binaries don't ship in the
GPU builds at this commit):
  - baseline          --spec-type none
  - ngram-mod         --spec-type ngram-mod (best of thc1006)
  - ngram-cache       --spec-type ngram-cache + --lookup-cache-dynamic
                       (worst of thc1006; closest to the optimization.md
                       proposal so worth re-measuring)

Prompts:
  - chat   short conversational continuation (the "no-collapse" case)
  - code   HumanEval-style python completion (the "collapse" case)

Server-driven: llama-bench has no spec-decode flags, so we drive
llama-server via /completion with seeded deterministic generation.

Output: results/csv/daily_driver_phase9_lookup_<tag>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_daily_driver import (  # type: ignore
    GGUF_CPU,
    REPO_ROOT,
    CSV_DIR,
    TRASH_ROOT,
    sample_power_online,
)


SERVER_EXE = REPO_ROOT / "llama.cpp" / "build-cpu" / "bin" / "llama-server.exe"
HOST = "127.0.0.1"
PORT = 8765  # avoid clashing with a user-running 8080
HEALTH_URL = f"http://{HOST}:{PORT}/health"
COMPLETION_URL = f"http://{HOST}:{PORT}/completion"


PROMPTS = {
    "chat": (
        "User: I'm planning a weekend trip to a quiet coastal town. "
        "What are three things I should bring and why?\nAssistant:"
    ),
    "code": (
        "def fibonacci(n):\n"
        "    # Return the nth Fibonacci number using memoization.\n"
        "    # Use a dictionary as the memo. Handle n <= 1 as base cases.\n"
        "    "
    ),
}


CONFIGS = [
    # (label, extra args list)
    ("baseline",     []),
    ("ngram_mod",    ["--spec-type", "ngram-mod", "--draft", "8"]),
    ("ngram_cache",  ["--spec-type", "ngram-cache", "--draft", "8",
                      "--lookup-cache-dynamic", "_LOOKUP_BIN_"]),
]


def wait_for_health(timeout_s: int) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def post_completion(prompt: str, n_predict: int, timeout_s: int) -> dict | None:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 0,
        "cache_prompt": False,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        COMPLETION_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
        return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    POST failed: {e}", flush=True)
        return None


def start_server(extra_args: list[str], threads: int, ctx: int,
                 log_path: Path, lookup_bin: Path) -> subprocess.Popen:
    args = [str(SERVER_EXE),
            "-m", str(GGUF_CPU),
            "-t", str(threads),
            "-c", str(ctx),
            "--host", HOST,
            "--port", str(PORT)]
    # substitute placeholder for lookup-cache-dynamic path
    resolved = []
    for a in extra_args:
        if a == "_LOOKUP_BIN_":
            # ensure parent dir exists; file may or may not exist yet
            lookup_bin.parent.mkdir(parents=True, exist_ok=True)
            resolved.append(str(lookup_bin))
        else:
            resolved.append(a)
    args.extend(resolved)
    print(f"  cmd: {' '.join(args)}", flush=True)

    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print("    terminate timed out; killing", flush=True)
        proc.kill()
        proc.wait(timeout=10)


def parse_timings(resp: dict) -> dict:
    """Extract t/s and AR-like fields from /completion response."""
    out = {
        "predicted_n": resp.get("tokens_predicted") or resp.get("tokens_evaluated") or 0,
        "prompt_n": resp.get("tokens_evaluated") or 0,
        "tg_tps": "", "pp_tps": "",
        "draft_n": "", "accept_n": "", "accept_pct": "",
    }
    t = resp.get("timings", {}) or {}
    if "predicted_per_second" in t and t["predicted_per_second"] is not None:
        out["tg_tps"] = f"{float(t['predicted_per_second']):.3f}"
    if "prompt_per_second" in t and t["prompt_per_second"] is not None:
        out["pp_tps"] = f"{float(t['prompt_per_second']):.3f}"
    if "predicted_n" in t:
        out["predicted_n"] = t["predicted_n"]
    if "prompt_n" in t:
        out["prompt_n"] = t["prompt_n"]
    # Some llama-server versions surface speculative stats; capture if present
    spec = resp.get("speculative") or t.get("speculative") or {}
    if isinstance(spec, dict):
        if "n_drafted" in spec:
            out["draft_n"] = spec["n_drafted"]
        if "n_accept" in spec:
            out["accept_n"] = spec["n_accept"]
        if out["draft_n"] not in ("", 0) and out["accept_n"] not in ("", 0):
            try:
                out["accept_pct"] = f"{100.0 * float(out['accept_n']) / float(out['draft_n']):.2f}"
            except Exception:
                pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--health-timeout-s", type=int, default=180)
    parser.add_argument("--gen-timeout-s", type=int, default=300)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    tag = args.tag or time.strftime("%Y-%m-%d_%H%M%S")
    csv_path = CSV_DIR / f"daily_driver_phase9_lookup_{tag}.csv"
    log_root = TRASH_ROOT / f"daily_driver_phase9_lookup_{tag}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"CSV: {csv_path}\nLogs: {log_root}\n")

    rows: list[dict] = []
    lookup_bin = log_root / "lookup_dynamic.bin"

    for label, extra in CONFIGS:
        print(f"=== config: {label} ===", flush=True)
        log_path = log_root / f"server_{label}.log"
        proc = start_server(extra, args.threads, args.ctx, log_path, lookup_bin)
        try:
            print(f"  waiting for health (<= {args.health_timeout_s}s)...", flush=True)
            if not wait_for_health(args.health_timeout_s):
                rc = proc.poll()
                print(f"  health timed out (rc={rc}); skipping config", flush=True)
                for kind in PROMPTS:
                    rows.append({"config": label, "prompt_kind": kind,
                                 "tg_tps": "", "pp_tps": "",
                                 "predicted_n": 0, "prompt_n": 0,
                                 "draft_n": "", "accept_n": "", "accept_pct": "",
                                 "wall_s": 0.0, "ok": 0,
                                 "notes": "server health timeout"})
                continue
            print("  ready.", flush=True)

            for kind, prompt in PROMPTS.items():
                print(f"  prompt={kind} (n_predict={args.n_predict})", flush=True)
                t0 = time.perf_counter()
                resp = post_completion(prompt, args.n_predict, args.gen_timeout_s)
                wall = time.perf_counter() - t0
                if resp is None:
                    rows.append({"config": label, "prompt_kind": kind,
                                 "tg_tps": "", "pp_tps": "",
                                 "predicted_n": 0, "prompt_n": 0,
                                 "draft_n": "", "accept_n": "", "accept_pct": "",
                                 "wall_s": f"{wall:.1f}", "ok": 0,
                                 "notes": "post failed"})
                    continue
                t = parse_timings(resp)
                row = {
                    "config": label, "prompt_kind": kind,
                    "tg_tps": t["tg_tps"], "pp_tps": t["pp_tps"],
                    "predicted_n": t["predicted_n"], "prompt_n": t["prompt_n"],
                    "draft_n": t["draft_n"], "accept_n": t["accept_n"],
                    "accept_pct": t["accept_pct"],
                    "wall_s": f"{wall:.1f}", "ok": 1, "notes": "",
                }
                rows.append(row)
                print(f"    tg={t['tg_tps']} t/s  pp={t['pp_tps']} t/s  "
                      f"predicted={t['predicted_n']}  wall={wall:.1f}s  "
                      f"accept={t['accept_pct']}%", flush=True)
                # save full response for forensics
                (log_root / f"resp_{label}_{kind}.json").write_text(
                    json.dumps(resp, indent=2), encoding="utf-8")
        finally:
            print("  stopping server...", flush=True)
            stop_server(proc)
            time.sleep(2)

    # write CSV
    cols = ["config", "prompt_kind", "tg_tps", "pp_tps",
            "predicted_n", "prompt_n",
            "draft_n", "accept_n", "accept_pct",
            "wall_s", "ok", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # quick summary table to stdout
    print("\nsummary:")
    print(f"  {'config':<14} {'kind':<6} {'tg_tps':>10} {'pp_tps':>10} {'wall_s':>8}")
    for r in rows:
        print(f"  {r['config']:<14} {r['prompt_kind']:<6} {r['tg_tps']:>10} {r['pp_tps']:>10} {r['wall_s']:>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
