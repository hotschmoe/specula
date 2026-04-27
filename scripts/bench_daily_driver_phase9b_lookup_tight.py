"""Phase 9b: tightened n-gram lookup bench.

Phase 9 surprised us: ngram_mod showed +14% on chat, no collapse on
code — the opposite of thc1006's RTX-3090 finding. But Phase 9 had
two confounds:
  (a) baseline ran cold, ngram_mod ran with warm OS file cache
      (Phase 6 showed up to +9% from that alone)
  (b) only one prompt per kind, no long-context probe

Phase 9b removes both:
  - A warm-up pass per config (untimed) before recorded trials
  - 3 trials per (config, prompt) — report median
  - Adds an ~8k-token synthetic agent transcript to probe d≈8k TG
    (the depth the daily-driver agent loop actually lives at)
  - cache_prompt=true so trials 2-3 hit the prefix cache (we record
    only TG t/s, which is independent of whether prefill ran)

Configs: baseline, ngram_mod (CPU + Q4_K_M only — spec-decode binaries
ship in build-cpu only).

Output: results/csv/daily_driver_phase9b_lookup_tight_<tag>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
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
PORT = 8765
HEALTH_URL = f"http://{HOST}:{PORT}/health"
COMPLETION_URL = f"http://{HOST}:{PORT}/completion"


CONFIGS = [
    ("baseline",  []),
    ("ngram_mod", ["--spec-type", "ngram-mod", "--draft", "8"]),
]


def make_long_prompt(target_chars: int = 28000) -> str:
    """Synthesize an agent-style transcript: code + JSON + chat, repeated.

    Mirrors the daily-driver workload: tool-call envelopes, function bodies,
    error tracebacks, prose. Deterministic so trials are comparable.
    """
    blocks = []

    code_chunk = (
        "def fibonacci(n: int) -> int:\n"
        "    \"\"\"Return the nth Fibonacci number with memoization.\"\"\"\n"
        "    cache = {0: 0, 1: 1}\n"
        "    def helper(k: int) -> int:\n"
        "        if k in cache:\n"
        "            return cache[k]\n"
        "        cache[k] = helper(k - 1) + helper(k - 2)\n"
        "        return cache[k]\n"
        "    return helper(n)\n"
        "\n"
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        if arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
        "\n"
    )

    json_chunk = (
        "{\n"
        '  "tool": "read_file",\n'
        '  "args": {"path": "src/lib/foo.py"},\n'
        '  "result": {\n'
        '    "ok": true,\n'
        '    "content": "def foo(x):\\n    return x + 1\\n"\n'
        "  }\n"
        "},\n"
        "{\n"
        '  "tool": "run_tests",\n'
        '  "args": {"pattern": "tests/test_foo.py"},\n'
        '  "result": {\n'
        '    "ok": false,\n'
        '    "stdout": "FAILED tests/test_foo.py::test_increment - AssertionError: assert 2 == 3"\n'
        "  }\n"
        "},\n"
    )

    chat_chunk = (
        "User: The test is failing because foo(1) returns 2, not 3. "
        "What did I miss?\n"
        "Assistant: The test asserts foo(1) == 3, but your foo adds 1, "
        "so foo(1) == 2. Either the test is wrong or foo should return "
        "x + 2. Which one matches the spec?\n\n"
    )

    n_blocks = max(1, target_chars // (len(code_chunk) + len(json_chunk) + len(chat_chunk)))
    for _ in range(n_blocks + 1):
        blocks.append(code_chunk)
        blocks.append(json_chunk)
        blocks.append(chat_chunk)

    body = "".join(blocks)[:target_chars]
    body += (
        "\n\nUser: Given everything above, write a single Python function "
        "that combines the patterns shown — recursive memoization plus "
        "iterative bisection — into a function `solve(arr, target)` that "
        "returns the index, or -1 if absent. Explain your reasoning briefly.\n"
        "Assistant:"
    )
    return body


PROMPTS = {
    "chat_short": (
        "User: I'm planning a weekend trip to a quiet coastal town. "
        "What are three things I should bring and why?\nAssistant:"
    ),
    "code_short": (
        "def fibonacci(n):\n"
        "    # Return the nth Fibonacci number using memoization.\n"
        "    # Use a dictionary as the memo. Handle n <= 1 as base cases.\n"
        "    "
    ),
    "agent_long_8k": make_long_prompt(28000),
}


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


def post_completion(prompt: str, n_predict: int, timeout_s: int,
                    cache_prompt: bool) -> dict | None:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 0,
        "cache_prompt": cache_prompt,
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
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"      POST failed: {e}", flush=True)
        return None


def start_server(extra_args: list[str], threads: int, ctx: int,
                 log_path: Path) -> subprocess.Popen:
    args = [str(SERVER_EXE),
            "-m", str(GGUF_CPU),
            "-t", str(threads),
            "-c", str(ctx),
            "--host", HOST,
            "--port", str(PORT)]
    args.extend(extra_args)
    print(f"  cmd: {' '.join(args)}", flush=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        args, stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def parse_t(resp: dict) -> tuple[float | None, float | None, int, int]:
    """Return (tg_tps, pp_tps, predicted_n, prompt_n)."""
    t = resp.get("timings", {}) or {}
    tg = t.get("predicted_per_second")
    pp = t.get("prompt_per_second")
    n_pred = t.get("predicted_n", resp.get("tokens_predicted", 0)) or 0
    n_prompt = t.get("prompt_n", resp.get("tokens_evaluated", 0)) or 0
    return (
        float(tg) if tg is not None else None,
        float(pp) if pp is not None else None,
        int(n_pred), int(n_prompt),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--health-timeout-s", type=int, default=180)
    parser.add_argument("--gen-timeout-s", type=int, default=600)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    tag = args.tag or time.strftime("%Y-%m-%d_%H%M%S")
    csv_path = CSV_DIR / f"daily_driver_phase9b_lookup_tight_{tag}.csv"
    log_root = TRASH_ROOT / f"daily_driver_phase9b_lookup_tight_{tag}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"CSV: {csv_path}\nLogs: {log_root}\n")

    long_n_chars = len(PROMPTS["agent_long_8k"])
    print(f"long prompt: {long_n_chars} chars (~{long_n_chars // 4} approx tokens)\n")

    rows: list[dict] = []

    for label, extra in CONFIGS:
        print(f"=== config: {label} ===", flush=True)
        log_path = log_root / f"server_{label}.log"
        proc = start_server(extra, args.threads, args.ctx, log_path)
        try:
            print(f"  waiting for health (<= {args.health_timeout_s}s)...", flush=True)
            if not wait_for_health(args.health_timeout_s):
                print(f"  health timed out (rc={proc.poll()}); skipping config")
                for kind in PROMPTS:
                    rows.append({"config": label, "prompt_kind": kind, "trial": -1,
                                 "tg_tps": "", "pp_tps": "", "predicted_n": 0,
                                 "prompt_n": 0, "wall_s": 0.0, "ok": 0,
                                 "notes": "server health timeout"})
                continue
            print("  ready.", flush=True)

            # Warm-up: short, untimed; primes file cache + samplers + slot
            print("  warmup...", flush=True)
            _ = post_completion(PROMPTS["chat_short"], 32,
                                args.gen_timeout_s, cache_prompt=True)

            for kind, prompt in PROMPTS.items():
                print(f"  prompt={kind}", flush=True)
                tg_samples: list[float] = []
                for trial in range(args.n_trials):
                    t0 = time.perf_counter()
                    resp = post_completion(prompt, args.n_predict,
                                           args.gen_timeout_s, cache_prompt=True)
                    wall = time.perf_counter() - t0
                    if resp is None:
                        rows.append({"config": label, "prompt_kind": kind,
                                     "trial": trial, "tg_tps": "", "pp_tps": "",
                                     "predicted_n": 0, "prompt_n": 0,
                                     "wall_s": f"{wall:.1f}", "ok": 0,
                                     "notes": "post failed"})
                        continue
                    tg, pp, n_pred, n_prompt = parse_t(resp)
                    if tg is not None:
                        tg_samples.append(tg)
                    rows.append({"config": label, "prompt_kind": kind,
                                 "trial": trial,
                                 "tg_tps": f"{tg:.3f}" if tg is not None else "",
                                 "pp_tps": f"{pp:.3f}" if pp is not None else "",
                                 "predicted_n": n_pred, "prompt_n": n_prompt,
                                 "wall_s": f"{wall:.1f}", "ok": 1, "notes": ""})
                    print(f"    trial {trial}: tg={tg:.2f} t/s pp={pp:.1f} t/s "
                          f"n_pred={n_pred} n_prompt={n_prompt} wall={wall:.1f}s",
                          flush=True)
                    # save just trial 0 response for forensics
                    if trial == 0:
                        (log_root / f"resp_{label}_{kind}_trial0.json").write_text(
                            json.dumps(resp, indent=2), encoding="utf-8")
                if tg_samples:
                    med = statistics.median(tg_samples)
                    rng = max(tg_samples) - min(tg_samples)
                    print(f"    median tg = {med:.2f} t/s  (n={len(tg_samples)}, "
                          f"range={rng:.2f})", flush=True)
        finally:
            print("  stopping server...", flush=True)
            stop_server(proc)
            time.sleep(2)

    cols = ["config", "prompt_kind", "trial", "tg_tps", "pp_tps",
            "predicted_n", "prompt_n", "wall_s", "ok", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # summary: median per (config, prompt)
    print("\nsummary (median tg_tps):")
    print(f"  {'config':<14} {'prompt':<14} {'tg_med':>8} {'tg_min':>8} {'tg_max':>8} {'n':>3}")
    keyed: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if not r["tg_tps"] or not r["ok"]:
            continue
        keyed.setdefault((r["config"], r["prompt_kind"]), []).append(float(r["tg_tps"]))
    for (cfg, kind), vals in keyed.items():
        med = statistics.median(vals)
        print(f"  {cfg:<14} {kind:<14} {med:>8.3f} {min(vals):>8.3f} "
              f"{max(vals):>8.3f} {len(vals):>3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
