"""Phase 9c: ngram_mod validation at d=32k.

Phase 9b found ngram_mod CPU = +28% at d~9525. The canonical-config
flip from Vulkan back to CPU at long ctx hinges on whether the win
holds at d=32k (the depth where Phase 8 measured Vulkan = 16.89 t/s).

Single prompt (agent_long_32k, ~32k tokens), 2 configs, 3 trials each.
Trial 0 is cold prefill (~16 min wall at PP=33 t/s); trials 1-2 hit
the prefix cache and just measure TG at d=32k.

Server-driven; mirrors phase 9b's structure with a bigger prompt and
larger ctx.
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
from bench_daily_driver_phase9b_lookup_tight import (  # type: ignore
    make_long_prompt,
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
        COMPLETION_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"      POST failed: {e}", flush=True)
        return None


def start_server(extra: list[str], threads: int, ctx: int,
                 log_path: Path) -> subprocess.Popen:
    args = [str(SERVER_EXE), "-m", str(GGUF_CPU),
            "-t", str(threads), "-c", str(ctx),
            "--host", HOST, "--port", str(PORT)]
    args.extend(extra)
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
    parser.add_argument("--ctx", type=int, default=49152)  # 32k prompt + headroom
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--n-trials", type=int, default=2)  # 1 cold + 1 warm
    parser.add_argument("--target-chars", type=int, default=80000,  # ~27k tokens (this corpus tokenizes at ~3 chars/token, denser than the 4 chars/tok rule of thumb)
                        help="prompt size in chars; previous run with 128k chars gave 43k tokens, way past the 32k goal")
    parser.add_argument("--health-timeout-s", type=int, default=240)
    parser.add_argument("--gen-timeout-s", type=int, default=5400)  # 90 min — at d~27k cold PP runs 40-50 min, headroom matters
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-power-check", action="store_true")
    args = parser.parse_args()

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    tag = args.tag or time.strftime("%Y-%m-%d_%H%M%S")
    csv_path = CSV_DIR / f"daily_driver_phase9c_lookup_d32k_{tag}.csv"
    log_root = TRASH_ROOT / f"daily_driver_phase9c_lookup_d32k_{tag}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"CSV: {csv_path}\nLogs: {log_root}\n")

    long_prompt = make_long_prompt(args.target_chars)
    print(f"long prompt: {len(long_prompt)} chars (~{len(long_prompt) // 4} approx tokens)\n")
    (log_root / "agent_long_32k_prompt.txt").write_text(long_prompt, encoding="utf-8")

    rows: list[dict] = []

    for label, extra in CONFIGS:
        print(f"=== config: {label} ===", flush=True)
        log_path = log_root / f"server_{label}.log"
        proc = start_server(extra, args.threads, args.ctx, log_path)
        try:
            print(f"  waiting for health (<= {args.health_timeout_s}s)...", flush=True)
            if not wait_for_health(args.health_timeout_s):
                print(f"  health timed out (rc={proc.poll()}); skipping config")
                rows.append({"config": label, "trial": -1,
                             "tg_tps": "", "pp_tps": "", "predicted_n": 0,
                             "prompt_n": 0, "wall_s": 0.0, "ok": 0,
                             "notes": "server health timeout"})
                continue
            print("  ready.", flush=True)
            print("  warmup (small prompt)...", flush=True)
            _ = post_completion("Hello.", 16, args.gen_timeout_s, cache_prompt=True)

            print(f"  agent_long_32k ({args.n_trials} trials)", flush=True)
            tg_samples: list[float] = []
            for trial in range(args.n_trials):
                t0 = time.perf_counter()
                resp = post_completion(long_prompt, args.n_predict,
                                       args.gen_timeout_s, cache_prompt=True)
                wall = time.perf_counter() - t0
                if resp is None:
                    rows.append({"config": label, "trial": trial,
                                 "tg_tps": "", "pp_tps": "",
                                 "predicted_n": 0, "prompt_n": 0,
                                 "wall_s": f"{wall:.1f}", "ok": 0,
                                 "notes": "post failed"})
                    continue
                tg, pp, n_pred, n_prompt = parse_t(resp)
                if tg is not None:
                    tg_samples.append(tg)
                rows.append({"config": label, "trial": trial,
                             "tg_tps": f"{tg:.3f}" if tg is not None else "",
                             "pp_tps": f"{pp:.3f}" if pp is not None else "",
                             "predicted_n": n_pred, "prompt_n": n_prompt,
                             "wall_s": f"{wall:.1f}", "ok": 1, "notes": ""})
                kind = "cold" if n_prompt > 1000 else "warm"
                print(f"    trial {trial} ({kind}): tg={tg:.3f} t/s pp={pp:.1f} t/s "
                      f"n_pred={n_pred} n_prompt={n_prompt} wall={wall:.1f}s",
                      flush=True)
                if trial == 0:
                    (log_root / f"resp_{label}_trial0.json").write_text(
                        json.dumps(resp, indent=2), encoding="utf-8")
            if tg_samples:
                med = statistics.median(tg_samples)
                rng = max(tg_samples) - min(tg_samples)
                warm = [s for s, r in zip(tg_samples, rows[-len(tg_samples):])
                        if int(r.get("prompt_n", 0)) <= 1000]
                cold = [s for s, r in zip(tg_samples, rows[-len(tg_samples):])
                        if int(r.get("prompt_n", 0)) > 1000]
                print(f"    median tg = {med:.3f} t/s  (n={len(tg_samples)}, range={rng:.3f})",
                      flush=True)
                if cold:
                    print(f"    cold tg  = {statistics.median(cold):.3f} (n={len(cold)})",
                          flush=True)
                if warm:
                    print(f"    warm tg  = {statistics.median(warm):.3f} (n={len(warm)})",
                          flush=True)
        finally:
            print("  stopping server...", flush=True)
            stop_server(proc)
            time.sleep(2)

    cols = ["config", "trial", "tg_tps", "pp_tps", "predicted_n", "prompt_n",
            "wall_s", "ok", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    print("\nsummary (all trials):")
    for r in rows:
        print(f"  {r['config']:<10} trial={r['trial']:<2} "
              f"tg={r['tg_tps']:>8} pp={r['pp_tps']:>8} "
              f"n_prompt={r['prompt_n']:>6} wall={r['wall_s']:>6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
