"""Phase 10: TTFT after a tool-call — delta-prefill bench (lever B).

The agent UX question: when each turn extends the prior conversation
by a small delta (tool result + new user message), how much does the
prefix cache actually save? Phase 9b confirmed prefix-cache works
(re-prefilled only 4 of 9525 tokens after a slot hit) but didn't
quantify across backends or measure wall-time impact at long ctx.

Workload: a 5-turn agent loop simulation
  turn 0 (cold)  : ~16k-token "session start" — full prefill
  turn 1         : turn 0 + assistant_response_0 + delta (~500 tokens)
  turn 2         : turn 1 + assistant_response_1 + delta (~500 tokens)
  turn 3         : turn 2 + assistant_response_2 + delta (~500 tokens)
  turn 4         : turn 3 + assistant_response_3 + delta (~500 tokens)

Generation: 200 tokens per turn. Each delta represents tool result +
new user prompt at typical agent-harness sizes. cache_prompt=true so
the slot retains state across turns.

Configs:
  cpu_baseline       CPU + Q4_K_M (-t 8)
  cpu_ngram_mod      CPU + Q4_K_M + ngram-mod (added in Phase 9b/c)
  vulkan_baseline    Vulkan + MXFP4_MOE + ngl=99 + canonical knobs

Headline metrics:
  - prompt_n per turn (gold: ~500 on turns 1+, ~16k on turn 0)
  - prompt_per_second per turn
  - wall_s per turn (TTFT proxy — what the user actually feels)

Output: results/csv/daily_driver_phase10_delta_prefill_<tag>.csv
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
    GGUF_GPU,
    REPO_ROOT,
    CSV_DIR,
    TRASH_ROOT,
    sample_power_online,
)
from bench_daily_driver_phase9b_lookup_tight import (  # type: ignore
    make_long_prompt,
)


HOST = "127.0.0.1"
PORT = 8765
HEALTH_URL = f"http://{HOST}:{PORT}/health"
COMPLETION_URL = f"http://{HOST}:{PORT}/completion"


CPU_SERVER  = REPO_ROOT / "llama.cpp" / "build-cpu"    / "bin" / "llama-server.exe"
VK_SERVER   = REPO_ROOT / "llama.cpp" / "build-vulkan" / "bin" / "llama-server.exe"


CONFIGS = [
    {
        "label": "cpu_baseline",
        "exe": CPU_SERVER,
        "model": GGUF_CPU,
        "extra": ["-t", "8"],
        "env": {},
    },
    {
        "label": "cpu_ngram_mod",
        "exe": CPU_SERVER,
        "model": GGUF_CPU,
        "extra": ["-t", "8", "--spec-type", "ngram-mod", "--draft", "8"],
        "env": {},
    },
    {
        "label": "vulkan_baseline",
        "exe": VK_SERVER,
        "model": GGUF_GPU,
        "extra": ["-ngl", "99"],
        "env": {"GGML_VK_DISABLE_F16": "1", "GGML_VK_PREFER_HOST_MEMORY": "1"},
    },
]


# Each delta represents one agent turn: tool result + new user message.
# Made deterministic and varied enough to exercise both prefix-cache hit
# and a small bit of n-gram repetition.
DELTA_TEMPLATES = [
    "\n\n[Tool: read_file] Result: {file_count} files matched. Listing the first 3:\n"
    "  - src/lib/auth.py (412 lines, last modified 2026-04-12)\n"
    "  - src/lib/sessions.py (188 lines, last modified 2026-04-15)\n"
    "  - tests/test_auth.py (94 lines, last modified 2026-04-15)\n\n"
    "User: Now read the third file and tell me what's in test_authenticate_user.\n",

    "\n\n[Tool: read_file] Result: tests/test_auth.py contents below.\n"
    "```python\nimport pytest\nfrom src.lib.auth import authenticate_user\n\n"
    "def test_authenticate_user():\n    assert authenticate_user('alice', 'pw') is True\n"
    "    assert authenticate_user('alice', 'wrong') is False\n```\n\n"
    "User: That test only checks the happy path — what about empty inputs and SQL injection?\n",

    "\n\n[Tool: run_tests] Result: tests/test_auth.py PASSED in 0.12s.\n"
    "  - test_authenticate_user PASSED\n  - test_session_expiry SKIPPED (todo)\n\n"
    "User: Skipped tests are bad. What's blocking test_session_expiry?\n",

    "\n\n[Tool: grep] Pattern 'session_expiry' found in 4 files:\n"
    "  - src/lib/sessions.py:42 (function def)\n"
    "  - tests/test_sessions.py:11 (import)\n"
    "  - docs/api.md:88 (reference)\n  - CHANGELOG.md:5 (note)\n\n"
    "User: Open sessions.py at line 42 and explain what the function does.\n",
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
        "cache_prompt": True,
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


def start_server(cfg: dict, ctx: int, log_path: Path) -> subprocess.Popen:
    args = [str(cfg["exe"]), "-m", str(cfg["model"]),
            "-c", str(ctx), "--host", HOST, "--port", str(PORT)]
    args.extend(cfg["extra"])
    env = os.environ.copy()
    env.update(cfg["env"])
    print(f"  cmd: {' '.join(args)}", flush=True)
    if cfg["env"]:
        print(f"  env: {cfg['env']}", flush=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        args, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
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


def parse_t(resp: dict) -> dict:
    t = resp.get("timings", {}) or {}
    return {
        "tg_tps": float(t["predicted_per_second"]) if t.get("predicted_per_second") is not None else None,
        "pp_tps": float(t["prompt_per_second"]) if t.get("prompt_per_second") is not None else None,
        "predicted_n": int(t.get("predicted_n", resp.get("tokens_predicted", 0)) or 0),
        "prompt_n": int(t.get("prompt_n", resp.get("tokens_evaluated", 0)) or 0),
    }


def build_initial(target_chars: int) -> str:
    return make_long_prompt(target_chars)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-state", choices=("ac", "bat"), required=True)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--initial-chars", type=int, default=48000,
                        help="initial session prompt size in chars (~16k tokens at this corpus)")
    parser.add_argument("--turn-n-predict", type=int, default=200)
    parser.add_argument("--health-timeout-s", type=int, default=240)
    parser.add_argument("--gen-timeout-s", type=int, default=2400)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-power-check", action="store_true")
    parser.add_argument("--configs", default="cpu_baseline,cpu_ngram_mod,vulkan_baseline",
                        help="comma-separated subset of config labels")
    args = parser.parse_args()

    online = sample_power_online()
    if args.power_state == "ac" and online is False and not args.skip_power_check:
        print("ERROR: --power-state ac but PowerOnline=False.")
        return 2

    selected = set(s.strip() for s in args.configs.split(","))
    configs = [c for c in CONFIGS if c["label"] in selected]
    if not configs:
        print(f"ERROR: no configs match {selected}")
        return 2

    tag = args.tag or time.strftime("%Y-%m-%d_%H%M%S")
    csv_path = CSV_DIR / f"daily_driver_phase10_delta_prefill_{tag}.csv"
    log_root = TRASH_ROOT / f"daily_driver_phase10_delta_prefill_{tag}"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"CSV: {csv_path}\nLogs: {log_root}\n")

    initial = build_initial(args.initial_chars)
    print(f"initial session prompt: {len(initial)} chars (~{len(initial) // 3} tokens)\n")
    (log_root / "initial.txt").write_text(initial, encoding="utf-8")

    rows: list[dict] = []

    for cfg in configs:
        label = cfg["label"]
        print(f"=== config: {label} ===", flush=True)
        log_path = log_root / f"server_{label}.log"
        proc = start_server(cfg, args.ctx, log_path)
        try:
            print(f"  waiting for health (<= {args.health_timeout_s}s)...", flush=True)
            if not wait_for_health(args.health_timeout_s):
                print(f"  health timed out (rc={proc.poll()}); skipping config")
                rows.append({"config": label, "turn": -1, "tg_tps": "", "pp_tps": "",
                             "predicted_n": 0, "prompt_n": 0,
                             "delta_chars": 0, "wall_s": 0.0, "ok": 0,
                             "notes": "server health timeout"})
                continue
            print("  ready.", flush=True)

            # Build turn prompts: turn 0 = initial; subsequent turns append
            # the previous response (we use the actual response content from
            # the prior request) + the next delta template.
            current_prompt = initial
            for turn_idx in range(1 + len(DELTA_TEMPLATES)):
                turn_label = "cold_session" if turn_idx == 0 else f"turn_{turn_idx}"
                delta_chars = (len(current_prompt) - len(initial)) if turn_idx > 0 else 0
                print(f"  {turn_label} (n_predict={args.turn_n_predict}, "
                      f"prompt_chars={len(current_prompt)}, delta_chars={delta_chars})",
                      flush=True)
                t0 = time.perf_counter()
                resp = post_completion(current_prompt, args.turn_n_predict,
                                       args.gen_timeout_s)
                wall = time.perf_counter() - t0
                if resp is None:
                    rows.append({"config": label, "turn": turn_idx,
                                 "tg_tps": "", "pp_tps": "",
                                 "predicted_n": 0, "prompt_n": 0,
                                 "delta_chars": delta_chars,
                                 "wall_s": f"{wall:.1f}", "ok": 0,
                                 "notes": "post failed"})
                    break
                t = parse_t(resp)
                row = {
                    "config": label, "turn": turn_idx,
                    "tg_tps": f"{t['tg_tps']:.3f}" if t["tg_tps"] is not None else "",
                    "pp_tps": f"{t['pp_tps']:.3f}" if t["pp_tps"] is not None else "",
                    "predicted_n": t["predicted_n"], "prompt_n": t["prompt_n"],
                    "delta_chars": delta_chars,
                    "wall_s": f"{wall:.1f}", "ok": 1, "notes": "",
                }
                rows.append(row)
                print(f"    tg={t['tg_tps']:.2f} t/s  pp={t['pp_tps']:.1f} t/s  "
                      f"prompt_n={t['prompt_n']:>5}  predicted_n={t['predicted_n']}  "
                      f"wall={wall:.1f}s", flush=True)
                # save resp for forensics
                (log_root / f"resp_{label}_{turn_label}.json").write_text(
                    json.dumps(resp, indent=2), encoding="utf-8")
                # Build next prompt: append assistant response + next delta
                if turn_idx < len(DELTA_TEMPLATES):
                    response_content = resp.get("content", "")
                    delta = DELTA_TEMPLATES[turn_idx]
                    current_prompt = current_prompt + response_content + delta
        finally:
            print("  stopping server...", flush=True)
            stop_server(proc)
            time.sleep(2)

    cols = ["config", "turn", "tg_tps", "pp_tps", "predicted_n", "prompt_n",
            "delta_chars", "wall_s", "ok", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Summary: TTFT per turn per config
    print("\nsummary (TTFT per turn):")
    print(f"  {'config':<18} {'turn':>4} {'prompt_n':>8} {'pp_tps':>8} {'tg_tps':>8} {'wall_s':>8}")
    for r in rows:
        print(f"  {r['config']:<18} {r['turn']:>4} {str(r['prompt_n']):>8} "
              f"{r['pp_tps']:>8} {r['tg_tps']:>8} {r['wall_s']:>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
