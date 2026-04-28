"""SQ1 Path B — multi-round naive serial spec-decode.

Real driver loop. For each round:
  1. NPU drafts K tokens given the accumulated context.
  2. Target serial-verifies position-by-position via /completion
     (n_predict=1 per call, cache_prompt=true so the target's KV is
     reused across the K verify calls within a round).
  3. Commit draft[:first_mismatch] + [target_token_at_mismatch], OR
     all K + 1 target-bonus token if no mismatch in K positions.
  4. Stop on EOS, max committed tokens, or max rounds.

NPU re-prefills the accumulated context each round (sidecar has no
rewind op yet — adding one is its own follow-up). Per-round wall:
  npu = pp_full_ctx + K * decode_step
  target = K serial /completion calls (each ~1 token of new compute
           via cache_prompt; first call has 1-shot prefill of the new
           context delta vs cached).

This is **not faster than CPU-only** for any K — see findings.md
Pre-Path-B/C analysis. The point is to MEASURE steady-state accept
rate over many rounds and confirm the loop architecture works.

Run:
  scripts/serve_target_14b.ps1                                  # terminal 1
  .venv/Scripts/python.exe last_side_quest/sq1_heterogeneous/demo_path_b.py \
      --target-url http://127.0.0.1:8081 --k 8 --rounds 8 --ctx-tier 2048
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
BUNDLE_TOKENIZER = BUNDLE_DIR / "tokenizer.json"

# Qwen3 tokenizer EOS ids — stop generation if any of these are emitted.
QWEN3_EOS_IDS = {151643, 151645}  # <|endoftext|>, <|im_end|>


def spawn_npu_sidecar(ctx_tier: int):
    import subprocess
    cmd = [
        sys.executable, str(REPO_ROOT / "npu_engine" / "sidecar.py"),
        "--mode", "serve", "--ctx-tier", str(ctx_tier), "--start-mode", "ar1",
    ]
    print(f"  spawning: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, encoding="utf-8",
    )
    while True:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read()
            print(f"  NPU sidecar died:\n{stderr}", file=sys.stderr)
            return None
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [npu] non-JSON: {line}")
            continue
        if evt.get("event") == "ready":
            print(f"  NPU ready: startup={evt['startup_s']:.1f}s "
                  f"per_part={evt.get('start_per_part_s')}")
            return proc
        print(f"  [npu] {evt}")


def npu_request(proc, req: dict) -> dict:
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    rsp_line = proc.stdout.readline()
    if not rsp_line:
        raise RuntimeError("NPU sidecar closed stdout")
    return json.loads(rsp_line.strip())


def target_one_token(target_url: str, prompt_ids: list[int],
                     slot_id: int = 0) -> tuple[int, float, dict]:
    """POST /completion with n_predict=1, return (token_id, wall_s, raw_body).

    `cache_prompt=true` and a stable `id_slot` mean the server reuses
    its KV cache across calls — only the new tail compute counts.
    """
    body = {
        "prompt": prompt_ids,
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "return_tokens": True,
        "cache_prompt": True,
        "id_slot": slot_id,
    }
    t = time.perf_counter()
    r = requests.post(f"{target_url.rstrip('/')}/completion",
                      json=body, timeout=300)
    wall_s = time.perf_counter() - t
    r.raise_for_status()
    out = r.json()
    tokens = out.get("tokens") or []
    if not tokens:
        # Fallback: re-tokenize content (rare; flag in caller)
        content = out.get("content", "")
        tok = Tokenizer.from_file(str(BUNDLE_TOKENIZER))
        tokens = tok.encode(content).ids[:1]
    return tokens[0], wall_s, out


def target_baseline_generate(target_url: str, prompt_ids: list[int],
                             n_tokens: int, slot_id: int = 0
                             ) -> tuple[list[int], float, dict]:
    body = {
        "prompt": prompt_ids,
        "n_predict": n_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "return_tokens": True,
        "cache_prompt": True,
        "id_slot": slot_id,
    }
    t = time.perf_counter()
    r = requests.post(f"{target_url.rstrip('/')}/completion",
                      json=body, timeout=600)
    wall_s = time.perf_counter() - t
    r.raise_for_status()
    out = r.json()
    return out.get("tokens") or [], wall_s, out


def run_spec_decode(npu, target_url: str, prompt_ids: list[int],
                    K: int, max_rounds: int, max_committed: int,
                    tokenizer: Tokenizer, target_slot: int = 0):
    """Multi-round naive serial spec-decode. Returns a stats dict."""
    context_ids = list(prompt_ids)
    committed_ids: list[int] = []
    per_round = []
    total_npu_pp_s = 0.0
    total_npu_tg_s = 0.0
    total_target_s = 0.0
    t_loop_start = time.perf_counter()
    stop_reason = "max_rounds"

    for round_i in range(max_rounds):
        # ---- 1. NPU draft K
        npu_rsp = npu_request(npu, {
            "op": "draft", "id": f"sq1b-r{round_i}",
            "prompt_ids": context_ids, "n_draft": K,
        })
        if not npu_rsp.get("ok"):
            stop_reason = f"npu_draft_failed: {npu_rsp}"
            break
        draft_ids = npu_rsp["draft_ids"]
        npu_pp_s = npu_rsp["pp_wall_s"]
        npu_tg_s = npu_rsp["tg_wall_s"]
        total_npu_pp_s += npu_pp_s
        total_npu_tg_s += npu_tg_s

        # ---- 2. Target serial-verify position by position
        verify_ids = []
        verify_walls = []
        first_mismatch = None
        for i in range(K):
            check_ctx = context_ids + draft_ids[:i]
            t_id, t_wall, _ = target_one_token(
                target_url, check_ctx, slot_id=target_slot)
            verify_ids.append(t_id)
            verify_walls.append(t_wall)
            total_target_s += t_wall
            if t_id != draft_ids[i]:
                first_mismatch = i
                break

        # ---- 3. Decide what to commit this round
        if first_mismatch is None:
            # All K matched. Sample one bonus token from target at K-th position
            # to keep moving (the target's natural next token after K accepts).
            bonus_ctx = context_ids + draft_ids
            t_id, t_wall, _ = target_one_token(
                target_url, bonus_ctx, slot_id=target_slot)
            verify_ids.append(t_id)
            verify_walls.append(t_wall)
            total_target_s += t_wall
            this_round_commit = list(draft_ids) + [t_id]
            n_accepted = K
        else:
            # Mismatch at first_mismatch. Commit draft[:i] + target_token.
            this_round_commit = list(draft_ids[:first_mismatch]) + [verify_ids[-1]]
            n_accepted = first_mismatch

        # ---- 4. Append, stop checks
        committed_ids.extend(this_round_commit)
        context_ids.extend(this_round_commit)

        per_round.append({
            "round": round_i, "K": K,
            "draft_ids": draft_ids,
            "verify_ids": verify_ids,
            "first_mismatch": (first_mismatch if first_mismatch is not None else K),
            "n_accepted": n_accepted,
            "committed": this_round_commit,
            "npu_pp_s": npu_pp_s, "npu_tg_s": npu_tg_s,
            "target_walls_s": verify_walls,
            "target_round_s": sum(verify_walls),
        })

        if any(t in QWEN3_EOS_IDS for t in this_round_commit):
            stop_reason = "eos"
            break
        if len(committed_ids) >= max_committed:
            stop_reason = "max_committed"
            break

    total_loop_s = time.perf_counter() - t_loop_start

    return {
        "stop_reason": stop_reason,
        "rounds": len(per_round),
        "committed_ids": committed_ids,
        "n_committed": len(committed_ids),
        "per_round": per_round,
        "total_npu_pp_s": total_npu_pp_s,
        "total_npu_tg_s": total_npu_tg_s,
        "total_target_s": total_target_s,
        "total_loop_s": total_loop_s,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-url", default="http://127.0.0.1:8081")
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rounds", type=int, default=8,
                   help="max number of spec-decode rounds")
    p.add_argument("--max-committed", type=int, default=64,
                   help="stop after this many committed tokens")
    p.add_argument("--ctx-tier", type=int, default=2048,
                   choices=(512, 1024, 2048, 3072, 4096))
    p.add_argument("--baseline", action="store_true",
                   help="also run target-only baseline of same n_tokens for comparison")
    p.add_argument("--csv", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt_text = args.prompt
    else:
        prompt_text = (
            "You are an expert Python developer. Complete the following function:\n\n"
            "def fibonacci(n: int) -> int:\n"
            '    """Return the nth Fibonacci number using O(n) time."""\n'
        )

    tokenizer = Tokenizer.from_file(str(BUNDLE_TOKENIZER))
    prompt_ids = tokenizer.encode(prompt_text).ids

    print("=== SQ1 Path B — multi-round naive serial spec-decode ===")
    print(f"  target_url    : {args.target_url}")
    print(f"  K             : {args.k}")
    print(f"  max_rounds    : {args.rounds}")
    print(f"  max_committed : {args.max_committed}")
    print(f"  ctx tier      : {args.ctx_tier}")
    print(f"  prompt_tokens : {len(prompt_ids)}")
    print()

    print("--- Checking target reachability ---")
    try:
        r = requests.get(f"{args.target_url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        print(f"  target health: {r.json()}")
    except Exception as e:
        print(f"  ERROR: target server not reachable ({e})", file=sys.stderr)
        return 2

    # ---- Optional baseline first (separate slot so it doesn't pollute
    # spec-decode's KV cache).
    baseline_stats = None
    if args.baseline:
        print(f"\n--- Target-only baseline ({args.max_committed} tokens) ---")
        b_tokens, b_wall, _ = target_baseline_generate(
            args.target_url, prompt_ids, args.max_committed, slot_id=99)
        n_b = len(b_tokens)
        rate = n_b / b_wall if b_wall > 0 else 0.0
        print(f"  generated {n_b} tokens in {b_wall:.2f}s -> {rate:.2f} t/s")
        print(f"  text: {tokenizer.decode(b_tokens)!r}")
        baseline_stats = {
            "n_tokens": n_b, "wall_s": b_wall, "rate_t_s": rate,
            "tokens": b_tokens,
        }

    print("\n--- Starting NPU sidecar ---")
    npu = spawn_npu_sidecar(args.ctx_tier)
    if npu is None:
        return 2

    try:
        print(f"\n--- Running spec-decode loop ---")
        stats = run_spec_decode(
            npu, args.target_url, prompt_ids,
            K=args.k, max_rounds=args.rounds,
            max_committed=args.max_committed,
            tokenizer=tokenizer, target_slot=0,
        )

        # Summary
        print(f"\n=== Loop summary ===")
        print(f"  stop_reason     : {stats['stop_reason']}")
        print(f"  rounds          : {stats['rounds']}")
        print(f"  committed_total : {stats['n_committed']}")
        if stats["rounds"] > 0:
            mean_acc = sum(r["n_accepted"] for r in stats["per_round"]) / stats["rounds"]
            print(f"  mean_accept/K   : {mean_acc:.2f} / {args.k} "
                  f"({100*mean_acc/args.k:.0f}%)")
        print(f"  total NPU pp_s  : {stats['total_npu_pp_s']:.2f}")
        print(f"  total NPU tg_s  : {stats['total_npu_tg_s']:.2f}")
        print(f"  total target_s  : {stats['total_target_s']:.2f}")
        print(f"  total loop_s    : {stats['total_loop_s']:.2f}")
        if stats["total_loop_s"] > 0:
            rate = stats["n_committed"] / stats["total_loop_s"]
            print(f"  spec-decode t/s : {rate:.2f}")
        if baseline_stats:
            print(f"  baseline t/s    : {baseline_stats['rate_t_s']:.2f}")
            if baseline_stats["rate_t_s"] > 0:
                speedup = (stats["n_committed"] / stats["total_loop_s"]) / baseline_stats["rate_t_s"]
                print(f"  speedup vs base : {speedup:.2f}x")

        # Per-round detail
        print(f"\n--- Per-round detail ---")
        for r in stats["per_round"]:
            preview = tokenizer.decode(r["committed"])[:50].replace("\n", "\\n")
            print(f"  r{r['round']:>2}: K={r['K']} matched={r['n_accepted']}/{r['K']} "
                  f"first_mm={r['first_mismatch']} "
                  f"npu={r['npu_pp_s']+r['npu_tg_s']:.2f}s "
                  f"tgt={r['target_round_s']:.2f}s "
                  f"committed={len(r['committed'])} "
                  f"-> {preview!r}")

        text = tokenizer.decode(stats["committed_ids"])
        print(f"\n--- Final committed text ({stats['n_committed']} tokens) ---")
        print(f"  {text!r}")

        if args.csv:
            import csv
            tag = args.tag or f"sq1_b_k{args.k}_r{args.rounds}_{int(time.time())}"
            row = {
                "tag": tag, "k": args.k, "max_rounds": args.rounds,
                "max_committed": args.max_committed,
                "ctx_tier": args.ctx_tier,
                "prompt_tokens": len(prompt_ids),
                "stop_reason": stats["stop_reason"],
                "rounds": stats["rounds"],
                "n_committed": stats["n_committed"],
                "total_npu_pp_s": stats["total_npu_pp_s"],
                "total_npu_tg_s": stats["total_npu_tg_s"],
                "total_target_s": stats["total_target_s"],
                "total_loop_s": stats["total_loop_s"],
                "spec_decode_t_s": stats["n_committed"] / stats["total_loop_s"]
                                   if stats["total_loop_s"] > 0 else 0,
                "baseline_t_s": (baseline_stats["rate_t_s"] if baseline_stats else None),
                "mean_accept_per_K": (
                    sum(r["n_accepted"] for r in stats["per_round"]) / stats["rounds"]
                    if stats["rounds"] > 0 else 0
                ),
            }
            csv_path = Path(args.csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            print(f"\n  CSV row appended: {csv_path}")

        return 0
    finally:
        try:
            npu.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            npu.stdin.flush()
        except Exception:
            pass
        try:
            npu.wait(timeout=10)
        except Exception:
            npu.kill()


if __name__ == "__main__":
    sys.exit(main())
