"""SQ1 Path C — REAL spec-decode with stateful NPU streams (rewind op landed).

Companion to demo_path_c.py. Same flow:
  NPU drafts K tokens, target batch-verifies, KV cache rewind on mismatch.

Differences from demo_path_c.py:
  * NPU side uses the new stateful stream API (stream_open / stream_decode /
    stream_truncate / stream_append) instead of the stateless `draft` op.
  * Per-round NPU cost drops from "full re-prefill of growing context"
    (the 73% wall-time share that capped Path C's speedup) to
    "K decode steps + 1 append". This is the headline SQ1 deliverable.

Driver runs in the SQ2 Prism x86_64 venv (.venv-aimet-x86) — that's
where llama-cpp-python wheels are buildable. NPU sidecar still runs
in the ARM64 .venv (it needs onnxruntime-qnn). Communication is the
same JSON-over-stdio pipe.

Per-round flow (stateful):
  1. NPU stream_decode(K) → K draft tokens. KV ends populated [0..L+K).
  2. Target llama_cpp.eval(draft_ids) — ONE batched forward pass producing
     logits at positions L-1..L+K-1.
  3. For each i in 0..K-1, target preferred at L+i = argmax(scores[L-1+i]).
     Find first mismatch i* (or none).
  4. If all accepted (no i*): bonus = argmax(scores[L+K-1]).
     - target.eval([bonus]) → updates target KV [0..L+K+1)
     - npu stream_append([bonus]) → updates NPU KV [0..L+K+1)
  5. Else mismatch at i*: target_token T = target_preds[i*].
     - target._ctx.kv_cache_seq_rm(-1, L + i*, -1); n_tokens = L + i*;
       target.eval([T]) → target KV [0..L+i*+1)
     - npu stream_truncate(L + i*); npu stream_append([T]) → NPU KV [0..L+i*+1)
  6. Loop until EOS / max_committed / max_rounds.

Run:
  last_side_quest/sq2_aimet_local/.venv-aimet-x86/Scripts/python.exe \\
      last_side_quest/sq1_heterogeneous/demo_path_c_stateful.py \\
      --k 8 --rounds 8 --ctx-tier 2048 --baseline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from tokenizers import Tokenizer
from llama_cpp import Llama

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
BUNDLE_TOKENIZER = BUNDLE_DIR / "tokenizer.json"
TARGET_GGUF = REPO_ROOT / "models" / "Qwen3-14B-Q4_K_M.gguf"

QWEN3_EOS_IDS = {151643, 151645}

NPU_VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
NPU_STREAM_ID = "spec"


def spawn_npu_sidecar(ctx_tier: int):
    cmd = [
        str(NPU_VENV_PYTHON), str(REPO_ROOT / "npu_engine" / "sidecar.py"),
        "--mode", "serve", "--ctx-tier", str(ctx_tier), "--start-mode", "ar1",
    ]
    print(f"  spawning: {' '.join(cmd)}")
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("PYTHONHOME", "PYTHONPATH",
                              "PYTHONSTARTUP", "VIRTUAL_ENV")}
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, encoding="utf-8",
        env=child_env,
    )
    while True:
        line = proc.stdout.readline()
        if not line:
            print(f"  NPU sidecar died:\n{proc.stderr.read()}", file=sys.stderr)
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--max-committed", type=int, default=64)
    p.add_argument("--ctx-tier", type=int, default=2048,
                   choices=(512, 1024, 2048, 3072, 4096))
    p.add_argument("--target-ctx", type=int, default=4096)
    p.add_argument("--target-threads", type=int, default=8)
    p.add_argument("--baseline", action="store_true")
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

    print("=== SQ1 Path C STATEFUL — real batched-verify spec-decode ===")
    print(f"  K             : {args.k}")
    print(f"  max_rounds    : {args.rounds}")
    print(f"  max_committed : {args.max_committed}")
    print(f"  npu ctx       : {args.ctx_tier}")
    print(f"  target ctx    : {args.target_ctx}")
    print(f"  target threads: {args.target_threads}")
    print(f"  prompt_tokens : {len(prompt_ids)}")
    print()

    print("--- Loading target Qwen3-14B-Q4_K_M (in-process) ---")
    t = time.perf_counter()
    llm = Llama(
        model_path=str(TARGET_GGUF),
        n_ctx=args.target_ctx,
        n_threads=args.target_threads,
        n_threads_batch=args.target_threads,
        n_batch=args.k + 64,
        logits_all=True,
        verbose=False,
    )
    print(f"  loaded in {time.perf_counter()-t:.1f}s. n_ctx={llm.n_ctx()}, "
          f"vocab={llm.n_vocab()}")

    baseline_stats = None
    if args.baseline:
        print(f"\n--- Target-only baseline ({args.max_committed} tokens) ---")
        llm.reset()
        t = time.perf_counter()
        out = llm.create_completion(
            prompt=prompt_text,
            max_tokens=args.max_committed,
            temperature=0.0, top_k=1, stream=False,
        )
        wall = time.perf_counter() - t
        b_text = out["choices"][0]["text"]
        b_tokens = tokenizer.encode(b_text).ids
        rate = len(b_tokens) / wall if wall > 0 else 0.0
        print(f"  generated {len(b_tokens)} tokens in {wall:.2f}s -> {rate:.2f} t/s")
        print(f"  text: {b_text!r}")
        baseline_stats = {"n_tokens": len(b_tokens), "wall_s": wall,
                          "rate_t_s": rate, "text": b_text}

    print("\n--- Starting NPU sidecar (ARM64 .venv) ---")
    npu = spawn_npu_sidecar(args.ctx_tier)
    if npu is None:
        return 2

    try:
        # Initial target prefill
        print(f"\n--- Target prefill (prompt={len(prompt_ids)} tokens) ---")
        llm.reset()
        t = time.perf_counter()
        llm.eval(prompt_ids)
        prefill_wall = time.perf_counter() - t
        print(f"  target prefill: {prefill_wall:.2f}s "
              f"({len(prompt_ids)/prefill_wall:.1f} tok/s)")

        # Initial NPU stream_open — once for the whole demo
        t = time.perf_counter()
        rsp = npu_request(npu, {
            "op": "stream_open", "id": "init", "stream_id": NPU_STREAM_ID,
            "prompt_ids": prompt_ids, "force_ar128": (len(prompt_ids) >= 128),
        })
        npu_open_wall = time.perf_counter() - t
        if not rsp.get("ok"):
            print(f"  NPU stream_open failed: {rsp}", file=sys.stderr)
            return 3
        print(f"  NPU stream_open: {npu_open_wall:.2f}s "
              f"({len(prompt_ids)/npu_open_wall:.1f} tok/s)")

        # Spec-decode loop
        L = len(prompt_ids)
        committed_ids: list[int] = []
        per_round = []
        total_npu_decode_s = 0.0
        total_npu_commit_s = 0.0
        total_target_eval_s = 0.0
        total_target_rewind_s = 0.0
        stop_reason = "max_rounds"
        t_loop_start = time.perf_counter()

        for round_i in range(args.rounds):
            # 1. NPU drafts K tokens via stream_decode (no re-prefill)
            t_npu = time.perf_counter()
            rsp = npu_request(npu, {
                "op": "stream_decode", "id": f"r{round_i}-decode",
                "stream_id": NPU_STREAM_ID, "max_new": args.k,
            })
            npu_decode_s = time.perf_counter() - t_npu
            if not rsp.get("ok"):
                stop_reason = f"npu_decode_failed: {rsp}"
                break
            draft_ids = rsp["generated_ids"]
            if len(draft_ids) < args.k:
                # Stream hit ctx_full mid-draft
                stop_reason = f"npu_short_draft (got {len(draft_ids)}/{args.k})"
                break
            total_npu_decode_s += npu_decode_s

            # 2. Target batched verify
            saved_L = llm.n_tokens
            assert saved_L == L, f"KV drift: llm.n_tokens={saved_L} vs L={L}"
            t_eval = time.perf_counter()
            llm.eval(list(draft_ids))
            eval_wall = time.perf_counter() - t_eval
            total_target_eval_s += eval_wall

            # 3. Find first mismatch
            first_mm = None
            target_preds = []
            for i in range(args.k):
                pred = int(np.argmax(llm.scores[L - 1 + i]))
                target_preds.append(pred)
                if pred != draft_ids[i]:
                    first_mm = i
                    break

            t_rewind = time.perf_counter()
            if first_mm is None:
                # All K accepted — sample bonus
                bonus = int(np.argmax(llm.scores[L + args.k - 1]))
                llm.eval([bonus])  # extends target KV to L+K+1
                # NPU: append the bonus (KV at L+K is already drafted; bonus
                # goes at L+K. NPU stream is at position L+K, kv.t=L+K).
                rsp = npu_request(npu, {
                    "op": "stream_append", "id": f"r{round_i}-append",
                    "stream_id": NPU_STREAM_ID, "append_ids": [bonus],
                })
                if not rsp.get("ok"):
                    stop_reason = f"npu_append_failed: {rsp}"
                    break
                this_round_commit = list(draft_ids) + [bonus]
                n_accepted = args.k
                new_L = L + args.k + 1
            else:
                target_token = target_preds[first_mm]
                # Target rewind
                llm._ctx.kv_cache_seq_rm(-1, L + first_mm, -1)
                llm.n_tokens = L + first_mm
                llm.eval([target_token])
                # NPU: truncate to L+first_mm, then append target_token
                rsp = npu_request(npu, {
                    "op": "stream_truncate", "id": f"r{round_i}-trunc",
                    "stream_id": NPU_STREAM_ID,
                    "new_position": L + first_mm,
                })
                if not rsp.get("ok"):
                    stop_reason = f"npu_truncate_failed: {rsp}"
                    break
                rsp = npu_request(npu, {
                    "op": "stream_append", "id": f"r{round_i}-append",
                    "stream_id": NPU_STREAM_ID, "append_ids": [target_token],
                })
                if not rsp.get("ok"):
                    stop_reason = f"npu_append_failed: {rsp}"
                    break
                this_round_commit = list(draft_ids[:first_mm]) + [target_token]
                n_accepted = first_mm
                new_L = L + first_mm + 1
            commit_wall = time.perf_counter() - t_rewind
            total_npu_commit_s += commit_wall  # incl target rewind too
            total_target_rewind_s += commit_wall  # historical name

            committed_ids.extend(this_round_commit)
            L = new_L
            assert llm.n_tokens == L, \
                f"post-rewind drift: llm.n_tokens={llm.n_tokens} vs L={L}"

            per_round.append({
                "round": round_i, "K": args.k,
                "n_accepted": n_accepted,
                "first_mismatch": (first_mm if first_mm is not None else args.k),
                "committed": this_round_commit,
                "npu_decode_s": npu_decode_s,
                "target_eval_s": eval_wall,
                "commit_s": commit_wall,
            })

            if any(t in QWEN3_EOS_IDS for t in this_round_commit):
                stop_reason = "eos"
                break
            if len(committed_ids) >= args.max_committed:
                stop_reason = "max_committed"
                break

        total_loop_s = time.perf_counter() - t_loop_start

        print(f"\n=== Loop summary ===")
        print(f"  stop_reason       : {stop_reason}")
        print(f"  rounds            : {len(per_round)}")
        print(f"  committed_total   : {len(committed_ids)}")
        if per_round:
            mean_acc = sum(r["n_accepted"] for r in per_round) / len(per_round)
            print(f"  mean_accept/K     : {mean_acc:.2f} / {args.k} "
                  f"({100*mean_acc/args.k:.0f}%)")
        print(f"  total NPU decode  : {total_npu_decode_s:.2f}")
        print(f"  total NPU commit  : {total_npu_commit_s:.2f}")
        print(f"  total target eval : {total_target_eval_s:.2f}")
        print(f"  target prefill    : {prefill_wall:.2f}")
        print(f"  npu open          : {npu_open_wall:.2f}")
        print(f"  total loop_s      : {total_loop_s:.2f}")
        if total_loop_s > 0:
            rate = len(committed_ids) / total_loop_s
            print(f"  spec-decode t/s   : {rate:.2f}")
        if baseline_stats:
            print(f"  baseline t/s      : {baseline_stats['rate_t_s']:.2f}")
            if baseline_stats["rate_t_s"] > 0:
                speedup = (len(committed_ids) / total_loop_s) / baseline_stats["rate_t_s"]
                print(f"  speedup vs base   : {speedup:.2f}x")

        print(f"\n--- Per-round detail ---")
        for r in per_round:
            preview = tokenizer.decode(r["committed"])[:50].replace("\n", "\\n")
            print(f"  r{r['round']:>2}: K={r['K']} matched={r['n_accepted']}/{r['K']} "
                  f"first_mm={r['first_mismatch']} "
                  f"npu_dec={r['npu_decode_s']:.2f}s "
                  f"tgt_eval={r['target_eval_s']:.3f}s "
                  f"commit={r['commit_s']:.3f}s "
                  f"-> {preview!r}")

        text = tokenizer.decode(committed_ids)
        print(f"\n--- Final committed text ({len(committed_ids)} tokens) ---")
        print(f"  {text!r}")

        if args.csv:
            import csv
            tag = args.tag or f"sq1_c_stateful_k{args.k}_r{args.rounds}_{int(time.time())}"
            row = {
                "tag": tag, "k": args.k, "max_rounds": args.rounds,
                "max_committed": args.max_committed,
                "ctx_tier": args.ctx_tier,
                "prompt_tokens": len(prompt_ids),
                "stop_reason": stop_reason,
                "rounds": len(per_round),
                "n_committed": len(committed_ids),
                "total_npu_decode_s": total_npu_decode_s,
                "total_npu_commit_s": total_npu_commit_s,
                "total_target_eval_s": total_target_eval_s,
                "target_prefill_s": prefill_wall,
                "npu_open_s": npu_open_wall,
                "total_loop_s": total_loop_s,
                "spec_decode_t_s": (len(committed_ids) / total_loop_s
                                    if total_loop_s > 0 else 0),
                "baseline_t_s": (baseline_stats["rate_t_s"] if baseline_stats else None),
                "mean_accept_per_K": (
                    sum(r["n_accepted"] for r in per_round) / len(per_round)
                    if per_round else 0
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
