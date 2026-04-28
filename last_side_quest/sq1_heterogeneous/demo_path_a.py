"""SQ1 Path A — plumbing-only heterogeneous demo.

NPU 4B drafts K tokens given a prompt; CPU 14B target generates K tokens
given the same prompt; driver prints side-by-side and computes naive
"draft matches target" rate at each position. NO spec-decode loop, NO
KV rewind — both backends run independently and we just diff outputs.

What this proves:
  - Both compute islands talk to each other end-to-end.
  - NPU draft tokens are coherent (or surface a tokenizer mismatch).
  - Position-1 match rate is a real number (the "would the target have
    produced the same first token" rate, the headline accept-rate
    upper bound for a real spec-decode loop).

What this DOESN'T prove:
  - Any throughput claim — both backends do their full work
    independently. This is plumbing, not perf.

Prereqs:
  - llama-server running Qwen3-14B-Q4_K_M on http://127.0.0.1:8081
    (use scripts/serve_target_14b.ps1).
  - venv with onnxruntime-qnn, numpy, tokenizers, requests, yaml.
  - Qualcomm Qwen3-4B bundle present (used by npu_engine/sidecar.py).

Run:
  .venv/Scripts/python.exe last_side_quest/sq1_heterogeneous/demo_path_a.py \\
      --target-url http://127.0.0.1:8081 --k 8 --ctx-tier 2048
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Windows cp1252 default stdout can't encode token strings that include
# Qwen's tokenizer special bytes (Ġ space marker, multi-byte UTF-8 in
# non-ASCII tokens). Force UTF-8 before any print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = (
    REPO_ROOT
    / "models"
    / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
BUNDLE_TOKENIZER = BUNDLE_DIR / "tokenizer.json"


DEFAULT_PROMPT = (
    "You are an expert Python developer. Complete the following function:\n\n"
    "def fibonacci(n: int) -> int:\n"
    '    """Return the nth Fibonacci number using O(n) time.\n\n'
    "    Args:\n"
    "        n: Non-negative integer.\n\n"
    "    Returns:\n"
    "        The nth Fibonacci number, where fibonacci(0) = 0 and fibonacci(1) = 1.\n"
    '    """\n'
)


def spawn_npu_sidecar(ctx_tier: int):
    """Launch npu_engine/sidecar.py as a subprocess and wait for ready."""
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "npu_engine" / "sidecar.py"),
        "--mode", "serve",
        "--ctx-tier", str(ctx_tier),
        "--start-mode", "ar1",
    ]
    print(f"  spawning: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8",
    )
    while True:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read()
            print(f"  NPU sidecar died before ready:\n{stderr}", file=sys.stderr)
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
                  f"mode={evt.get('start_mode')} "
                  f"per_part={evt.get('start_per_part_s')}")
            return proc
        # Other events get echoed.
        print(f"  [npu] {evt}")


def npu_request(proc, req: dict) -> dict:
    line = json.dumps(req) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    rsp_line = proc.stdout.readline()
    if not rsp_line:
        raise RuntimeError("NPU sidecar closed stdout")
    return json.loads(rsp_line.strip())


def target_completion(target_url: str, prompt_ids: list[int], n_predict: int) -> dict:
    """POST llama-server /completion with token-ID prompt and request
    deterministic K tokens. Returns the parsed body."""
    body = {
        "prompt": prompt_ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        # llama-server includes the generated tokens in `tokens` when
        # this is set; otherwise we'd have to re-tokenize text and risk
        # a roundtrip mismatch.
        "return_tokens": True,
    }
    t = time.perf_counter()
    r = requests.post(f"{target_url.rstrip('/')}/completion", json=body, timeout=300)
    wall_s = time.perf_counter() - t
    r.raise_for_status()
    out = r.json()
    out["__wall_s"] = wall_s
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-url", default="http://127.0.0.1:8081")
    p.add_argument("--prompt", default=None,
                   help="prompt text (default: a fixed Python coding task)")
    p.add_argument("--prompt-file", default=None,
                   help="path to a .txt file with the prompt")
    p.add_argument("--k", type=int, default=8,
                   help="number of tokens both sides predict")
    p.add_argument("--ctx-tier", type=int, default=2048,
                   choices=(512, 1024, 2048, 3072, 4096))
    p.add_argument("--csv", default=None, help="optional CSV output path")
    p.add_argument("--tag", default=None, help="tag for CSV row")
    args = p.parse_args()

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = DEFAULT_PROMPT

    if not BUNDLE_TOKENIZER.exists():
        print(f"ERROR: bundle tokenizer not found at {BUNDLE_TOKENIZER}", file=sys.stderr)
        return 2
    tokenizer = Tokenizer.from_file(str(BUNDLE_TOKENIZER))
    prompt_ids = tokenizer.encode(prompt_text).ids

    print(f"=== SQ1 Path A — plumbing-only heterogeneous demo ===")
    print(f"  target_url  : {args.target_url}")
    print(f"  ctx tier    : {args.ctx_tier}")
    print(f"  k           : {args.k}")
    print(f"  prompt_ids  : {len(prompt_ids)} tokens")
    preview = prompt_text[:80].replace("\n", "\\n")
    print(f"  prompt_text : '{preview}{'...' if len(prompt_text) > 80 else ''}'")
    print()

    # Reachability check on target before spawning NPU (15+ s startup).
    print("--- Checking target reachability ---")
    try:
        r = requests.get(f"{args.target_url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        print(f"  target health: {r.json()}")
    except Exception as e:
        print(f"  ERROR: target server not reachable at {args.target_url} ({e})", file=sys.stderr)
        print(f"  start it with: scripts/serve_target_14b.ps1", file=sys.stderr)
        return 2

    print("\n--- Starting NPU sidecar ---")
    npu = spawn_npu_sidecar(args.ctx_tier)
    if npu is None:
        return 2

    try:
        # 1. NPU drafts K tokens
        print(f"\n--- NPU draft (k={args.k}) ---")
        npu_rsp = npu_request(npu, {
            "op": "draft",
            "id": "demo-A",
            "prompt_ids": prompt_ids,
            "n_draft": args.k,
        })
        if not npu_rsp.get("ok"):
            print(f"  NPU draft failed: {npu_rsp}", file=sys.stderr)
            return 1
        draft_ids = npu_rsp["draft_ids"]
        npu_pp_s = npu_rsp["pp_wall_s"]
        npu_tg_s = npu_rsp["tg_wall_s"]
        print(f"  draft_ids   : {draft_ids}")
        print(f"  decoded     : {tokenizer.decode(draft_ids)!r}")
        print(f"  pp wall     : {npu_pp_s:.2f} s")
        print(f"  tg wall     : {npu_tg_s:.2f} s "
              f"({args.k / npu_tg_s:.2f} t/s)")

        # 2. Target generates K tokens given the same prompt
        print(f"\n--- Target (CPU 14B) generation (k={args.k}) ---")
        try:
            tgt = target_completion(args.target_url, prompt_ids, args.k)
        except Exception as e:
            print(f"  ERROR calling target: {e}", file=sys.stderr)
            return 1
        target_wall_s = tgt["__wall_s"]
        # llama-server returns tokens in different shapes across versions;
        # try a few likely keys.
        target_ids = (
            tgt.get("tokens")
            or [t.get("id") for t in tgt.get("completion_probabilities", []) if "id" in t]
            or []
        )
        if not target_ids:
            # Fallback to re-tokenizing the content. Note: this can give
            # a slightly different ID sequence due to BPE merge order on
            # whitespace-leading tokens. Flag it.
            content = tgt.get("content", "")
            target_ids = tokenizer.encode(content).ids[: args.k]
            print(f"  WARN: server didn't return tokens; re-tokenized content")
        target_ids = list(target_ids)[: args.k]
        target_text = tokenizer.decode(target_ids) if target_ids else tgt.get("content", "")
        print(f"  target_ids  : {target_ids}")
        print(f"  decoded     : {target_text!r}")
        print(f"  wall        : {target_wall_s:.2f} s "
              f"({args.k / target_wall_s:.2f} t/s)")
        timings = tgt.get("timings") or {}
        if timings:
            print(f"  timings (server-side): "
                  f"prompt_ms={timings.get('prompt_ms'):.0f} "
                  f"predicted_ms={timings.get('predicted_ms'):.0f} "
                  f"predicted_per_token_ms={timings.get('predicted_per_token_ms'):.1f}")

        # 3. Side-by-side
        print(f"\n--- Side-by-side at K={args.k} ---")
        print(f"  {'pos':>3}  {'NPU':>10}  {'target':>10}  {'match':>5}  "
              f"NPU_tok / target_tok")
        print("  " + "-" * 88)
        matches = []
        for i in range(args.k):
            d = draft_ids[i] if i < len(draft_ids) else None
            t = target_ids[i] if i < len(target_ids) else None
            ok = d is not None and t is not None and d == t
            matches.append(ok)
            d_str = tokenizer.id_to_token(d) if d is not None else "-"
            t_str = tokenizer.id_to_token(t) if t is not None else "-"
            print(f"  {i:>3}  {str(d):>10}  {str(t):>10}  {'YES' if ok else 'no':>5}  "
                  f"{d_str!r:<25} {t_str!r}")

        # First-mismatch position is the meaningful spec-decode metric:
        # in real spec-decode at this K, you'd accept first_mismatch_pos
        # tokens and resample at that position.
        first_mismatch = next(
            (i for i, m in enumerate(matches) if not m), len(matches)
        )
        n_match = sum(matches)
        print(f"\n  matches               : {n_match}/{args.k} ({100*n_match/args.k:.0f}%)")
        print(f"  first-mismatch index  : {first_mismatch}  "
              f"(real spec-decode would accept {first_mismatch}/{args.k} per round at K={args.k})")

        # 4. Optional CSV row
        if args.csv:
            import csv
            tag = args.tag or f"sq1_a_k{args.k}_cl{args.ctx_tier}_{int(time.time())}"
            row = {
                "tag": tag,
                "k": args.k,
                "ctx_tier": args.ctx_tier,
                "prompt_tokens": len(prompt_ids),
                "draft_ids": ";".join(map(str, draft_ids)),
                "target_ids": ";".join(map(str, target_ids)),
                "matches": ";".join("1" if m else "0" for m in matches),
                "n_match": n_match,
                "first_mismatch": first_mismatch,
                "npu_pp_s": npu_pp_s,
                "npu_tg_s": npu_tg_s,
                "target_wall_s": target_wall_s,
                "target_predicted_ms": (timings.get("predicted_ms") if timings else None),
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
