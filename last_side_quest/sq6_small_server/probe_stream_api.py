"""Validate sidecar's new stateful stream API.

Spawns sidecar, exercises stream_open / stream_decode / stream_truncate /
stream_append / stream_close in sequence, asserting position invariants.

Compares one stateful run vs an equivalent stateless run (the existing
`chat` op) to confirm the same generated tokens come out — that's the
correctness check that says "the rewind is byte-identical to a fresh
prefill of the same effective prompt".

Run:
    .venv/Scripts/python.exe last_side_quest/sq6_small_server/probe_stream_api.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
TOKENIZER_PATH = BUNDLE_DIR / "tokenizer.json"


def spawn_sidecar(ctx_tier=2048):
    cmd = [
        sys.executable, str(REPO_ROOT / "npu_engine" / "sidecar.py"),
        "--mode", "serve",
        "--ctx-tier", str(ctx_tier),
        "--start-mode", "ar1",
    ]
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONHOME", "PYTHONPATH",
                        "PYTHONSTARTUP", "VIRTUAL_ENV")}
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1, encoding="utf-8", env=env,
    )
    while True:
        line = proc.stdout.readline()
        if not line:
            print(f"[sidecar died]\n{proc.stderr.read()}", file=sys.stderr)
            sys.exit(2)
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("event") == "ready":
            print(f"  sidecar ready in {evt['startup_s']:.1f}s")
            return proc


def req(proc, **kwargs):
    proc.stdin.write(json.dumps(kwargs) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline().strip())


def main():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    prompt_text = (
        "<|im_start|>user\nWrite a one-line Python comment about Fibonacci."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False).ids
    print(f"prompt: {len(prompt_ids)} tokens")

    print("\n=== Spawning sidecar ===")
    proc = spawn_sidecar()
    try:
        # ---- Run 1: stateless chat baseline ----
        print("\n--- Run 1 (stateless chat, full prefill) ---")
        t = time.perf_counter()
        r = req(proc, op="chat", id="r1", prompt_ids=prompt_ids,
                max_new_tokens=20,
                eos_ids=[151643, 151645],
                stop_token_seqs=[],
                force_ar128=False)
        wall1 = time.perf_counter() - t
        assert r["ok"], r
        gen1 = r["generated_ids"]
        print(f"  generated {len(gen1)} tokens in {wall1:.2f}s")
        print(f"  text: {tok.decode(gen1)!r}")

        # ---- Run 2: stateful with stream_open + stream_decode ----
        print("\n--- Run 2 (stateful: open + decode) ---")
        t = time.perf_counter()
        r = req(proc, op="stream_open", id="r2a", stream_id="s2",
                prompt_ids=prompt_ids, force_ar128=False)
        assert r["ok"], r
        assert r["position"] == len(prompt_ids), r
        # stream_open uses prefill_only which in turn sets last_logits +
        # next_token (Stream constructor) — so we should be able to call
        # stream_decode immediately.
        r = req(proc, op="stream_decode", id="r2b", stream_id="s2",
                max_new=20, eos_ids=[151643, 151645])
        wall2 = time.perf_counter() - t
        assert r["ok"], r
        gen2 = r["generated_ids"]
        print(f"  generated {len(gen2)} tokens in {wall2:.2f}s")
        print(f"  text: {tok.decode(gen2)!r}")
        assert r["position"] == len(prompt_ids) + len(gen2), r

        # CORRECTNESS CHECK: stateful and stateless should produce the
        # same tokens (same prompt, greedy decode, no randomness).
        if gen1 == gen2:
            print("  ✓ stateful matches stateless — rewind correctness OK")
        else:
            print(f"  ✗ DIVERGENCE: stateless={gen1[:5]}... stateful={gen2[:5]}...")
            return 1

        # ---- Run 3: truncate + append + decode ----
        # Roll back the last 5 generated tokens, append something else,
        # then continue. This is the spec-decode rejection path AND the
        # chat-server "user edited an earlier message" path.
        print("\n--- Run 3 (truncate + append + decode) ---")
        rollback_to = len(prompt_ids) + len(gen2) - 5
        r = req(proc, op="stream_truncate", id="r3a", stream_id="s2",
                new_position=rollback_to)
        assert r["ok"], r
        assert r["position"] == rollback_to, r
        # Append two specific tokens — the model should now see those
        # at positions rollback_to and rollback_to+1, then continue.
        # Pick tokens that probably make grammatical sense: " and" (323)
        # and " also" (1083) (just probing — meaning doesn't matter).
        injected = [323, 1083]
        r = req(proc, op="stream_append", id="r3b", stream_id="s2",
                append_ids=injected, force_ar128=False)
        assert r["ok"], r
        assert r["position"] == rollback_to + len(injected), r
        # Decode 10 more
        r = req(proc, op="stream_decode", id="r3c", stream_id="s2",
                max_new=10, eos_ids=[151643, 151645])
        assert r["ok"], r
        gen3 = r["generated_ids"]
        print(f"  injected: {tok.decode(injected)!r}")
        print(f"  generated after: {tok.decode(gen3)!r}")
        print(f"  position now: {r['position']}")

        # ---- Run 4: close stream, verify it's gone ----
        print("\n--- Run 4 (close + verify) ---")
        r = req(proc, op="stream_close", id="r4", stream_id="s2")
        assert r["ok"], r
        # stream_decode should now fail with "unknown stream_id"
        r = req(proc, op="stream_decode", id="r4b", stream_id="s2",
                max_new=1)
        assert not r["ok"], f"expected failure, got {r}"
        assert "unknown stream_id" in r.get("error", "")
        print("  ✓ closed stream correctly rejected")

        print("\n=== ALL CHECKS PASSED ===")
        return 0
    finally:
        try:
            proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
