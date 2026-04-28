"""SQ1 sub-probe — verify the NPU bundle tokenizer matches the
running llama-server's tokenizer.

Critical for SQ1's heterogeneous demo: if the draft (NPU 4B,
Qualcomm Qwen3 bundle) and target (whatever llama-server is
serving) use different tokenizer vocabularies, draft_ids will
not align with target_ids and the demo's per-position match
will be meaningless garbage even when the models semantically
agree.

Strategy:
  - Encode a fixed set of probe strings via the bundle's
    `tokenizer.json` locally.
  - POST the same strings to llama-server's /tokenize endpoint.
  - Diff IDs position-by-position. Any mismatch means the
    server's vocab differs from the bundle's.

This is a yes/no answer — we don't try to characterize HOW they
differ; that would be a vocab-translation effort outside SQ1's
scope.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_TOKENIZER = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite" / "tokenizer.json"
)


PROBE_STRINGS = [
    "Hello, world!",
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n",
    "{\n  \"name\": \"Alice\",\n  \"age\": 30\n}",
    "The quick brown fox jumps over the lazy dog. ",
    # Tokens that Qwen sometimes treats specially
    "<|im_start|>user\nHi\n<|im_end|>\n",
    # Unicode / Chinese
    "Hello 世界 🌍",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-url", default="http://127.0.0.1:8081")
    args = p.parse_args()

    if not BUNDLE_TOKENIZER.exists():
        print(f"ERROR: bundle tokenizer not found at {BUNDLE_TOKENIZER}", file=sys.stderr)
        return 2
    bundle_tok = Tokenizer.from_file(str(BUNDLE_TOKENIZER))

    print(f"=== Tokenizer compat probe ===")
    print(f"  bundle: {BUNDLE_TOKENIZER}")
    print(f"  server: {args.target_url}")
    print()

    all_ok = True
    for i, s in enumerate(PROBE_STRINGS):
        bundle_ids = bundle_tok.encode(s).ids

        try:
            r = requests.post(
                f"{args.target_url.rstrip('/')}/tokenize",
                json={"content": s, "add_special": False},
                timeout=15,
            )
            r.raise_for_status()
            server_ids = r.json().get("tokens", [])
        except Exception as e:
            print(f"  probe {i}: ERROR talking to server: {e}", file=sys.stderr)
            return 1

        ok = bundle_ids == server_ids
        all_ok = all_ok and ok
        preview = s.replace("\n", "\\n")[:50]
        if len(s) > 50:
            preview = preview + "..."
        print(f"  probe {i}: {'OK ' if ok else 'MISMATCH'}  '{preview}'")
        if not ok:
            print(f"    bundle: {bundle_ids}")
            print(f"    server: {server_ids}")
            # Show first divergence index
            for j in range(min(len(bundle_ids), len(server_ids))):
                if bundle_ids[j] != server_ids[j]:
                    print(f"    first diff at index {j}: "
                          f"bundle={bundle_ids[j]} server={server_ids[j]}")
                    break
            if len(bundle_ids) != len(server_ids):
                print(f"    length diff: bundle={len(bundle_ids)} server={len(server_ids)}")

    print()
    if all_ok:
        print(f"=== COMPAT — all {len(PROBE_STRINGS)} probes pass ===")
        return 0
    print(f"=== INCOMPAT — at least one probe fails ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
