"""Phase 5 step 7 — first NPU-drafted token into llama.cpp verify.

Plumbing checkpoint, not a perf run. Exit criterion per docs/npu_scoping.md §7
step 7: one drafted token returned, one target token returned, accept/reject
decision logged.

Flow:

  1. Spawn llama-server (CPU build) against Qwen3-8B-Q4_K_M with greedy
     sampling pinned (temperature=0, top_k=1). Wait for /health.
  2. Sanity-probe: tokenize a short string via both the HF Qwen3-0.6B BPE
     tokenizer (what the NPU draft uses) and llama-server's /tokenize
     endpoint (what the target uses). Assert id streams match — confirms
     cross-model token-id compatibility, which is the precondition for
     comparing draft ids to target ids without a text round-trip.
  3. Reuse npu_vs_cpu_correctness.run_cpu_prefill_then_decode_to_511 to
     build a 511-token anchor state on the CPU ONNX. This lands us on
     numerically-validated ground — same anchor as the step 6 probe
     where Path A scored cos=0.9999.
  4. Run one NPU forward on Path A at anchor position 511 with
     anchor_next_id as input. Argmax of the NPU logits -> draft_id.
     This is the draft's prediction for position 512 given the
     512-token prefix (prefix_ids + [anchor_next_id]).
  5. Submit the same 512-id prefix to llama-server /completion with
     n_predict=1, temperature=0, top_k=1, seed=1, return_tokens=true.
     Read the single returned token -> target_id.
  6. Compare: accept = (draft_id == target_id). Log the verdict.
  7. Tear down the server.

We send raw ids (not text) to the /completion endpoint so no tokenizer
round-trip sits between draft and target — the comparison is purely at
the id level, matching llama.cpp's native spec-decode accept logic.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_spec_step7_plumbing.py
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import CONTEXT_MAX, LOGITS_OUTPUT_NAME  # noqa: E402
from npu_vs_cpu_correctness import (  # noqa: E402
    CONFIG_JSON,
    CPU_ONNX,
    PROMPT,
    TOKENIZER_JSON,
    load_cpu_session,
    load_npu_session,
    run_cpu_prefill_then_decode_to_511,
    single_step,
)

TARGET_MODEL = REPO_ROOT / "models" / "Qwen3-8B-Q4_K_M.gguf"
LLAMA_SERVER = REPO_ROOT / "llama.cpp" / "build-cpu" / "bin" / "llama-server.exe"
SERVER_LOG = REPO_ROOT / "results" / "phase5_step7_llama_server.log"

HOST = "127.0.0.1"
PORT = 8088
BASE_URL = f"http://{HOST}:{PORT}"

# Target needs room for the 512-token prompt + n_predict=1 + a little slack.
SERVER_CTX_SIZE = CONTEXT_MAX + 64
# Target decode uses CPU build; pin to the same 18 threads our Phase 2 CPU
# spec-decode baseline used so timings (if we care later) are comparable.
SERVER_THREADS = 18
HEALTH_TIMEOUT_S = 180


def _safe_repr(s: str) -> str:
    return repr(s.encode("ascii", "backslashreplace").decode("ascii"))


def http_post_json(url: str, payload: dict, timeout: float = 180.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_health(url: str, timeout_s: float) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def spawn_server(log_path: Path = SERVER_LOG) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(TARGET_MODEL),
        "--host", HOST,
        "--port", str(PORT),
        "-c", str(SERVER_CTX_SIZE),
        "-t", str(SERVER_THREADS),
        "--no-warmup",
    ]
    print(f"launching llama-server:\n    {' '.join(cmd)}")
    print(f"    log -> {log_path}")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def teardown_server(proc: subprocess.Popen, log_f) -> None:
    print("shutting down llama-server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_f.close()


def tokenizer_sanity_probe(tok_hf: Tokenizer) -> bool:
    """Verify HF Qwen3-0.6B BPE and llama-server's GGUF vocab agree on ids.

    Returns True iff the id streams match for our probe string. If they
    disagree, the id-based /completion comparison below is invalid and
    the user needs to investigate tokenizer divergence before step 8.
    """
    probe = "def fibonacci(n):\n    # Return the nth Fibonacci number"
    hf_ids = tok_hf.encode(probe).ids
    srv = http_post_json(
        f"{BASE_URL}/tokenize",
        {"content": probe, "add_special": False},
    )
    srv_ids = [int(t) for t in srv["tokens"]]
    print(f"  probe text        : {_safe_repr(probe)}")
    print(f"  HF ids ({len(hf_ids):2d})      : {hf_ids}")
    print(f"  server ids ({len(srv_ids):2d})  : {srv_ids}")
    match = hf_ids == srv_ids
    print(f"  tokenizers agree  : {match}")
    return match


def npu_draft_and_cpu_reference(
    path_key: str,
) -> tuple[int, int, list[int], int]:
    """Drive CPU prefill to past_len=511, then one NPU step at position 511.

    Returns (draft_id, cpu_ref_id, prefix_ids_511, anchor_next_id):
      * draft_id       — NPU argmax for position 512
      * cpu_ref_id     — CPU argmax for position 512 (step-6 reference)
      * prefix_ids_511 — the 511-token prefix (positions 0..510)
      * anchor_next_id — CPU greedy for position 511 (NOT yet in prefix)
    """
    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    print("\n--- loading CPU ONNX (FP32, dynamic past_len) ---")
    t0 = time.perf_counter()
    cpu_sess = load_cpu_session(CPU_ONNX)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    print("\n--- loading NPU Path A binary (FP16, fixed past_len=511) ---")
    t0 = time.perf_counter()
    npu_sess = load_npu_session(cfg, path_key)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")
    providers = npu_sess.get_providers()
    if not providers or providers[0] != "QNNExecutionProvider":
        raise RuntimeError(f"NPU session fell back: providers={providers}")

    print(f"\n--- CPU prefill + decode to past_len={CONTEXT_MAX - 1} ---")
    print(f"  prompt = {_safe_repr(PROMPT)}")
    anchor_past, anchor_next_id, generated = run_cpu_prefill_then_decode_to_511(
        cpu_sess, cfg, tok, PROMPT
    )
    anchor_position = CONTEXT_MAX - 1  # 511

    prompt_ids = tok.encode(PROMPT).ids
    # generated = [greedy at pos P, greedy at pos P+1, ..., greedy at pos 511]
    # prefix covers positions 0..510 = prompt_ids + generated[:-1]
    prefix_ids = list(prompt_ids) + list(generated[:-1])
    if len(prefix_ids) != CONTEXT_MAX - 1:
        raise AssertionError(
            f"prefix length {len(prefix_ids)} != {CONTEXT_MAX - 1}"
        )
    if generated[-1] != anchor_next_id:
        raise AssertionError(
            f"anchor_next_id {anchor_next_id} != generated[-1] {generated[-1]}"
        )

    print(f"\n--- single NPU forward at position {anchor_position} ---")
    cpu_logits, npu_logits, _, _ = single_step(
        cpu_sess, npu_sess, cfg, anchor_past, anchor_next_id, anchor_position, path_key
    )
    draft_id = int(np.argmax(npu_logits))
    cpu_ref_id = int(np.argmax(cpu_logits))
    print(f"  NPU draft id      : {draft_id}   -> {_safe_repr(tok.decode([draft_id]))}")
    print(f"  CPU 0.6B ref id   : {cpu_ref_id}   -> {_safe_repr(tok.decode([cpu_ref_id]))}")

    return draft_id, cpu_ref_id, prefix_ids, int(anchor_next_id)


def target_next_token(prefix_ids_512: list[int]) -> int:
    """Ask llama-server for the target's greedy next token."""
    if len(prefix_ids_512) != CONTEXT_MAX:
        raise AssertionError(
            f"target prefix length {len(prefix_ids_512)} != {CONTEXT_MAX}"
        )
    payload = {
        "prompt": prefix_ids_512,
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": False,
        "return_tokens": True,
    }
    t0 = time.perf_counter()
    resp = http_post_json(f"{BASE_URL}/completion", payload)
    elapsed = time.perf_counter() - t0

    tokens = resp.get("tokens") or []
    if not tokens:
        raise RuntimeError(
            f"/completion did not return tokens; response keys: {list(resp.keys())}"
        )
    target_id = int(tokens[0])
    timings = resp.get("timings", {})
    print(f"  elapsed           : {elapsed:.2f} s (server: prompt_n="
          f"{timings.get('prompt_n', '?')}, predicted_n="
          f"{timings.get('predicted_n', '?')})")
    content = resp.get("content", "")
    print(f"  target id         : {target_id}   -> content={_safe_repr(content)}")
    return target_id


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask"), default="patha")
    args = parser.parse_args()
    path_key = args.path

    print(f"=== step 7 - first NPU-drafted token into llama.cpp verify ({path_key}) ===\n")

    if not TARGET_MODEL.exists():
        print(f"ERROR: target GGUF missing at {TARGET_MODEL}")
        return 2
    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server.exe missing at {LLAMA_SERVER}")
        return 2

    proc, log_f = spawn_server()
    try:
        print(f"\n--- waiting for /health (timeout {HEALTH_TIMEOUT_S}s) ---")
    # keep all downstream prints cp1252-safe (no check/cross marks, em-dashes, etc.)
        if not wait_for_health(f"{BASE_URL}/health", HEALTH_TIMEOUT_S):
            print("ERROR: llama-server never came healthy")
            return 2
        print("  server healthy")

        print("\n--- tokenizer sanity probe (HF 0.6B vs server 8B) ---")
        tok_hf = Tokenizer.from_file(str(TOKENIZER_JSON))
        if not tokenizer_sanity_probe(tok_hf):
            print("ERROR: tokenizer id streams disagree - id-based "
                  "/completion comparison is invalid.")
            return 1

        print("\n--- NPU draft ---")
        draft_id, cpu_ref_id, prefix_ids_511, anchor_next_id = (
            npu_draft_and_cpu_reference(path_key)
        )

        # Full 512-id prefix the target sees for its own greedy prediction
        # at position 512.
        target_prefix = prefix_ids_511 + [anchor_next_id]

        print("\n--- target verify (llama-server, Qwen3-8B-Q4_K_M) ---")
        target_id = target_next_token(target_prefix)

        accept = draft_id == target_id
        draft_agrees_with_cpu_06 = draft_id == cpu_ref_id

        print("\n=== accept/reject verdict ===")
        print(f"  NPU draft (0.6B)           : {draft_id}   -> {_safe_repr(tok_hf.decode([draft_id]))}")
        print(f"  CPU reference (0.6B)       : {cpu_ref_id}   -> {_safe_repr(tok_hf.decode([cpu_ref_id]))}")
        print(f"  Target (8B via llama-srv)  : {target_id}   -> {_safe_repr(tok_hf.decode([target_id]))}")
        print(f"  draft == target (accept)   : {accept}")
        print(f"  draft == 0.6B-CPU (sanity) : {draft_agrees_with_cpu_06}")
        if accept:
            print("  interpretation             : draft accepted - target would skip its own compute.")
        else:
            print("  interpretation             : draft rejected - target's token wins this round.")

        print("\n=== STATUS: ok (one draft token + one target token + one accept/reject decision) ===")
        return 0
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        teardown_server(proc, log_f)


if __name__ == "__main__":
    sys.exit(main())
