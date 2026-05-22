"""Phase 1 config C2 — draft on Adreno GPU || verify on Hexagon NPU.

C2 = speculative decoding where:
  * DRAFT  : Qwen3-0.6B on the Adreno GPU via a llama.cpp OpenCL
             llama-server (/completion, raw token ids in/out).
  * VERIFY : Qwen3-4B w4a16 on the Hexagon NPU, in-process via the
             npu_engine/sidecar.py --serve subprocess (stream_open /
             stream_decode / stream_truncate / stream_append).
  * OVERLAP: round N+1's GPU draft overlaps round N's NPU verify via
             a ThreadPoolExecutor, mirroring
             scripts/npu_spec_outer_loop_async.py.

This is a NEW standalone driver. It does not modify the existing loop
or the sidecar.

--- Verify contract (NPU, in-process) -----------------------------------

verify(committed_ids, drafts) must return the TARGET model's greedy
(argmax) token at each of the k+1 positions:
  out[0]   = target argmax given committed_ids
  out[i+1] = target argmax given committed_ids + drafts[:i+1]

Built only from the sidecar's existing stream primitives:
  stream_open(prefix)              -> prefill committed_ids
  loop i = 0..k:
     stream_decode(max_new=1)      -> argmax at current position -> out[i]
     if i < k:
        stream_truncate(pos-1)     -> drop the just-decoded slot
        stream_append([drafts[i]]) -> ingest the draft token instead
The next stream_decode then yields the argmax given prefix+drafts[:i+1].

Then longest-common-prefix(drafts, out) -> accepted count j, and
out[j] is the bonus token (same accept logic as
npu_spec_outer_loop_async.py).

Run:
  .venv\\Scripts\\python.exe gpu_npu_sidequest\\scripts\\run_c2.py --sanity
  .venv\\Scripts\\python.exe gpu_npu_sidequest\\scripts\\run_c2.py -k 4 -n 128
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDEQUEST = REPO_ROOT / "gpu_npu_sidequest"
BUNDLE = (REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
          / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite")
SIDECAR = REPO_ROOT / "npu_engine" / "sidecar.py"
PYEXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

GPU_DRAFT_URL = "http://127.0.0.1:8089"
# cl512 tier: total KV slots capped at 511.
CTX_CAP = 511

# Fixed prompt — one technical paragraph (matches the native-specdecode
# baseline run's style; ~30 tokens).
FIXED_PROMPT = (
    "Speculative decoding accelerates language model inference by using "
    "a small draft model to propose multiple tokens which the larger "
    "target model then verifies in a single batched forward pass."
)


# ----------------------------- GPU draft side ----------------------------

def http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def draft_via_gpu_server(prefix_ids: list[int], k: int) -> list[int]:
    """POST committed prefix to the OpenCL llama-server; return k greedy
    draft token ids. cache_prompt=true so the shared prefix is reused
    cheaply across rounds."""
    resp = http_post(f"{GPU_DRAFT_URL}/completion", {
        "prompt": list(prefix_ids),
        "n_predict": k,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
        "return_tokens": True,
        "samplers": ["top_k"],
    })
    toks = resp.get("tokens") or []
    return [int(t) for t in toks][:k]


def gpu_server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{GPU_DRAFT_URL}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# ----------------------------- NPU verify side ---------------------------

class NpuSidecar:
    """Wraps the npu_engine/sidecar.py --serve subprocess; speaks the
    newline-delimited JSON protocol over stdin/stdout."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [str(PYEXE), str(SIDECAR), "--model", "qwen3-4b",
             "--mode", "serve", "--start-mode", "ar1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            bufsize=1, cwd=str(REPO_ROOT))
        self.startup_s = None
        # Wait for the "ready" event.
        line = self.proc.stdout.readline()
        evt = json.loads(line)
        if evt.get("event") != "ready":
            raise RuntimeError(f"sidecar did not become ready: {evt}")
        self.startup_s = evt.get("startup_s")

    def request(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sidecar closed unexpectedly")
        return json.loads(line)

    def shutdown(self):
        try:
            self.request({"op": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


VERIFY_SID = "c2-verify"


def verify_via_npu(sc: NpuSidecar, committed_ids: list[int],
                   drafts: list[int]) -> list[int]:
    """Run the 4B target on the NPU and return its greedy token at each
    of the len(drafts)+1 positions.

    out[0]   = target argmax given committed_ids
    out[i+1] = target argmax given committed_ids + drafts[:i+1]

    STATELESS variant: stream_open re-prefills the whole committed
    prefix every call. Correct but pays a full AR1 prefill per round —
    used only by the sanity check.
    """
    k = len(drafts)
    r = sc.request({"op": "stream_open", "stream_id": VERIFY_SID,
                    "prompt_ids": list(committed_ids)})
    if not r.get("ok"):
        raise RuntimeError(f"stream_open failed: {r}")
    pos = r["position"]
    return _verify_speculate(sc, VERIFY_SID, drafts, pos)


def _verify_speculate(sc: NpuSidecar, sid: str, drafts: list[int],
                      base_pos: int) -> list[int]:
    """Given a stream already positioned at `base_pos` (= len(committed)),
    run the k+1-token greedy speculation cycle and return out[0..k].

    The stream is LEFT positioned at base_pos on return (the speculation
    slots are truncated away) so the caller can append the actually-
    committed tokens to keep the KV in sync.
    """
    k = len(drafts)
    out: list[int] = []
    pos = base_pos
    for i in range(k + 1):
        d = sc.request({"op": "stream_decode", "stream_id": sid,
                        "max_new": 1})
        if not d.get("ok"):
            raise RuntimeError(f"stream_decode failed: {d}")
        gen = d.get("generated_ids") or []
        if not gen:
            raise RuntimeError(f"stream_decode produced nothing: {d}")
        out.append(int(gen[0]))
        pos = d["position"]  # advanced by 1
        if i < k:
            t = sc.request({"op": "stream_truncate", "stream_id": sid,
                            "new_position": pos - 1})
            if not t.get("ok"):
                raise RuntimeError(f"stream_truncate failed: {t}")
            a = sc.request({"op": "stream_append", "stream_id": sid,
                            "append_ids": [int(drafts[i])]})
            if not a.get("ok"):
                raise RuntimeError(f"stream_append failed: {a}")
            pos = a["position"]
    # Drop the whole speculation tail; leave the stream at base_pos.
    t = sc.request({"op": "stream_truncate", "stream_id": sid,
                    "new_position": base_pos})
    if not t.get("ok"):
        raise RuntimeError(f"final stream_truncate failed: {t}")
    return out


def npu_greedy_generate(sc: NpuSidecar, prompt_ids: list[int],
                        n: int) -> list[int]:
    """Pure NPU greedy continuation — the reference for the sanity check."""
    sid = "c2-ref"
    r = sc.request({"op": "stream_open", "stream_id": sid,
                    "prompt_ids": list(prompt_ids)})
    if not r.get("ok"):
        raise RuntimeError(f"stream_open failed: {r}")
    d = sc.request({"op": "stream_decode", "stream_id": sid, "max_new": n})
    if not d.get("ok"):
        raise RuntimeError(f"stream_decode failed: {d}")
    return [int(t) for t in d.get("generated_ids") or []]


# ------------------------------- accept logic ----------------------------

def longest_common_prefix(a: list[int], b: list[int]) -> int:
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]:
        j += 1
    return j


# ------------------------------- sanity check ----------------------------

def sanity_check(sc: NpuSidecar, prompt_ids: list[int]) -> bool:
    """The verify path must agree with the 4B's own greedy generation.

    Reference: npu_greedy_generate(prompt, N) -> ref[0..N-1].
    Test: feed verify the *correct* drafts (ref[:k]) for committed=prompt;
    verify must then return exactly ref[0..k] (k+1 tokens). And feeding
    WRONG drafts must make verify diverge at the wrong position.
    """
    print("\n=== SANITY CHECK: NPU verify vs 4B greedy ===", flush=True)
    N = 8
    ref = npu_greedy_generate(sc, prompt_ids, N)
    print(f"  4B greedy ({N} tok): {ref}", flush=True)

    k = 4
    # Test 1: correct drafts — verify must echo ref[0..k].
    correct_drafts = ref[:k]
    out = verify_via_npu(sc, prompt_ids, correct_drafts)
    expect = ref[:k + 1]
    ok1 = out == expect
    print(f"  test1 correct-drafts: drafts={correct_drafts}", flush=True)
    print(f"    verify out = {out}", flush=True)
    print(f"    expected   = {expect}   -> {'PASS' if ok1 else 'FAIL'}",
          flush=True)

    # Test 2: wrong draft at position 1 — LCP must stop at j=1.
    wrong = list(ref[:k])
    wrong[1] = (wrong[1] + 1) % 151936  # corrupt one token
    out2 = verify_via_npu(sc, prompt_ids, wrong)
    j = longest_common_prefix(wrong, out2)
    # out2[0] must still equal ref[0]; out2[1] is target's argmax given
    # the wrong token at pos1 — j should be 1 (accept only draft 0).
    ok2 = (out2[0] == ref[0]) and (j == 1)
    print(f"  test2 wrong-draft@1 : drafts={wrong}", flush=True)
    print(f"    verify out = {out2}  lcp_j={j}  "
          f"-> {'PASS' if ok2 else 'FAIL'}", flush=True)

    passed = ok1 and ok2
    print(f"  SANITY: {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


# ------------------------------- C2 main loop ----------------------------

def run_c2(sc: NpuSidecar, prompt_ids: list[int], k: int,
           n_predict: int) -> dict:
    """Async C2 spec-decode loop: GPU draft || NPU verify, overlapped.

    The NPU verify stream is opened ONCE on the prompt and kept live
    across rounds: each round only appends the newly-committed tokens
    (incremental ingest) instead of re-prefilling the whole prefix.

    Overlap: round N+1's GPU draft is pre-issued the instant round N's
    commits are known, so it runs concurrently with round N's NPU
    verify-stream sync + the next round's NPU speculation. Both phases
    release the GIL (HTTP socket / sidecar pipe), so the ThreadPoolExecutor
    overlap is real.
    """
    committed: list[int] = list(prompt_ids)
    prompt_len = len(prompt_ids)

    draft_wait_s = 0.0
    verify_wait_s = 0.0
    parallel_wall_s = 0.0
    rounds = []

    # Open the verify stream once on the prompt. verify_synced_pos tracks
    # how many committed tokens the stream's KV currently holds.
    r = sc.request({"op": "stream_open", "stream_id": VERIFY_SID,
                    "prompt_ids": list(prompt_ids)})
    if not r.get("ok"):
        raise RuntimeError(f"verify stream_open failed: {r}")
    verify_synced_pos = r["position"]  # == prompt_len

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-draft")

    t_gen0 = time.perf_counter()
    round_idx = 0
    try:
        # Pre-issue round 1 draft.
        pending_draft = executor.submit(
            draft_via_gpu_server, list(committed), k)
        while (len(committed) - prompt_len) < n_predict:
            round_idx += 1
            L = len(committed)
            if L + k + 1 > CTX_CAP:
                print(f"  round {round_idx}: near ctx cap "
                      f"(L={L}); stopping", flush=True)
                break

            t_round = time.perf_counter()

            # Join this round's draft (GPU).
            t_d0 = time.perf_counter()
            drafts = pending_draft.result()
            draft_wait_s += time.perf_counter() - t_d0
            if not drafts:
                print(f"  round {round_idx}: GPU draft empty; stop",
                      flush=True)
                break

            # --- Verify on NPU (main thread) ---
            t_v0 = time.perf_counter()
            # Sync the verify stream's KV up to the current committed
            # prefix by appending only the delta since last round.
            if verify_synced_pos < L:
                a = sc.request({"op": "stream_append",
                                "stream_id": VERIFY_SID,
                                "append_ids": committed[verify_synced_pos:L]})
                if not a.get("ok"):
                    raise RuntimeError(f"verify sync append failed: {a}")
                verify_synced_pos = a["position"]
            # Speculate: k+1 greedy target tokens; stream left at L.
            target_out = _verify_speculate(sc, VERIFY_SID, drafts, L)
            verify_wait_s += time.perf_counter() - t_v0

            j = longest_common_prefix(drafts, target_out)
            bonus = target_out[j]
            new_commits = drafts[:j] + [bonus]
            committed.extend(new_commits)

            # Pre-issue next round's draft NOW — overlaps with the next
            # round's NPU verify-stream sync + speculation.
            next_L = len(committed)
            will_continue = ((next_L - prompt_len) < n_predict
                             and next_L + k + 1 <= CTX_CAP)
            if will_continue:
                pending_draft = executor.submit(
                    draft_via_gpu_server, list(committed), k)

            parallel_wall_s += time.perf_counter() - t_round
            rounds.append({
                "round": round_idx, "L": L, "drafts": drafts,
                "target": target_out, "j": j, "bonus": bonus,
                "committed": len(new_commits),
            })
            print(f"  r{round_idx:03d} L={L:3d} k={len(drafts)} "
                  f"drafts={drafts} target={target_out} j={j} "
                  f"bonus={bonus} +{len(new_commits)}", flush=True)
            if not will_continue:
                break
    finally:
        executor.shutdown(wait=True)

    gen_s = time.perf_counter() - t_gen0
    decoded = len(committed) - prompt_len
    tps = decoded / gen_s if gen_s > 0 else 0.0
    total_drafted = sum(len(r["drafts"]) for r in rounds)
    total_accepted = sum(r["j"] for r in rounds)
    accept_rate = total_accepted / total_drafted if total_drafted else 0.0
    accept_per_round = total_accepted / len(rounds) if rounds else 0.0

    return {
        "k": k, "prompt_len": prompt_len, "decoded": decoded,
        "rounds": len(rounds), "wall_generate_s": gen_s,
        "decode_tps": tps,
        "total_drafted": total_drafted, "total_accepted": total_accepted,
        "accept_rate": accept_rate, "accept_per_round": accept_per_round,
        "draft_wait_s": draft_wait_s, "verify_wait_s": verify_wait_s,
        "parallel_wall_s": parallel_wall_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true",
                    help="run only the verify sanity check and exit")
    ap.add_argument("-k", "--draft-k", type=int, default=4)
    ap.add_argument("-n", "--n-predict", type=int, default=128)
    ap.add_argument("--ks", type=str, default=None,
                    help="comma-separated k sweep, e.g. 2,4,8")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(BUNDLE / "tokenizer.json"))
    prompt_ids = tok.encode(FIXED_PROMPT).ids
    print(f"prompt: {len(prompt_ids)} tokens", flush=True)

    if not gpu_server_healthy():
        print("ERROR: GPU draft server not healthy at "
              f"{GPU_DRAFT_URL}", flush=True)
        return 2
    print("GPU draft server: healthy", flush=True)

    print("starting NPU sidecar (4B w4a16, ~15s load)...", flush=True)
    t0 = time.perf_counter()
    sc = NpuSidecar()
    print(f"NPU sidecar ready ({time.perf_counter()-t0:.1f}s wall, "
          f"startup_s={sc.startup_s})", flush=True)

    try:
        if not sanity_check(sc, prompt_ids):
            print("\nABORT: verify sanity check FAILED — not running C2.",
                  flush=True)
            return 3
        if args.sanity:
            print("\n--sanity only: stopping after sanity check.",
                  flush=True)
            return 0

        ks = ([int(x) for x in args.ks.split(",")] if args.ks
              else [args.draft_k])
        results = []
        for k in ks:
            print(f"\n=== C2 RUN k={k} n_predict={args.n_predict} ===",
                  flush=True)
            summary = run_c2(sc, prompt_ids, k, args.n_predict)
            results.append(summary)
            print(f"  -> decoded={summary['decoded']} "
                  f"rounds={summary['rounds']} "
                  f"tok/s={summary['decode_tps']:.2f} "
                  f"accept_rate={summary['accept_rate']*100:.1f}% "
                  f"accept/round={summary['accept_per_round']:.2f}",
                  flush=True)
            print(f"  draft_wait={summary['draft_wait_s']:.2f}s "
                  f"verify_wait={summary['verify_wait_s']:.2f}s "
                  f"parallel_wall={summary['parallel_wall_s']:.2f}s",
                  flush=True)

        # Emit machine-readable JSON for the CSV writer.
        out_path = SIDEQUEST / "logs" / "c2_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nresults JSON -> {out_path}", flush=True)
        return 0
    finally:
        sc.shutdown()


if __name__ == "__main__":
    sys.exit(main())
