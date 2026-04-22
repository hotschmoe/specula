"""Phase 5 step 8 — NPU-drafted speculative decoding, first end-to-end run.

Sidecar-as-driver design (per docs/npu_scoping.md §6.6 option (a)):
  * NPU Path B-mask (Qwen3-0.6B) drafts k tokens per round.
  * llama-server (Qwen3-8B-Q4_K_M CPU) verifies via POST /completion
    with prompt = committed ids + drafted ids trimmed appropriately,
    n_predict = k + 1 (target greedy for positions L..L+k).
  * Longest-common-prefix accept rule, plus target's k+1-th token as
    the guaranteed bonus.

State invariant between rounds:
  * committed_ids       : list of token ids committed so far (length L)
  * npu_past            : 511-slot NPU past_kv, slots 0..L-1 real,
                          slots L..510 zero-padded
  * next_candidate      : argmax from the most recent NPU forward —
                          candidate for position L, NOT yet in past,
                          NOT yet committed.

Draft phase keeps k-1 past snapshots so any prefix of the draft ids
(0 through k) can be committed without rerunning those NPU steps.

First-cut scope:
  * Single humaneval prompt (p0 = fibonacci stub, 16 tokens).
  * k = 3 (Phase 2's CPU-spec optimum).
  * n_predict = 64 tokens of generation (enough to measure stable t/s).
  * Greedy only (temperature=0, top_k=1, seed=1).
  * Output: per-round log + summary (total tokens, decode t/s, mean
    accept rate). No CSV yet — this is the plumbing run.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_spec_outer_loop.py
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import CONTEXT_MAX  # noqa: E402
from npu_short_prompt_probe import (  # noqa: E402
    HUMANEVAL,
    load_prompt,
    npu_rearrange_present_to_past,
    npu_single_step_short_prompt,
    pad_cpu_past_to_npu,
)
from npu_spec_step7_plumbing import (  # noqa: E402
    BASE_URL,
    HEALTH_TIMEOUT_S,
    http_post_json,
    spawn_server,
    teardown_server,
    wait_for_health,
)
from npu_vs_cpu_correctness import (  # noqa: E402
    CONFIG_JSON,
    CPU_ONNX,
    TOKENIZER_JSON,
    load_cpu_session,
    load_npu_session,
)
from npu_short_prompt_probe import cpu_prefill  # noqa: E402


# Runtime path selector. Defaults to the fp16 pathbmask baseline; set
# SPECULA_NPU_PATH=pathb to target the rotary-hoisted Path B binary
# (w4a16-ready). Must be consistent with whatever compile --path value
# produced the currently-loaded .bin (see SPECULA_NPU_VARIANT).
PATH_KEY = os.environ.get("SPECULA_NPU_PATH", "pathbmask")


def _safe_repr(s: str) -> str:
    return repr(s.encode("ascii", "backslashreplace").decode("ascii"))


def draft_k_tokens(
    npu_sess,
    npu_past_at_round_start: dict,
    next_candidate: int,
    round_start_committed: int,
    k: int,
    n_layers: int,
) -> tuple[list[int], list[dict]]:
    """Draft k tokens on the NPU, returning k drafts and k past snapshots.

    `past_snapshots[i]` is the NPU past after absorbing drafts[0..i-1]
    (valid_past_len = round_start_committed + i). Index 0 is the input
    past (no drafts absorbed); index k-1 has absorbed drafts[0..k-2]
    and is the deepest rollback the outer loop can do directly.

    The j == k case (all drafts accepted) needs one more snapshot that
    has absorbed drafts[k-1]; the outer loop materialises it lazily via
    `materialize_snapshot_k` so we don't pay that NPU call when j < k.
    """
    L = round_start_committed
    drafts: list[int] = [next_candidate]
    past_snapshots: list[dict] = [npu_past_at_round_start]

    for i in range(k - 1):
        logits, outputs, out_names = npu_single_step_short_prompt(
            npu_sess,
            past_snapshots[-1],
            drafts[i],
            position=L + i,
            valid_past_len=L + i,
            path_key=PATH_KEY,
        )
        past_snapshots.append(npu_rearrange_present_to_past(
            outputs, out_names, n_layers, old_valid_past_len=L + i,
        ))
        drafts.append(int(np.argmax(logits)))

    return drafts, past_snapshots


def materialize_snapshot_k(
    npu_sess,
    snapshot_k_minus_1: dict,
    draft_k_minus_1: int,
    round_start_committed: int,
    k: int,
    n_layers: int,
) -> dict:
    """Produce past_snapshots[k] on demand (only called when j == k).

    Absorbs drafts[k-1] into past_snapshots[k-1] so its valid_past_len
    grows from L+k-1 to L+k. Logits produced by this step are the
    "k+1-th draft" — discarded since the outer loop only uses k drafts.
    """
    pos = round_start_committed + k - 1
    _, outputs, out_names = npu_single_step_short_prompt(
        npu_sess,
        snapshot_k_minus_1,
        draft_k_minus_1,
        position=pos,
        valid_past_len=pos,
        path_key=PATH_KEY,
    )
    return npu_rearrange_present_to_past(
        outputs, out_names, n_layers, old_valid_past_len=pos,
    )


def verify_via_target(committed_ids: list[int], k: int) -> list[int]:
    """POST committed ids to /completion, return target's k+1 greedy tokens."""
    payload = {
        "prompt": committed_ids,
        "n_predict": k + 1,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": True,
        "return_tokens": True,
    }
    resp = http_post_json(f"{BASE_URL}/completion", payload, timeout=180.0)
    tokens = resp.get("tokens") or []
    if len(tokens) < k + 1:
        raise RuntimeError(
            f"/completion returned {len(tokens)} tokens, expected {k + 1}; "
            f"response keys: {list(resp.keys())}"
        )
    return [int(t) for t in tokens[: k + 1]]


def longest_common_prefix(drafts: list[int], target: list[int]) -> int:
    """Number of matching drafts (0..k). Bonus target token is at index j."""
    j = 0
    for d, t in zip(drafts, target):
        if d == t:
            j += 1
        else:
            break
    return j


def absorb_bonus(
    npu_sess,
    past_at_valid_L_plus_j: dict,
    bonus_id: int,
    bonus_position: int,
    n_layers: int,
) -> tuple[int, dict]:
    """Absorb the bonus token into past_kv and produce the next_candidate.

    Input past has valid_past_len = bonus_position (covers 0..bonus_position-1).
    After this call, past covers 0..bonus_position (valid = bonus_position+1)
    and next_candidate is the argmax for position bonus_position+1.
    """
    logits, outputs, out_names = npu_single_step_short_prompt(
        npu_sess,
        past_at_valid_L_plus_j,
        bonus_id,
        position=bonus_position,
        valid_past_len=bonus_position,
        path_key=PATH_KEY,
    )
    next_past = npu_rearrange_present_to_past(
        outputs, out_names, n_layers, old_valid_past_len=bonus_position
    )
    next_candidate = int(np.argmax(logits))
    return next_candidate, next_past


def run_spec_decode(
    cpu_sess,
    npu_sess,
    cfg: dict,
    tok: Tokenizer,
    prompt_ids: list[int],
    k: int,
    n_predict_target: int,
) -> dict:
    """Main spec-decode loop. Returns metrics dict."""
    n_layers = cfg["num_hidden_layers"]

    # 1. CPU prefill: one shot through the base ONNX to initialize past + get
    #    the first draft candidate for position P.
    print(f"\n--- CPU prefill (P={len(prompt_ids)}) ---")
    t0 = time.perf_counter()
    cpu_past, first_candidate = cpu_prefill(cpu_sess, cfg, prompt_ids)
    print(f"  elapsed           : {time.perf_counter() - t0:.2f} s")
    print(f"  first candidate   : {first_candidate}  -> "
          f"{_safe_repr(tok.decode([first_candidate]))}")

    # 2. Convert CPU past -> NPU shape (zero-padded to 511).
    npu_past = pad_cpu_past_to_npu(cpu_past, len(prompt_ids), cfg)

    committed_ids: list[int] = list(prompt_ids)
    next_candidate = first_candidate
    prompt_len = len(prompt_ids)

    round_metrics: list[dict] = []
    draft_wall_s = 0.0
    verify_wall_s = 0.0
    absorb_wall_s = 0.0

    t_gen_start = time.perf_counter()
    round_idx = 0

    while (len(committed_ids) - prompt_len) < n_predict_target:
        round_idx += 1
        L = len(committed_ids)

        # Safety: keep us inside the NPU's compiled context window.
        if L + k + 1 > CONTEXT_MAX - 1:
            print(f"  round {round_idx}: near context limit (L={L}, "
                  f"CONTEXT_MAX-1={CONTEXT_MAX - 1}); stopping")
            break

        # 3. Draft k tokens on NPU.
        t0 = time.perf_counter()
        drafts, past_snapshots = draft_k_tokens(
            npu_sess, npu_past, next_candidate, L, k, n_layers
        )
        draft_wall_s += time.perf_counter() - t0

        # 4. Target verify.
        t0 = time.perf_counter()
        target_ids = verify_via_target(committed_ids, k)
        verify_wall_s += time.perf_counter() - t0

        # 5. Accept-reject.
        j = longest_common_prefix(drafts, target_ids)
        bonus_id = target_ids[j]
        new_commits = drafts[:j] + [bonus_id]

        # 6. Update state: absorb bonus into past, pick up next_candidate.
        #    Lazy: only pay the snapshot-k NPU call when j == k.
        t0 = time.perf_counter()
        if j < k:
            pre_bonus_past = past_snapshots[j]
        else:
            pre_bonus_past = materialize_snapshot_k(
                npu_sess, past_snapshots[k - 1], drafts[k - 1], L, k, n_layers
            )
        next_candidate, npu_past = absorb_bonus(
            npu_sess,
            pre_bonus_past,
            bonus_id,
            bonus_position=L + j,
            n_layers=n_layers,
        )
        absorb_wall_s += time.perf_counter() - t0

        committed_ids.extend(new_commits)

        # Log round.
        accepted_drafts = drafts[:j]
        rejected_draft = drafts[j] if j < k else None
        round_metrics.append({
            "round": round_idx,
            "L_start": L,
            "drafts": drafts,
            "target": target_ids,
            "j": j,
            "bonus": bonus_id,
            "committed_this_round": len(new_commits),
        })
        print(
            f"  r{round_idx:03d} L={L:3d} drafts={drafts} target={target_ids[:k]}"
            f"+b={target_ids[k]} j={j} bonus={bonus_id} "
            f"acc={_safe_repr(tok.decode(accepted_drafts)) if accepted_drafts else '<none>'} "
            f"new={_safe_repr(tok.decode(new_commits))}"
        )

    t_gen_elapsed = time.perf_counter() - t_gen_start
    decoded = len(committed_ids) - prompt_len
    decode_tps = decoded / t_gen_elapsed if t_gen_elapsed > 0 else 0.0

    total_drafted = k * len(round_metrics)
    total_accepted = sum(r["j"] for r in round_metrics)
    mean_accept = total_accepted / total_drafted if total_drafted else 0.0

    summary = {
        "prompt_len": prompt_len,
        "decoded": decoded,
        "rounds": len(round_metrics),
        "k": k,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "mean_accept_rate": mean_accept,
        "wall_generate_s": t_gen_elapsed,
        "wall_draft_s": draft_wall_s,
        "wall_verify_s": verify_wall_s,
        "wall_absorb_s": absorb_wall_s,
        "decode_tps": decode_tps,
        "generated_text": tok.decode(committed_ids[prompt_len:]),
    }
    return summary


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("-k", "--draft-k", type=int, default=3)
    parser.add_argument("-n", "--n-predict", type=int, default=64)
    args = parser.parse_args()

    print(f"=== step 8 - NPU-drafted spec decode (prompt_idx={args.prompt_idx}, "
          f"k={args.draft_k}, n_predict={args.n_predict}) ===\n")

    if not HUMANEVAL.exists():
        print(f"ERROR: {HUMANEVAL} missing")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    prompt = load_prompt(args.prompt_idx)
    prompt_ids = tok.encode(prompt).ids
    print(f"prompt (humaneval p{args.prompt_idx}, {len(prompt_ids)} tokens):")
    print(f"  {_safe_repr(prompt)}")

    # Pre-compute budget check.
    budget = CONTEXT_MAX - 1 - args.draft_k - 1
    if len(prompt_ids) + args.n_predict > budget:
        print(f"WARNING: prompt_len + n_predict = "
              f"{len(prompt_ids) + args.n_predict} exceeds NPU budget "
              f"({budget} incl. per-round k+1 slack); will stop early")

    proc, log_f = spawn_server(REPO_ROOT / "results" / "phase5_step8_llama_server.log")
    try:
        print(f"\n--- waiting for /health (timeout {HEALTH_TIMEOUT_S}s) ---")
        if not wait_for_health(f"{BASE_URL}/health", HEALTH_TIMEOUT_S):
            print("ERROR: llama-server never came healthy")
            return 2
        print("  server healthy")

        print("\n--- loading CPU ONNX (FP32, dynamic past_len) ---")
        t0 = time.perf_counter()
        cpu_sess = load_cpu_session(CPU_ONNX)
        print(f"  loaded in {time.perf_counter() - t0:.1f} s")

        print("\n--- loading NPU Path B-mask binary ---")
        t0 = time.perf_counter()
        npu_sess = load_npu_session(cfg, PATH_KEY)
        print(f"  loaded in {time.perf_counter() - t0:.1f} s")
        providers = npu_sess.get_providers()
        if not providers or providers[0] != "QNNExecutionProvider":
            print(f"ERROR: NPU session fell back: {providers}")
            return 2

        print(f"\n--- spec decode loop (k={args.draft_k}) ---")
        summary = run_spec_decode(
            cpu_sess, npu_sess, cfg, tok, prompt_ids,
            k=args.draft_k, n_predict_target=args.n_predict,
        )

        print("\n=== summary ===")
        print(f"  prompt tokens         : {summary['prompt_len']}")
        print(f"  decoded tokens        : {summary['decoded']}")
        print(f"  rounds                : {summary['rounds']}")
        print(f"  k                     : {summary['k']}")
        print(f"  mean accept rate      : "
              f"{summary['mean_accept_rate'] * 100:.1f}% "
              f"({summary['total_accepted']}/{summary['total_drafted']})")
        print(f"  wall generate         : {summary['wall_generate_s']:.2f} s")
        print(f"    NPU draft total     : {summary['wall_draft_s']:.2f} s")
        print(f"    target verify total : {summary['wall_verify_s']:.2f} s")
        print(f"    NPU absorb total    : {summary['wall_absorb_s']:.2f} s")
        print(f"  decode rate           : "
              f"{summary['decode_tps']:.2f} t/s (reference: phase 2 CPU-spec "
              f"k=3 = 40.2 t/s; CPU baseline TG = 25.91 t/s)")
        print(f"\n  generated text        :\n"
              f"    {_safe_repr(summary['generated_text'])}")
        return 0
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        teardown_server(proc, log_f)


if __name__ == "__main__":
    sys.exit(main())
