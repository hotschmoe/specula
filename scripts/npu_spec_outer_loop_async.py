"""Phase 5.5 Lever A — async NPU-draft ↔ target-verify overlap.

Per-round structure (vs sync baseline in npu_spec_outer_loop.py):

    sync     : [draft k-1 NPU calls] -> [HTTP verify] -> [absorb NPU call]
    async    : [draft || verify]                       -> [absorb NPU call]
               ^-- max(draft_ms, verify_ms)

Design decisions confirmed by scripts/npu_gil_probe.py:

  * sess.run() releases the GIL (probe 2 overlap score 0.98), so a
    ThreadPoolExecutor is sufficient — no subprocess split.
  * HTP serializes concurrent sessions on one queue (probe 3 score -0.07),
    so no point in two-session parallelism. Use a single session.

Single worker pins NPU work to one thread. Main thread issues the
/completion HTTP call (which also releases GIL on socket.recv), so the
two phases overlap naturally.

Optional R2 (draft-p-min early-exit): if `--p-min > 0.0`, stop drafting
mid-round when the draft's argmax softmax probability falls below
threshold. Saves NPU calls on low-confidence streaks (e.g. `flatten` p5).

Run:
    .venv\\Scripts\\python.exe scripts\\npu_spec_outer_loop_async.py \\
        --prompt-idx 0 -k 2 -n 64
    .venv\\Scripts\\python.exe scripts\\npu_spec_outer_loop_async.py \\
        --prompt-idx 0 -k 2 -n 64 --p-min 0.80
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import CONTEXT_MAX  # noqa: E402
from npu_short_prompt_probe import (  # noqa: E402
    HUMANEVAL,
    cpu_prefill,
    load_prompt,
    npu_rearrange_present_to_past,
    npu_single_step_short_prompt,
    pad_cpu_past_to_npu,
)
from npu_spec_step7_plumbing import (  # noqa: E402
    BASE_URL,
    HEALTH_TIMEOUT_S,
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
from npu_spec_outer_loop import (  # noqa: E402
    PATH_KEY,
    _safe_repr,
    absorb_bonus,
    longest_common_prefix,
    materialize_snapshot_k,
    verify_via_target,
)


def argmax_and_max_prob(logits: np.ndarray) -> tuple[int, float]:
    """Numerically stable: return (argmax_id, softmax_prob_at_argmax)."""
    flat = logits.astype(np.float32).flatten()
    idx = int(np.argmax(flat))
    shifted = flat - flat.max()
    e = np.exp(shifted)
    return idx, float(e[idx] / e.sum())


def draft_k_tokens_pmin(
    npu_sess,
    npu_past_at_round_start: dict,
    next_candidate: int,
    round_start_committed: int,
    k: int,
    n_layers: int,
    p_min: float = 0.0,
) -> tuple[list[int], list[dict]]:
    """Like draft_k_tokens() but optionally early-exits when argmax prob drops
    below p_min. Returns (drafts, past_snapshots) where len(drafts) <= k.

    past_snapshots[i] is the past BEFORE absorbing drafts[i] (valid_past_len
    = round_start_committed + i). len(past_snapshots) == len(drafts).
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
        )
        past_snapshots.append(
            npu_rearrange_present_to_past(
                outputs, out_names, n_layers, old_valid_past_len=L + i,
            )
        )
        if p_min > 0.0:
            next_id, p_max = argmax_and_max_prob(logits)
            drafts.append(next_id)
            if p_max < p_min:
                break
        else:
            drafts.append(int(np.argmax(logits)))

    return drafts, past_snapshots


def run_spec_decode_async(
    cpu_sess,
    npu_sess,
    cfg: dict,
    tok: Tokenizer,
    prompt_ids: list[int],
    k: int,
    n_predict_target: int,
    p_min: float = 0.0,
) -> dict:
    """Async variant of run_spec_decode: overlaps draft phase with verify."""
    n_layers = cfg["num_hidden_layers"]

    print(f"\n--- CPU prefill (P={len(prompt_ids)}) ---")
    t0 = time.perf_counter()
    cpu_past, first_candidate = cpu_prefill(cpu_sess, cfg, prompt_ids)
    print(f"  elapsed           : {time.perf_counter() - t0:.2f} s")
    print(f"  first candidate   : {first_candidate}  -> "
          f"{_safe_repr(tok.decode([first_candidate]))}")

    npu_past = pad_cpu_past_to_npu(cpu_past, len(prompt_ids), cfg)
    committed_ids: list[int] = list(prompt_ids)
    next_candidate = first_candidate
    prompt_len = len(prompt_ids)

    round_metrics: list[dict] = []
    # Wait-time accounting: time from f.result() call to return. If draft
    # finished first, draft_wait_s is ~0; the "real" parallel window shows
    # up in verify_wait_s instead. Either way, parallel_wall_s is the
    # ground truth per-round overlap wall (max of the two phases, as
    # observed from main thread).
    draft_wait_s = 0.0
    verify_wait_s = 0.0
    absorb_wall_s = 0.0
    parallel_wall_s = 0.0
    early_exits = 0

    # Single-worker executor pins NPU work to one thread; main thread runs
    # HTTP verify in parallel. Both release the GIL during their blocking
    # sections so the overlap is real.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npu-draft")

    t_gen_start = time.perf_counter()
    round_idx = 0
    try:
        while (len(committed_ids) - prompt_len) < n_predict_target:
            round_idx += 1
            L = len(committed_ids)

            if L + k + 1 > CONTEXT_MAX - 1:
                print(f"  round {round_idx}: near context limit (L={L}, "
                      f"CONTEXT_MAX-1={CONTEXT_MAX - 1}); stopping")
                break

            t_round_start = time.perf_counter()

            # Fire draft phase on the executor; run verify on main thread
            # concurrently.
            f_drafts = executor.submit(
                draft_k_tokens_pmin,
                npu_sess, npu_past, next_candidate, L, k, n_layers, p_min,
            )

            t_v0 = time.perf_counter()
            target_ids = verify_via_target(committed_ids, k)
            verify_wait_s += time.perf_counter() - t_v0

            t_d0 = time.perf_counter()
            drafts, past_snapshots = f_drafts.result()
            draft_wait_s += time.perf_counter() - t_d0

            parallel_wall_s += time.perf_counter() - t_round_start

            actual_k = len(drafts)
            if actual_k < k:
                early_exits += 1

            # Longest-common-prefix over however many drafts we actually
            # produced. Target provided k+1 tokens regardless of early
            # exit; extras just go unused.
            j = longest_common_prefix(drafts, target_ids)
            bonus_id = target_ids[j]
            new_commits = drafts[:j] + [bonus_id]

            # Absorb bonus. Sequential — depends on j.
            t_a0 = time.perf_counter()
            if j < actual_k:
                pre_bonus_past = past_snapshots[j]
            else:
                # j == actual_k: need to absorb drafts[actual_k - 1]
                # beyond past_snapshots[actual_k - 1]. materialize_snapshot_k
                # does exactly this; pass actual_k as the "k" parameter.
                pre_bonus_past = materialize_snapshot_k(
                    npu_sess,
                    past_snapshots[actual_k - 1],
                    drafts[actual_k - 1],
                    L,
                    actual_k,
                    n_layers,
                )
            next_candidate, npu_past = absorb_bonus(
                npu_sess,
                pre_bonus_past,
                bonus_id,
                bonus_position=L + j,
                n_layers=n_layers,
            )
            absorb_wall_s += time.perf_counter() - t_a0

            committed_ids.extend(new_commits)

            accepted_drafts = drafts[:j]
            round_metrics.append({
                "round": round_idx,
                "L_start": L,
                "drafts": drafts,
                "target": target_ids,
                "j": j,
                "bonus": bonus_id,
                "committed_this_round": len(new_commits),
                "early_exit": actual_k < k,
            })
            ee_mark = "*" if actual_k < k else " "
            print(
                f"  r{round_idx:03d} L={L:3d} drafts={drafts}{ee_mark} "
                f"target={target_ids[:k]}+b={target_ids[k]} j={j} bonus={bonus_id} "
                f"acc={_safe_repr(tok.decode(accepted_drafts)) if accepted_drafts else '<none>'} "
                f"new={_safe_repr(tok.decode(new_commits))}"
            )
    finally:
        executor.shutdown(wait=True)

    t_gen_elapsed = time.perf_counter() - t_gen_start
    decoded = len(committed_ids) - prompt_len
    decode_tps = decoded / t_gen_elapsed if t_gen_elapsed > 0 else 0.0

    total_drafted = sum(len(r["drafts"]) for r in round_metrics)
    total_accepted = sum(r["j"] for r in round_metrics)
    mean_accept = total_accepted / total_drafted if total_drafted else 0.0

    summary = {
        "mode": "async",
        "prompt_len": prompt_len,
        "decoded": decoded,
        "rounds": len(round_metrics),
        "k": k,
        "p_min": p_min,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "mean_accept_rate": mean_accept,
        "early_exits": early_exits,
        "wall_generate_s": t_gen_elapsed,
        "wall_draft_wait_s": draft_wait_s,
        "wall_verify_wait_s": verify_wait_s,
        "wall_parallel_s": parallel_wall_s,
        "wall_absorb_s": absorb_wall_s,
        "decode_tps": decode_tps,
        "generated_text": tok.decode(committed_ids[prompt_len:]),
    }
    return summary


def run_spec_decode_async_pipelined(
    cpu_sess,
    npu_sess,
    cfg: dict,
    tok: Tokenizer,
    prompt_ids: list[int],
    k: int,
    n_predict_target: int,
    p_min: float = 0.0,
) -> dict:
    """Design (2) — pipelined variant of design (1).

    On top of design (1)'s draft || verify overlap, pre-issues round
    N+1's HTTP /completion during round N's absorb. committed_ids_{N+1}
    is known the moment round N's accept check completes, and HTTP is
    deterministic (no rollback logic needed). This folds absorb and
    next-round verify into a single wall-clock max(), so steady-state
    per-round cost drops from max(draft, verify) + absorb to
    ~draft + max(absorb, verify).

    Two executors:
      * npu_pool (1 worker) pins NPU work to one thread
      * http_pool (1 worker) lets HTTP run concurrent with NPU
    """
    n_layers = cfg["num_hidden_layers"]

    print(f"\n--- CPU prefill (P={len(prompt_ids)}) ---")
    t0 = time.perf_counter()
    cpu_past, first_candidate = cpu_prefill(cpu_sess, cfg, prompt_ids)
    print(f"  elapsed           : {time.perf_counter() - t0:.2f} s")
    print(f"  first candidate   : {first_candidate}  -> "
          f"{_safe_repr(tok.decode([first_candidate]))}")

    npu_past = pad_cpu_past_to_npu(cpu_past, len(prompt_ids), cfg)
    committed_ids: list[int] = list(prompt_ids)
    next_candidate = first_candidate
    prompt_len = len(prompt_ids)

    round_metrics: list[dict] = []
    draft_wait_s = 0.0
    verify_wait_s = 0.0
    absorb_wall_s = 0.0
    parallel_wall_s = 0.0
    early_exits = 0
    # Fraction of rounds where pending_verify was already done when we
    # awaited it — this is the pipeline "cache hit" rate.
    prewarmed_verify_count = 0

    npu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npu")
    http_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="http")

    t_gen_start = time.perf_counter()
    round_idx = 0

    # Pre-issue the FIRST verify HTTP before any round starts. This
    # bootstraps the pipeline so round 1 already benefits from the
    # overlap.
    pending_verify = http_pool.submit(verify_via_target, list(committed_ids), k)

    try:
        while (len(committed_ids) - prompt_len) < n_predict_target:
            round_idx += 1
            L = len(committed_ids)

            if L + k + 1 > CONTEXT_MAX - 1:
                print(f"  round {round_idx}: near context limit (L={L}, "
                      f"CONTEXT_MAX-1={CONTEXT_MAX - 1}); stopping")
                break

            t_round_start = time.perf_counter()

            # Fire round-N draft. Main thread awaits pending verify.
            f_drafts = npu_pool.submit(
                draft_k_tokens_pmin,
                npu_sess, npu_past, next_candidate, L, k, n_layers, p_min,
            )

            t_v0 = time.perf_counter()
            if pending_verify.done():
                prewarmed_verify_count += 1
            target_ids = pending_verify.result()
            verify_wait_s += time.perf_counter() - t_v0

            t_d0 = time.perf_counter()
            drafts, past_snapshots = f_drafts.result()
            draft_wait_s += time.perf_counter() - t_d0

            parallel_wall_s += time.perf_counter() - t_round_start

            actual_k = len(drafts)
            if actual_k < k:
                early_exits += 1

            j = longest_common_prefix(drafts, target_ids)
            bonus_id = target_ids[j]
            new_commits = drafts[:j] + [bonus_id]
            committed_ids_next = committed_ids + new_commits

            # Pre-issue the NEXT round's verify concurrent with absorb.
            # Skip if we're about to exit the loop (no need to pay HTTP
            # for a round we won't run).
            will_continue = (len(committed_ids_next) - prompt_len) < n_predict_target \
                and (L + len(new_commits) + k + 1) <= (CONTEXT_MAX - 1)
            if will_continue:
                pending_verify = http_pool.submit(
                    verify_via_target, list(committed_ids_next), k,
                )
            else:
                pending_verify = None

            # Absorb on npu_pool. Tested both routes: submitting to npu_pool
            # beat main-thread-direct by ~5% empirically (11.14 vs 10.53 t/s
            # at k=2 sweep). Counterintuitive but consistent — keeping NPU
            # work on one dedicated thread seems to reduce GIL thrash with
            # the http_pool's verify call. Main thread awaits via future
            # (GIL released during wait).
            t_a0 = time.perf_counter()
            if j < actual_k:
                pre_bonus_past = past_snapshots[j]
            else:
                pre_bonus_past = npu_pool.submit(
                    materialize_snapshot_k,
                    npu_sess, past_snapshots[actual_k - 1],
                    drafts[actual_k - 1], L, actual_k, n_layers,
                ).result()
            f_absorb = npu_pool.submit(
                absorb_bonus,
                npu_sess, pre_bonus_past, bonus_id,
                L + j, n_layers,
            )
            next_candidate, npu_past = f_absorb.result()
            absorb_wall_s += time.perf_counter() - t_a0

            committed_ids = committed_ids_next

            accepted_drafts = drafts[:j]
            round_metrics.append({
                "round": round_idx,
                "L_start": L,
                "drafts": drafts,
                "target": target_ids,
                "j": j,
                "bonus": bonus_id,
                "committed_this_round": len(new_commits),
                "early_exit": actual_k < k,
            })
            ee_mark = "*" if actual_k < k else " "
            print(
                f"  r{round_idx:03d} L={L:3d} drafts={drafts}{ee_mark} "
                f"target={target_ids[:k]}+b={target_ids[k]} j={j} bonus={bonus_id} "
                f"acc={_safe_repr(tok.decode(accepted_drafts)) if accepted_drafts else '<none>'} "
                f"new={_safe_repr(tok.decode(new_commits))}"
            )
    finally:
        # Flush any in-flight verify so the HTTP pool can shut down cleanly.
        if pending_verify is not None and not pending_verify.done():
            try:
                pending_verify.result(timeout=30.0)
            except Exception:
                pass
        http_pool.shutdown(wait=True)
        npu_pool.shutdown(wait=True)

    t_gen_elapsed = time.perf_counter() - t_gen_start
    decoded = len(committed_ids) - prompt_len
    decode_tps = decoded / t_gen_elapsed if t_gen_elapsed > 0 else 0.0

    total_drafted = sum(len(r["drafts"]) for r in round_metrics)
    total_accepted = sum(r["j"] for r in round_metrics)
    mean_accept = total_accepted / total_drafted if total_drafted else 0.0

    summary = {
        "mode": "async-pipelined",
        "prompt_len": prompt_len,
        "decoded": decoded,
        "rounds": len(round_metrics),
        "k": k,
        "p_min": p_min,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "mean_accept_rate": mean_accept,
        "early_exits": early_exits,
        "prewarmed_verify_rounds": prewarmed_verify_count,
        "wall_generate_s": t_gen_elapsed,
        "wall_draft_wait_s": draft_wait_s,
        "wall_verify_wait_s": verify_wait_s,
        "wall_parallel_s": parallel_wall_s,
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
    parser.add_argument("-k", "--draft-k", type=int, default=2)
    parser.add_argument("-n", "--n-predict", type=int, default=64)
    parser.add_argument("--p-min", type=float, default=0.0,
                        help="R2 early-exit threshold on draft argmax prob (0 disables)")
    parser.add_argument("--pipelined", action="store_true",
                        help="Use design (2) pipelined variant (pre-issues next-round verify during absorb)")
    args = parser.parse_args()

    print(f"=== Phase 5.5 Lever A — async NPU-spec "
          f"(prompt_idx={args.prompt_idx}, k={args.draft_k}, "
          f"n_predict={args.n_predict}, p_min={args.p_min}) ===\n")

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

    budget = CONTEXT_MAX - 1 - args.draft_k - 1
    if len(prompt_ids) + args.n_predict > budget:
        print(f"WARNING: prompt_len + n_predict = "
              f"{len(prompt_ids) + args.n_predict} exceeds NPU budget "
              f"({budget} incl. per-round k+1 slack); will stop early")

    proc, log_f = spawn_server(REPO_ROOT / "results" / "phase5_5_async_llama_server.log")
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

        mode_label = "async-pipelined (design 2)" if args.pipelined else "async (design 1)"
        print(f"\n--- {mode_label} spec decode loop (k={args.draft_k}, "
              f"p_min={args.p_min}) ---")
        runner = run_spec_decode_async_pipelined if args.pipelined else run_spec_decode_async
        summary = runner(
            cpu_sess, npu_sess, cfg, tok, prompt_ids,
            k=args.draft_k, n_predict_target=args.n_predict,
            p_min=args.p_min,
        )

        print("\n=== summary ===")
        print(f"  mode                  : {summary.get('mode', 'async')}")
        print(f"  prompt tokens         : {summary['prompt_len']}")
        print(f"  decoded tokens        : {summary['decoded']}")
        print(f"  rounds                : {summary['rounds']}")
        print(f"  k / p_min             : {summary['k']} / {summary['p_min']}")
        print(f"  early exits (R2)      : {summary['early_exits']} / {summary['rounds']}")
        if "prewarmed_verify_rounds" in summary:
            print(f"  prewarmed verify hits : "
                  f"{summary['prewarmed_verify_rounds']} / {summary['rounds']} "
                  f"(higher = pipeline overlap working)")
        print(f"  mean accept rate      : "
              f"{summary['mean_accept_rate'] * 100:.1f}% "
              f"({summary['total_accepted']}/{summary['total_drafted']})")
        print(f"  wall generate         : {summary['wall_generate_s']:.2f} s")
        print(f"    parallel (draft||v) : {summary['wall_parallel_s']:.2f} s")
        print(f"    of which draft wait : {summary['wall_draft_wait_s']:.2f} s")
        print(f"    of which verify     : {summary['wall_verify_wait_s']:.2f} s")
        print(f"    NPU absorb total    : {summary['wall_absorb_s']:.2f} s")
        print(f"  decode rate           : "
              f"{summary['decode_tps']:.2f} t/s (baseline phase 5 k=2 = 7.98 t/s; "
              f"phase 2 CPU-spec k=3 = 40.2 t/s)")
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
