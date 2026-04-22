"""Phase 5 step 9 — sweep NPU-drafted spec decode over (prompts x k).

One long-lived process does:
  * spawns ONE llama-server (Qwen3-8B-Q4_K_M, CPU, greedy)
  * loads ONE CPU ORT session (Qwen3-0.6B optimum ONNX)
  * loads ONE NPU QNN session (Path B-mask binary)
  * iterates (prompt_idx, k) cells, running run_spec_decode per cell
  * writes results/spec-npu-humaneval-{timestamp}.csv with a schema
    matching results/spec-cpu-*.csv from Phase 2 so comparison with the
    CPU-spec baseline is a 1:1 column lookup.

Between cells: no server restart, no session reload. Target's
cache_prompt is per-slot and would fight us across prompt switches
if we aren't careful; we set cache_prompt=False on each /completion
call to avoid poisoning (small prompts anyway).

Budget: 10 prompts x 4 k values x ~40 s/cell ≈ 25-35 min.
Pure-run budget: ~20 min if each prompt's NPU draft is close to our
p0 measurement (65 tokens generated in 10 s -> 256 tokens ≈ 40 s).

Run:
    .venv\\Scripts\\python.exe scripts\\sweep_npu_spec.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import json
import sys
import time
import traceback
from pathlib import Path

from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import CONTEXT_MAX  # noqa: E402
from npu_short_prompt_probe import HUMANEVAL, load_prompt  # noqa: E402
from npu_spec_outer_loop import PATH_KEY, run_spec_decode  # noqa: E402
from npu_spec_outer_loop_async import (  # noqa: E402
    run_spec_decode_async,
    run_spec_decode_async_pipelined,
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


CSV_FIELDS = [
    "timestamp",
    "target",
    "draft",
    "backend",
    "threads",
    "draft_max",
    "n_predict_req",
    "prompt_idx",
    "n_drafted",
    "n_accept",
    "accept_pct",
    "decoded_tokens",
    "decoded_seconds",
    "decoded_tok_per_sec",
    "encoded_tok_per_sec",
    "wallclock_s",
]


def _safe_repr(s: str, max_len: int = 50) -> str:
    r = repr(s.encode("ascii", "backslashreplace").decode("ascii"))
    return r if len(r) <= max_len else r[: max_len - 3] + "...'"


def _ts() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="microseconds")


def run_sweep(
    cpu_sess,
    npu_sess,
    cfg: dict,
    tok: Tokenizer,
    prompt_indices: list[int],
    k_values: list[int],
    n_predict: int,
    csv_path: Path,
    mode: str = "sync",
    p_min: float = 0.0,
) -> list[dict]:
    """Run the full (prompt x k) matrix, emitting one CSV row per cell."""
    rows: list[dict] = []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()

        total_cells = len(prompt_indices) * len(k_values)
        cell_idx = 0
        sweep_t0 = time.perf_counter()

        for prompt_idx in prompt_indices:
            prompt = load_prompt(prompt_idx)
            prompt_ids = tok.encode(prompt).ids
            print(f"\n==== prompt p{prompt_idx} ({len(prompt_ids)} toks): "
                  f"{_safe_repr(prompt)} ====")

            for k in k_values:
                cell_idx += 1
                cell_t0 = time.perf_counter()
                print(f"\n-- cell {cell_idx}/{total_cells}: "
                      f"p{prompt_idx} k={k} n_predict={n_predict} --")

                budget = CONTEXT_MAX - 1 - k - 1
                if len(prompt_ids) + n_predict > budget:
                    print(f"   SKIP: prompt_len + n_predict "
                          f"({len(prompt_ids) + n_predict}) > budget {budget}")
                    continue

                try:
                    if mode == "async":
                        summary = run_spec_decode_async(
                            cpu_sess, npu_sess, cfg, tok, prompt_ids,
                            k=k, n_predict_target=n_predict, p_min=p_min,
                        )
                    elif mode == "async-pipelined":
                        summary = run_spec_decode_async_pipelined(
                            cpu_sess, npu_sess, cfg, tok, prompt_ids,
                            k=k, n_predict_target=n_predict, p_min=p_min,
                        )
                    else:
                        summary = run_spec_decode(
                            cpu_sess, npu_sess, cfg, tok, prompt_ids,
                            k=k, n_predict_target=n_predict,
                        )
                except Exception:
                    traceback.print_exc()
                    print("   cell FAILED - continuing sweep")
                    continue

                row = {
                    "timestamp": _ts(),
                    "target": "Qwen3-8B-Q4_K_M.gguf",
                    "draft": "qwen3_0_6b_draft_v81_ctx512.pathbmask.bin",
                    "backend": "npu-draft+cpu-target",
                    "threads": 18,
                    "draft_max": k,
                    "n_predict_req": n_predict,
                    "prompt_idx": prompt_idx,
                    "n_drafted": summary["total_drafted"],
                    "n_accept": summary["total_accepted"],
                    "accept_pct": round(
                        100.0 * summary["mean_accept_rate"], 3
                    ),
                    "decoded_tokens": summary["decoded"],
                    "decoded_seconds": round(summary["wall_generate_s"], 3),
                    "decoded_tok_per_sec": round(summary["decode_tps"], 3),
                    "encoded_tok_per_sec": 0.0,  # not measured; prompt prefill is tiny
                    "wallclock_s": round(
                        time.perf_counter() - cell_t0, 2
                    ),
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(f"   -> accept {row['accept_pct']:.1f}% "
                      f"decode {row['decoded_tok_per_sec']:.2f} t/s  "
                      f"wall {row['wallclock_s']:.1f}s")

        sweep_elapsed = time.perf_counter() - sweep_t0
        print(f"\n==== sweep complete: {len(rows)}/{total_cells} cells in "
              f"{sweep_elapsed / 60:.1f} min ====")

    return rows


def print_summary_table(rows: list[dict]) -> None:
    """Aggregate rows into the Phase-2-style winner table per k."""
    if not rows:
        print("\n(no rows, nothing to summarise)")
        return
    from collections import defaultdict
    per_k: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        per_k[r["draft_max"]].append(r)

    print("\n--- aggregate (per k, across all prompts) ---")
    print(f"{'k':>3}  {'n_cells':>7}  {'mean_accept':>11}  "
          f"{'mean_decode_tps':>15}  {'best_decode_tps':>15}  "
          f"{'worst_decode_tps':>16}")
    for k in sorted(per_k.keys()):
        cells = per_k[k]
        mean_accept = sum(c["accept_pct"] for c in cells) / len(cells)
        decode_tps = [c["decoded_tok_per_sec"] for c in cells]
        print(f"{k:>3}  {len(cells):>7}  {mean_accept:>10.2f}%  "
              f"{sum(decode_tps) / len(decode_tps):>15.2f}  "
              f"{max(decode_tps):>15.2f}  {min(decode_tps):>16.2f}")

    global_best = max(rows, key=lambda r: r["decoded_tok_per_sec"])
    global_worst = min(rows, key=lambda r: r["decoded_tok_per_sec"])
    print(f"\n  best  cell : p{global_best['prompt_idx']} k={global_best['draft_max']} "
          f"-> {global_best['decoded_tok_per_sec']:.2f} t/s "
          f"(accept {global_best['accept_pct']:.1f}%)")
    print(f"  worst cell : p{global_worst['prompt_idx']} k={global_worst['draft_max']} "
          f"-> {global_worst['decoded_tok_per_sec']:.2f} t/s "
          f"(accept {global_worst['accept_pct']:.1f}%)")


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, nargs="*", default=list(range(10)),
                        help="humaneval prompt indices (default: 0..9)")
    parser.add_argument("--k-values", type=int, nargs="*", default=[2, 3, 4, 8],
                        help="draft_max values (default: 2 3 4 8)")
    parser.add_argument("-n", "--n-predict", type=int, default=256,
                        help="tokens to generate per cell (default: 256 for Phase-2 parity)")
    parser.add_argument("--mode", choices=("sync", "async", "async-pipelined"), default="sync",
                        help="outer-loop mode: sync (baseline), async (Lever A design 1), "
                             "async-pipelined (Lever A design 2)")
    parser.add_argument("--p-min", type=float, default=0.0,
                        help="R2 early-exit threshold (async only; 0 disables)")
    args = parser.parse_args()

    print("=== NPU-spec sweep ===")
    print(f"  prompts    : {args.prompts}")
    print(f"  k-values   : {args.k_values}")
    print(f"  n_predict  : {args.n_predict}")
    print(f"  mode       : {args.mode}  p_min={args.p_min}")

    if not HUMANEVAL.exists():
        print(f"ERROR: {HUMANEVAL} missing")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_tag = args.mode if args.p_min == 0.0 else f"{args.mode}-pmin{args.p_min:g}"
    csv_path = (
        REPO_ROOT / "results"
        / f"spec-npu-Qwen3-8B-Q4_K_M-vs-Qwen3-0.6B-pathbmask-{mode_tag}-{ts}.csv"
    )
    print(f"  csv out    : {csv_path}")

    proc, log_f = spawn_server(
        REPO_ROOT / "results" / f"spec-npu-server-{ts}.log"
    )
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

        rows = run_sweep(
            cpu_sess, npu_sess, cfg, tok,
            prompt_indices=args.prompts, k_values=args.k_values,
            n_predict=args.n_predict, csv_path=csv_path,
            mode=args.mode, p_min=args.p_min,
        )
        print_summary_table(rows)
        return 0 if rows else 1
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        teardown_server(proc, log_f)


if __name__ == "__main__":
    sys.exit(main())
