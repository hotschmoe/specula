"""Phase 5.5 Lever A prerequisite — does ORT-QNN session.run() release the GIL?

Lever A (async NPU-draft ↔ target-verify overlap) only works if Python can
run a thread during sess.run(). Three probes, fastest to slowest to
interpret:

1. **Busy-counter probe.** One thread spins `x += 1` for the duration of a
   sess.run() on another thread. If the counter advances meaningfully
   during the NPU call, the GIL is released for at least part of the call
   window. Baseline: same counter loop run alone for the same wall time.
   Ratio `threaded_counter_advance / solo_counter_advance` tells us the
   GIL-release fraction directly.

2. **Sleep-overlap probe (the actual use case).** time.sleep(T) + sess.run()
   scheduled concurrently. If both release the GIL (time.sleep always does),
   wall should be max(T, ort_ms). If sess.run() holds the GIL, wall ≈
   T + ort_ms. T chosen to match CPU target verify per round (~157 ms).

3. **Two-session overlap probe.** Two loaded sessions, each `sess.run()` in
   its own thread. Probes whether NPU hardware itself can overlap two
   draft calls, independent of GIL. This is secondary — Lever A only
   needs probe 2, but probe 3 tells us whether a future R2-style
   tree-verify could push more NPU work into a single round.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_gil_probe.py
"""

from __future__ import annotations

import functools
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import (  # noqa: E402
    CONTEXT_MAX,
    _bin_path,
    _config_json,
    _wrapper_path,
    build_ep_context_wrapper,
    build_zero_feed,
    load_wrapper,
)

import json


PATH_KEY = "patha"
N_WARMUP = 3
N_TIMED = 5

# Approximate target-verify wall per round at k=3, n_predict=4, from the
# Phase 5 baseline breakdown (157 ms target / round).
SIMULATED_VERIFY_MS = 157


def load_session(path_key: str):
    """Reuse the step-5 load path exactly."""
    bin_p = _bin_path(path_key)
    wrap = _wrapper_path(path_key)
    cfg_json = _config_json(path_key)
    if not bin_p.exists():
        raise FileNotFoundError(f"missing {bin_p} — run earlier phase-5 steps first")
    if not wrap.exists():
        with cfg_json.open() as f:
            cfg = json.load(f)
        build_ep_context_wrapper(cfg, bin_p, wrap, path_key)
    return load_wrapper(wrap)


def single_step_latency(sess, feed, n: int) -> float:
    """Median ms per sess.run() over n calls."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(np.median(samples))


def probe1_busy_counter(sess, feed, per_call_ms: float) -> dict:
    """A Python-side busy loop + a sess.run() in parallel.

    If the GIL stays locked throughout sess.run(), the counter thread
    cannot advance while the NPU call is active. If the GIL releases,
    the counter advances at ~solo-rate for whatever fraction of the NPU
    wall is spent in GIL-free native code.
    """
    print("\n--- probe 1: busy-counter (GIL-release fraction) ---")

    # 1a. Solo rate — pure-Python throughput over the same window.
    window_s = per_call_ms / 1000.0
    stop = threading.Event()
    counter = [0]

    def spin():
        local = 0
        while not stop.is_set():
            local += 1
        counter[0] = local

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(window_s)
    stop.set()
    t.join()
    solo_count = counter[0]
    print(f"  solo counter (window={per_call_ms:.1f} ms)  : {solo_count:>12d}")

    # 1b. Concurrent with sess.run(). Reset counter state.
    stop = threading.Event()
    counter = [0]

    def spin2():
        local = 0
        while not stop.is_set():
            local += 1
        counter[0] = local

    t = threading.Thread(target=spin2, daemon=True)
    t.start()
    t0 = time.perf_counter_ns()
    sess.run(None, feed)
    t1 = time.perf_counter_ns()
    stop.set()
    t.join()
    concurrent_count = counter[0]
    npu_ms = (t1 - t0) / 1e6
    print(f"  concurrent counter during NPU             : {concurrent_count:>12d}")
    print(f"  NPU wall this call                        : {npu_ms:.2f} ms")
    # Rescale solo to the actual observed NPU wall (the solo window was
    # the median estimate; the concurrent call may have been longer).
    solo_scaled = int(solo_count * (npu_ms / per_call_ms))
    gil_release_frac = concurrent_count / solo_scaled if solo_scaled > 0 else 0.0
    print(f"  solo counter scaled to this wall          : {solo_scaled:>12d}")
    print(f"  GIL-release fraction (concurrent / solo)  : {gil_release_frac:.3f}")
    return {
        "per_call_ms": per_call_ms,
        "solo_count": solo_count,
        "concurrent_count": concurrent_count,
        "npu_ms": npu_ms,
        "solo_scaled": solo_scaled,
        "gil_release_frac": gil_release_frac,
    }


def probe2_sleep_overlap(sess, feed, per_call_ms: float, verify_ms: int = SIMULATED_VERIFY_MS) -> dict:
    """The actual Lever A use case shaped: time.sleep(verify) || sess.run(draft).

    time.sleep is guaranteed to release the GIL. If sess.run() also releases,
    wall ≈ max(verify_ms, per_call_ms). If not, wall ≈ verify_ms + per_call_ms.
    """
    print(f"\n--- probe 2: sleep({verify_ms} ms) || sess.run() ---")

    # Serial baseline for reference.
    t0 = time.perf_counter_ns()
    time.sleep(verify_ms / 1000.0)
    sess.run(None, feed)
    serial_ms = (time.perf_counter_ns() - t0) / 1e6

    def sleeper():
        time.sleep(verify_ms / 1000.0)

    def npu_caller():
        sess.run(None, feed)

    t0 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_sleep = ex.submit(sleeper)
        f_npu = ex.submit(npu_caller)
        f_sleep.result()
        f_npu.result()
    parallel_ms = (time.perf_counter_ns() - t0) / 1e6

    expected_max = max(verify_ms, per_call_ms)
    expected_sum = verify_ms + per_call_ms
    overlap_score = (expected_sum - parallel_ms) / max(1.0, (expected_sum - expected_max))
    print(f"  per-call NPU ms (median)                  : {per_call_ms:.2f}")
    print(f"  serial (sleep + npu)                      : {serial_ms:.2f} ms")
    print(f"  parallel (sleep || npu)                   : {parallel_ms:.2f} ms")
    print(f"  expected if fully overlapped (max)        : {expected_max:.2f} ms")
    print(f"  expected if no overlap (sum)              : {expected_sum:.2f} ms")
    print(f"  overlap score (1.0=full, 0.0=none)        : {overlap_score:.3f}")
    return {
        "per_call_ms": per_call_ms,
        "verify_ms": verify_ms,
        "serial_ms": serial_ms,
        "parallel_ms": parallel_ms,
        "expected_max": expected_max,
        "expected_sum": expected_sum,
        "overlap_score": overlap_score,
    }


def probe3_two_sessions(path_key: str, per_call_ms: float) -> dict:
    """Two separate sessions in two threads. Does the HTP NPU overlap at all?"""
    print("\n--- probe 3: two sessions, two threads, same NPU ---")
    print("  loading session A ...")
    sess_a = load_session(path_key)
    feed_a = build_zero_feed(sess_a)
    print("  loading session B ...")
    sess_b = load_session(path_key)
    feed_b = build_zero_feed(sess_b)

    # Warm each.
    for _ in range(2):
        sess_a.run(None, feed_a)
        sess_b.run(None, feed_b)

    def run_a():
        sess_a.run(None, feed_a)

    def run_b():
        sess_b.run(None, feed_b)

    # Serial baseline: 2 consecutive calls.
    t0 = time.perf_counter_ns()
    sess_a.run(None, feed_a)
    sess_b.run(None, feed_b)
    serial_ms = (time.perf_counter_ns() - t0) / 1e6

    # Parallel.
    t0 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(run_a)
        fb = ex.submit(run_b)
        fa.result()
        fb.result()
    parallel_ms = (time.perf_counter_ns() - t0) / 1e6

    overlap_score = (serial_ms - parallel_ms) / max(1.0, (serial_ms - per_call_ms))
    print(f"  serial (A then B)                         : {serial_ms:.2f} ms")
    print(f"  parallel (A || B)                         : {parallel_ms:.2f} ms")
    print(f"  if NPU fully overlaps two sessions        : ~{per_call_ms:.2f} ms")
    print(f"  if NPU serializes on one queue            : ~{serial_ms:.2f} ms")
    print(f"  overlap score (1.0=full, 0.0=none)        : {overlap_score:.3f}")
    return {
        "serial_ms": serial_ms,
        "parallel_ms": parallel_ms,
        "per_call_ms": per_call_ms,
        "overlap_score": overlap_score,
    }


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    print(f"=== NPU GIL-release probe (path={PATH_KEY}) ===\n")
    print("--- load wrapper ---")
    sess = load_session(PATH_KEY)
    feed = build_zero_feed(sess)

    print(f"\n--- warm + measure per-call NPU latency ({N_WARMUP} warmup, {N_TIMED} timed) ---")
    for _ in range(N_WARMUP):
        sess.run(None, feed)
    per_call_ms = single_step_latency(sess, feed, N_TIMED)
    print(f"  median per-call                           : {per_call_ms:.2f} ms")

    r1 = probe1_busy_counter(sess, feed, per_call_ms)
    r2 = probe2_sleep_overlap(sess, feed, per_call_ms)
    r3 = probe3_two_sessions(PATH_KEY, per_call_ms)

    print("\n=== verdict ===")
    # Primary decision criterion: probe 2 — it shapes exactly like Lever A.
    if r2["overlap_score"] >= 0.80:
        verdict = "GO"
        reason = (
            f"probe 2 overlap score {r2['overlap_score']:.2f} >= 0.80 -- sess.run() "
            f"releases GIL and async target-verify || draft is viable."
        )
    elif r2["overlap_score"] >= 0.40:
        verdict = "PARTIAL"
        reason = (
            f"probe 2 overlap score {r2['overlap_score']:.2f} between 0.40 and 0.80 -- "
            f"sess.run() releases GIL for part of the call. Lever A gains something "
            f"but below the 37% prediction; check probe 1 fraction."
        )
    else:
        verdict = "NO-GO via threads"
        reason = (
            f"probe 2 overlap score {r2['overlap_score']:.2f} < 0.40 -- sess.run() "
            f"holds GIL. Need subprocess-per-draft or an ORT run_async callback."
        )
    print(f"verdict  : {verdict}")
    print(f"reason   : {reason}")
    print(f"probe 1 GIL-release fraction              : {r1['gil_release_frac']:.3f}")
    print(f"probe 2 overlap score (sleep || npu)      : {r2['overlap_score']:.3f}")
    print(f"probe 3 overlap score (two sessions)      : {r3['overlap_score']:.3f}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
