"""Phase 5.5 R4 — probe ORT-QNN zero-copy + shared-memory options.

Two questions:

1. **What provider options does ORT-QNN 1.24.4 accept silently?**
   ORT doesn't expose an introspection API for EP options — unknown
   keys are either silently ignored or cause the session load to fail.
   We try each candidate with a trivial load + single forward pass and
   measure per-call latency to see if anything moved.

2. **Does ORT IOBinding cut copy overhead?** IOBinding lets us feed
   pre-allocated buffers directly to ORT, skipping the numpy->ORT
   tensor wrap on every call. For our Path A wrapper (58 inputs, most
   of which are large past_kv tensors), this could be a material win
   even if QNN doesn't do true zero-copy underneath — ORT-side
   overhead alone is measurable.

Baseline is the plain `sess.run(None, feed)` at 72.7 ms/call median
from the GIL probe. Any variant that lands in [65, 75] ms is
indistinguishable (noise). Variants that shave >=5 ms are worth
folding into production; anything >10 ms is a real win.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_zerocopy_probe.py
"""

from __future__ import annotations

import functools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import (  # noqa: E402
    CONTEXT_MAX,
    HTP_ARCH,
    SOC_MODEL,
    _bin_path,
    _config_json,
    _wrapper_path,
    build_ep_context_wrapper,
    build_zero_feed,
)


PATH_KEY = "patha"
N_WARMUP = 3
N_TIMED = 10


# Candidate options to probe. Pair each with a short hypothesis + what
# we expect to see if it works.
#
# Source: piecing together Microsoft's ORT-QNN EP doc pages + Qualcomm
# QAIRT 2.42 provider-options reference. Not all of these exist on
# 1.24.4 — probe empirically.
CANDIDATE_OPTION_SETS: list[tuple[str, dict, str]] = [
    (
        "baseline",
        {},
        "canonical config from npu_load_qwen3_bin.py; everything below adds on top",
    ),
    (
        "+enable_htp_weight_sharing=1",
        {"enable_htp_weight_sharing": "1"},
        "cross-session weight dedup — irrelevant for single-session but probe anyway",
    ),
    (
        "+htp_graph_finalization_optimization_mode=3",
        {"htp_graph_finalization_optimization_mode": "3"},
        "max finalization optimization (ORT docs: 0-3 scale, 3 most aggressive)",
    ),
    (
        "+rpc_control_latency=100",
        {"rpc_control_latency": "100"},
        "lower RPC latency budget (us) — relevant for per-call QNN dispatch",
    ),
    (
        "+rpc_control_latency=10",
        {"rpc_control_latency": "10"},
        "even lower",
    ),
    (
        "+vtcm_mb=8",
        {"vtcm_mb": "8"},
        "bump VTCM allocation",
    ),
    (
        "+qnn_context_priority=high",
        {"qnn_context_priority": "high"},
        "higher QNN dispatch priority",
    ),
    (
        "+htp_shared_memory=1",
        {"htp_shared_memory": "1"},
        "the direct zero-copy hint if exposed",
    ),
    (
        "+enable_htp_shared_memory_allocator=1",
        {"enable_htp_shared_memory_allocator": "1"},
        "alternate name seen in some ORT docs",
    ),
]


def base_provider_options() -> dict:
    backend = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
    return {
        "backend_path": str(backend),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }


def try_load(extra_opts: dict, wrapper_onnx: Path, silent_stderr: bool = True) -> tuple[ort.InferenceSession | None, str]:
    """Attempt to load with the given extra options. Returns (session, error_msg).

    ORT logs unknown keys at severity 3+ (WARNING). We set log_severity_level
    to 0 (VERBOSE) when probing so we can see acceptance/rejection, but mute
    it for production calls.
    """
    opts = base_provider_options()
    opts.update(extra_opts)

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # warnings + up; keeps output quiet

    try:
        sess = ort.InferenceSession(
            str(wrapper_onnx),
            sess_options=sess_opts,
            providers=[("QNNExecutionProvider", opts)],
        )
        return sess, ""
    except Exception as e:
        return None, repr(e)


def measure_latency(sess: ort.InferenceSession, n_warmup: int, n_timed: int) -> dict:
    feed = build_zero_feed(sess)
    for _ in range(n_warmup):
        sess.run(None, feed)
    samples = []
    for _ in range(n_timed):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return {
        "n": n_timed,
        "min": float(min(samples)),
        "median": float(np.median(samples)),
        "mean": float(np.mean(samples)),
        "max": float(max(samples)),
        "p95": float(np.percentile(samples, 95)),
    }


def measure_iobinding(sess: ort.InferenceSession, n_warmup: int, n_timed: int) -> dict:
    """Use ORT's IOBinding: pre-allocate all inputs in CPU ORT tensors,
    bind them once, reuse across calls. Skips the numpy->ORT wrap cost
    on every call.
    """
    feed = build_zero_feed(sess)
    # Wrap each numpy array as an ORT tensor (OrtValue) on CPU. This does
    # a single copy up front; subsequent run_with_iobinding doesn't re-wrap.
    ort_inputs: dict[str, ort.OrtValue] = {}
    for name, arr in feed.items():
        ort_inputs[name] = ort.OrtValue.ortvalue_from_numpy(arr, "cpu")

    binding = sess.io_binding()
    for name, ov in ort_inputs.items():
        binding.bind_ortvalue_input(name, ov)
    # Outputs: let ORT allocate on CPU for now (simplest; the copy-back is
    # the same as plain run).
    for out in sess.get_outputs():
        binding.bind_output(out.name, "cpu")

    # Warm.
    for _ in range(n_warmup):
        sess.run_with_iobinding(binding)
    # Timed.
    samples = []
    for _ in range(n_timed):
        t0 = time.perf_counter_ns()
        sess.run_with_iobinding(binding)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return {
        "n": n_timed,
        "min": float(min(samples)),
        "median": float(np.median(samples)),
        "mean": float(np.mean(samples)),
        "max": float(max(samples)),
        "p95": float(np.percentile(samples, 95)),
    }


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    print(f"=== R4 zero-copy probe (ORT {ort.__version__}, path={PATH_KEY}) ===\n")

    bin_p = _bin_path(PATH_KEY)
    wrap = _wrapper_path(PATH_KEY)
    cfg_json = _config_json(PATH_KEY)
    if not bin_p.exists():
        print(f"ERROR: {bin_p} missing")
        return 2
    if not wrap.exists():
        with cfg_json.open() as f:
            cfg = json.load(f)
        build_ep_context_wrapper(cfg, bin_p, wrap, PATH_KEY)

    results: list[dict] = []
    print(f"{'label':48s}  {'status':10s}  {'median_ms':>10s}  {'p95_ms':>8s}  {'min_ms':>8s}")
    print("-" * 100)

    for label, extras, note in CANDIDATE_OPTION_SETS:
        sess, err = try_load(extras, wrap)
        if sess is None:
            status = "LOAD_FAIL"
            med = p95 = mn = float("nan")
            error_suffix = f"  ({err[:60]})"
        else:
            try:
                stats = measure_latency(sess, N_WARMUP, N_TIMED)
                status = "OK"
                med = stats["median"]
                p95 = stats["p95"]
                mn = stats["min"]
                error_suffix = ""
            except Exception as e:
                status = "RUN_FAIL"
                med = p95 = mn = float("nan")
                error_suffix = f"  ({repr(e)[:60]})"
            finally:
                # Explicit del so the next session can rebind exclusive driver
                # resources if any are held.
                try:
                    del sess
                except Exception:
                    pass
        results.append({"label": label, "status": status, "median": med, "note": note})
        print(f"{label:48s}  {status:10s}  {med:>10.2f}  {p95:>8.2f}  {mn:>8.2f}{error_suffix}")

    # IOBinding probe uses the baseline session configuration.
    print("\n--- IOBinding probe (baseline config + run_with_iobinding) ---")
    sess, err = try_load({}, wrap)
    if sess is None:
        print(f"  LOAD_FAIL: {err}")
    else:
        stats_plain = measure_latency(sess, N_WARMUP, N_TIMED)
        stats_iob = measure_iobinding(sess, N_WARMUP, N_TIMED)
        print(f"{'mode':32s}  {'median_ms':>10s}  {'p95_ms':>8s}  {'min_ms':>8s}")
        print("-" * 70)
        print(f"{'plain sess.run':32s}  {stats_plain['median']:>10.2f}  "
              f"{stats_plain['p95']:>8.2f}  {stats_plain['min']:>8.2f}")
        print(f"{'run_with_iobinding':32s}  {stats_iob['median']:>10.2f}  "
              f"{stats_iob['p95']:>8.2f}  {stats_iob['min']:>8.2f}")
        delta_pct = (stats_iob['median'] - stats_plain['median']) / stats_plain['median'] * 100
        print(f"\n  iobinding delta: {delta_pct:+.1f}% median")
        if delta_pct <= -5.0:
            print("  VERDICT: IOBinding wins — fold into production outer loop.")
        elif delta_pct >= +5.0:
            print("  VERDICT: IOBinding regressed — skip.")
        else:
            print("  VERDICT: within noise — likely no real-world win.")

    print("\n=== summary ===")
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        baseline = next((r for r in ok_results if r["label"] == "baseline"), None)
        if baseline is not None:
            print(f"baseline median: {baseline['median']:.2f} ms")
            print(f"{'label':48s}  {'delta':>8s}")
            print("-" * 60)
            for r in ok_results:
                if r["label"] == "baseline":
                    continue
                delta = r["median"] - baseline["median"]
                flag = ""
                if delta <= -5.0:
                    flag = "  <-- adopt"
                elif delta >= 5.0:
                    flag = "  <-- REGRESSED"
                print(f"{r['label']:48s}  {delta:+7.2f}{flag}")
    rejected = [r for r in results if r["status"] != "OK"]
    if rejected:
        print(f"\nrejected options ({len(rejected)}):")
        for r in rejected:
            print(f"  {r['label']:48s}  {r['status']}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
