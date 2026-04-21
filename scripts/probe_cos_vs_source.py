"""Phase 5 step 6 - CPU-ORT cos-vs-source probe for rewritten Qwen3.

Compares a candidate ONNX (stage / path A / path B-mask) against the
optimum source on CPU-ORT. Gate: cos >= 0.9999 AND argmax match on
the single-step probe. Any candidate that fails this gate must not
be transferred to the aarch64 side -- we'd reproduce the session 9
"silently corrupted" failure mode.

Probe design:

    past_kv : all zeros, shape [1, 8, 511, 128] per layer (FP16)
    input_ids   : [[151643]]  (Qwen3 BOS)
    position_ids: [[511]]     (last slot of the 512-wide decode window)
    attention_mask: [[1] * 512] FP matching (only needed for source)

Rationale for past=511 zeros + BOS at position 511 rather than
"zero-KV at position 0":

    The staged/folded graphs pin attention_mask to length 512 as an
    initializer. A past=0 probe exercises a length-1 mask window in
    the source graph but a length-512 mask window in the staged
    graph, and the internal causal-mask builder reacts to mask length
    in ways that make the two NOT numerically equivalent at that
    boundary. Using past=511 puts total_len=512 in both graphs, so
    they see the same shape regime end-to-end.

    Zero-valued past KV gives softmax(QK^T) of all-zeros = uniform
    attention weights; weighted sum of zero V tensors = zero
    contribution from past. The current token is therefore the only
    meaningful signal, which is what we want for a structural probe.

Pass/fail semantics:

    cos >= 0.9999         => rewrite is numerically equivalent
    argmax match          => top-1 prediction preserved
    cos in [0.99, 0.9999) => suspicious, investigate before trusting
    cos < 0.99            => rewrite is broken, do not ship

Run:

    python scripts/probe_cos_vs_source.py \\
        --source models/qwen3-0.6b-optimum/model.onnx \\
        --candidate models/qwen3-0.6b-staged/model.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent

# Qwen3-0.6B structural constants.
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
CTX_LEN = 512
PAST_LEN = CTX_LEN - 1  # 511
BOS_TOKEN = 151643


def make_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    # Disable ORT's own graph-level optimizations so we compare the
    # as-written graph, not an ORT-rewritten variant. Matters for
    # verifying that our protobuf edits are what's being measured.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    print(f"  session loaded in {time.perf_counter() - t0:.2f} s "
          f"({len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs)")
    return sess


def detect_input_dtype(sess: ort.InferenceSession, name: str) -> np.dtype:
    for inp in sess.get_inputs():
        if inp.name == name:
            # ORT type strings: 'tensor(float16)', 'tensor(float)',
            # 'tensor(int64)', 'tensor(bool)', etc.
            t = inp.type
            if "float16" in t:
                return np.float16
            if "float" in t:
                return np.float32
            if "int64" in t:
                return np.int64
            if "int32" in t:
                return np.int32
            if "bool" in t:
                return np.bool_
    return np.float32


def build_feed(sess: ort.InferenceSession, for_source: bool) -> dict:
    """Construct the probe input dict.

    for_source=True graphs still have attention_mask as a runtime input;
    rewritten graphs have it promoted to an initializer. We detect this
    by looking at the session's declared inputs.
    """
    input_names = {i.name for i in sess.get_inputs()}

    kv_dtype = detect_input_dtype(sess, "past_key_values.0.key")
    # Past KV of shape [1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM] all zeros.
    past_kv_shape = (1, NUM_KV_HEADS, PAST_LEN, HEAD_DIM)

    feed: dict = {}
    for i in range(NUM_LAYERS):
        feed[f"past_key_values.{i}.key"] = np.zeros(past_kv_shape, dtype=kv_dtype)
        feed[f"past_key_values.{i}.value"] = np.zeros(past_kv_shape, dtype=kv_dtype)

    feed["input_ids"] = np.array([[BOS_TOKEN]], dtype=np.int64)
    feed["position_ids"] = np.array([[PAST_LEN]], dtype=np.int64)

    if "attention_mask" in input_names:
        # Source graph (attention_mask still a runtime input).
        feed["attention_mask"] = np.ones((1, CTX_LEN), dtype=np.int64)
    # else: staged/fold graphs have it as a constant initializer; no
    # runtime input to supply.

    if "attention_bias" in input_names:
        # Path B-mask graph. Additive causal mask; since total_len=512
        # and we're at the last position (511), all previous positions
        # are valid -> all zeros for this probe. FP16.
        feed["attention_bias"] = np.zeros((1, 1, 1, CTX_LEN), dtype=np.float16)

    # Report what we built so mismatches with the session are easy to
    # spot.
    print(f"  feed: {len(feed)} tensors")
    missing = [i.name for i in sess.get_inputs() if i.name not in feed]
    extra = [n for n in feed if n not in input_names]
    if missing:
        print(f"  WARNING: session expects inputs not in feed: {missing}")
    if extra:
        print(f"  NOTE: feed contains inputs not on session: {extra}")
    return feed


def get_logits(sess: ort.InferenceSession, outputs: list) -> np.ndarray:
    for i, o in enumerate(sess.get_outputs()):
        if o.name == "logits":
            return outputs[i]
    raise RuntimeError("no output named 'logits' on session")


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Cosine + argmax + top-5 overlap + max |diff| on last-position logits.

    Both inputs shape [1, seq, vocab]. We compare the last row ([:, -1, :]).
    """
    a_v = a[0, -1].astype(np.float32)
    b_v = b[0, -1].astype(np.float32)
    cos = float(np.dot(a_v, b_v) / (np.linalg.norm(a_v) * np.linalg.norm(b_v) + 1e-30))
    max_abs = float(np.max(np.abs(a_v - b_v)))
    argmax_a = int(np.argmax(a_v))
    argmax_b = int(np.argmax(b_v))
    top5_a = set(np.argsort(-a_v)[:5].tolist())
    top5_b = set(np.argsort(-b_v)[:5].tolist())
    top5_overlap = len(top5_a & top5_b)
    return {
        "cos": cos,
        "max_abs_diff": max_abs,
        "argmax_source": argmax_a,
        "argmax_candidate": argmax_b,
        "argmax_match": argmax_a == argmax_b,
        "top5_overlap": top5_overlap,
    }


def grade(cos: float, argmax_match: bool) -> str:
    if cos >= 0.9999 and argmax_match:
        return "PASS"
    if cos >= 0.99 and argmax_match:
        return "SUSPICIOUS"
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "model.onnx",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path to write a JSON report for the handoff",
    )
    args = parser.parse_args()

    print(f"source    : {args.source}")
    print(f"candidate : {args.candidate}")

    print("\nloading source session ...")
    src_sess = make_session(args.source)
    print("\nloading candidate session ...")
    cand_sess = make_session(args.candidate)

    print("\nbuilding feed for source ...")
    src_feed = build_feed(src_sess, for_source=True)
    print("\nbuilding feed for candidate ...")
    cand_feed = build_feed(cand_sess, for_source=False)

    print("\nrunning source ...")
    t0 = time.perf_counter()
    src_out = src_sess.run(None, src_feed)
    print(f"  source run   : {(time.perf_counter() - t0) * 1000:.0f} ms")

    print("running candidate ...")
    t0 = time.perf_counter()
    cand_out = cand_sess.run(None, cand_feed)
    print(f"  candidate run: {(time.perf_counter() - t0) * 1000:.0f} ms")

    src_logits = get_logits(src_sess, src_out)
    cand_logits = get_logits(cand_sess, cand_out)
    print(f"\nsource logits shape   : {src_logits.shape} ({src_logits.dtype})")
    print(f"candidate logits shape: {cand_logits.shape} ({cand_logits.dtype})")

    result = compare(src_logits, cand_logits)
    verdict = grade(result["cos"], result["argmax_match"])

    print("\n=== result ===")
    for k, v in result.items():
        print(f"  {k:20s} {v}")
    print(f"  {'verdict':20s} {verdict}")

    if args.report:
        payload = {
            "source": str(args.source),
            "candidate": str(args.candidate),
            **result,
            "verdict": verdict,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2))
        print(f"  wrote report to {args.report}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
