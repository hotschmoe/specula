"""Phase 5 step 6 - CPU-vs-NPU correctness probe.

Question this answers: does the Qwen3-0.6B HTP context binary
compute the same next-token distribution as an ORT-CPU reference
running the base FP32 ONNX on the same KV state?

Strategy:

  1. Drive CPU prefill + decode on the base ONNX
     (`models/qwen3-0.6b-onnx/onnx/model.onnx`, dynamic past_len,
     FP32 KV, INT64 ids) until past_len = 511 -- the fixed past_len
     the NPU graph was compiled for. The 511-slot KV is what the
     NPU always expects.
  2. Single-step comparison: at past_len=511, run one more decode
     step on BOTH backends with identical past_kv + input_ids +
     position_ids. Compare logits (argmax, top-5 overlap,
     cosine similarity, max |delta|).
  3. Multi-step greedy with sliding-window KV: from the same
     anchor, decode N tokens on each backend. Drop slot 0 of the
     512-slot present to get the next 511-slot past. Track how
     often the two backends pick the same greedy token.

Why the base ONNX (not the nomask variant) on the CPU reference:
the base ONNX carries the real attention_mask and a standard
dotted-name IO, giving us FP32 ground truth on any past_len.
The nomask variant was a structural workaround for HTP's lack
of BOOL support; using it as the CPU reference would bake in
the same mask-is-always-all-ones assumption we're trying to
validate.

Numerical expectations (informational, not strict tolerances):
  * NPU was compiled with `--quantize_full_type float16` ->
    interior math is FP16; CPU is FP32. Single-step |delta| on
    logits can reach a few tenths; cosine sim should stay > 0.99.
  * Argmax usually matches, top-5 usually overlaps 4/5 or 5/5.
  * Multi-step greedy can diverge after several steps even when
    each single step is "close enough" -- small logit noise
    flips the argmax at one ambiguous position and both streams
    then condition on different tokens. A 70%+ per-step match
    rate over 16 tokens with both streams staying recognisably
    English is the bar for this step.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_vs_cpu_correctness.py --path patha
    .venv\\Scripts\\python.exe scripts\\npu_vs_cpu_correctness.py --path pathbmask
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from npu_load_qwen3_bin import (  # noqa: E402
    CONTEXT_MAX,
    LOGITS_OUTPUT_NAME,
    build_ep_context_wrapper,
    load_wrapper,
)

# Use the optimum export, NOT the `qwen3-0.6b-onnx/` community export.
# The community export uses com.microsoft::GroupQueryAttention +
# com.microsoft::RotaryEmbedding with their own KV-cache semantics --
# internally ring-buffered, RoPE applied in-op. The NPU binary was
# compiled from the optimum export which uses standard-ONNX ops with
# RoPE applied explicitly to K before storage. Taking KV from GQA and
# handing it to the standard-ops NPU graph gave catastrophic
# disagreement (cosine sim 0.55, top-5 overlap 0/5). The optimum
# export shares lineage with the NPU's source ONNX, so CPU KV state
# is directly interchangeable with NPU KV state.
CPU_ONNX_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
CPU_ONNX = CPU_ONNX_DIR / "model.onnx"
TOKENIZER_JSON = CPU_ONNX_DIR / "tokenizer.json"
CONFIG_JSON = CPU_ONNX_DIR / "config.json"


def _npu_bin(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}.bin"


def _npu_wrapper(path_key: str) -> Path:
    return REPO_ROOT / "models" / f"qwen3_0_6b_draft_v81_ctx{CONTEXT_MAX}.{path_key}.wrapper.onnx"


PROMPT = "The Snapdragon X2 Elite Extreme is"
N_GREEDY_STEPS = 16  # multi-step comparison length


def _safe_repr(s: str) -> str:
    """Windows console is cp1252; repr() any tokenizer output through ASCII
    backslash-escapes so a stray emoji / CJK char doesn't crash the print."""
    return repr(s.encode("ascii", "backslashreplace").decode("ascii"))


def load_cpu_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def load_npu_session(cfg: dict, path_key: str) -> ort.InferenceSession:
    bin_path = _npu_bin(path_key)
    wrapper_path = _npu_wrapper(path_key)
    if not wrapper_path.exists():
        build_ep_context_wrapper(cfg, bin_path, wrapper_path, path_key)
    return load_wrapper(wrapper_path)


def cpu_build_feed(
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    past_kv: dict[str, np.ndarray],
    total_seq_len: int,
) -> dict[str, np.ndarray]:
    """CPU ONNX: INT64 ids/positions, attention_mask over [past + current]."""
    feed: dict[str, np.ndarray] = {
        "input_ids": input_ids.astype(np.int64),
        "position_ids": position_ids.astype(np.int64),
        "attention_mask": np.ones((1, total_seq_len), dtype=np.int64),
    }
    feed.update(past_kv)
    return feed


def cpu_present_to_past(
    outputs: list[np.ndarray],
    name_to_idx: dict[str, int],
    n_layers: int,
) -> dict[str, np.ndarray]:
    """Carry present.i.{key,value} back as past_key_values.i.{key,value}."""
    past: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        past[f"past_key_values.{i}.key"] = outputs[name_to_idx[f"present.{i}.key"]]
        past[f"past_key_values.{i}.value"] = outputs[name_to_idx[f"present.{i}.value"]]
    return past


def slide_cpu_past(past: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop slot 0 so past_len returns to 511 after each sliding-window step."""
    return {k: v[:, :, 1:, :] for k, v in past.items()}


def npu_past_from_cpu(
    cpu_past: dict[str, np.ndarray], n_layers: int
) -> dict[str, np.ndarray]:
    """Translate CPU past (dotted names, FP32) -> NPU past (underscored, FP32).

    Shapes are already [1, 8, 511, 128] if the CPU session has been
    fed exactly 511 past slots, so we only rename and assert dtype.
    """
    npu_past: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        k = cpu_past[f"past_key_values.{i}.key"]
        v = cpu_past[f"past_key_values.{i}.value"]
        assert k.dtype == np.float32 and v.dtype == np.float32, (
            f"layer {i}: expected float32 KV, got {k.dtype}/{v.dtype}"
        )
        assert k.shape[2] == CONTEXT_MAX - 1, (
            f"layer {i}: expected past_len={CONTEXT_MAX - 1}, got {k.shape[2]}"
        )
        npu_past[f"past_key_values_{i}_key"] = k
        npu_past[f"past_key_values_{i}_value"] = v
    return npu_past


def npu_build_feed(
    input_id: int,
    position: int,
    npu_past: dict[str, np.ndarray],
    path_key: str,
) -> dict[str, np.ndarray]:
    """NPU: INT32 ids/positions [1,1]; past_kv FP32 [1,8,511,128].

    Path B-mask additionally requires an `attention_bias` FP32 input of
    shape [1,1,1,512]. For the decode-only regime (fully-valid window)
    the bias is all zeros; the causal / padding cases would use a
    lower-triangular 0.0 / -65504.0 matrix but that's not needed for
    the current sd.npu draft path.
    """
    feed: dict[str, np.ndarray] = {
        "input_ids": np.array([[input_id]], dtype=np.int32),
        "position_ids": np.array([[position]], dtype=np.int32),
    }
    if path_key == "pathbmask":
        feed["attention_bias"] = np.zeros((1, 1, 1, CONTEXT_MAX), dtype=np.float32)
    feed.update(npu_past)
    return feed


def npu_present_to_past(
    outputs: list[np.ndarray],
    npu_out_names: list[str],
    n_layers: int,
) -> dict[str, np.ndarray]:
    """output_0 = logits; output_{2i+1,2i+2} = present.i.{key,value}.

    The NPU returns 512-slot present; drop slot 0 to get the next
    511-slot past (sliding window).
    """
    past: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        k = outputs[npu_out_names.index(f"output_{2 * i + 1}")]
        v = outputs[npu_out_names.index(f"output_{2 * i + 2}")]
        past[f"past_key_values_{i}_key"] = k[:, :, 1:, :]
        past[f"past_key_values_{i}_value"] = v[:, :, 1:, :]
    return past


def compare_logits(
    cpu_logits: np.ndarray, npu_logits: np.ndarray, top_k: int = 5
) -> dict[str, float | int | list[int]]:
    """Single-step logit comparison.

    cpu_logits/npu_logits are both shape [vocab]. We report argmax
    agreement, top-k overlap size, cosine similarity, and max |delta|.
    """
    cpu = cpu_logits.astype(np.float32).ravel()
    npu = npu_logits.astype(np.float32).ravel()
    assert cpu.shape == npu.shape, f"vocab mismatch: {cpu.shape} vs {npu.shape}"

    cpu_argmax = int(np.argmax(cpu))
    npu_argmax = int(np.argmax(npu))
    cpu_topk = np.argsort(-cpu)[:top_k].tolist()
    npu_topk = np.argsort(-npu)[:top_k].tolist()
    overlap = len(set(cpu_topk) & set(npu_topk))

    diff = cpu - npu
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(cpu) * np.linalg.norm(npu))
    cos = float(np.dot(cpu, npu) / denom) if denom > 0 else 0.0

    return {
        "cpu_argmax": cpu_argmax,
        "npu_argmax": npu_argmax,
        "argmax_match": int(cpu_argmax == npu_argmax),
        "cpu_top5": cpu_topk,
        "npu_top5": npu_topk,
        "top5_overlap": overlap,
        "max_abs_diff": max_abs,
        "cosine_sim": cos,
    }


def run_cpu_prefill_then_decode_to_511(
    sess: ort.InferenceSession,
    cfg: dict,
    tok: Tokenizer,
    prompt: str,
) -> tuple[dict[str, np.ndarray], int, list[int]]:
    """Prefill the prompt + greedy-decode on CPU until past_len == 511.

    Returns (past_kv_511, next_token_id, generated_ids). The caller
    then hands `next_token_id` + `past_kv_511` to both CPU and NPU
    for the single-step comparison.

    The last greedy argmax becomes the `next_token_id` to feed into
    the comparison step; at that point past already covers 511
    positions (0..510) and the token sits at position 511.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    out_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {n: i for i, n in enumerate(out_names)}

    prompt_ids = tok.encode(prompt).ids
    seq_len = len(prompt_ids)
    print(f"  prompt tokens ({seq_len}): {prompt_ids}")

    # Prefill the prompt.
    empty_kv = {
        f"past_key_values.{i}.{k}": np.zeros(
            (1, n_kv, 0, head_dim), dtype=np.float32
        )
        for i in range(n_layers)
        for k in ("key", "value")
    }
    feed = cpu_build_feed(
        input_ids=np.array([prompt_ids], dtype=np.int64),
        position_ids=np.arange(seq_len, dtype=np.int64)[None, :],
        past_kv=empty_kv,
        total_seq_len=seq_len,
    )
    outputs = sess.run(None, feed)
    past_kv = cpu_present_to_past(outputs, name_to_idx, n_layers)
    past_len = seq_len
    next_id = int(np.argmax(outputs[name_to_idx["logits"]][0, -1]))
    generated: list[int] = [next_id]

    # Decode until past_len + 1 == 511 + 1, i.e., past_len == 511 with
    # next_id ready at position 511. That's the anchor for comparison.
    print(
        f"  CPU decode: growing past_len {past_len} -> {CONTEXT_MAX - 1} "
        f"({CONTEXT_MAX - 1 - past_len} steps)..."
    )
    t0 = time.perf_counter()
    while past_len < CONTEXT_MAX - 1:
        feed = cpu_build_feed(
            input_ids=np.array([[next_id]], dtype=np.int64),
            position_ids=np.array([[past_len]], dtype=np.int64),
            past_kv=past_kv,
            total_seq_len=past_len + 1,
        )
        outputs = sess.run(None, feed)
        past_kv = cpu_present_to_past(outputs, name_to_idx, n_layers)
        next_id = int(np.argmax(outputs[name_to_idx["logits"]][0, -1]))
        generated.append(next_id)
        past_len += 1
    elapsed = time.perf_counter() - t0
    print(f"  decoded {len(generated) - 1} tokens in {elapsed:.1f} s")

    # At this point: past covers positions 0..510 (511 slots),
    # next_id is the token at position 511 (not yet committed).
    return past_kv, next_id, generated


def single_step(
    cpu_sess: ort.InferenceSession,
    npu_sess: ort.InferenceSession,
    cfg: dict,
    cpu_past: dict[str, np.ndarray],
    input_id: int,
    position: int,
    path_key: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Run one decode step on both backends. Return (cpu_logits, npu_logits,
    cpu_outputs, npu_outputs) for callers that need the present KV too."""
    n_layers = cfg["num_hidden_layers"]

    cpu_out_names = [o.name for o in cpu_sess.get_outputs()]
    cpu_name_to_idx = {n: i for i, n in enumerate(cpu_out_names)}
    # attention_mask covers [past_len + 1] -- derive past_len from the KV
    # directly rather than from `position`, so zero-KV diagnostics at
    # position != past_len still pass a correctly-sized mask.
    past_len = next(iter(cpu_past.values())).shape[2]
    cpu_feed = cpu_build_feed(
        input_ids=np.array([[input_id]], dtype=np.int64),
        position_ids=np.array([[position]], dtype=np.int64),
        past_kv=cpu_past,
        total_seq_len=past_len + 1,
    )
    cpu_outputs = cpu_sess.run(None, cpu_feed)
    cpu_logits = cpu_outputs[cpu_name_to_idx["logits"]][0, -1]

    npu_out_names = [o.name for o in npu_sess.get_outputs()]
    npu_past = npu_past_from_cpu(cpu_past, n_layers)
    npu_feed = npu_build_feed(input_id, position, npu_past, path_key)
    npu_outputs = npu_sess.run(None, npu_feed)
    npu_logits = npu_outputs[npu_out_names.index(LOGITS_OUTPUT_NAME)][0, -1]

    return cpu_logits, npu_logits, cpu_outputs, npu_outputs


def multi_step_sliding(
    cpu_sess: ort.InferenceSession,
    npu_sess: ort.InferenceSession,
    cfg: dict,
    tok: Tokenizer,
    anchor_past: dict[str, np.ndarray],
    anchor_next_id: int,
    anchor_position: int,
    n_steps: int,
    path_key: str,
) -> dict:
    """From (past_511, next_id, position) run N greedy steps on each backend
    in lock-step with sliding-window KV. Each backend drives its own KV so
    after step 1 the two KVs diverge slightly; that divergence + FP16 drift
    is what we're measuring.

    Returns per-step match flags and the two generated token streams.
    """
    n_layers = cfg["num_hidden_layers"]
    cpu_out_names = [o.name for o in cpu_sess.get_outputs()]
    cpu_name_to_idx = {n: i for i, n in enumerate(cpu_out_names)}
    npu_out_names = [o.name for o in npu_sess.get_outputs()]

    cpu_past = {k: v.copy() for k, v in anchor_past.items()}
    npu_past = npu_past_from_cpu(cpu_past, n_layers)

    cpu_stream: list[int] = []
    npu_stream: list[int] = []
    matches: list[int] = []
    position = anchor_position
    cpu_in = anchor_next_id
    npu_in = anchor_next_id

    for step in range(n_steps):
        # CPU step. Past stays at 511 slots due to sliding window; mask
        # must cover 511 past + 1 new = 512 regardless of absolute position.
        past_len = next(iter(cpu_past.values())).shape[2]
        cpu_feed = cpu_build_feed(
            input_ids=np.array([[cpu_in]], dtype=np.int64),
            position_ids=np.array([[position]], dtype=np.int64),
            past_kv=cpu_past,
            total_seq_len=past_len + 1,
        )
        cpu_out = cpu_sess.run(None, cpu_feed)
        cpu_present = cpu_present_to_past(cpu_out, cpu_name_to_idx, n_layers)
        cpu_past = slide_cpu_past(cpu_present)
        cpu_next = int(np.argmax(cpu_out[cpu_name_to_idx["logits"]][0, -1]))

        # NPU step.
        npu_feed = npu_build_feed(npu_in, position, npu_past, path_key)
        npu_out = npu_sess.run(None, npu_feed)
        npu_past = npu_present_to_past(npu_out, npu_out_names, n_layers)
        npu_next = int(
            np.argmax(npu_out[npu_out_names.index(LOGITS_OUTPUT_NAME)][0, -1])
        )

        cpu_stream.append(cpu_next)
        npu_stream.append(npu_next)
        matches.append(int(cpu_next == npu_next))
        cpu_in = cpu_next
        npu_in = npu_next
        position += 1

    return {
        "cpu_stream": cpu_stream,
        "npu_stream": npu_stream,
        "matches": matches,
        "match_rate": sum(matches) / max(1, len(matches)),
        "cpu_text": tok.decode(cpu_stream),
        "npu_text": tok.decode(npu_stream),
    }


def zero_kv_diagnostic(
    cpu_sess: ort.InferenceSession,
    npu_sess: ort.InferenceSession,
    cfg: dict,
    tok: Tokenizer,
    bos_id: int,
    position: int,
    path_key: str,
) -> None:
    """Control test: feed BOS + all-zero past_kv to both backends and compare.

    This isolates "is the NPU graph broken on its own?" from "does KV
    hand-off work?". With zero KV, attention output is zero (zero V),
    and the hidden state reduces to MLP-through-layers of the token
    embedding. Both backends should produce the same logits to within
    FP16 drift.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_511 = CONTEXT_MAX - 1

    # CPU: zero KV at past_len=511 (matches NPU's fixed window).
    cpu_past = {
        f"past_key_values.{i}.{k}": np.zeros(
            (1, n_kv, past_511, head_dim), dtype=np.float32
        )
        for i in range(n_layers)
        for k in ("key", "value")
    }
    cpu_logits, npu_logits, _, _ = single_step(
        cpu_sess, npu_sess, cfg, cpu_past, bos_id, position, path_key
    )
    stats = compare_logits(cpu_logits, npu_logits)
    print(
        f"  [zero-KV, token={bos_id} (BOS), pos={position}] "
        f"cpu_max={float(np.max(cpu_logits)):+.2f} "
        f"npu_max={float(np.max(npu_logits)):+.2f} "
        f"cos={stats['cosine_sim']:.4f} "
        f"top5_overlap={stats['top5_overlap']}/5 "
        f"argmax_match={bool(stats['argmax_match'])}"
    )
    print(f"    cpu argmax: {stats['cpu_argmax']} -> {_safe_repr(tok.decode([stats['cpu_argmax']]))}")
    print(f"    npu argmax: {stats['npu_argmax']} -> {_safe_repr(tok.decode([stats['npu_argmax']]))}")


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask"), required=True)
    args = parser.parse_args()
    path_key = args.path

    print(f"=== step 6 - CPU vs NPU correctness ({path_key}) ===\n")

    npu_bin = _npu_bin(path_key)

    if not CPU_ONNX.exists():
        print(f"ERROR: {CPU_ONNX} missing (run download_qwen3_onnx.py)")
        return 2
    if not npu_bin.exists():
        print(f"ERROR: {npu_bin} missing (need AI Hub step 4 output)")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    print("--- loading CPU ONNX (FP32, dynamic past_len) ---")
    t0 = time.perf_counter()
    cpu_sess = load_cpu_session(CPU_ONNX)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    print("\n--- loading NPU binary (FP16 interior, fixed past_len=511) ---")
    t0 = time.perf_counter()
    npu_sess = load_npu_session(cfg, path_key)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")
    providers = npu_sess.get_providers()
    if not providers or providers[0] != "QNNExecutionProvider":
        print(f"ERROR: NPU session fell back to {providers}")
        return 2

    print("\n--- zero-KV diagnostic (isolate graph vs KV-handoff failures) ---")
    bos_id = cfg.get("bos_token_id", 151643)
    zero_kv_diagnostic(cpu_sess, npu_sess, cfg, tok, bos_id=bos_id, position=0, path_key=path_key)
    zero_kv_diagnostic(cpu_sess, npu_sess, cfg, tok, bos_id=bos_id, position=511, path_key=path_key)
    # Also try a non-BOS token -- catches cases where BOS is a special
    # path in one graph but not the other.
    zero_kv_diagnostic(cpu_sess, npu_sess, cfg, tok, bos_id=785, position=0, path_key=path_key)

    print("\n--- CPU prefill + decode to past_len=511 ---")
    anchor_past, anchor_next_id, generated = run_cpu_prefill_then_decode_to_511(
        cpu_sess, cfg, tok, PROMPT
    )
    anchor_position = CONTEXT_MAX - 1  # 511
    anchor_text = PROMPT + tok.decode(generated)
    print(f"\n  CPU-generated continuation ({len(generated)} toks, first 120 chars):")
    print(f"    {_safe_repr(anchor_text[: len(PROMPT) + 120])}")
    print(f"  anchor next-token id (position {anchor_position}): {anchor_next_id}")

    print("\n--- single-step logit comparison at position 511 ---")
    cpu_logits, npu_logits, _, _ = single_step(
        cpu_sess, npu_sess, cfg, anchor_past, anchor_next_id, anchor_position, path_key
    )
    single = compare_logits(cpu_logits, npu_logits)
    print(f"  cpu argmax      : {single['cpu_argmax']}  -> {_safe_repr(tok.decode([single['cpu_argmax']]))}")
    print(f"  npu argmax      : {single['npu_argmax']}  -> {_safe_repr(tok.decode([single['npu_argmax']]))}")
    print(f"  argmax match    : {bool(single['argmax_match'])}")
    print(f"  top-5 overlap   : {single['top5_overlap']} / 5")
    print(f"    cpu top5      : {single['cpu_top5']}")
    print(f"    npu top5      : {single['npu_top5']}")
    print(f"  cosine sim      : {single['cosine_sim']:.6f}")
    print(f"  max |delta|     : {single['max_abs_diff']:.4f}")

    print(f"\n--- multi-step greedy ({N_GREEDY_STEPS} steps, sliding-window KV) ---")
    multi = multi_step_sliding(
        cpu_sess,
        npu_sess,
        cfg,
        tok,
        anchor_past,
        anchor_next_id,
        anchor_position,
        n_steps=N_GREEDY_STEPS,
        path_key=path_key,
    )
    print(f"  CPU stream ids  : {multi['cpu_stream']}")
    print(f"  NPU stream ids  : {multi['npu_stream']}")
    print(f"  per-step match  : {multi['matches']}")
    print(f"  match rate      : {multi['match_rate'] * 100:.1f}%")
    print(f"  CPU text        : {_safe_repr(multi['cpu_text'])}")
    print(f"  NPU text        : {_safe_repr(multi['npu_text'])}")

    # Success criteria (informational, not gated): cosine > 0.99 on the
    # single step AND match-rate >= 50% AND NPU text looks like English.
    cos_ok = single["cosine_sim"] > 0.99
    rate_ok = multi["match_rate"] >= 0.5
    npu_has_alpha = any(c.isalpha() for c in multi["npu_text"])
    healthy = cos_ok and rate_ok and npu_has_alpha

    print("\n=== STATUS ===")
    print(f"  cosine > 0.99           : {cos_ok} ({single['cosine_sim']:.4f})")
    print(f"  greedy match-rate >= 50%: {rate_ok} ({multi['match_rate'] * 100:.1f}%)")
    print(f"  NPU stream is text      : {npu_has_alpha}")
    print(f"  overall                 : {'ok' if healthy else 'needs investigation'}")
    return 0 if healthy else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
