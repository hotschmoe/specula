"""Phase 5 step 3b - validate Qwen3-0.6B ONNX end-to-end on ORT-CPU.

Exit criterion substitution: the scoping doc called for PyTorch-logit
comparison within 1e-3, but torch has no cp312 win_arm64 wheel. Instead
we do a coherence check -- greedy-decode 32 tokens from a fixed prompt
and confirm the output is recognisable text. If the ONNX is malformed
the decode diverges or produces repeated/garbage tokens.

Run:
    .venv\\Scripts\\python.exe scripts\\validate_qwen3_onnx_cpu.py

Exit codes:
    0 - session ran, output is coherent
    1 - output looks degenerate (all same token, non-text, etc.)
    2 - hard failure
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3-0.6b-onnx"
MODEL_ONNX = MODEL_DIR / "onnx" / "model.onnx"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"
CONFIG_JSON = MODEL_DIR / "config.json"

PROMPT = "The Snapdragon X2 Elite Extreme is"
N_NEW_TOKENS = 32


def summarize_io(sess: ort.InferenceSession) -> None:
    print("\n--- ONNX IO signature (first 5 of each) ---")
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  inputs  ({len(inputs)} total):")
    for x in inputs[:5]:
        print(f"    {x.name:40s} {x.type:20s} {x.shape}")
    print(f"    ... ({len(inputs) - 5} more)") if len(inputs) > 5 else None
    print(f"  outputs ({len(outputs)} total):")
    for x in outputs[:5]:
        print(f"    {x.name:40s} {x.type:20s} {x.shape}")
    print(f"    ... ({len(outputs) - 5} more)") if len(outputs) > 5 else None


def build_empty_kv(cfg: dict) -> dict:
    """Return dict of zero-length KV tensors suitable as first-step past_*."""
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    empty = np.zeros((1, n_kv, 0, head_dim), dtype=np.float32)
    feed = {}
    for i in range(n_layers):
        feed[f"past_key_values.{i}.key"] = empty
        feed[f"past_key_values.{i}.value"] = empty
    return feed


def main() -> int:
    if not MODEL_ONNX.exists():
        print(f"model not found at {MODEL_ONNX}")
        print("  run scripts/download_qwen3_onnx.py first")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    print(f"model arch          : {cfg.get('model_type', '?')}")
    print(f"num_hidden_layers   : {cfg['num_hidden_layers']}")
    print(f"num_attention_heads : {cfg['num_attention_heads']}")
    print(f"num_key_value_heads : {cfg.get('num_key_value_heads', cfg['num_attention_heads'])}")
    print(f"hidden_size         : {cfg['hidden_size']}")
    print(f"head_dim            : {cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])}")
    print(f"vocab_size          : {cfg['vocab_size']}")

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    eos_id = cfg.get("eos_token_id")
    if isinstance(eos_id, list):
        eos_ids = set(eos_id)
    else:
        eos_ids = {eos_id} if eos_id is not None else set()
    print(f"eos_token_ids       : {sorted(eos_ids)}")

    # ORT-CPU only for this validation. No QNN yet; we need to confirm
    # the ONNX itself is well-formed before worrying about NPU compile.
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"\nloading {MODEL_ONNX.name} on CPU ...")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(MODEL_ONNX),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    print(f"session loaded in {time.perf_counter() - t0:.2f} s")
    summarize_io(sess)

    # Tokenize prompt.
    enc = tok.encode(PROMPT)
    input_ids = np.array([enc.ids], dtype=np.int64)
    seq_len = input_ids.shape[1]
    print(f"\nprompt              : {PROMPT!r}")
    print(f"prompt tokens ({seq_len}): {enc.ids}")
    print(f"decoded back        : {tok.decode(enc.ids)!r}")

    # --- Prefill ---
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    feed.update(build_empty_kv(cfg))

    print(f"\n--- prefill ({seq_len} tokens) ---")
    t0 = time.perf_counter()
    outputs = sess.run(None, feed)
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"prefill latency     : {prefill_ms:.1f} ms  ({prefill_ms / seq_len:.1f} ms/tok)")

    output_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {name: i for i, name in enumerate(output_names)}
    logits = outputs[name_to_idx["logits"]]
    print(f"logits shape        : {logits.shape}")

    # Greedy next token.
    next_id = int(np.argmax(logits[0, -1]))

    # Build generator loop.
    generated = [next_id]
    past_len = seq_len

    # Carry present_* back as past_*.
    def present_to_past(outs: list) -> dict:
        new_past = {}
        for i in range(cfg["num_hidden_layers"]):
            new_past[f"past_key_values.{i}.key"] = outs[name_to_idx[f"present.{i}.key"]]
            new_past[f"past_key_values.{i}.value"] = outs[name_to_idx[f"present.{i}.value"]]
        return new_past

    past_kv = present_to_past(outputs)

    print(f"\n--- decode ({N_NEW_TOKENS} tokens) ---")
    t0 = time.perf_counter()
    for _ in range(N_NEW_TOKENS - 1):
        if next_id in eos_ids:
            break
        step_ids = np.array([[next_id]], dtype=np.int64)
        step_pos = np.array([[past_len]], dtype=np.int64)
        step_mask = np.ones((1, past_len + 1), dtype=np.int64)

        feed = {
            "input_ids": step_ids,
            "attention_mask": step_mask,
            "position_ids": step_pos,
        }
        feed.update(past_kv)

        outputs = sess.run(None, feed)
        logits = outputs[name_to_idx["logits"]]
        next_id = int(np.argmax(logits[0, -1]))
        generated.append(next_id)
        past_kv = present_to_past(outputs)
        past_len += 1
    decode_ms = (time.perf_counter() - t0) * 1000
    n_decoded = len(generated)
    print(f"decode total        : {decode_ms:.1f} ms  ({decode_ms / n_decoded:.1f} ms/tok)")
    print(f"decode t/s          : {n_decoded / (decode_ms / 1000):.2f}")

    text = tok.decode(generated)
    full = PROMPT + text
    print(f"\n--- generation ---\n{full}\n---")

    # Coherence heuristics.
    unique_ratio = len(set(generated)) / max(1, len(generated))
    has_text = any(ch.isalpha() for ch in text)
    is_coherent = unique_ratio > 0.3 and has_text
    print(f"unique-token ratio  : {unique_ratio:.2f}  (>0.3 for coherent)")
    print(f"contains alpha chars: {has_text}")

    if is_coherent:
        print("\n=== STATUS: ok (ONNX export runs and produces coherent text) ===")
        return 0
    print("\n=== STATUS: degenerate output, ONNX may be broken ===")
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
