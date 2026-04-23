"""Differential diagnostic: w4a16-local* vs fp16-local on identical feeds.

Motivation: cos(NPU_w4a16_tfe, CPU_fp32) = 0.36 on fib-p0. Same probe
returns cos=0.9999 for fp16-local vs CPU_fp32. So we know:
- fp16-local path: correct end-to-end.
- w4a16-local path: ~0.6 cos gap to both CPU and (by proxy) fp16-local.

This probe eliminates the CPU reference and compares the two NPU
binaries directly at the logit level. Answers:
1. Is the gap MOSTLY quant error? (cos(fp16_logits, w4a16_logits) ≈ cos(CPU, w4a16))
2. Is the gap consistent per-layer? (compare present_0_key between
   fp16 and dequanted w4a16 first)

Run:
    .venv/Scripts/python.exe scripts/probe_w4a16_vs_fp16_differential.py \
        --w4a16-variant w4a16-local-tfe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load(variant: str, path_key: str, cfg: dict):
    # Re-import npu_load_qwen3_bin under a fresh VARIANT — module globals
    # are computed at import time, so we clear the module cache and set the
    # env var before re-importing.
    os.environ["SPECULA_NPU_VARIANT"] = variant
    for m in list(sys.modules):
        if m.startswith("npu_load_qwen3_bin") or m.startswith("npu_vs_cpu_correctness"):
            del sys.modules[m]
    from npu_load_qwen3_bin import (  # noqa: E402
        IS_LOCAL_W4A16,
        LOGITS_OUTPUT_NAME,
        _encodings_path,
        build_ep_context_wrapper,
        load_wrapper,
        load_quant_specs,
        dequant_from_uint16,
        quant_to_uint16,
    )
    from npu_vs_cpu_correctness import _npu_bin, _npu_wrapper  # noqa: E402

    bin_path = _npu_bin(path_key)
    wrapper = _npu_wrapper(path_key)
    if not wrapper.exists():
        build_ep_context_wrapper(cfg, bin_path, wrapper, path_key)
    sess = load_wrapper(wrapper)
    specs = None
    if IS_LOCAL_W4A16:
        specs = load_quant_specs(_encodings_path(path_key),
                                 [x.name for x in sess.get_inputs()] + [x.name for x in sess.get_outputs()])
    return sess, specs, LOGITS_OUTPUT_NAME, dequant_from_uint16, quant_to_uint16


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w4a16-variant", default="w4a16-local-tfe")
    parser.add_argument("--prompt-idx", type=int, default=0)
    args = parser.parse_args()

    CONFIG_JSON = REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "config.json"
    TOKENIZER_JSON = REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "tokenizer.json"
    HUMANEVAL = REPO_ROOT / "prompts" / "humaneval_subset.jsonl"
    CPU_ONNX = REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "model.onnx"

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    with HUMANEVAL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.prompt_idx:
                prompt = json.loads(line)["prompt"]
                break
    prompt_ids = tok.encode(prompt).ids
    print(f"prompt p{args.prompt_idx}, {len(prompt_ids)} tokens")

    # CPU prefill so both NPU sessions see the same past.
    print("loading CPU ONNX ...")
    os.environ["SPECULA_NPU_CTX"] = "256"
    for m in list(sys.modules):
        if m.startswith("npu_load_qwen3_bin") or m.startswith("npu_vs_cpu_correctness") or m.startswith("npu_short_prompt"):
            del sys.modules[m]
    from npu_short_prompt_probe import cpu_prefill, pad_cpu_past_to_npu, build_masked_bias  # noqa: E402
    from npu_vs_cpu_correctness import load_cpu_session  # noqa: E402
    from npu_load_qwen3_bin import rope_tables, CONTEXT_MAX  # noqa: E402

    cpu = load_cpu_session(CPU_ONNX)
    cpu_past, next_id = cpu_prefill(cpu, cfg, prompt_ids)
    print(f"next_id={next_id}, past_len={len(prompt_ids)}")

    # Build the fp32-interior feed once.
    npu_past = pad_cpu_past_to_npu(cpu_past, len(prompt_ids), cfg)
    feed_fp: dict[str, np.ndarray] = {
        "input_ids": np.array([[next_id]], dtype=np.int32),
        "attention_bias": build_masked_bias(len(prompt_ids)),
    }
    cos_t, sin_t = rope_tables(len(prompt_ids))
    feed_fp["position_ids_cos"] = cos_t
    feed_fp["position_ids_sin"] = sin_t
    feed_fp.update(npu_past)

    # --- fp16-local session ---
    print(f"\nloading fp16-local ...")
    sess_fp16, _, logits_name_fp16, _, _ = _load("fp16-local", "pathb", cfg)
    out_names_fp16 = [o.name for o in sess_fp16.get_outputs()]
    t0 = time.perf_counter_ns()
    outs_fp16 = sess_fp16.run(None, feed_fp)
    t1 = time.perf_counter_ns()
    logits_fp16 = outs_fp16[out_names_fp16.index(logits_name_fp16)][0, -1].astype(np.float32)
    print(f"  fp16-local logits: shape={logits_fp16.shape} "
          f"range=[{logits_fp16.min():.3f}, {logits_fp16.max():.3f}]  {(t1-t0)/1e6:.1f} ms")

    # --- w4a16-local-* session ---
    print(f"\nloading {args.w4a16_variant} ...")
    sess_w, specs_w, logits_name_w, dequant_w, quant_w = _load(args.w4a16_variant, "pathb", cfg)
    out_names_w = [o.name for o in sess_w.get_inputs() + sess_w.get_outputs()]
    # Quantize feed per specs_w
    feed_q = {}
    for name, arr in feed_fp.items():
        sp = specs_w.get(name)
        feed_q[name] = quant_w(arr, sp) if sp is not None else arr
    out_names_w = [o.name for o in sess_w.get_outputs()]
    t0 = time.perf_counter_ns()
    outs_w = sess_w.run(None, feed_q)
    t1 = time.perf_counter_ns()
    logits_w_raw = outs_w[out_names_w.index(logits_name_w)][0, -1]
    logits_w = dequant_w(logits_w_raw, specs_w[logits_name_w]).astype(np.float32)
    print(f"  {args.w4a16_variant} logits: shape={logits_w.shape} "
          f"range=[{logits_w.min():.3f}, {logits_w.max():.3f}]  {(t1-t0)/1e6:.1f} ms")

    # --- Differential ---
    def cos(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    c = cos(logits_fp16, logits_w)
    max_abs = float(np.max(np.abs(logits_fp16 - logits_w)))
    fp16_top5 = np.argsort(-logits_fp16)[:5].tolist()
    w_top5 = np.argsort(-logits_w)[:5].tolist()
    overlap = len(set(fp16_top5) & set(w_top5))
    print(f"\n=== fp16-local vs {args.w4a16_variant} (logits) ===")
    print(f"  cos = {c:.6f}")
    print(f"  max |delta| = {max_abs:.3f}")
    print(f"  fp16 top-5 = {fp16_top5}")
    print(f"  w4a16 top-5 = {w_top5}")
    print(f"  top-5 overlap = {overlap}/5")

    # Walk every layer's present K/V to find where quant error grows past
    # tolerance. Just-the-new-token slot: position 255 (seq_k-1 = CONTEXT_MAX-1).
    out_names_w_list = [o.name for o in sess_w.get_outputs()]
    print(f"\n=== per-layer present K/V cos (fp16 vs {args.w4a16_variant}) — ===")
    print(f"{'layer':>5} {'key_cos':>10} {'value_cos':>10} {'k_maxabs':>10} {'v_maxabs':>10}")
    n_layers = cfg["num_hidden_layers"]
    for i in range(n_layers):
        for kind, col in (("key", "k"), ("value", "v")):
            pass
        # Outputs are [batch=1, n_kv, seq_k, head_dim]; take batch 0 and
        # the new-token slot (seq_k axis idx -1).
        k_fp = outs_fp16[out_names_fp16.index(f"present_{i}_key")][0]
        v_fp = outs_fp16[out_names_fp16.index(f"present_{i}_value")][0]
        k_w = dequant_w(outs_w[out_names_w_list.index(f"present_{i}_key")][0],
                        specs_w[f"present_{i}_key"]).astype(np.float32)
        v_w = dequant_w(outs_w[out_names_w_list.index(f"present_{i}_value")][0],
                        specs_w[f"present_{i}_value"]).astype(np.float32)
        k_fp_new = k_fp[:, -1, :].ravel()
        k_w_new = k_w[:, -1, :].ravel()
        v_fp_new = v_fp[:, -1, :].ravel()
        v_w_new = v_w[:, -1, :].ravel()
        k_cos = cos(k_fp_new, k_w_new)
        v_cos = cos(v_fp_new, v_w_new)
        k_maxabs = float(np.max(np.abs(k_fp_new - k_w_new)))
        v_maxabs = float(np.max(np.abs(v_fp_new - v_w_new)))
        print(f"{i:>5} {k_cos:>10.6f} {v_cos:>10.6f} {k_maxabs:>10.3f} {v_maxabs:>10.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
