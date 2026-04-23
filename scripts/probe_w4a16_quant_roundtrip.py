"""Diagnostic: is the w4a16-local quant layer itself broken, or is the model?

The correctness probe on fib-p0 returned cos_sim=0.33 — too low to be
pure quant noise on well-calibrated uint16 (relative error should be
<0.1% per tensor). Possibilities:

  1. Our quant/dequant formula doesn't match what the binary expects.
  2. encodings.json scale/offset values aren't what the binary baked in.
  3. Some specific input (attention_bias, cos/sin) is being misfed.
  4. Multi-tensor compound error is worse than expected.

This script runs quant->dequant round trips on the ACTUAL fp32 feed
values the probe computed (reconstructed from a CPU prefill), per-tensor,
and reports per-tensor relative error + saturation count. Cheap to run,
tells us whether the problem is the quant layer or something downstream.

Run:
    SPECULA_NPU_CTX=256 SPECULA_NPU_VARIANT=w4a16-local \
        .venv/Scripts/python.exe scripts/probe_w4a16_quant_roundtrip.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import (  # noqa: E402
    CONTEXT_MAX,
    _encodings_path,
    dequant_tensor,
    load_quant_specs,
    quant_tensor,
    rope_tables,
)
from npu_short_prompt_probe import build_masked_bias, cpu_prefill, load_prompt  # noqa: E402
from npu_vs_cpu_correctness import (  # noqa: E402
    CONFIG_JSON,
    CPU_ONNX,
    TOKENIZER_JSON,
    load_cpu_session,
)


def _roundtrip_stats(name: str, arr: np.ndarray, spec) -> dict:
    # Per-tensor dispatch so full-quant-IO variants round-trip uint8
    # past_kv and uint16 everything-else through the right codec.
    q = quant_tensor(arr, spec)
    back = dequant_tensor(q, spec)
    diff = arr - back
    rng = max(abs(float(arr.min())), abs(float(arr.max())), 1e-9)
    at_min = int((q == 0).sum())
    at_max = int((q == spec.qmax).sum())
    return {
        "name": name,
        "shape": arr.shape,
        "bitwidth": spec.bitwidth,
        "fp32_min": float(arr.min()),
        "fp32_max": float(arr.max()),
        "cal_min": (0 + spec.offset) * spec.scale,
        "cal_max": (spec.qmax + spec.offset) * spec.scale,
        "max_abs_err": float(np.max(np.abs(diff))),
        "rel_rms_err": float(np.sqrt(np.mean(diff * diff)) / rng),
        "saturated_lo_pct": 100.0 * at_min / arr.size,
        "saturated_hi_pct": 100.0 * at_max / arr.size,
    }


def main() -> int:
    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    prompt = load_prompt(0)
    prompt_ids = tok.encode(prompt).ids
    prompt_len = len(prompt_ids)
    print(f"prompt p0, {prompt_len} tokens")

    print("loading CPU ONNX ...")
    cpu_sess = load_cpu_session(CPU_ONNX)
    cpu_past, next_id = cpu_prefill(cpu_sess, cfg, prompt_ids)
    print(f"CPU prefill OK, next_id={next_id}")

    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_len = CONTEXT_MAX - 1
    pad_slots = past_len - prompt_len

    # Rebuild what the probe feeds to the NPU for the first step.
    feed: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        k = cpu_past[f"past_key_values.{i}.key"]
        v = cpu_past[f"past_key_values.{i}.value"]
        pad = np.zeros((1, n_kv, pad_slots, head_dim), dtype=np.float32)
        feed[f"past_key_values_{i}_key"] = np.concatenate([k.astype(np.float32), pad], axis=2)
        feed[f"past_key_values_{i}_value"] = np.concatenate([v.astype(np.float32), pad], axis=2)
    feed["attention_bias"] = build_masked_bias(prompt_len)
    cos, sin = rope_tables(prompt_len)
    feed["position_ids_cos"] = cos
    feed["position_ids_sin"] = sin

    print("loading quant specs ...")
    enc_path = _encodings_path("pathb")
    specs = load_quant_specs(enc_path, list(feed.keys()))

    print("\nper-tensor round-trip stats:")
    print(f"{'name':32s} {'fp32_min':>10s} {'fp32_max':>10s} {'cal_lo':>10s} {'cal_hi':>10s} "
          f"{'maxabs':>10s} {'relRMS%':>9s} {'satLo%':>7s} {'satHi%':>7s}")
    results = []
    for name, arr in feed.items():
        spec = specs[name]
        s = _roundtrip_stats(name, arr, spec)
        results.append(s)
        show = (
            s["saturated_lo_pct"] > 0.1
            or s["saturated_hi_pct"] > 0.1
            or s["rel_rms_err"] > 0.01
            or name in ("attention_bias", "position_ids_cos", "position_ids_sin")
            or "_0_" in name
        )
        if show:
            print(
                f"{name:32s} "
                f"{s['fp32_min']:10.4f} {s['fp32_max']:10.4f} "
                f"{s['cal_min']:10.4f} {s['cal_max']:10.4f} "
                f"{s['max_abs_err']:10.4g} {100 * s['rel_rms_err']:9.4f} "
                f"{s['saturated_lo_pct']:7.2f} {s['saturated_hi_pct']:7.2f}"
            )

    # Aggregate: worst-case saturated past_kv layer (decode layer 0 is famous
    # for large activation ranges in transformer nets)
    worst_sat = max(r["saturated_lo_pct"] + r["saturated_hi_pct"] for r in results)
    worst_err = max(r["rel_rms_err"] for r in results)
    print(f"\nworst saturation pct = {worst_sat:.2f}%   worst rel RMS err = {worst_err * 100:.3f}%")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
