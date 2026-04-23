"""Steady-state NPU per-step latency across variants.

The session-18 probes reported ~50 ms first-call for every shotgun
variant, vs ~28-30 ms we had measured historically for baseline
w4a16-local + fp16-local. On battery, single-call, unclear whether
that's cold HTP, thermal, or per-variant real.

This harness loads one variant at a time, does N_WARMUP warm-up runs
to prime the HTP context cache, then reports median/min/max over
N_MEASURE calls. Same inputs across all variants (fib-p0 CPU prefill
+ first decode step at position 16, with quantized feed for w4a16*).

Run:
    .venv\\Scripts\\python.exe scripts\\probe_npu_steady_state_latency.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

N_WARMUP = 5
N_MEASURE = 25

VARIANTS = [
    ("w4a16-local",      "baseline w4a16 (tf)"),
    ("w4a16-local-tfe",  "enhanced activation cal"),
    ("w4a16-local-pr",   "w4 per-row weight quant"),
    ("w4a16-local-mse",  "mse activation cal"),
    ("w8a16-local",      "w8 weights"),
    ("w8a16-local-pr",   "w8 per-row weight quant"),
    ("fp16-local",       "no PTQ (reference)"),
]


def _fresh_import_for(variant: str):
    os.environ["SPECULA_NPU_CTX"] = "256"
    os.environ["SPECULA_NPU_VARIANT"] = variant
    for m in list(sys.modules):
        if m.startswith("npu_"):
            del sys.modules[m]
    from npu_load_qwen3_bin import (  # noqa: E402
        _encodings_path,
        _config_json,
        build_ep_context_wrapper,
        load_wrapper,
        load_quant_specs,
        quant_to_uint16,
        dequant_from_uint16,
        IS_LOCAL_W4A16,
        LOGITS_OUTPUT_NAME,
        rope_tables,
        CONTEXT_MAX,
    )
    from npu_short_prompt_probe import cpu_prefill, pad_cpu_past_to_npu, build_masked_bias  # noqa: E402
    from npu_vs_cpu_correctness import load_cpu_session, CPU_ONNX, CONFIG_JSON, _npu_bin, _npu_wrapper  # noqa: E402
    return dict(
        CPU_ONNX=CPU_ONNX, CONFIG_JSON=CONFIG_JSON, _npu_bin=_npu_bin, _npu_wrapper=_npu_wrapper,
        _encodings_path=_encodings_path, build_ep_context_wrapper=build_ep_context_wrapper,
        load_wrapper=load_wrapper, load_quant_specs=load_quant_specs,
        quant_to_uint16=quant_to_uint16, dequant_from_uint16=dequant_from_uint16,
        IS_LOCAL_W4A16=IS_LOCAL_W4A16, LOGITS_OUTPUT_NAME=LOGITS_OUTPUT_NAME,
        rope_tables=rope_tables, CONTEXT_MAX=CONTEXT_MAX,
        cpu_prefill=cpu_prefill, pad_cpu_past_to_npu=pad_cpu_past_to_npu,
        build_masked_bias=build_masked_bias, load_cpu_session=load_cpu_session,
    )


def main() -> int:
    # CPU prefill once at the start; reuse past across every variant.
    api = _fresh_import_for("w8a16-local")  # just to grab CPU reference paths
    with api["CONFIG_JSON"].open() as f:
        cfg = json.load(f)
    HUMANEVAL = REPO / "prompts" / "humaneval_subset.jsonl"
    with HUMANEVAL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                prompt = json.loads(line)["prompt"]
                break
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(REPO / "models" / "qwen3-0.6b-optimum" / "tokenizer.json"))
    prompt_ids = tok.encode(prompt).ids
    print(f"fib-p0 prompt_len={len(prompt_ids)}, running {N_WARMUP} warmup + {N_MEASURE} measured calls per variant\n")

    print("loading CPU ONNX ...")
    cpu_sess = api["load_cpu_session"](api["CPU_ONNX"])
    cpu_past, next_id = api["cpu_prefill"](cpu_sess, cfg, prompt_ids)
    del cpu_sess

    print(f"{'variant':24s} {'median':>8s} {'min':>8s} {'max':>8s} {'p95':>8s}")
    for variant, desc in VARIANTS:
        api = _fresh_import_for(variant)
        bin_path = api["_npu_bin"]("pathb")
        if not bin_path.exists():
            print(f"{variant:24s} SKIP (missing binary)")
            continue
        wrapper = api["_npu_wrapper"]("pathb")
        if not wrapper.exists():
            api["build_ep_context_wrapper"](cfg, bin_path, wrapper, "pathb")
        sess = api["load_wrapper"](wrapper)
        in_names = [x.name for x in sess.get_inputs()]
        out_names = [x.name for x in sess.get_outputs()]
        specs = None
        if api["IS_LOCAL_W4A16"]:
            specs = api["load_quant_specs"](api["_encodings_path"]("pathb"), in_names + out_names)

        # Build the fp32 feed for a fresh step at position len(prompt_ids)
        npu_past = api["pad_cpu_past_to_npu"](cpu_past, len(prompt_ids), cfg)
        feed_fp: dict[str, np.ndarray] = {
            "input_ids": np.array([[next_id]], dtype=np.int32),
            "attention_bias": api["build_masked_bias"](len(prompt_ids)),
        }
        cos_t, sin_t = api["rope_tables"](len(prompt_ids))
        feed_fp["position_ids_cos"] = cos_t
        feed_fp["position_ids_sin"] = sin_t
        feed_fp.update(npu_past)

        # Quantize once if needed — steady-state comparison is on session.run(), not per-step quant.
        if specs is not None:
            feed = {n: (api["quant_to_uint16"](a, specs[n]) if n in specs else a) for n, a in feed_fp.items()}
        else:
            feed = feed_fp

        # Warmup then measure
        for _ in range(N_WARMUP):
            sess.run(None, feed)
        timings = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter_ns()
            sess.run(None, feed)
            timings.append((time.perf_counter_ns() - t0) / 1e6)
        timings.sort()
        median = timings[len(timings) // 2]
        mn = timings[0]
        mx = timings[-1]
        p95 = timings[int(len(timings) * 0.95)]
        print(f"{variant:24s} {median:7.2f}ms {mn:7.2f}ms {mx:7.2f}ms {p95:7.2f}ms  {desc}")

        del sess

    return 0


if __name__ == "__main__":
    sys.exit(main())
