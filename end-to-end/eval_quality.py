"""Numerical quality-evaluation harness for an AIMET-quantized pathb bundle.

Track 0 of `docs/e2e_optimizations.md`. The other optimization tracks
(selective fp32 unquantize, op-config alignment, RMSNorm fusion, ...)
all need a *trusted* metric to A/B against; this script is that metric.

What it does
------------
Given an AIMET stage-6 output dir — the one holding the exported
QuantSim graph::

    <aimet-dir>/<prefix>.onnx        # clean FP graph (no QDQ embedded)
    <aimet-dir>/<prefix>.encodings   # AIMET 1.0.0 list-schema encodings
    <aimet-dir>/<prefix>.data        # external weights

it evaluates quantized-vs-FP first-decode quality over a SET of short
oracle prompts (not the single prompt the in-pipeline probe uses):

  - first-decode logit cosine similarity  cos(fp_logits, q_logits)
  - top-1 argmax agreement (and the decoded tokens)

reported per-prompt and aggregated (mean / min cos, argmax match rate).

How it builds the two sessions
------------------------------
This mirrors `lib/aimet.py`'s stage-9 probe:

  * FP session  — a plain `onnxruntime.InferenceSession` over the
    exported `<prefix>.onnx`. AIMET's `sim.export()` writes the *clean*
    FP graph (no QuantizeLinear/DequantizeLinear nodes), so running it
    directly == the FP reference.
  * Quantized session — rebuild a `QuantizationSimModel` from that same
    clean ONNX, then `load_encodings_to_sim()` the exported
    `.encodings`. `sim.session` is then the fake-quantized graph, i.e.
    exactly what stage-9's probe ran against `sim.session` in-process.

The AR=1 decode loop, RoPE cache, and pathb attention-bias / KV-cache
layout are all copied from the stage-9 probe so the numbers stay
directly comparable to the `cos_fp_q` written into `aimet_info.json`.

Usage
-----
    source /workspace/venvs/aimet-2.26-cu121-py310/bin/activate
    python end-to-end/eval_quality.py \
        --aimet-dir /workspace/runs/qwen3_4b_w4a16/06_aimet_w8a16 \
        --export-prefix qwen3-4b_pathb_w8a16 \
        --model-path /workspace/models/Qwen3-4B \
        --precision w8a16 --ctx 512 \
        --report-out /workspace/runs/eval_quality_4b_w8a16.json

It is also importable: call `eval_quality(...)` for the structured
report dict, or `format_markdown(report)` for the human summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


# --- env bootstrap (must run before importing onnxruntime / aimet) -------
# Mirrors /workspace/runs/run_pipeline.sh: the CUDA / cuDNN shared libs
# ship inside the venv's nvidia/* packages and the process needs them on
# LD_LIBRARY_PATH or the onnxruntime-gpu CUDA EP fails to load. This is a
# no-op when the libs are already discoverable (e.g. CPU-only run, or the
# launcher script already exported it).
def _ensure_nvidia_libs() -> None:
    venv = os.environ.get("VIRTUAL_ENV") or sys.prefix
    nvidia_root = Path(venv) / "lib" / "python3.10" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return
    lib_dirs = [str(p) for p in nvidia_root.glob("*/lib") if p.is_dir()]
    if not lib_dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    have = set(cur.split(":")) if cur else set()
    new = [d for d in lib_dirs if d not in have]
    if new:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new + ([cur] if cur else []))


_ensure_nvidia_libs()

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402


# --- oracle prompt set ---------------------------------------------------
# Diverse short prompts: factual recall, arithmetic, code, prose,
# structured text. The first is the canonical stage-9 probe prompt so a
# direct apples-to-apples comparison with `aimet_info.json`'s cos_fp_q is
# always available in the per-prompt table.
ORACLE_PROMPTS: list[str] = [
    "The capital of France is",            # stage-9 probe prompt (anchor)
    "2 + 2 =",                             # arithmetic
    "The opposite of hot is",              # simple semantics
    "def add(a, b):\n    return",          # python code
    "Water is made of hydrogen and",       # factual recall
    "Once upon a time, there was a",       # narrative prose
    "The first president of the United States was",  # factual recall
    "import numpy as np\narr = np.",       # code continuation
    "Roses are red, violets are",          # completion / idiom
    "The largest planet in our solar system is",  # factual recall
]


# --- helpers (copied / adapted from lib/aimet.py stage-9 probe) ----------
def _logits_name(sess: ort.InferenceSession) -> str:
    for o in sess.get_outputs():
        if "logit" in o.name.lower():
            return o.name
    return sess.get_outputs()[0].name


def _present_in_order(sess: ort.InferenceSession, n_layers: int) -> list[str]:
    outs = [o.name for o in sess.get_outputs()]
    ks = sorted([n for n in outs if "present" in n and ".key" in n],
                key=lambda x: int(''.join(c for c in x.split(".key")[0] if c.isdigit())))
    vs = sorted([n for n in outs if "present" in n and ".value" in n],
                key=lambda x: int(''.join(c for c in x.split(".value")[0] if c.isdigit())))
    out: list[str] = []
    for i in range(n_layers):
        out.append(ks[i])
        out.append(vs[i])
    return out


def _decode_last(sess: ort.InferenceSession, token_ids: list[int], *,
                  n_layers: int, ctx: int, rope_cos: np.ndarray,
                  rope_sin: np.ndarray) -> np.ndarray:
    """Run an AR=1 decode over `token_ids`; return last-position logits.

    Verbatim pathb cache + attention-bias layout from the stage-9 probe:
    current token at slot (ctx-1), past KVs grow backward, the unused
    front slots are masked with -65504.
    """
    past_names = [i.name for i in sess.get_inputs()
                  if i.name.startswith("past_key_values.")]
    kv_shape = {i.name: [d if isinstance(d, int) else 1 for d in i.shape]
                for i in sess.get_inputs()
                if i.name.startswith("past_key_values.")}
    logits_name = _logits_name(sess)
    present_names = _present_in_order(sess, n_layers)
    all_outs = [logits_name] + present_names
    past = {n: np.zeros(kv_shape[n], dtype=np.float32) for n in past_names}
    last = None
    for pos, t in enumerate(token_ids):
        input_ids = np.array([[t]], dtype=np.int64)
        position_ids = np.array([[pos]], dtype=np.int64)
        ab = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
        ab[..., ctx - 1 - pos:] = 0.0
        feeds = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_bias": ab,
            "position_ids_cos": rope_cos[pos:pos + 1][None, ...].astype(np.float32),
            "position_ids_sin": rope_sin[pos:pos + 1][None, ...].astype(np.float32),
        }
        feeds.update(past)
        outs = sess.run(all_outs, feeds)
        last = outs[0]
        past = {p_n: outs[1 + i][..., -(ctx - 1):, :].astype(np.float32, copy=False)
                for i, p_n in enumerate(past_names)}
    return last


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    f = a.flatten().astype(np.float64)
    q = b.flatten().astype(np.float64)
    return float(np.dot(f, q) / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-12))


# --- session construction ------------------------------------------------
def _build_fp_session(onnx_path: Path,
                      providers: list[str]) -> ort.InferenceSession:
    """Plain ORT session over the exported clean FP graph."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), providers=providers,
                                sess_options=so)


def _build_quant_session(onnx_path: Path, encodings_path: Path, *,
                         precision: str, providers: list[str]):
    """Rebuild the AIMET QuantSim from the exported clean ONNX + encodings.

    Returns the `QuantizationSimModel`; `.session` is the fake-quantized
    graph — identical to the `sim.session` the stage-9 probe evaluated.
    """
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
    from aimet_onnx.common.defs import QuantScheme

    if precision == "w8a16":
        param_type, activation_type = "int8", "int16"
    elif precision == "w4a16":
        param_type, activation_type = "int4", "int16"
    else:
        raise ValueError(f"unknown precision {precision!r} (want w8a16 / w4a16)")

    model_proto = onnx.load(str(onnx_path), load_external_data=True)
    sim = QuantizationSimModel(
        model_proto,
        param_type=param_type,
        activation_type=activation_type,
        quant_scheme=QuantScheme.min_max,
        providers=providers,
    )
    # Load the exported encodings onto the freshly-built quantizers.
    # strict=False lets quantizer settings realign to the encodings
    # (the export may carry per-tensor / bitwidth overrides — e.g. the
    # V/O w8 pin or embedding int16 pin — that a default-built QSM does
    # not reproduce). disable_missing_quantizers mirrors a real export:
    # any quantizer with no encoding is turned off.
    load_encodings_to_sim(sim, str(encodings_path), strict=False,
                          disable_missing_quantizers=True)
    return sim


# --- main eval -----------------------------------------------------------
def eval_quality(*, aimet_dir: Path, export_prefix: str, model_path: Path,
                 precision: str, ctx: int, prompts: Optional[list[str]] = None,
                 cuda: bool = True) -> dict:
    """Evaluate quantized-vs-FP first-decode quality.

    Returns a structured report dict (also JSON-serialisable).
    """
    aimet_dir = Path(aimet_dir)
    model_path = Path(model_path)
    prompts = list(prompts) if prompts else list(ORACLE_PROMPTS)

    onnx_path = aimet_dir / f"{export_prefix}.onnx"
    encodings_path = aimet_dir / f"{export_prefix}.encodings"
    if not onnx_path.exists():
        raise FileNotFoundError(f"exported QuantSim ONNX not found: {onnx_path}")
    if not encodings_path.exists():
        raise FileNotFoundError(f"AIMET encodings not found: {encodings_path}")

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if cuda else ["CPUExecutionProvider"])

    # tokenizer + config + RoPE — same recipe as lib/aimet.py stage 1.
    from transformers import AutoTokenizer, AutoConfig
    from lib.rope import build_rope_cache

    tok = AutoTokenizer.from_pretrained(str(model_path))
    cfg = AutoConfig.from_pretrained(str(model_path))
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    rope_theta = float(cfg.rope_theta)
    n_layers = int(cfg.num_hidden_layers)
    rope_cos, rope_sin = build_rope_cache(rope_theta, head_dim, ctx + 64)

    report: dict = {
        "aimet_dir": str(aimet_dir),
        "export_prefix": export_prefix,
        "model_path": str(model_path),
        "precision": precision,
        "ctx": ctx,
        "onnx_path": str(onnx_path),
        "encodings_path": str(encodings_path),
        "n_layers": n_layers,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "n_prompts": len(prompts),
        "prompts": [],
    }

    t0 = time.time()
    print(f"[eval] building FP session over {onnx_path.name} ...", flush=True)
    fp_sess = _build_fp_session(onnx_path, providers)
    report["fp_providers"] = fp_sess.get_providers()
    print(f"[eval] FP session built ({time.time() - t0:.1f}s) "
          f"providers={fp_sess.get_providers()}", flush=True)

    t0 = time.time()
    print(f"[eval] rebuilding QuantSim + loading encodings "
          f"({encodings_path.name}) ...", flush=True)
    sim = _build_quant_session(onnx_path, encodings_path,
                               precision=precision, providers=providers)
    q_sess = sim.session
    print(f"[eval] quantized session ready ({time.time() - t0:.1f}s)", flush=True)

    # Per-prompt first-decode cos + argmax.
    per_prompt: list[dict] = []
    for idx, text in enumerate(prompts):
        ids = tok(text, return_tensors="np").input_ids[0].tolist()
        t0 = time.time()
        fp_logits = _decode_last(fp_sess, ids, n_layers=n_layers, ctx=ctx,
                                 rope_cos=rope_cos, rope_sin=rope_sin)
        q_logits = _decode_last(q_sess, ids, n_layers=n_layers, ctx=ctx,
                                rope_cos=rope_cos, rope_sin=rope_sin)
        cos = _cosine(fp_logits, q_logits)
        fp_arg = int(np.argmax(fp_logits[0, 0]))
        q_arg = int(np.argmax(q_logits[0, 0]))
        fp_tok = tok.decode([fp_arg])
        q_tok = tok.decode([q_arg])
        match = fp_arg == q_arg
        entry = {
            "prompt": text,
            "n_tokens": len(ids),
            "cos_fp_q": cos,
            "fp_argmax": fp_arg,
            "q_argmax": q_arg,
            "fp_token": fp_tok,
            "q_token": q_tok,
            "argmax_match": match,
            "wall_s": time.time() - t0,
        }
        per_prompt.append(entry)
        print(f"[eval] prompt {idx + 1}/{len(prompts)}  cos={cos:.6f}  "
              f"argmax_match={match}  fp={fp_tok!r} q={q_tok!r}  "
              f"({entry['wall_s']:.1f}s)", flush=True)

    report["prompts"] = per_prompt

    cos_vals = [p["cos_fp_q"] for p in per_prompt]
    matches = [p["argmax_match"] for p in per_prompt]
    report["aggregate"] = {
        "mean_cos": float(np.mean(cos_vals)),
        "min_cos": float(np.min(cos_vals)),
        "max_cos": float(np.max(cos_vals)),
        "median_cos": float(np.median(cos_vals)),
        "std_cos": float(np.std(cos_vals)),
        "argmax_match_rate": float(np.mean(matches)),
        "argmax_match_count": int(np.sum(matches)),
        "n_prompts": len(per_prompt),
    }
    # cos >= 0.99 is the success gate from docs/e2e_optimizations.md.
    report["gate"] = {
        "cos_threshold": 0.99,
        "mean_cos_pass": report["aggregate"]["mean_cos"] >= 0.99,
        "min_cos_pass": report["aggregate"]["min_cos"] >= 0.99,
    }
    return report


# --- markdown report -----------------------------------------------------
def format_markdown(report: dict) -> str:
    agg = report["aggregate"]
    gate = report["gate"]
    lines: list[str] = []
    lines.append("# Quantized-vs-FP quality evaluation\n")
    lines.append(f"- aimet dir   : `{report['aimet_dir']}`")
    lines.append(f"- export prefix: `{report['export_prefix']}`")
    lines.append(f"- model path  : `{report['model_path']}`")
    lines.append(f"- precision   : `{report['precision']}`   ctx: `{report['ctx']}`"
                 f"   layers: `{report['n_layers']}`")
    lines.append(f"- FP providers: `{report.get('fp_providers')}`")
    lines.append("")
    lines.append("## Per-prompt first-decode quality\n")
    lines.append("| # | prompt | tok | cos(fp,q) | argmax match | fp tok | q tok |")
    lines.append("|--:|---|--:|--:|:--:|---|---|")
    for i, p in enumerate(report["prompts"], start=1):
        short = p["prompt"].replace("\n", "\\n")
        if len(short) > 38:
            short = short[:35] + "..."
        mark = "yes" if p["argmax_match"] else "NO"
        lines.append(f"| {i} | `{short}` | {p['n_tokens']} | "
                     f"{p['cos_fp_q']:.4f} | {mark} | "
                     f"`{p['fp_token']!r}` | `{p['q_token']!r}` |")
    lines.append("")
    lines.append("## Aggregate\n")
    lines.append(f"- mean cos     : **{agg['mean_cos']:.4f}**")
    lines.append(f"- median cos   : {agg['median_cos']:.4f}")
    lines.append(f"- min / max cos: {agg['min_cos']:.4f} / {agg['max_cos']:.4f}")
    lines.append(f"- cos std      : {agg['std_cos']:.4f}")
    lines.append(f"- argmax match : {agg['argmax_match_count']}/{agg['n_prompts']} "
                 f"({agg['argmax_match_rate'] * 100:.1f}%)")
    lines.append("")
    lines.append("## Gate (cos >= 0.99)\n")
    mean_pass = "PASS" if gate["mean_cos_pass"] else "FAIL"
    min_pass = "PASS" if gate["min_cos_pass"] else "FAIL"
    lines.append(f"- mean cos >= 0.99 : **{mean_pass}**")
    lines.append(f"- min  cos >= 0.99 : **{min_pass}**")
    lines.append("")
    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Evaluate quantized-vs-FP first-decode logit quality "
                    "for an AIMET stage-6 output bundle.")
    p.add_argument("--aimet-dir", type=Path, required=True,
                   help="AIMET stage-6 output dir holding "
                        "<prefix>.onnx + .encodings + .data")
    p.add_argument("--export-prefix", required=True,
                   help="filename prefix of the exported QuantSim graph "
                        "(e.g. qwen3-4b_pathb_w8a16)")
    p.add_argument("--model-path", type=Path, required=True,
                   help="HF model dir for tokenizer + config "
                        "(e.g. /workspace/models/Qwen3-4B)")
    p.add_argument("--precision", choices=["w8a16", "w4a16"], required=True,
                   help="quantization precision the bundle was built at")
    p.add_argument("--ctx", type=int, default=512,
                   help="pathb context length the graph was pinned to")
    p.add_argument("--report-out", type=Path, default=None,
                   help="optional path to write the JSON report to "
                        "(a sibling .md is written alongside)")
    p.add_argument("--no-cuda", action="store_true",
                   help="force CPU execution provider only")
    args = p.parse_args(argv)

    report = eval_quality(
        aimet_dir=args.aimet_dir,
        export_prefix=args.export_prefix,
        model_path=args.model_path,
        precision=args.precision,
        ctx=args.ctx,
        cuda=not args.no_cuda,
    )

    md = format_markdown(report)
    print()
    print(md)

    if args.report_out:
        out = args.report_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        md_out = out.with_suffix(".md")
        md_out.write_text(md)
        print(f"[wrote {out}]")
        print(f"[wrote {md_out}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
