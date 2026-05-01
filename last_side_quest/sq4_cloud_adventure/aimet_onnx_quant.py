"""SQ4 M1 step 5 — aimet_onnx PTQ on the pathb-pinned ONNX.

Companion to ``aimet_quant.py`` (the torch driver used for runs 1a/d/e).
This script switches to ``aimet_onnx.QuantizationSimModel`` so we can
quantize the **pathb-rewritten ONNX directly** — bypassing the
P1 tensor-name mismatch that caused 80% float-fallback in the
qairt-converter run during the m1e end-to-end attempt.

Pipeline:

    pathb-pinned ONNX  (model.onnx + model.data, AR=1 ctx=512)
        ↓
    onnxruntime FP session — generate real calibration tuples by
    walking calibration prompts token-by-token (real KVs, real
    position_ids_cos/sin from the Qwen3 RoPE recipe with rope_theta).
        ↓
    aimet_onnx.QuantizationSimModel (w8a16 default, --precision flag
    flips to w4a16)
        ↓
    compute_encodings over the pre-collected cal samples
        ↓
    sim.export → ONNX with Q/DQ + encodings.json next to it

Usage:

    python aimet_onnx_quant.py \\
      --src-dir /workspace/sq4_m1_pathb/qwen3-0.6b-pathb-ctx512 \\
      --tokenizer /workspace/models/Qwen3-0.6B \\
      --output-dir runs/m1_pathb_w8a16_aimet_onnx \\
      --precision w8a16 \\
      --num-cal-samples 128

Cal data is gathered on-the-fly from a small bundled prompt set so
we don't depend on HF datasets. ~128 samples × 1 forward at AR=1
takes <30 s on A40.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]


# Small cal prompt bank — mix of code, prose, dialogue, structured
# output. Kept inline so we never depend on HF datasets at run time.
CAL_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps in the sun.",
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Q: What is the capital of France?\nA: The capital of France is Paris.",
    "import numpy as np\narr = np.zeros((3, 4), dtype=np.float32)\nprint(arr.shape)",
    "In machine learning, gradient descent is an iterative optimization algorithm.",
    "The user said: hello, can you help me debug this Python script please?",
    "<html>\n  <body>\n    <h1>Welcome</h1>\n    <p>Hello world</p>\n  </body>\n</html>",
    "{\"name\": \"alice\", \"age\": 30, \"items\": [\"apple\", \"banana\", \"cherry\"]}",
    "She walked through the silent forest, listening to the wind in the leaves.",
    "Theorem: Every prime number greater than 2 is odd. Proof: suppose p is prime and even.",
    "git commit -m 'fix: handle empty input gracefully'\ngit push origin main",
    "The matrix product AB is defined when the columns of A equal the rows of B.",
    "User: How do I sort a list?\nAssistant: Use the sorted() built-in or list.sort() in place.",
    "After the rain stopped, a rainbow appeared above the distant mountains.",
    "TODO: refactor authentication to use JWT instead of session cookies for the API",
    "She opened the letter and read the first line three times before continuing.",
]


def build_rope_cache(rope_theta: float, head_dim: int, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard Qwen3 RoPE cos/sin cache, FP32, shape [max_pos, head_dim]."""
    # Half-dim inverse frequencies.
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_pos, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)  # [max_pos, half]
    # Match transformers convention: [pos, head_dim] = concat(freqs, freqs)
    emb = np.concatenate([freqs, freqs], axis=-1)  # [max_pos, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def make_zero_feeds(sess: ort.InferenceSession) -> dict[str, np.ndarray]:
    """One-shot zero-filled feeds in the right shapes (KV all zeros)."""
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.type in ("tensor(int64)",):
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def collect_kv_output_names(sess: ort.InferenceSession) -> list[str]:
    """Return the present.<i>.<key|value> output names in graph order."""
    names = [o.name for o in sess.get_outputs()
             if o.name.startswith("present.")]
    return names


def kv_input_names(sess: ort.InferenceSession) -> list[str]:
    return [i.name for i in sess.get_inputs()
            if i.name.startswith("past_key_values.")]


def cal_iter(
    sess: ort.InferenceSession,
    tokenizer,
    prompts: list[str],
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    ctx: int,
    max_samples: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Walk prompts step-by-step; yield (input feeds) at each AR=1 decode.

    Each yielded dict is a complete input set valid for the pinned graph
    at AR=1 ctx=512. KV cache grows position-by-position as we consume
    `present.*` outputs and stuff them into the next step's
    `past_key_values.*` inputs.
    """
    head_dim = rope_cos.shape[-1]
    past_names = kv_input_names(sess)
    present_names = collect_kv_output_names(sess)
    assert len(past_names) == len(present_names), (
        f"input/output KV mismatch: {len(past_names)} vs {len(present_names)}"
    )

    # KV input shape from the graph.
    kv_shape: dict[str, list[int]] = {}
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values."):
            kv_shape[inp.name] = [d if isinstance(d, int) else 1 for d in inp.shape]

    yielded = 0
    for prompt in prompts:
        if yielded >= max_samples:
            return
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()
        if len(ids) > ctx - 1:
            ids = ids[: ctx - 1]
        # Reset KV
        past = {n: np.zeros(kv_shape[n], dtype=np.float32) for n in past_names}
        for pos, tok in enumerate(ids):
            if pos >= ctx - 1 or yielded >= max_samples:
                break
            input_ids = np.array([[tok]], dtype=np.int64)
            position_ids = np.array([[pos]], dtype=np.int64)
            # Additive attention bias: zeros in the live range, -inf in the future.
            # Pathb cache layout: current token lands at slot (ctx-1);
            # real past KVs grow backward from slot (ctx-2). Slots
            # 0..(ctx-2-pos) are uninitialised zeros — mask them.
            attn_bias = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
            visible_start = ctx - 1 - pos
            attn_bias[..., visible_start:] = 0.0
            cos_step = rope_cos[pos: pos + 1][None, ...]  # [1,1,head_dim]
            sin_step = rope_sin[pos: pos + 1][None, ...]
            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_bias": attn_bias,
                "position_ids_cos": cos_step.astype(np.float32),
                "position_ids_sin": sin_step.astype(np.float32),
            }
            feeds.update(past)
            yield feeds
            yielded += 1
            # Step forward: run the FP session to advance KV. This
            # session is the *unquantized* model; its KV is the cleanest
            # cal target we can supply to QSM.
            outs = sess.run(present_names, feeds)
            # Build next step's `past_key_values.*` inputs by trimming
            # `present.*` outputs to the past_len slot. The pinned graph
            # has past_seq_len = ctx-1 = 511, and present_seq_len = ctx
            # = 512 (it concats the new token's KV onto past). For the
            # next step at position pos+1, past_seq_len is again ctx-1
            # = 511, so we slice present[..., -(ctx-1):, :].
            past_len = ctx - 1
            new_past = {}
            for past_name, present_arr in zip(past_names, outs):
                # present is [batch, kv_heads, present_seq_len, head_dim]
                # take last past_len positions for next step's past
                trimmed = present_arr[..., -past_len:, :]
                new_past[past_name] = trimmed.astype(np.float32, copy=False)
            past = new_past


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path, required=True,
                   help="pathb-pinned ONNX directory (model.onnx + model.data)")
    p.add_argument("--tokenizer", type=Path, required=True,
                   help="HF tokenizer dir (uses model_id config for rope_theta if needed)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--precision", choices=("w4a16", "w8a16"), default="w8a16")
    p.add_argument("--num-cal-samples", type=int, default=128)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--quant-scheme",
                   choices=("min_max", "post_training_tf_enhanced",
                            "post_training_percentile"),
                   default="post_training_tf_enhanced")
    p.add_argument("--export-prefix", default=None,
                   help="filename prefix for sim.export; default derived")
    p.add_argument("--cuda", action="store_true",
                   help="use CUDAExecutionProvider for both FP cal session and QSM session")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest preamble (always written, even if a stage crashes).
    manifest: dict = {
        "src_dir": str(args.src_dir),
        "tokenizer": str(args.tokenizer),
        "output_dir": str(args.output_dir),
        "precision": args.precision,
        "num_cal_samples": args.num_cal_samples,
        "ctx": args.ctx,
        "quant_scheme": args.quant_scheme,
        "argv": sys.argv,
        "stages": {},
    }
    manifest_path = args.output_dir / "manifest.json"

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    # ---- Stage 1: load tokenizer + config ----
    t0 = time.time()
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    cfg = AutoConfig.from_pretrained(str(args.tokenizer))
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    rope_theta = float(cfg.rope_theta)
    print(f"[stage 1] tokenizer + config loaded ({time.time() - t0:.1f}s) "
          f"head_dim={head_dim} rope_theta={rope_theta}")
    manifest["stages"]["1_tokenizer_config"] = {
        "wall_s": time.time() - t0, "head_dim": head_dim, "rope_theta": rope_theta,
    }
    save_manifest()

    # ---- Stage 2: build FP onnxruntime session ----
    t0 = time.time()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.cuda else ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    fp_sess = ort.InferenceSession(str(args.src_dir / "model.onnx"),
                                    providers=providers, sess_options=so)
    actual_providers = fp_sess.get_providers()
    print(f"[stage 2] FP session built ({time.time() - t0:.1f}s) providers={actual_providers}")
    manifest["stages"]["2_fp_session"] = {
        "wall_s": time.time() - t0, "providers": actual_providers,
    }
    save_manifest()

    # ---- Stage 3: gather cal samples ----
    t0 = time.time()
    rope_cos, rope_sin = build_rope_cache(rope_theta, head_dim, args.ctx + 64)
    cal_samples: list[dict[str, np.ndarray]] = []
    n_prompts = 0
    for sample in cal_iter(fp_sess, tok, CAL_PROMPTS, rope_cos, rope_sin,
                           args.ctx, args.num_cal_samples):
        cal_samples.append(sample)
        if len(cal_samples) % 16 == 0:
            print(f"  cal: {len(cal_samples)} samples")
    print(f"[stage 3] gathered {len(cal_samples)} cal samples ({time.time() - t0:.1f}s)")
    manifest["stages"]["3_cal_gather"] = {
        "wall_s": time.time() - t0, "n_samples": len(cal_samples),
    }
    save_manifest()
    # Drop the FP session; AIMET will build its own.
    del fp_sess

    # ---- Stage 4: build QuantSim ----
    t0 = time.time()
    from aimet_onnx.quantsim import QuantizationSimModel
    from aimet_onnx.common.defs import QuantScheme

    if args.precision == "w8a16":
        param_type = "int8"
        activation_type = "int16"
    elif args.precision == "w4a16":
        param_type = "int4"
        activation_type = "int16"
    else:
        raise ValueError(f"unknown precision {args.precision}")

    quant_scheme_map = {
        "min_max": QuantScheme.min_max,
        "post_training_tf_enhanced": QuantScheme.post_training_tf_enhanced,
        "post_training_percentile": QuantScheme.post_training_percentile,
    }
    qsim_providers = providers
    print(f"[stage 4] building QuantSim "
          f"param={param_type} activation={activation_type} scheme={args.quant_scheme} ...")
    model_proto = onnx.load(str(args.src_dir / "model.onnx"), load_external_data=True)
    sim = QuantizationSimModel(
        model_proto,
        param_type=param_type,
        activation_type=activation_type,
        quant_scheme=quant_scheme_map[args.quant_scheme],
        providers=qsim_providers,
    )
    print(f"[stage 4] QuantSim built ({time.time() - t0:.1f}s)")
    manifest["stages"]["4_qsim_build"] = {
        "wall_s": time.time() - t0,
        "param_type": param_type, "activation_type": activation_type,
    }
    save_manifest()

    # ---- Stage 5: compute encodings ----
    t0 = time.time()
    sim.compute_encodings(cal_samples)
    print(f"[stage 5] compute_encodings ({time.time() - t0:.1f}s)")
    manifest["stages"]["5_compute_encodings"] = {"wall_s": time.time() - t0}
    save_manifest()

    # ---- Stage 6: export ONNX + encodings (BEFORE probe, so the artifact
    # lands even if the probe stage crashes) ----
    t0 = time.time()
    prefix = args.export_prefix or f"qwen3_0p6b_pathb_{args.precision}"
    sim.export(str(args.output_dir), filename_prefix=prefix)
    print(f"[stage 6] sim.export → {args.output_dir}/{prefix}.onnx + .encodings ({time.time() - t0:.1f}s)")
    manifest["stages"]["6_export"] = {
        "wall_s": time.time() - t0, "prefix": prefix,
    }
    save_manifest()

    # ---- Stage 7: cos probe vs FP on a held-out prompt (informational; failure non-fatal) ----
    try:
        t0 = time.time()
        probe_text = "The capital of France is"
        probe_ids = tok(probe_text, return_tensors="np").input_ids[0].tolist()
        fp_sess = ort.InferenceSession(str(args.src_dir / "model.onnx"),
                                        providers=providers, sess_options=so)
        past_names = kv_input_names(fp_sess)
        kv_shape: dict[str, list[int]] = {
            i.name: [d if isinstance(d, int) else 1 for d in i.shape]
            for i in fp_sess.get_inputs() if i.name.startswith("past_key_values.")
        }

        def _logits_output_name(sess):
            for o in sess.get_outputs():
                if o.name.endswith("logits") or "logit" in o.name.lower():
                    return o.name
            return sess.get_outputs()[0].name

        def _present_output_names(sess, n_layers: int):
            outs = [o.name for o in sess.get_outputs()]
            ks = [n for n in outs if "present" in n and ".key" in n]
            vs = [n for n in outs if "present" in n and ".value" in n]
            ks.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split(".key")[0]))))
            vs.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split(".value")[0]))))
            present = []
            for i in range(n_layers):
                present.append(ks[i] if i < len(ks) else None)
                present.append(vs[i] if i < len(vs) else None)
            return present

        n_layers = cfg.num_hidden_layers

        def _decode_last_logits(sess):
            logits_name = _logits_output_name(sess)
            present_names = _present_output_names(sess, n_layers)
            all_outs = [logits_name] + present_names
            past = {n: np.zeros(kv_shape[n], dtype=np.float32) for n in past_names}
            last_logits = None
            for pos, tok_id in enumerate(probe_ids):
                input_ids = np.array([[tok_id]], dtype=np.int64)
                position_ids = np.array([[pos]], dtype=np.int64)
                attn_bias = np.full((1, 1, 1, args.ctx), -65504.0, dtype=np.float32)
                visible_start = args.ctx - 1 - pos
                attn_bias[..., visible_start:] = 0.0
                cos_step = rope_cos[pos: pos + 1][None, ...].astype(np.float32)
                sin_step = rope_sin[pos: pos + 1][None, ...].astype(np.float32)
                feeds = {
                    "input_ids": input_ids, "position_ids": position_ids,
                    "attention_bias": attn_bias,
                    "position_ids_cos": cos_step, "position_ids_sin": sin_step,
                }
                feeds.update(past)
                outs = sess.run(all_outs, feeds)
                last_logits = outs[0]
                past = {
                    p_n: outs[1 + i][..., -(args.ctx - 1):, :].astype(np.float32, copy=False)
                    for i, p_n in enumerate(past_names)
                }
            return last_logits

        fp_logits = _decode_last_logits(fp_sess)
        q_logits = _decode_last_logits(sim.session)
        fp_flat = fp_logits.flatten().astype(np.float64)
        q_flat = q_logits.flatten().astype(np.float64)
        cos = float(np.dot(fp_flat, q_flat) / (np.linalg.norm(fp_flat) * np.linalg.norm(q_flat) + 1e-12))
        fp_argmax = int(np.argmax(fp_logits[0, 0]))
        q_argmax = int(np.argmax(q_logits[0, 0]))
        fp_token = tok.decode([fp_argmax])
        q_token = tok.decode([q_argmax])
        print(f"[stage 7] probe '{probe_text}' ({time.time() - t0:.1f}s)\n"
              f"  cos(fp, q)        = {cos:.6f}\n"
              f"  fp last-pos argmax = {fp_argmax!r:>8} -> {fp_token!r}\n"
              f"  q  last-pos argmax = {q_argmax!r:>8} -> {q_token!r}\n"
              f"  argmax match      = {fp_argmax == q_argmax}")
        manifest["stages"]["7_probe"] = {
            "wall_s": time.time() - t0,
            "cos_fp_q": cos,
            "fp_argmax": fp_argmax, "q_argmax": q_argmax,
            "fp_token": fp_token, "q_token": q_token,
            "argmax_match": fp_argmax == q_argmax,
        }
        save_manifest()
        del fp_sess
    except Exception as e:
        import traceback
        print(f"[stage 7] probe FAILED (non-fatal): {e}")
        traceback.print_exc()
        manifest["stages"]["7_probe"] = {"error": str(e)}
        save_manifest()

    print(f"\n[done] manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
