"""AIMET aimet_onnx PTQ stage with the full Qualcomm-quality recipe.

Order: build QSM → SEQ_MSE (per-tensor weight scale search) → AdaScale
(per-block scale tuning, gradient-based) → compute_encodings (final
activation observation) → V/O w8 pin (optional, for w4a16 only) →
export.

Each optimizer runs on its own block of cal samples — they all consume
the same iterable so we materialize the cal list up front.

V/O pin: identifies attention `v_proj` and `o_proj` weight quantizers
(matching the Qwen3 graph's optimum-export naming) and overrides their
bitwidth post-encodings to w8. Mitigates the W4A16 V/O collapse without
needing a full mixed-precision rebuild.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort

from .cal import cal_iter
from .rope import build_rope_cache


def _detect_vo_proj_weight_tensors(model: onnx.ModelProto) -> list[str]:
    """Find every initializer name that looks like attn V or O proj weight.

    Optimum-export Qwen3 names them e.g.
      onnx::MatMul_<id>  (raw)
    but after AIMET wraps them, the QSM tracks them via Q/DQ pairs with
    the original op's input_name. The reliable way is to walk the graph
    nodes and find /self_attn/v_proj/MatMul + /self_attn/o_proj/MatMul,
    then return the weight (input[1]) of each.
    """
    found: list[str] = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        if not node.name:
            continue
        if node.name.endswith("/self_attn/v_proj/MatMul") or node.name.endswith("/self_attn/o_proj/MatMul"):
            if len(node.input) >= 2:
                found.append(node.input[1])  # weight is the second input
    return found


def _bump_vo_to_w8(qsim, vo_weight_names: list[str]) -> int:
    """Override the param-quantizer bitwidth for V/O proj weights to 8.

    Walks `qsim.qc_quantize_op_dict` (or the equivalent quantsim-internal
    map; aimet_onnx uses `qsim._param_names_to_quantize_info_pairs` style
    APIs depending on version). Returns count bumped.

    aimet_onnx 2.26 stores per-tensor quantizers under
    `qsim.qc_quantize_op_dict` (a dict from tensor name → QcQuantizeOp).
    Mutate the bitwidth field on each match.
    """
    bumped = 0
    qmap = getattr(qsim, "qc_quantize_op_dict", None)
    if qmap is None:
        # 2.26 falls back to .quant_info or named quantizer accessors
        for attr in ("quant_info", "param_quantizers"):
            qmap = getattr(qsim, attr, None)
            if qmap:
                break
    if qmap is None:
        print("  [V/O pin] WARNING: could not locate quantizer map on QSM; skip")
        return 0
    for w in vo_weight_names:
        q = qmap.get(w)
        if q is None:
            continue
        # qcQuantizeOp has bitwidth property; aimet_onnx 2.26 stores as
        # an int attribute or via setEncodingInfo. Try the documented path.
        try:
            q.bitwidth = 8
            bumped += 1
        except AttributeError:
            try:
                q.set_bitwidth(8)
                bumped += 1
            except Exception as e:
                print(f"  [V/O pin] could not bump {w}: {e}")
    return bumped


def run_aimet(
    *,
    src_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    precision: str,
    ctx: int,
    num_cal_samples: int,
    use_seq_mse: bool,
    seq_mse_candidates: int,
    use_ada_scale: bool,
    ada_scale_iters: int,
    use_vo_pin_w8: bool,
    quant_scheme: str,
    cuda: bool,
    log_path: Path,
    export_prefix: str,
) -> dict:
    """Run the AIMET stage. Returns a dict of metrics + paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "src_dir": str(src_dir),
        "output_dir": str(output_dir),
        "precision": precision,
        "ctx": ctx,
        "num_cal_samples": num_cal_samples,
        "use_seq_mse": use_seq_mse,
        "seq_mse_candidates": seq_mse_candidates,
        "use_ada_scale": use_ada_scale,
        "ada_scale_iters": ada_scale_iters,
        "use_vo_pin_w8": use_vo_pin_w8,
        "quant_scheme": quant_scheme,
        "stages": {},
    }

    def _log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # ---- 1. tokenizer + config + RoPE ----
    t0 = time.time()
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path))
    cfg = AutoConfig.from_pretrained(str(tokenizer_path))
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    rope_theta = float(cfg.rope_theta)
    rope_cos, rope_sin = build_rope_cache(rope_theta, head_dim, ctx + 64)
    _log(f"[aimet 1] tokenizer + config + RoPE ({time.time() - t0:.1f}s) "
         f"head_dim={head_dim} rope_theta={rope_theta}")
    info["stages"]["1_tokenizer"] = {
        "wall_s": time.time() - t0, "head_dim": head_dim, "rope_theta": rope_theta,
    }

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    so = ort.SessionOptions(); so.log_severity_level = 3

    # ---- 2. FP session for cal generation ----
    t0 = time.time()
    fp_sess = ort.InferenceSession(str(src_dir / "model.onnx"),
                                    providers=providers, sess_options=so)
    _log(f"[aimet 2] FP session built ({time.time() - t0:.1f}s) providers={fp_sess.get_providers()}")
    info["stages"]["2_fp_session"] = {"wall_s": time.time() - t0,
                                       "providers": fp_sess.get_providers()}

    # ---- 3. gather cal samples ----
    t0 = time.time()
    cal_samples: list[dict[str, np.ndarray]] = []
    for sample in cal_iter(fp_sess, tok, rope_cos, rope_sin, ctx, num_cal_samples):
        cal_samples.append(sample)
        if len(cal_samples) % 32 == 0:
            _log(f"  cal: {len(cal_samples)} samples")
    _log(f"[aimet 3] gathered {len(cal_samples)} cal samples ({time.time() - t0:.1f}s)")
    info["stages"]["3_cal_gather"] = {"wall_s": time.time() - t0, "n_samples": len(cal_samples)}
    del fp_sess

    # ---- 4. build QuantSim ----
    t0 = time.time()
    from aimet_onnx.quantsim import QuantizationSimModel
    from aimet_onnx.common.defs import QuantScheme
    if precision == "w8a16":
        param_type, activation_type = "int8", "int16"
    elif precision == "w4a16":
        param_type, activation_type = "int4", "int16"
    else:
        raise ValueError(f"unknown precision {precision}")
    quant_scheme_map = {
        "min_max": QuantScheme.min_max,
        "post_training_tf_enhanced": QuantScheme.post_training_tf_enhanced,
        "post_training_percentile": QuantScheme.post_training_percentile,
    }
    _log(f"[aimet 4] building QuantSim "
         f"param={param_type} activation={activation_type} scheme={quant_scheme}")
    model_proto = onnx.load(str(src_dir / "model.onnx"), load_external_data=True)
    sim = QuantizationSimModel(
        model_proto,
        param_type=param_type, activation_type=activation_type,
        quant_scheme=quant_scheme_map[quant_scheme],
        providers=providers,
    )
    _log(f"[aimet 4] QuantSim built ({time.time() - t0:.1f}s)")
    info["stages"]["4_qsim_build"] = {"wall_s": time.time() - t0,
                                       "param_type": param_type, "activation_type": activation_type}

    # ---- 5. SEQ_MSE (optional, run BEFORE compute_encodings) ----
    if use_seq_mse:
        t0 = time.time()
        from aimet_onnx import apply_seq_mse
        _log(f"[aimet 5] SEQ_MSE candidates={seq_mse_candidates} samples={len(cal_samples)} ...")
        apply_seq_mse(sim, cal_samples, num_candidates=seq_mse_candidates)
        _log(f"[aimet 5] SEQ_MSE done ({time.time() - t0:.1f}s)")
        info["stages"]["5_seq_mse"] = {"wall_s": time.time() - t0,
                                        "num_candidates": seq_mse_candidates}

    # ---- 6. AdaScale (optional, run BEFORE compute_encodings) ----
    if use_ada_scale:
        t0 = time.time()
        from aimet_onnx.experimental.adascale.adascale_optimizer import (
            apply_adascale, AdaScaleModelConfig,
        )
        _log(f"[aimet 6] AdaScale model_type=qwen3 iters={ada_scale_iters} samples={len(cal_samples)} ...")
        adascale_cfg = AdaScaleModelConfig(model_type="qwen3")
        apply_adascale(sim, cal_samples, adascale_cfg, num_iterations=ada_scale_iters)
        _log(f"[aimet 6] AdaScale done ({time.time() - t0:.1f}s)")
        info["stages"]["6_ada_scale"] = {"wall_s": time.time() - t0,
                                          "num_iterations": ada_scale_iters}

    # ---- 7. compute_encodings (locks in activation encodings) ----
    t0 = time.time()
    sim.compute_encodings(cal_samples)
    _log(f"[aimet 7] compute_encodings ({time.time() - t0:.1f}s)")
    info["stages"]["7_compute_encodings"] = {"wall_s": time.time() - t0}

    # ---- 8. V/O w8 pin (optional, for w4a16) ----
    if use_vo_pin_w8:
        t0 = time.time()
        vo_names = _detect_vo_proj_weight_tensors(model_proto)
        _log(f"[aimet 8] V/O pin: detected {len(vo_names)} V/O proj weights "
             f"(expect 2 × num_layers); bumping bitwidth to 8 ...")
        bumped = _bump_vo_to_w8(sim, vo_names)
        _log(f"[aimet 8] V/O pin: bumped {bumped}/{len(vo_names)} ({time.time() - t0:.1f}s)")
        info["stages"]["8_vo_pin_w8"] = {"wall_s": time.time() - t0,
                                          "detected": len(vo_names), "bumped": bumped}

    # ---- 9. probe cos vs FP on a held-out prompt (informational) ----
    try:
        t0 = time.time()
        probe_text = "The capital of France is"
        probe_ids = tok(probe_text, return_tensors="np").input_ids[0].tolist()
        fp_sess = ort.InferenceSession(str(src_dir / "model.onnx"),
                                        providers=providers, sess_options=so)
        past_names = [i.name for i in fp_sess.get_inputs() if i.name.startswith("past_key_values.")]
        kv_shape = {i.name: [d if isinstance(d, int) else 1 for d in i.shape]
                    for i in fp_sess.get_inputs() if i.name.startswith("past_key_values.")}

        def _logits_name(s):
            for o in s.get_outputs():
                if "logit" in o.name.lower():
                    return o.name
            return s.get_outputs()[0].name

        def _present_in_order(s, n_layers):
            outs = [o.name for o in s.get_outputs()]
            ks = sorted([n for n in outs if "present" in n and ".key" in n],
                        key=lambda x: int(''.join(c for c in x.split(".key")[0] if c.isdigit())))
            vs = sorted([n for n in outs if "present" in n and ".value" in n],
                        key=lambda x: int(''.join(c for c in x.split(".value")[0] if c.isdigit())))
            out: list[str] = []
            for i in range(n_layers):
                out.append(ks[i]); out.append(vs[i])
            return out

        n_layers = cfg.num_hidden_layers

        def _decode_last(s):
            logits_name = _logits_name(s)
            present_names = _present_in_order(s, n_layers)
            all_outs = [logits_name] + present_names
            past = {n: np.zeros(kv_shape[n], dtype=np.float32) for n in past_names}
            last = None
            for pos, t in enumerate(probe_ids):
                input_ids = np.array([[t]], dtype=np.int64)
                position_ids = np.array([[pos]], dtype=np.int64)
                ab = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
                ab[..., ctx - 1 - pos:] = 0.0
                feeds = {
                    "input_ids": input_ids, "position_ids": position_ids,
                    "attention_bias": ab,
                    "position_ids_cos": rope_cos[pos:pos+1][None, ...].astype(np.float32),
                    "position_ids_sin": rope_sin[pos:pos+1][None, ...].astype(np.float32),
                }
                feeds.update(past)
                outs = s.run(all_outs, feeds)
                last = outs[0]
                past = {p_n: outs[1+i][..., -(ctx-1):, :].astype(np.float32, copy=False)
                        for i, p_n in enumerate(past_names)}
            return last

        fp_logits = _decode_last(fp_sess)
        q_logits = _decode_last(sim.session)
        f = fp_logits.flatten().astype(np.float64)
        q = q_logits.flatten().astype(np.float64)
        cos = float(np.dot(f, q) / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-12))
        fp_arg = int(np.argmax(fp_logits[0, 0])); q_arg = int(np.argmax(q_logits[0, 0]))
        fp_tok = tok.decode([fp_arg]); q_tok = tok.decode([q_arg])
        _log(f"[aimet 9] probe '{probe_text}' ({time.time() - t0:.1f}s)\n"
             f"  cos(fp, q)        = {cos:.6f}\n"
             f"  fp last-pos argmax = {fp_arg!r:>8} -> {fp_tok!r}\n"
             f"  q  last-pos argmax = {q_arg!r:>8} -> {q_tok!r}\n"
             f"  argmax match      = {fp_arg == q_arg}")
        info["stages"]["9_probe"] = {
            "wall_s": time.time() - t0, "cos_fp_q": cos,
            "fp_argmax": fp_arg, "q_argmax": q_arg,
            "fp_token": fp_tok, "q_token": q_tok,
            "argmax_match": fp_arg == q_arg,
        }
        del fp_sess
    except Exception as e:
        import traceback
        _log(f"[aimet 9] probe FAILED (non-fatal): {e}")
        traceback.print_exc()
        info["stages"]["9_probe"] = {"error": str(e)}

    # ---- 10. export ONNX + encodings ----
    t0 = time.time()
    sim.export(str(output_dir), filename_prefix=export_prefix)
    _log(f"[aimet 10] sim.export → {output_dir}/{export_prefix}.onnx + .encodings ({time.time() - t0:.1f}s)")
    info["stages"]["10_export"] = {"wall_s": time.time() - t0, "prefix": export_prefix}

    # write done marker
    info_path = output_dir / "aimet_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    info["info_path"] = str(info_path)
    return info
