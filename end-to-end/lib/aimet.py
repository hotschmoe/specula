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


def _patch_apply_adascale_for_pathb_kv() -> None:
    """Override AdaScale.apply_adascale to recognize HF-style past_kv naming.

    aimet_onnx 2.26's apply_adascale matches block KV inputs by substring:

        if (
            f"past_key_{idx}_in" in name
            or f"past_value_{idx}_in" in name
        ):
            block_kv_tensor_names.append(name)

    But optimum-cli exports Qwen3 past KVs as `past_key_values.{idx}.key`
    and `past_key_values.{idx}.value`. With the original substring check,
    block_kv_tensor_names is empty for every block, so the extracted
    onnx subgraph for the block is missing past_kv inputs — and onnx2torch
    later sees them as `ValueType.UNKNOWN`, crashing with
    `RuntimeError: Got unexpected input value type (ValueType.UNKNOWN)`.

    This patch swaps the substring check to match `past_key_values.{idx}.`
    which catches both .key and .value in one filter. Idempotent.
    """
    import aimet_onnx.experimental.adascale.adascale_optimizer as ao_mod
    if getattr(ao_mod.AdaScale, "_pathb_kv_patched", False):
        return

    # Imports the original function uses.
    import contextlib, copy, gc, os, tempfile
    from pathlib import Path
    import numpy as np
    import torch
    import onnx
    from aimet_onnx.experimental.adascale.adascale_optimizer import (
        AdaScale, AdaScaleModelConfig, _logger,
    )
    from aimet_onnx.experimental.adascale.find_blocks import (
        get_decoder_blocks_end_points,
    )
    from aimet_onnx.experimental.adascale.activation_sampler import ActivationSampler
    from aimet_onnx.utils import get_torch_device
    from aimet_onnx.quantsim import QuantizationSimModel

    @classmethod
    def apply_adascale_pathb(cls, sim, inputs, adascale_model_config, num_iterations=1500):
        """Verbatim copy of upstream apply_adascale with one fix:
        block_kv_tensor_names matches `past_key_values.{idx}.` instead of
        `past_key_{idx}_in` / `past_value_{idx}_in`."""
        with cls._disable_activation_quantizers(sim):
            sim._compute_param_encodings(overwrite=False)

            blocks_end_points = get_decoder_blocks_end_points(
                sim, adascale_model_config.model_type
            )
            device = get_torch_device(sim.session)
            graph_input_names = [inp.name for inp in sim.session.get_inputs()]
            if graph_input_names != list(inputs[0].keys()):
                raise ValueError(
                    "Graph input names do not match the keys in the provided inputs."
                )

            common_input_names = []
            for name in graph_input_names:
                if "attention" in name:
                    common_input_names.append(name)
                if "position" in name:
                    common_input_names.append(name)

            del sim.session
            gc.collect()
            torch.cuda.empty_cache()

            with tempfile.TemporaryDirectory() as tempdir:
                fp32_model = copy.deepcopy(sim.model.model)
                fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)
                model_path = os.path.join(tempdir, "model.onnx")
                onnx.save_model(
                    fp32_model, model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=Path(model_path).name + ".data",
                )

                for idx in range(len(blocks_end_points)):
                    _logger.info("Optimizing block: %d", idx)
                    block_kv_tensor_names = []
                    # ---- PATCH: HF-optimum naming for past KVs ----
                    for name in graph_input_names:
                        if f"past_key_values.{idx}." in name:
                            block_kv_tensor_names.append(name)
                    # ----------------------------------------------
                    block_input_names = list(common_input_names)  # fresh copy per block
                    if len(block_kv_tensor_names) > 0:
                        if len(block_kv_tensor_names) != 2:
                            raise RuntimeError(
                                f"Unable to find both past_key and past_value for block {idx}. "
                                f"Got {block_kv_tensor_names!r}"
                            )
                        block_input_names.extend(block_kv_tensor_names)

                    qsim_sess = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        sim.model.model,
                        sim.providers,
                    )
                    fp_inputs, qsim_inputs = [], []
                    for input_ in inputs:
                        qsim_inputs.append(qsim_sess.sample_acts(input_))
                    qsim_sess.restore_graph()
                    del qsim_sess

                    fp32_sampler = ActivationSampler(
                        blocks_end_points[idx][0].inputs[0].name,
                        model_path, sim.providers, tempdir,
                    )
                    for input_ in inputs:
                        fp_inputs.append(fp32_sampler.sample_acts(input_))
                    fp32_sampler.restore_graph()
                    del fp32_sampler

                    fp_input_list = []
                    qsim_input_list = []
                    for i in range(len(fp_inputs)):
                        fp_list, qsim_list = [], []
                        fp_list.append(fp_inputs[i])
                        qsim_list.append(qsim_inputs[i])
                        for name in block_input_names:
                            fp_list.append(inputs[i][name])
                            qsim_list.append(inputs[i][name])
                        fp_input_list.append(fp_list)
                        qsim_input_list.append(qsim_list)

                    block_input_output_names = AdaScale.get_block_start_end_name(
                        blocks_end_points, idx, block_input_names
                    )
                    AdaScale.optimize_adascale_block(
                        sim, fp_input_list, qsim_input_list,
                        block_input_output_names,
                        adascale_model_config.beta_gamma_lr,
                        adascale_model_config.scales_lr,
                        num_iterations, device,
                    )
                    del fp_input_list, qsim_input_list, fp_inputs, qsim_inputs
                    gc.collect()
                    torch.cuda.empty_cache()

                sim._rebuild_session()

    ao_mod.AdaScale.apply_adascale = apply_adascale_pathb
    ao_mod.AdaScale._pathb_kv_patched = True
    # The module also binds `apply_adascale = AdaScale.apply_adascale` at
    # module-load time (line 441 of adascale_optimizer.py). Replacing the
    # classmethod alone leaves the module-level free function pointing at
    # the original; we have to rebind that too.
    ao_mod.apply_adascale = ao_mod.AdaScale.apply_adascale
    print("[adascale-patch] overrode AdaScale.apply_adascale + module-level apply_adascale "
          "for HF-style past_key_values.{i}.{key,value} naming")


def _patch_onnx2torch_reduce_mean_v18() -> None:
    """Register a v18 ReduceMean handler in onnx2torch's converter registry.

    aimet_onnx 2.26's `experimental.adascale.onnx2torch_ext` only
    registers ReduceMean v1/v11/v13 — the optimum-cli export emits
    opset 18 where ReduceMean's `axes` moved from attribute to optional
    input (mirroring what ReduceSum did at v13). Without this patch,
    `apply_adascale` crashes with:

        NotImplementedError: Converter is not implemented (
          OperationDescription(domain='', operation_type='ReduceMean',
          version=18))

    The handler below mirrors onnx2torch's existing ReduceSum v13
    implementation: read `axes` from input[1] when it's a constant
    initializer (which it is for Qwen3's RMSNorm — fixed `[ -1 ]`),
    reuse `OnnxReduceStaticAxes` for the actual reduction.
    """
    try:
        from onnx2torch.node_converters.registry import (
            add_converter, OperationDescription, _CONVERTER_REGISTRY,
        )
        from onnx2torch.onnx_node import OnnxNode
        from onnx2torch.onnx_graph import OnnxGraph
        from onnx2torch.utils.common import (
            OperationConverterResult, OnnxMapping, onnx_mapping_from_node,
        )
        from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
        from onnx2torch.utils.common import get_const_value
    except Exception as e:
        print(f"[adascale-patch] WARNING: cannot import onnx2torch internals "
              f"({e}); skipping ReduceMean v18 patch")
        return

    from onnx import defs as onnx_defs
    desc = OperationDescription(
        domain=onnx_defs.ONNX_DOMAIN,
        operation_type="ReduceMean",
        version=18,
    )
    if desc in _CONVERTER_REGISTRY:
        return  # already patched

    @add_converter(operation_type="ReduceMean", version=18)
    def _reduce_mean_v18(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
        keepdims: int = node.attributes.get("keepdims", 1)
        # noop_with_empty_axes attr was added at v18 too; default 0.
        # Resolve `axes` from input[1] if present + constant; else None.
        axes = None
        if len(node.input_values) >= 2 and node.input_values[1]:
            try:
                axes_t = get_const_value(node.input_values[1], graph)
                axes = axes_t.tolist()
            except (KeyError, AttributeError):
                axes = None
        return OperationConverterResult(
            torch_module=OnnxReduceStaticAxes(
                operation_type="ReduceMean", axes=axes, keepdims=keepdims,
            ),
            onnx_mapping=OnnxMapping(
                inputs=(node.input_values[0],),
                outputs=node.output_values,
            ),
        )

    print("[adascale-patch] registered ReduceMean v18 handler")


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
    # apply_seq_mse asserts the sim was built with the min_max scheme
    # ("Use TF quant-scheme with sequential MSE"). Force when needed.
    effective_quant_scheme = quant_scheme
    if use_seq_mse and quant_scheme != "min_max":
        _log(f"[aimet 4] WARNING: forcing quant_scheme=min_max because SEQ_MSE "
             f"requires it (you asked for {quant_scheme})")
        effective_quant_scheme = "min_max"
    quant_scheme_map = {
        "min_max": QuantScheme.min_max,
        "post_training_tf_enhanced": QuantScheme.post_training_tf_enhanced,
        "post_training_percentile": QuantScheme.post_training_percentile,
    }
    _log(f"[aimet 4] building QuantSim "
         f"param={param_type} activation={activation_type} scheme={effective_quant_scheme}")
    model_proto = onnx.load(str(src_dir / "model.onnx"), load_external_data=True)
    sim = QuantizationSimModel(
        model_proto,
        param_type=param_type, activation_type=activation_type,
        quant_scheme=quant_scheme_map[effective_quant_scheme],
        providers=providers,
    )
    _log(f"[aimet 4] QuantSim built ({time.time() - t0:.1f}s)")
    info["stages"]["4_qsim_build"] = {"wall_s": time.time() - t0,
                                       "param_type": param_type,
                                       "activation_type": activation_type,
                                       "effective_quant_scheme": effective_quant_scheme}

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
        # Three patches required to make AdaScale work on optimum-cli's
        # Qwen3 export:
        #   (a) onnx2torch needs a ReduceMean v18 handler (RMSNorm).
        #   (b) AdaScale's KV-input-name matching expects Qualcomm
        #       qai_hub naming (`past_key_{idx}_in`); optimum-cli uses
        #       HF naming (`past_key_values.{idx}.{key,value}`).
        #   (c) AdaScale.ADASCALE_PARAM_BW is hardcoded to 4 (with a
        #       literal TODO comment in the source). For w8a16 this
        #       writes `offset=-8` into bw=8 encoding slots, which
        #       qairt-converter rejects ("offset must be 0 or
        #       -2^(bw-1)") AND tunes weights with the wrong bw budget
        #       (cos regression 0.613 → 0.510 vs SEQ_MSE-only).
        _patch_onnx2torch_reduce_mean_v18()
        _patch_apply_adascale_for_pathb_kv()
        import aimet_onnx.experimental.adascale.adascale_optimizer as _ao_mod
        # Derive the actual weight bw from the QSM rather than parsing
        # `precision` — single source of truth, future-proofs against
        # any new precision mode we add (w16a16, mixed, etc.) and
        # catches drift where `precision` and the QSM were configured
        # inconsistently. Walk qc_quantize_op_dict and grab the bw from
        # any quantizer that's tied to a graph initializer (=> a param).
        init_names = {init.name for init in sim.model.model.graph.initializer}
        weight_bw = None
        for name, q in sim.qc_quantize_op_dict.items():
            if name in init_names and hasattr(q, "bitwidth"):
                weight_bw = int(q.bitwidth)
                break
        if weight_bw is None:
            # Defensive fallback — shouldn't happen for a working QSM.
            weight_bw = 8 if precision == "w8a16" else 4
            _log(f"[adascale-patch] WARNING: could not derive weight bw from QSM "
                 f"(no quantizer found over an initializer); falling back to {weight_bw} "
                 f"based on --precision {precision}")
        prev_bw = _ao_mod.AdaScale.ADASCALE_PARAM_BW
        _ao_mod.AdaScale.ADASCALE_PARAM_BW = weight_bw
        _log(f"[adascale-patch] AdaScale.ADASCALE_PARAM_BW = {weight_bw} "
             f"(was {prev_bw}; derived from QSM, matches --precision {precision})")
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
