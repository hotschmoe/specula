"""AIMET aimet_onnx PTQ stage with the full Qualcomm-quality recipe.

Order: build QSM → V/O w8 pin (4b, w4a16) → embed pin (4c, sub-int16
weights) → P1 precision config (4d, w4a16) → SEQ_MSE (per-tensor weight
scale search) → AdaScale (per-block scale tuning, gradient-based) →
compute_encodings (final activation observation) → export.

Each optimizer runs on its own block of cal samples — they all consume
the same iterable so we materialize the cal list up front.

V/O pin: identifies attention `v_proj` and `o_proj` weight quantizers
(matching the Qwen3 graph's optimum-export naming) and overrides their
bitwidth post-encodings to w8. Mitigates the W4A16 V/O collapse without
needing a full mixed-precision rebuild.

P1 (docs/qai_hub_recipe.md §(c) P1): for the w4a16 path, ports
Qualcomm's precision config — int8-symmetric in/out-tied KV cache, 16x8
attention matmuls, int8 per-channel lm_head — by importing
`_apply_int8_kv_cache_tying_and_lm_head` from qai-hub-models. Applied
pre-SEQ_MSE so the int8/16x8 scales are searched at the final bitwidth.
The 4b V/O pin is likely made redundant by P1's 16x8 rule and the 4c
embed pin overlaps P1's int8 lm_head — both kept conservatively pending
GPU validation; see the stage 4d comment in `run_aimet`.
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
            # ---- PATCH (2026-05-21): declare shared-preamble leaf inputs ----
            # The optimum-cli + pathb graph routes one shared preamble
            # (embedding, RoPE/mask/shape prep) into EVERY decoder block's
            # extracted subgraph — a `/model/Shape` reads seq-len off
            # `past_key_values.0.*`, and the embedding `Gather` consumes
            # `input_ids`. With the upstream attention/position-only filter
            # those leaves are undeclared, so `onnx.utils.Extractor` produces
            # a block with dangling inputs and `onnx2torch.convert` dies with
            # `Got unexpected input value type (ValueType.UNKNOWN)`. Declaring
            # them makes every block's extraction self-contained. Verified on
            # the opset-17 0.6B graph: blocks 0/5/27 all `convert()` clean.
            for name in graph_input_names:
                if (name == "input_ids"
                        or name.startswith("past_key_values.0.")):
                    if name not in common_input_names:
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
                    # block 0's KV is past_key_values.0.* — already added to
                    # common_input_names as a shared-preamble leaf above.
                    # Dedup, preserving order, so the extracted block doesn't
                    # declare a duplicate input (which desyncs the fed input
                    # list length below).
                    block_input_names = list(dict.fromkeys(block_input_names))

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


def _detect_embedding_weight(model: onnx.ModelProto) -> list[str]:
    """Find the embedding-table weight (input[0] of the embed Gather).

    A Gather is a row copy — no arithmetic — so QNN requires its output
    encoding to equal its input (table) encoding. With w4a16 the table
    defaults to int4/per-channel while the Gather output activation is
    int16/per-tensor; qnn-context-binary-generator then rejects the op
    (`/model/embed_tokens/Gather ... offset -8, expected -136`). The
    table must be pinned to match the int16 per-tensor activation.
    """
    found: list[str] = []
    for node in model.graph.node:
        if node.op_type == "Gather" and node.name and "embed" in node.name.lower():
            if node.input:
                found.append(node.input[0])  # data tensor = embedding table
    return found


def _pin_embedding_w16(qsim, emb_weight_names: list[str]) -> int:
    """Pin embedding-table weight quantizers to int16, per-tensor — so the
    embed Gather's table encoding matches its int16 per-tensor output."""
    qmap = getattr(qsim, "qc_quantize_op_dict", None) or {}
    pinned = 0
    for w in emb_weight_names:
        q = qmap.get(w)
        if q is None:
            continue
        try:
            q.bitwidth = 16
        except Exception:
            try:
                q.set_bitwidth(16)
            except Exception:
                continue
        # per-tensor (the Gather output activation is per-tensor)
        try:
            q.enable_per_channel_quantization = False
        except Exception:
            pass
        pinned += 1
    return pinned


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


def _build_kv_io_map(sim) -> dict[str, str]:
    """Build the past->present KV tensor map for OUR pathb graph.

    docs/qai_hub_recipe.md §(c) P1. qai-hub-models' `_get_kv_io_map`
    matches `"past_key"`/`"past_value"` substrings in BOTH graph input
    AND output names. That works for Qualcomm's own export (their KV
    outputs are `past_key_*_out` etc.) but NOT for ours: our pathb
    rewrite (`lib/split.py`) names KV INPUTS `past_key_values.{i}.key`
    / `past_key_values.{i}.value` (these contain "past_key"/"past_value"
    — fine) and KV OUTPUTS `present.{i}.key` / `present.{i}.value`
    (these do NOT contain "past_key" — so `_get_kv_io_map` would yield
    an empty/wrong map). We therefore build the map ourselves by
    inspecting the actual QuantSim graph inputs/outputs and pairing
    `past_key_values.{i}.{key,value}` with `present.{i}.{key,value}`.

    Returns {input_tensor_name: output_tensor_name}. Only pairs where
    BOTH names are present in the graph are included (so the downstream
    `_get_enabled_quantizer` calls — which raise KeyError on a missing
    tensor — are safe).
    """
    graph = sim.model.model.graph
    in_names = {t.name for t in graph.input}
    out_names = {t.name for t in graph.output}

    # Discover layer indices from the `past_key_values.{i}.key` inputs.
    layer_ids: set[int] = set()
    for name in in_names:
        if name.startswith("past_key_values.") and name.endswith(".key"):
            try:
                layer_ids.add(int(name.split(".")[1]))
            except (IndexError, ValueError):
                continue

    kv_io_map: dict[str, str] = {}
    for i in sorted(layer_ids):
        for kind in ("key", "value"):
            in_t = f"past_key_values.{i}.{kind}"
            out_t = f"present.{i}.{kind}"
            if in_t in in_names and out_t in out_names:
                kv_io_map[in_t] = out_t
    return kv_io_map


def _apply_w4a16_precision_config(sim, log) -> dict:
    """Port Qualcomm's w4a16 precision config into our QuantSim.

    docs/qai_hub_recipe.md §(c) P1. Replicates qai-hub-models
    `_shared/llm/model.py::_configure_quant_sim` (w4a16 branch) which
    calls `_apply_int8_kv_cache_tying_and_lm_head(sim, kv_io_map)` from
    `_shared/llm/_utils.py`. That function (imported here, not
    reimplemented) does, in order:
      1. tie Concat quantizers + rebuild session,
      2. KV-cache I/O tensors -> int8 symmetric,
      3. lm_head weights -> int8 per-channel,
      4. tie KV in/out quantizers (cache round-trips without
         re-quantizing),
      5. `_set_matmul_second_input_to_8b` -> attention BMMs run 16x8
         (16-bit one operand, 8-bit the other; 8-bit propagated
         upstream through Concat/Transpose/Reshape/Slice/Div).

    The ONLY adaptation we make is the kv_io_map: see `_build_kv_io_map`
    (our pathb KV outputs are `present.*`, which qai-hub's
    `_get_kv_io_map` substring match would miss).

    Call site: this MUST run right after the QuantSim is built and
    BEFORE SEQ_MSE / AdaScale / compute_encodings — that is the order
    qai-hub uses (`create_quantsim` -> `_configure_quant_sim` happens
    inside `_build_quantsim`, strictly before `quantize()` runs
    seq_mse/adascale/compute_encodings). Bitwidths must be pinned
    before the scale search so the w8/int8 scales are searched and the
    encodings observed at the final bitwidth.
    """
    from qai_hub_models.models._shared.llm._utils import (
        _apply_int8_kv_cache_tying_and_lm_head,
    )

    kv_io_map = _build_kv_io_map(sim)
    log(f"[aimet 4d] P1: built kv_io_map with {len(kv_io_map)} entries "
        f"(pathb past_key_values.* -> present.*)")
    if not kv_io_map:
        log("[aimet 4d] WARNING: kv_io_map is empty — no KV inputs/outputs "
            "matched; skipping KV int8 tying (lm_head + 16x8 still applied)")

    _apply_int8_kv_cache_tying_and_lm_head(sim, kv_io_map, use_16x8_matmuls=True)
    log("[aimet 4d] P1: applied int8-tied KV cache + int8 lm_head + 16x8 "
        "attention matmuls")
    return {"kv_io_map_entries": len(kv_io_map)}


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
    model_info=None,  # lib.model_config.ModelInfo (optional for back-compat)
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
    # Detect V/O proj weight names on the CLEAN graph — QuantizationSimModel
    # mutates model_proto in place (wraps weights in QcQuantizeOp), so a
    # v_proj/o_proj MatMul's input[1] becomes `<weight>_qdq` afterwards and
    # no longer matches qc_quantize_op_dict keys. Detecting post-build is
    # why the V/O w8 pin silently bumped 0/N. (2026-05-21)
    vo_weight_names = _detect_vo_proj_weight_tensors(model_proto)
    emb_weight_names = _detect_embedding_weight(model_proto)
    # ---- P0 fix (docs/qai_hub_recipe.md §(c) P0): pass an explicit AIMET
    # config file. With config_file=None AIMET falls back to its built-in
    # default_config_per_channel.json whose supergroup_pass_list is just
    # ["MatmulAdd"] — it has NO RMSNorm pass, so QuantSim puts int16
    # activation quantizers on every RMSNorm intermediate (the `x²` tensor
    # @ range ~5e6, ReduceMean/Sqrt/Add/Div outputs). int16 over those
    # ranges annihilates signal → probe cos ~0.55.
    #
    # We vendor Qualcomm's default_config_llama.json (copied verbatim from
    # qai_hub_models/utils/aimet/) as lib/aimet_config_llama.json. Its
    # supergroup_pass_list is ["LayerNormalization","RMSNormalization"],
    # which triggers aimet_onnx's RMSNormalization graph pass — that pass
    # matches the decomposed Pow/ReduceMean/Add/Sqrt/Div/Mul cluster and
    # disables the output quantizers on every intermediate (HTP float
    # fallback, no QNN_Convert). It also brings the op-type exclusion set
    # and Softmax/Sigmoid [0,1] constraints. Compatible with param_type
    # int4/int8 + activation int16 (Qualcomm ships it with int4).
    #
    # The path is resolved relative to this lib/ dir, not cwd.
    _aimet_config_path = str(Path(__file__).resolve().parent / "aimet_config_llama.json")
    _log(f"[aimet 4] using config_file={_aimet_config_path}")
    # Mirror qai_hub_models _shared/llm/model.py::_build_quantsim — tie
    # Concat quantizers and skip Slice/Constant outputs. All three module
    # attributes verified present in aimet_onnx 2.26 quantsim.
    from aimet_onnx import quantsim as _qs_mod
    _qs_mod.op_types_to_tie_qtzrs = ["Concat"]
    _qs_mod._tie_qtzrs = True
    if "Slice" not in _qs_mod.op_outputs_to_ignore:
        _qs_mod.op_outputs_to_ignore.append("Slice")
    if "Constant" not in _qs_mod.op_outputs_to_ignore:
        _qs_mod.op_outputs_to_ignore.append("Constant")
    sim = QuantizationSimModel(
        model_proto,
        param_type=param_type, activation_type=activation_type,
        quant_scheme=quant_scheme_map[effective_quant_scheme],
        config_file=_aimet_config_path,
        providers=providers,
    )
    _log(f"[aimet 4] QuantSim built ({time.time() - t0:.1f}s); "
         f"detected {len(vo_weight_names)} V/O proj weights pre-build")
    info["stages"]["4_qsim_build"] = {"wall_s": time.time() - t0,
                                       "param_type": param_type,
                                       "activation_type": activation_type,
                                       "effective_quant_scheme": effective_quant_scheme}

    # ---- 4b. V/O w8 pin — bump BEFORE SEQ_MSE so the w8 scales are
    # searched/calibrated at the pinned bitwidth (bumping after
    # compute_encodings would leave w4-derived encodings on w8
    # quantizers). w4a16-only mitigation for the V-projection collapse.
    if use_vo_pin_w8:
        t0 = time.time()
        bumped = _bump_vo_to_w8(sim, vo_weight_names)
        _log(f"[aimet 4b] V/O pin: bumped {bumped}/{len(vo_weight_names)} "
             f"V/O proj weights to w8 (pre-SEQ_MSE) ({time.time() - t0:.1f}s)")
        info["stages"]["4b_vo_pin_w8"] = {"wall_s": time.time() - t0,
                                          "detected": len(vo_weight_names),
                                          "bumped": bumped}

    # ---- 4c. embedding-table pin — int16 per-tensor, pre-SEQ_MSE.
    # Only matters when weights would otherwise be < 16-bit (w4a16):
    # the embed Gather needs its table encoding == its int16 output.
    if param_type != "int16":
        t0 = time.time()
        pinned = _pin_embedding_w16(sim, emb_weight_names)
        _log(f"[aimet 4c] embedding pin: {pinned}/{len(emb_weight_names)} "
             f"embed-table weights → int16 per-tensor ({time.time() - t0:.1f}s)")
        info["stages"]["4c_embedding_pin"] = {"detected": len(emb_weight_names),
                                              "pinned": pinned}

    # ---- 4d. P1 — Qualcomm w4a16 precision config (docs/qai_hub_recipe.md
    # §(c) P1): int8-symmetric in/out-tied KV cache + 16x8 attention
    # matmuls + int8 per-channel lm_head. Ported by IMPORTING the qai-hub
    # functions (`_apply_int8_kv_cache_tying_and_lm_head` from
    # qai_hub_models/.../_shared/llm/_utils.py) — not reimplemented — so
    # we track Qualcomm's recipe exactly. The only local adaptation is the
    # kv_io_map (`_build_kv_io_map`): our pathb graph names KV outputs
    # `present.{i}.*`, which qai-hub's `_get_kv_io_map` substring match
    # ("past_key"/"past_value") would miss.
    #
    # Call order: this runs AFTER the QuantSim build and BEFORE SEQ_MSE /
    # AdaScale / compute_encodings — mirroring qai-hub, where
    # `_configure_quant_sim` (inside `create_quantsim`) applies the
    # precision config strictly before `quantize()` runs the optimizers.
    # Bitwidths must be pinned before the scale search so the int8/16x8
    # scales are searched and the encodings observed at the final
    # bitwidth.
    #
    # SCOPE — w4a16 only. qai-hub's `_configure_quant_sim` applies
    # `_apply_int8_kv_cache_tying_and_lm_head` ONLY on the `Precision.w4a16`
    # branch; there is no w8a16 branch in qai-hub's LLM recipe (Qualcomm
    # ships Qwen3-4B as w4a16). int8 KV + 16x8 matmuls would very likely
    # also help our w8a16 path (smaller KV cache, attention range), but
    # qai-hub gives us no w8a16 reference to match, so per the task's
    # "match what qai-hub does per precision" we keep P1 w4a16-only and
    # leave this comment. Revisit for w8a16 once w4a16 is GPU-validated.
    #
    # OVERLAP with the 4b V/O pin and 4c embed pin (left in place,
    # conservatively, pending GPU validation — see docs/qai_hub_recipe.md
    # §(c) P1):
    #  - V/O pin (4b): P1's 16x8 matmul rule (`_set_matmul_second_input_to_8b`)
    #    is the structural fix for the same V/O collapse `_bump_vo_to_w8`
    #    works around, so 4b is *likely redundant* now. NOT removed —
    #    needs on-GPU confirmation, which is unavailable here.
    #  - embed pin (4c): DIRECT CONFLICT. `_pin_embedding_w16` sets the
    #    embed-table weight to int16 per-tensor; P1's `_set_lm_head_to_8b`
    #    sets the lm_head weight to int8 per-channel. Qwen3 ties
    #    embeddings, so lm_head weight == embed table — both target the
    #    same initializer. We resolve in favour of Qualcomm's recipe:
    #    stage 4d runs AFTER 4c, so `_set_lm_head_to_8b` (int8 per-channel)
    #    is applied last and WINS for the tied lm_head/embed weight. This
    #    matches Qualcomm's shipped bundle (part4 lm_head is int8; ours
    #    was int16 — the +38% part4 size delta noted in
    #    e2e_optimizations.md). The 4c pin still covers any embed-table
    #    weight that is NOT also a lm_head input; for fully-tied Qwen3
    #    that set is empty, so 4c becomes effectively a no-op here — kept
    #    for non-tied models and pending validation.
    if precision == "w4a16":
        t0 = time.time()
        p1_info = _apply_w4a16_precision_config(sim, _log)
        _log(f"[aimet 4d] P1 precision config applied ({time.time() - t0:.1f}s)")
        info["stages"]["4d_p1_precision"] = {"wall_s": time.time() - t0, **p1_info}

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
        # AdaScale model_type: pull from family config when ModelInfo
        # is available, fall back to "qwen3" for callers that haven't
        # passed it (back-compat for the Qwen3-only e2e legacy path).
        adascale_model_type = "qwen3"
        if model_info is not None:
            adascale_model_type = model_info.family.aimet_adascale_model_type
        _log(f"[aimet 6] AdaScale model_type={adascale_model_type} "
             f"iters={ada_scale_iters} samples={len(cal_samples)} ...")
        adascale_cfg = AdaScaleModelConfig(model_type=adascale_model_type)
        apply_adascale(sim, cal_samples, adascale_cfg, num_iterations=ada_scale_iters)
        _log(f"[aimet 6] AdaScale done ({time.time() - t0:.1f}s)")
        info["stages"]["6_ada_scale"] = {"wall_s": time.time() - t0,
                                          "num_iterations": ada_scale_iters}

    # ---- 7. compute_encodings (locks in activation encodings) ----
    t0 = time.time()
    sim.compute_encodings(cal_samples)
    _log(f"[aimet 7] compute_encodings ({time.time() - t0:.1f}s)")
    info["stages"]["7_compute_encodings"] = {"wall_s": time.time() - t0}

    # ---- 7b. RMSNorm-internal quantizers are handled structurally by
    # the RMSNormalization supergroup pass (config_file in stage 4,
    # docs/qai_hub_recipe.md P0) — no post-encodings manual disable. ----

    # ---- 8. V/O w8 pin (optional, for w4a16) ----
    # V/O w8 pin already applied at stage 4b (pre-SEQ_MSE) — see above.

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
