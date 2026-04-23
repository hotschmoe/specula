"""Phase 0 of the Qualcomm Qwen3-4B reproduction effort: record an oracle.

Drives the 4-part Qualcomm Qwen3-4B Genie w4a16 bundle through
prompt prefill + N generation steps using AR1 / CL512, recording
per-step logits and generated tokens. Output is the ground truth
that any future specula-built Qwen3-4B w4a16 must match (logit cos
on the same prompt) for the pipeline to be considered verified.

This is a CPU-cheap harness (no torch, no transformers) — all model
ops happen on the X2E HTP via ORT-QNN. The harness only handles
tokenization, position-id arithmetic, quantize/dequant, and
KV-cache stitching.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
        scripts/qualcomm_qwen3_4b_oracle.py --gen-steps 8

Output:
    results/qualcomm_qwen3_4b_oracle.npz   - per-step logits + tokens
    results/qualcomm_qwen3_4b_oracle.md    - human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
SOC_MODEL = "88"
HTP_ARCH = "81"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
CTX_LEN = 512
PAST_LEN = CTX_LEN - 1  # 511
HIDDEN_DIM = 2560
VOCAB_SIZE = 151936
ROPE_THETA = 1_000_000.0


def quant_uint16(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """dequant convention: f = (q + offset) * scale  =>  q = round(f/scale) - offset."""
    q = np.round(x_fp32 / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_uint16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.int32) + offset).astype(np.float32) * scale


def half_dim_rope_quantized(
    pos: int, scale: float, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Qualcomm-style half-dim cos/sin tables for a single position.
    Shape [1, 1, 1, 64], uint16-quantized with given scale/offset."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = pos * inv_freq  # [64]
    cos_h = np.cos(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    sin_h = np.sin(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    return quant_uint16(cos_h, scale, offset), quant_uint16(sin_h, scale, offset)


def attention_mask_quantized(pos: int, scale: float, offset: int) -> np.ndarray:
    """Build the CL=512 attention mask for AR1 decode at position `pos`.

    KV layout: past_kv has 511 slots (chronological, left-aligned — slot
    0 holds position 0, slot t-1 holds position t-1, slots t..510 are
    zero-filled and unused). The model internally concatenates the
    current step's K/V (1 slot) onto the past, producing 512 attention
    slots: [past_511 | current_1].

    Therefore at step t (0-indexed):
      - past slots 0..t-1   -> attend (valid history)
      - past slots t..510   -> block (unused, all-zero KV)
      - current slot 511    -> attend (always, this is the live token)

    Quantized representation: q=65535 -> dequant=0.0 (attend);
    q=0 -> dequant=(0-65535)*0.0015259 ~= -100 (mask). The dequantized
    mask is added to attention scores before softmax."""
    mask = np.zeros((1, 1, 1, CTX_LEN), dtype=np.uint16)
    if pos > 0:
        mask[..., :pos] = 65535  # past positions 0..pos-1
    mask[..., -1] = 65535  # current slot 511
    _ = scale, offset  # values are fixed by metadata; included for symmetry
    return mask


def build_part_cfg(metadata: dict) -> dict:
    """Pull AR1/CL512 IO specs from the bundle's metadata.yaml."""
    cfg = {}
    for part in (1, 2, 3, 4):
        comp_name = f"ar1_cl512_{part}_of_4"
        comp = metadata["components"][comp_name]
        # The wrapper.onnx must use UNDERSCORED tensor names — that's how
        # the QAIRT compiler stored them in the binary. metadata.yaml has
        # the SLASH form (original ONNX), so we translate.
        def to_underscore(name: str) -> str:
            if name.startswith("/"):
                # /model/model/.../foo_output_0  ->  _model_model_..._foo_output_0
                return name.replace("/", "_").replace(".", "_")
            return name
        cfg[part] = {
            "bin": f"qwen3_4b_part_{part}_of_4.bin",
            "graph_name": f"token_ar1_cl512_{part}_of_4",
            "inputs": [
                {
                    "name": to_underscore(name),
                    "metaname": name,
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype"],
                    # int32 input_ids has no quant params — leave as None.
                    "scale": spec.get("quantization_parameters", {}).get("scale"),
                    "offset": spec.get("quantization_parameters", {}).get("offset"),
                }
                for name, spec in comp["inputs"].items()
            ],
            "outputs": [
                {
                    "name": to_underscore(name),
                    "metaname": name,
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype"],
                    "scale": spec.get("quantization_parameters", {}).get("scale"),
                    "offset": spec.get("quantization_parameters", {}).get("offset"),
                }
                for name, spec in comp["outputs"].items()
            ],
        }
    return cfg


_DTYPE_PROTO = {
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "float32": TensorProto.FLOAT,
}
_DTYPE_NUMPY = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "float32": np.float32,
}


def build_wrapper(part_cfg: dict, wrapper_path: Path) -> None:
    inputs_decl = [
        helper.make_tensor_value_info(io["name"], _DTYPE_PROTO[io["dtype"]], io["shape"])
        for io in part_cfg["inputs"]
    ]
    outputs_decl = [
        helper.make_tensor_value_info(io["name"], _DTYPE_PROTO[io["dtype"]], io["shape"])
        for io in part_cfg["outputs"]
    ]
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs_decl],
        outputs=[v.name for v in outputs_decl],
        name=part_cfg["graph_name"],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=part_cfg["bin"],
        source="Qnn",
    )
    graph = helper.make_graph(
        nodes=[node],
        name=f"qualcomm_qwen3_4b_oracle_part{part_cfg['graph_name'][-6]}",
        inputs=inputs_decl,
        outputs=outputs_decl,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        producer_name="specula-oracle",
    )
    model.ir_version = 10
    onnx.save(model, str(wrapper_path))


def load_session(wrapper_path: Path) -> ort.InferenceSession:
    backend = Path(ort.__file__).parent / "capi" / "QnnHtp.dll"
    provider_options = {
        "backend_path": str(backend),
        "htp_performance_mode": "burst",
        "soc_model": SOC_MODEL,
        "htp_arch": HTP_ARCH,
        "enable_htp_fp16_precision": "1",
    }
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(
        str(wrapper_path),
        sess_options=sess_opts,
        providers=[("QNNExecutionProvider", provider_options)],
    )
    if sess.get_providers()[0] != "QNNExecutionProvider":
        raise RuntimeError(f"session fell back to {sess.get_providers()[0]}")
    return sess


class KVStore:
    """Persistent uint8 KV cache, shape [8, 1, 128, t] for keys and
    [8, 1, t, 128] for values, per layer. Grows by 1 timestep per
    decode call.

    Per-layer scale/offset are FIXED (Qualcomm chose them at compile
    time) — verified that past_kv_in scale/offset == past_kv_out
    scale/offset for every layer, so we can concat raw uint8 across
    steps without requantizing."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Stored as fixed-size [8, 1, 128, PAST_LEN] / [8, 1, PAST_LEN, 128]
        # filled with the "zero" code for each layer's offset (q=128 since
        # offset=-128 always for KV per Qualcomm). The model only attends
        # to indices 0..pos-1 (mask blocks the rest), so the unused slots'
        # values don't matter — but they must be at q=128 so dequant=0,
        # which keeps any accidental contribution to attention scores at
        # zero.
        self.keys: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, HEAD_DIM, PAST_LEN), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        self.values: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, PAST_LEN, HEAD_DIM), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        self.t = 0  # next free slot index in [0, PAST_LEN)

    def stitch_step(self, k_outs: list[np.ndarray], v_outs: list[np.ndarray]) -> None:
        if self.t >= PAST_LEN:
            raise RuntimeError(f"KV cache full at t={self.t}; ctx={CTX_LEN}")
        for i in range(self.num_layers):
            # k_out shape: [8, 1, 128, 1]
            self.keys[i][..., self.t : self.t + 1] = k_outs[i]
            # v_out shape: [8, 1, 1, 128]
            self.values[i][:, :, self.t : self.t + 1, :] = v_outs[i]
        self.t += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-steps", type=int, default=8,
                        help="number of generation steps after prompt prefill")
    parser.add_argument("--prompt-file", type=str,
                        default=str(BUNDLE_DIR / "sample_prompt.txt"))
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "results" / "qualcomm_qwen3_4b_oracle"))
    args = parser.parse_args()

    print(f"=== Qualcomm Qwen3-4B w4a16 oracle (AR1 / CL512) ===")
    print(f"bundle: {BUNDLE_DIR}")
    print(f"prompt file: {args.prompt_file}")

    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts = build_part_cfg(metadata)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(prompt_text).ids
    print(f"prompt: {len(prompt_text)} chars, {len(prompt_ids)} tokens")
    print(f"prefill positions 0..{len(prompt_ids) - 1}, "
          f"generate positions {len(prompt_ids)}..{len(prompt_ids) + args.gen_steps - 1}")

    # Build + load 4 sessions.
    print("\n--- building wrappers and loading sessions ---")
    sessions: dict[int, ort.InferenceSession] = {}
    t0 = time.perf_counter()
    for part_idx in (1, 2, 3, 4):
        wrapper_path = BUNDLE_DIR / f"oracle_part{part_idx}.wrapper.onnx"
        build_wrapper(parts[part_idx], wrapper_path)
        t_load = time.perf_counter()
        sessions[part_idx] = load_session(wrapper_path)
        print(f"  part {part_idx}: wrapper={wrapper_path.name}, "
              f"loaded in {time.perf_counter() - t_load:.1f} s")
    print(f"all 4 parts loaded in {time.perf_counter() - t0:.1f} s")

    # Capture per-input metadata for fast quantization at run time.
    # cos/sin/mask scales are constant across parts 2/3/4 (verified earlier).
    cos_scale = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["scale"]
    cos_offset = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["offset"]
    mask_scale = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["scale"]
    mask_offset = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["offset"]
    logits_io = next(io for io in parts[4]["outputs"] if io["name"] == "logits")
    logits_scale = logits_io["scale"]
    logits_offset = logits_io["offset"]

    # Per-layer KV input names are needed to address feed entries.
    def layer_input_name(part_idx: int, kv: str, layer: int) -> str:
        return f"past_{kv}_{layer}_in"

    def layer_output_name(part_idx: int, kv: str, layer: int) -> str:
        return f"past_{kv}_{layer}_out"

    # Hidden-state handoff names (underscored).
    HIDDEN_FROM_PART1 = "_model_model_embed_tokens_Gather_output_0"
    HIDDEN_FROM_PART2 = "_model_model_layers_11_Add_1_output_0"
    HIDDEN_FROM_PART3 = "_model_model_layers_23_Add_1_output_0"

    kv = KVStore(NUM_LAYERS)

    all_logits_uint16: list[np.ndarray] = []
    all_logits_fp32: list[np.ndarray] = []
    all_argmax: list[int] = []
    all_step_tokens: list[int] = []  # what was fed in
    all_decoded: list[str] = []      # what was the next token (for gen steps)
    step_latency_ms: list[float] = []

    # Schedule: feed prompt[0..len-1] one token at a time, then for
    # generation feed argmax of previous step's logits.
    total_steps = len(prompt_ids) + args.gen_steps
    next_token = prompt_ids[0]
    print(f"\n--- decoding {total_steps} steps ---")
    for step in range(total_steps):
        is_prefill = step < len(prompt_ids)
        token_in = prompt_ids[step] if is_prefill else next_token
        all_step_tokens.append(token_in)
        position = step  # KV slot to write into for this step

        # Quantize position-dependent inputs.
        cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
        mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

        t_step = time.perf_counter()

        # --- part 1: input_ids -> embedding ---
        feed1 = {"input_ids": np.array([[token_in]], dtype=np.int32)}
        emb_out = sessions[1].run([HIDDEN_FROM_PART1], feed1)[0]

        # --- part 2: layers 0..11 ---
        feed2 = {
            HIDDEN_FROM_PART1: emb_out,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(0, LAYERS_PER_PART):
            feed2[layer_input_name(2, "key", layer)] = kv.keys[layer]
            feed2[layer_input_name(2, "value", layer)] = kv.values[layer]
        out_names_2 = [HIDDEN_FROM_PART2] + [
            layer_output_name(2, kvtype, layer)
            for layer in range(0, LAYERS_PER_PART)
            for kvtype in ("key", "value")
        ]
        out_2 = sessions[2].run(out_names_2, feed2)
        hidden_after_p2 = out_2[0]
        # Stitch parts of out_2 into KV store (12 layers * 2)
        new_keys_p2 = []
        new_vals_p2 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p2.append(out_2[1 + 2 * i])
            new_vals_p2.append(out_2[1 + 2 * i + 1])

        # --- part 3: layers 12..23 ---
        feed3 = {
            HIDDEN_FROM_PART2: hidden_after_p2,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART):
            feed3[layer_input_name(3, "key", layer)] = kv.keys[layer]
            feed3[layer_input_name(3, "value", layer)] = kv.values[layer]
        out_names_3 = [HIDDEN_FROM_PART3] + [
            layer_output_name(3, kvtype, layer)
            for layer in range(LAYERS_PER_PART, 2 * LAYERS_PER_PART)
            for kvtype in ("key", "value")
        ]
        out_3 = sessions[3].run(out_names_3, feed3)
        hidden_after_p3 = out_3[0]
        new_keys_p3 = []
        new_vals_p3 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p3.append(out_3[1 + 2 * i])
            new_vals_p3.append(out_3[1 + 2 * i + 1])

        # --- part 4: layers 24..35 + lm_head -> logits ---
        feed4 = {
            HIDDEN_FROM_PART3: hidden_after_p3,
            "attention_mask": mask_q,
            "position_ids_cos": cos_q,
            "position_ids_sin": sin_q,
        }
        for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS):
            feed4[layer_input_name(4, "key", layer)] = kv.keys[layer]
            feed4[layer_input_name(4, "value", layer)] = kv.values[layer]
        out_names_4 = ["logits"] + [
            layer_output_name(4, kvtype, layer)
            for layer in range(2 * LAYERS_PER_PART, NUM_LAYERS)
            for kvtype in ("key", "value")
        ]
        out_4 = sessions[4].run(out_names_4, feed4)
        logits_uint16 = out_4[0]
        new_keys_p4 = []
        new_vals_p4 = []
        for i in range(LAYERS_PER_PART):
            new_keys_p4.append(out_4[1 + 2 * i])
            new_vals_p4.append(out_4[1 + 2 * i + 1])

        # Stitch all 36 layers' new K/V into the persistent cache.
        kv.stitch_step(
            new_keys_p2 + new_keys_p3 + new_keys_p4,
            new_vals_p2 + new_vals_p3 + new_vals_p4,
        )

        step_latency_ms.append((time.perf_counter() - t_step) * 1000)

        logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
        argmax_id = int(np.argmax(logits_fp32))
        all_logits_uint16.append(logits_uint16.squeeze())
        all_logits_fp32.append(logits_fp32.squeeze())
        all_argmax.append(argmax_id)

        # Always seed next_token from the most recent argmax so the
        # first generation step uses the prediction from the last
        # prefill step (positions are 0-indexed, so argmax at pos N
        # predicts the token at pos N+1).
        next_token = argmax_id
        if not is_prefill:
            tok_str = tokenizer.id_to_token(argmax_id) or f"<id {argmax_id}>"
            all_decoded.append(tok_str)
            print(f"  step {step:3d} (gen)    pos={position:3d}  "
                  f"in={token_in:6d}  argmax={argmax_id:6d}  "
                  f"tok={tok_str!r}  {step_latency_ms[-1]:.1f} ms")
        else:
            print(f"  step {step:3d} (prefill) pos={position:3d}  "
                  f"in={token_in:6d}  (next argmax preview={argmax_id:6d})  "
                  f"{step_latency_ms[-1]:.1f} ms")

    # Concatenate outputs and save.
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.out).with_suffix(".npz")
    md_path = Path(args.out).with_suffix(".md")

    np.savez_compressed(
        npz_path,
        logits_uint16=np.stack(all_logits_uint16),  # [steps, vocab]
        logits_fp32=np.stack(all_logits_fp32),
        step_tokens=np.array(all_step_tokens, dtype=np.int64),
        argmax_tokens=np.array(all_argmax, dtype=np.int64),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        step_latency_ms=np.array(step_latency_ms),
        logits_scale=np.array(logits_scale),
        logits_offset=np.array(logits_offset),
    )

    decoded_text = tokenizer.decode(all_argmax[len(prompt_ids):])
    md_path.write_text(
        "# Qualcomm Qwen3-4B w4a16 oracle trace\n\n"
        f"- bundle: `{BUNDLE_DIR.relative_to(REPO_ROOT)}`\n"
        f"- mode: AR1 / CL512, greedy argmax\n"
        f"- prompt: {len(prompt_ids)} tokens, generation steps: {args.gen_steps}\n"
        f"- prefill mean latency: {np.mean(step_latency_ms[:len(prompt_ids)]):.1f} ms/step\n"
        f"- generation mean latency: "
        f"{np.mean(step_latency_ms[len(prompt_ids):]):.1f} ms/step\n"
        f"- logits quant: scale={logits_scale}, offset={logits_offset}\n\n"
        "## Generated tokens\n\n"
        + "\n".join(
            f"{i}. id={tid}  tok={tokenizer.id_to_token(tid)!r}"
            for i, tid in enumerate(all_argmax[len(prompt_ids):])
        )
        + f"\n\n## Decoded continuation\n\n```\n{decoded_text}\n```\n"
        + f"\n## Saved\n\n- {npz_path}\n- {md_path}\n",
        encoding="utf-8",
    )

    print(f"\n--- saved ---")
    print(f"  {npz_path}")
    print(f"  {md_path}")
    print(f"\ngeneration text:\n{decoded_text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
