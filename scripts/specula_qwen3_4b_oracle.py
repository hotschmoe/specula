"""End-to-end oracle harness for our 4-part Qwen3-4B w4a16 bundle.

Drives the 4 EPContext wrappers (specula_qwen3_4b_part{1..4}.wrapper.onnx)
through prompt prefill + N generation steps via ORT-QNN 2.1 on the X2E
HTP. Dequantizes boundary tensors across parts (part-N output -> part-
N+1 input) because the per-tensor scale/offset is not guaranteed to
match across parts — qairt-quantizer calibrated each part independently.
Past-KV cache is kept in fp32 internally; each step re-quants into the
consuming part's input encoding.

Gate: cosine(first-decode-step logits, Qualcomm oracle first-decode
logits) >= 0.95.

Run:
    .venv-ort21/Scripts/python.exe scripts/specula_qwen3_4b_oracle.py \\
        --gen-steps 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn
from tokenizers import Tokenizer


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "phase5_qwen3_4b_bundle"
TOKENIZER = (REPO / "models" / "qualcomm-qwen3-4b-ref"
             / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
             / "tokenizer.json")
PROMPT_FILE = (REPO / "models" / "qualcomm-qwen3-4b-ref"
               / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
               / "sample_prompt.txt")
QUALCOMM_ORACLE = REPO / "results" / "qualcomm_qwen3_4b_oracle.npz"

NUM_LAYERS = 36
LAYERS_PER_PART = 12
HIDDEN = 2560
VOCAB = 151936
CTX = 512
PAST = CTX - 1
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0

EMBED_HIDDEN = "_model_embed_tokens_Gather_output_0"
L11_HIDDEN = "_model_layers_11_Add_1_output_0"
L23_HIDDEN = "_model_layers_23_Add_1_output_0"


def quant_u8(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """uint8 variant of quant_u16 for the Phase 5n uint8 KV cache."""
    q = np.round(x_fp32.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 255).astype(np.uint8)


def dequant_u8(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q.astype(np.float32) + float(offset)) * float(scale)


def quant_u16(x_fp32: np.ndarray, scale: float, offset: int) -> np.ndarray:
    """dequant convention: f = (q + offset) * scale ->
    q = round(f/scale) - offset. Clamp to uint16."""
    q = np.round(x_fp32.astype(np.float64) / scale) - offset
    return np.clip(q, 0, 65535).astype(np.uint16)


def dequant_u16(q: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return ((q.astype(np.int64) + int(offset)).astype(np.float64) * scale).astype(np.float32)


def rope_half_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    """Phase 5o: half-dim cos/sin [1,1,64] for graph inputs matching
    Qualcomm's genie bundle. The graph internally concats to [1,1,128]."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    cos = np.cos(freqs).astype(np.float32).reshape(1, 1, HEAD_DIM // 2)
    sin = np.sin(freqs).astype(np.float32).reshape(1, 1, HEAD_DIM // 2)
    return cos, sin


def rope_full_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
    cos = np.cos(emb).reshape(1, 1, HEAD_DIM)
    sin = np.sin(emb).reshape(1, 1, HEAD_DIM)
    return cos, sin


def attention_bias_at(position: int) -> np.ndarray:
    bias = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float32)
    bias[..., :position] = 0.0
    bias[..., -1] = 0.0
    return bias


def load_quant_maps(results_dir: Path) -> dict[int, dict[str, tuple[float, int]]]:
    """Returns {part_idx: {tensor_name: (scale, offset)}} for all
    boundary IO tensors of each part."""
    out: dict[int, dict[str, tuple[float, int]]] = {}
    for part in (1, 2, 3, 4):
        j = json.loads((results_dir / f"qwen3_4b_part{part}.w4a16-local.dlc.json").read_text())
        part_map: dict[str, tuple[float, int]] = {}
        for name, t in j["graph"]["tensors"].items():
            q = t.get("quant_params", {}).get("scale_offset", {})
            if q.get("is_fixed_point") and q.get("bitwidth"):
                part_map[name] = (float(q["scale"]), int(q["offset"]))
        out[part] = part_map
    return out


def name_to_wrapper(name: str) -> str:
    """ONNX/DLC names are slash-form; wrapper/HTP names are underscore-form
    with leading slash stripped. Match our wrapper-declared port names."""
    if name.startswith("/"):
        return name.replace("/", "_").replace(".", "_")
    return name.replace(".", "_")


def mk_session(wrapper: Path, qnn_devs: list, htp_dll: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_provider_for_devices(qnn_devs, {
        "backend_path": str(htp_dll),
        "htp_performance_mode": "burst",
        "soc_model": "88",
        "htp_arch": "81",
        "enable_htp_fp16_precision": "1",
    })
    return ort.InferenceSession(str(wrapper), sess_options=so)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument("--bundle-dir", type=Path, default=RESULTS)
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--oracle-npz", type=Path, default=QUALCOMM_ORACLE)
    parser.add_argument("--out", type=Path,
                        default=REPO / "results" / "specula_qwen3_4b_oracle")
    args = parser.parse_args()

    print("=== specula Qwen3-4B 4-part w4a16 oracle (AR1 / CL512) ===")
    print(f"bundle: {args.bundle_dir}")

    # Load scale/offset maps up front.
    qmaps = load_quant_maps(args.bundle_dir)
    # Convenient shared scales (take from part2; identical across parts 2/3/4).
    cos_scale, cos_offset = qmaps[2]["position_ids_cos"]
    sin_scale, sin_offset = qmaps[2]["position_ids_sin"]
    bias_scale, bias_offset = qmaps[2]["attention_bias"]
    logits_scale, logits_offset = qmaps[4]["logits"]
    # Seam hidden scales per consuming part.
    p1_embed_out = qmaps[1][f"/model/embed_tokens/Gather_output_0"]
    p2_embed_in = qmaps[2][f"/model/embed_tokens/Gather_output_0"]
    p2_l11_out = qmaps[2][f"/model/layers.11/Add_1_output_0"]
    p3_l11_in = qmaps[3][f"/model/layers.11/Add_1_output_0"]
    p3_l23_out = qmaps[3][f"/model/layers.23/Add_1_output_0"]
    p4_l23_in = qmaps[4][f"/model/layers.23/Add_1_output_0"]

    print(f"quant bridges (scale, offset):")
    print(f"  embed: part1_out={p1_embed_out}  part2_in={p2_embed_in}")
    print(f"  L11:   part2_out={p2_l11_out}    part3_in={p3_l11_in}")
    print(f"  L23:   part3_out={p3_l23_out}    part4_in={p4_l23_in}")

    # Initialize QNN EP.
    ort.register_execution_provider_library("QNNExecutionProvider",
                                            onnxruntime_qnn.get_library_path())
    qnn_devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not qnn_devs:
        print("FATAL: no QNN devices visible")
        return 2
    htp_dll = Path(onnxruntime_qnn.LIB_DIR_FULL_PATH) / "QnnHtp.dll"

    print(f"\nloading 4 HTP sessions ...")
    t0 = time.perf_counter()
    sessions: list[ort.InferenceSession] = []
    for p in (1, 2, 3, 4):
        wrapper = args.bundle_dir / f"specula_qwen3_4b_part{p}.wrapper.onnx"
        t_p = time.perf_counter()
        sessions.append(mk_session(wrapper, qnn_devs, htp_dll))
        print(f"  part{p}: {time.perf_counter()-t_p:.1f}s")
    print(f"all 4 loaded in {time.perf_counter()-t0:.1f}s")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt_text = args.prompt_file.read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(prompt_text).ids
    total_steps = len(prompt_ids) + args.gen_steps
    print(f"\nprompt: {len(prompt_ids)} tokens, gen steps: {args.gen_steps}")

    # Persistent fp32 KV cache: layer L key shape [1,8,PAST,128], value same.
    past_k = [np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)
              for _ in range(NUM_LAYERS)]
    past_v = [np.zeros((1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32)
              for _ in range(NUM_LAYERS)]

    all_logits_fp32: list[np.ndarray] = []
    all_argmax: list[int] = []
    all_step_tokens: list[int] = []
    all_decoded: list[str] = []
    step_latency_ms: list[float] = []

    next_token = prompt_ids[0]
    for step in range(total_steps):
        is_prefill = step < len(prompt_ids)
        token_in = prompt_ids[step] if is_prefill else next_token
        all_step_tokens.append(token_in)
        position = step

        cos_fp32, sin_fp32 = rope_half_dim(position)
        bias_fp32 = attention_bias_at(position)
        cos_u16 = quant_u16(cos_fp32, cos_scale, cos_offset)
        sin_u16 = quant_u16(sin_fp32, sin_scale, sin_offset)
        bias_u16 = quant_u16(bias_fp32, bias_scale, bias_offset)

        t_step = time.perf_counter()

        # --- part 1: input_ids (int32) -> embed_hidden (uint16) ---
        p1_feed = {"input_ids": np.array([[token_in]], dtype=np.int32)}
        embed_u16_p1 = sessions[0].run([EMBED_HIDDEN], p1_feed)[0]
        embed_fp32 = dequant_u16(embed_u16_p1, *p1_embed_out)
        embed_u16_p2 = quant_u16(embed_fp32, *p2_embed_in)

        def run_decode_part(sess_idx: int, layer_start: int, layer_end: int,
                            hidden_name: str, hidden_u16: np.ndarray,
                            hidden_out_name: str) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
            qm = qmaps[sess_idx + 1]
            feed: dict[str, np.ndarray] = {
                hidden_name: hidden_u16,
                "attention_bias": bias_u16,
                "position_ids_cos": cos_u16,
                "position_ids_sin": sin_u16,
            }
            for li in range(layer_start, layer_end + 1):
                k_name = f"past_key_values_{li}_key"
                v_name = f"past_key_values_{li}_value"
                k_scale, k_off = qm[f"past_key_values.{li}.key"]
                v_scale, v_off = qm[f"past_key_values.{li}.value"]
                feed[k_name] = quant_u8(past_k[li], k_scale, k_off)
                feed[v_name] = quant_u8(past_v[li], v_scale, v_off)

            out_names = [hidden_out_name]
            for li in range(layer_start, layer_end + 1):
                out_names.append(f"present_{li}_key")
                out_names.append(f"present_{li}_value")

            outs = sessions[sess_idx].run(out_names, feed)
            hidden_out = outs[0]  # uint16 in this part's output scale
            new_keys_fp32: list[np.ndarray] = []
            new_vals_fp32: list[np.ndarray] = []
            for i, li in enumerate(range(layer_start, layer_end + 1)):
                k_out_u16 = outs[1 + 2 * i]
                v_out_u16 = outs[1 + 2 * i + 1]
                k_scale, k_off = qm[f"present.{li}.key"]
                v_scale, v_off = qm[f"present.{li}.value"]
                new_keys_fp32.append(dequant_u8(k_out_u16, k_scale, k_off))
                new_vals_fp32.append(dequant_u8(v_out_u16, v_scale, v_off))
            return hidden_out, new_keys_fp32, new_vals_fp32

        # --- part 2: layers 0..11 ---
        l11_u16_p2, keys_p2, vals_p2 = run_decode_part(
            1, 0, 11, EMBED_HIDDEN, embed_u16_p2, L11_HIDDEN,
        )
        l11_fp32 = dequant_u16(l11_u16_p2, *p2_l11_out)
        l11_u16_p3 = quant_u16(l11_fp32, *p3_l11_in)

        # --- part 3: layers 12..23 ---
        l23_u16_p3, keys_p3, vals_p3 = run_decode_part(
            2, 12, 23, L11_HIDDEN, l11_u16_p3, L23_HIDDEN,
        )
        l23_fp32 = dequant_u16(l23_u16_p3, *p3_l23_out)
        l23_u16_p4 = quant_u16(l23_fp32, *p4_l23_in)

        # --- part 4: layers 24..35 + lm_head ---
        logits_u16, keys_p4, vals_p4 = run_decode_part(
            3, 24, 35, L23_HIDDEN, l23_u16_p4, "logits",
        )
        logits_fp32 = dequant_u16(logits_u16, logits_scale, logits_offset).squeeze()
        argmax_id = int(np.argmax(logits_fp32))

        # Stitch present KV: the HF concat is [past_511 | new_1], so the
        # NEW token's K/V sits at slot 511 (the last of present's 512 slots).
        # Write it into past_k/v at chronological slot `position`.
        all_new_keys = keys_p2 + keys_p3 + keys_p4   # 36 tensors
        all_new_vals = vals_p2 + vals_p3 + vals_p4
        for li in range(NUM_LAYERS):
            past_k[li][:, :, position:position + 1, :] = all_new_keys[li][:, :, -1:, :]
            past_v[li][:, :, position:position + 1, :] = all_new_vals[li][:, :, -1:, :]

        step_latency_ms.append((time.perf_counter() - t_step) * 1000)
        all_logits_fp32.append(logits_fp32)
        all_argmax.append(argmax_id)
        next_token = argmax_id
        if not is_prefill:
            tok_str = tokenizer.id_to_token(argmax_id) or f"<id {argmax_id}>"
            all_decoded.append(tok_str)
            print(f"  step {step:3d} (gen)    pos={position:3d}  "
                  f"in={token_in:6d}  argmax={argmax_id:6d}  "
                  f"tok={tok_str!r}  {step_latency_ms[-1]:.1f} ms")
        else:
            print(f"  step {step:3d} (prefill) pos={position:3d}  "
                  f"in={token_in:6d}  (preview argmax={argmax_id:6d})  "
                  f"{step_latency_ms[-1]:.1f} ms")

    # Compare first decode step logits to Qualcomm oracle if present.
    cos_gate = None
    if args.oracle_npz.exists():
        oracle = np.load(args.oracle_npz)
        if "logits_fp32" in oracle.files:
            oracle_logits = oracle["logits_fp32"]  # [steps, vocab]
            first_decode = len(prompt_ids)  # step index of first gen
            if first_decode < oracle_logits.shape[0] and first_decode < len(all_logits_fp32):
                ours = all_logits_fp32[first_decode].astype(np.float64).ravel()
                theirs = oracle_logits[first_decode].astype(np.float64).ravel()
                denom = np.linalg.norm(ours) * np.linalg.norm(theirs)
                cos_gate = float(np.dot(ours, theirs) / denom) if denom > 0 else float("nan")
                print(f"\nfirst-decode logit cosine vs Qualcomm oracle: {cos_gate:.6f}")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        logits_fp32=np.stack(all_logits_fp32),
        argmax_tokens=np.array(all_argmax, dtype=np.int64),
        step_tokens=np.array(all_step_tokens, dtype=np.int64),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        step_latency_ms=np.array(step_latency_ms),
    )
    decoded = tokenizer.decode(all_argmax[len(prompt_ids):])
    md = args.out.with_suffix(".md")
    md.write_text(
        f"# Specula Qwen3-4B 4-part w4a16 oracle\n\n"
        f"- prompt: {len(prompt_ids)} tokens, gen: {args.gen_steps}\n"
        f"- prefill mean: {np.mean(step_latency_ms[:len(prompt_ids)]):.1f} ms\n"
        f"- gen mean: {np.mean(step_latency_ms[len(prompt_ids):]):.1f} ms\n"
        f"- vs Qualcomm oracle first-decode cos: "
        f"{cos_gate if cos_gate is not None else 'not computed'}\n\n"
        f"## Decoded\n```\n{decoded}\n```\n",
        encoding="utf-8",
    )
    print(f"\nsaved: {args.out.with_suffix('.npz')}")
    print(f"       {md}")
    print(f"\ngeneration: {decoded!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
