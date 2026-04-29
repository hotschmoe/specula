"""Qwen2.5-7B w8a16 NPU oracle — sister to qualcomm_qwen3_4b_oracle.py.

Drives the 6-partition Qualcomm Qwen2.5-7B-Instruct w8a16 Genie bundle
through prompt prefill + N decode steps using AR1 / CL=4096, recording
per-step logits and generated tokens. Same shape as the 4B oracle —
different model, different layer split, different IO names.

Architectural deltas from 4B (validated via `qnn-context-binary-utility`
JSON dumps in `last_side_quest/sq6_small_server/7b_bundle_metadata/`):

  | knob              | Qwen3-4B (w4a16)  | Qwen2.5-7B (w8a16)  |
  |-------------------|-------------------|---------------------|
  | num_layers        | 36                | 28                  |
  | layer-parts       | 3 (12/12/12)      | 5 (6/6/6/6/4)       |
  | total parts       | 4 (1 embed + 3)   | 6 (1 embed + 5)     |
  | num_kv_heads      | 8                 | 4                   |
  | head_dim          | 128               | 128 (same)          |
  | hidden_dim        | 2560              | 3584                |
  | vocab             | 151936            | 152064              |
  | rope_theta        | 1e6               | 1e6 (same)          |
  | ctx tiers         | {512..4096}       | {4096}              |

  Hidden-state name pattern also differs (4B has `model_model_*`,
  7B has `model_*` — different ONNX wrapping in upstream).

The 7B bundle ships ONLY .bin files + tokenizer.json + genie_config.json
— no metadata.yaml. We rebuild the tensor specs at runtime by reading
`qnn-context-binary-utility` JSON dumps stored alongside the bundle
metadata in `last_side_quest/sq6_small_server/7b_bundle_metadata/`. To
regenerate those JSON dumps:

  for i in 1..6:
      qnn-context-binary-utility \
          --context_binary qwen2_5_7b_instruct_w8a16_part_${i}_of_6.bin \
          --json_file ${i}.json

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
        npu_engine/qualcomm_qwen2_5_7b_oracle.py --gen-steps 8

Output:
    results/qualcomm_qwen2_5_7b_oracle.npz   - per-step logits + tokens
    results/qualcomm_qwen2_5_7b_oracle.md    - human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Reuse model-agnostic plumbing from the 4B oracle. These don't depend
# on dim/layer constants:
#   - quant_uint16/dequant_uint16: pure
#   - build_wrapper: takes a part_cfg dict, model-neutral
#   - load_session: takes a wrapper path, model-neutral
#   - _DTYPE_PROTO/_DTYPE_NUMPY: shared dtype maps
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    build_wrapper,
    dequant_uint16,
    load_session,
    quant_uint16,
    _DTYPE_NUMPY,
    _DTYPE_PROTO,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen2_5-7b-ref"
    / "qwen2_5_7b_instruct-genie-w8a16-qualcomm_snapdragon_x2_elite"
)
QNN_METADATA_DIR = (
    REPO_ROOT / "last_side_quest" / "sq6_small_server" / "7b_bundle_metadata"
)
SOC_MODEL = "88"
HTP_ARCH = "81"

# 7B bundle was AI-Hub-built against QAIRT 2.45.40 (per part*.json
# `buildId: v2.45.0.260326154327`). Our venv ORT 1.24.4 ships QAIRT 2.42
# which fails the .bin load with QNN error 5000. Override to the
# system-installed QAIRT 2.45.40 DLL — works in-process.
QAIRT_2_45_BACKEND = Path(
    "C:/Qualcomm/AIStack/QAIRT/2.45.40.260406/lib/aarch64-windows-msvc/QnnHtp.dll"
)

NUM_LAYERS = 28
NUM_KV_HEADS = 4  # Qwen2.5-7B GQA: 28 attention heads, 4 KV heads
HEAD_DIM = 128
HALF_HEAD_DIM = HEAD_DIM // 2
HIDDEN_DIM = 3584
VOCAB_SIZE = 152064
CTX_LEN = 4096  # 7B bundle ships only this tier
PAST_LEN = CTX_LEN - 1  # 4095
ROPE_THETA = 1_000_000.0

# AR128 prefill — same batch size as 4B (a Qualcomm bundle convention).
AR128_BATCH = 128
PAST_LEN_AR128 = CTX_LEN - AR128_BATCH  # 3968

# 6-partition layout. Part 1 is embed-only; parts 2..5 each own 6
# transformer layers; part 6 owns 4 layers + lm_head.
NUM_PARTS = 6
PART_LAYER_RANGES = {
    2: (0, 6),
    3: (6, 12),
    4: (12, 18),
    5: (18, 24),
    6: (24, 28),
}

# Hidden-state handoff names per partition. These come from the
# upstream ONNX export's tensor names — they're stable per Qwen2.5-7B
# but distinct from 4B (Qwen3) wrapping.
HIDDEN_FROM_PART1 = "_model_embed_tokens_Gather_Gather_output_0"
HIDDEN_FROM_PART2 = "_model_layers_5_Add_1_Add_output_0"
HIDDEN_FROM_PART3 = "_model_layers_11_Add_1_Add_output_0"
HIDDEN_FROM_PART4 = "_model_layers_17_Add_1_Add_output_0"
HIDDEN_FROM_PART5 = "_model_layers_23_Add_1_Add_output_0"

PART_HIDDEN_IN = {
    2: HIDDEN_FROM_PART1,
    3: HIDDEN_FROM_PART2,
    4: HIDDEN_FROM_PART3,
    5: HIDDEN_FROM_PART4,
    6: HIDDEN_FROM_PART5,
}
PART_HIDDEN_OUT = {
    2: HIDDEN_FROM_PART2,
    3: HIDDEN_FROM_PART3,
    4: HIDDEN_FROM_PART4,
    5: HIDDEN_FROM_PART5,
    # part 6 emits "logits" instead of a hidden state
}


def half_dim_rope_quantized(
    pos: int, scale: float, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = pos * inv_freq
    cos_h = np.cos(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    sin_h = np.sin(freqs).reshape(1, 1, 1, HALF_HEAD_DIM)
    return quant_uint16(cos_h, scale, offset), quant_uint16(sin_h, scale, offset)


def half_dim_rope_quantized_ar128(
    p_base: int, scale: float, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    positions = np.arange(p_base, p_base + AR128_BATCH, dtype=np.float32).reshape(-1, 1)
    freqs = positions * inv_freq.reshape(1, -1)
    cos_h = np.cos(freqs).reshape(1, 1, AR128_BATCH, HALF_HEAD_DIM)
    sin_h = np.sin(freqs).reshape(1, 1, AR128_BATCH, HALF_HEAD_DIM)
    return quant_uint16(cos_h, scale, offset), quant_uint16(sin_h, scale, offset)


def attention_mask_quantized(
    pos: int, scale: float, offset: int, ctx_len: int = CTX_LEN
) -> np.ndarray:
    mask = np.zeros((1, 1, 1, ctx_len), dtype=np.uint16)
    if pos > 0:
        mask[..., :pos] = 65535
    mask[..., -1] = 65535
    _ = scale, offset
    return mask


def attention_mask_quantized_ar128(
    p_base: int, scale: float, offset: int, ctx_len: int = CTX_LEN
) -> np.ndarray:
    past_len = ctx_len - AR128_BATCH
    mask = np.zeros((1, 1, AR128_BATCH, ctx_len), dtype=np.uint16)
    if p_base > 0:
        mask[..., :p_base] = 65535
    causal = np.tri(AR128_BATCH, AR128_BATCH, k=0, dtype=np.uint16) * 65535
    mask[0, 0, :, past_len:] = causal
    _ = scale, offset
    return mask


# QNN dtype string -> 4B-oracle-style short dtype name.
_QNN_DTYPE_TO_SHORT = {
    "QNN_DATATYPE_UFIXED_POINT_8": "uint8",
    "QNN_DATATYPE_UFIXED_POINT_16": "uint16",
    "QNN_DATATYPE_INT_32": "int32",
    "QNN_DATATYPE_FLOAT_32": "float32",
}


def _io_spec_from_qnn(t: dict) -> dict:
    """Convert one tensor entry from `qnn-context-binary-utility` JSON
    to the 4B-oracle's per-IO spec dict shape."""
    info = t["info"]
    qp = info.get("quantizeParams", {}).get("scaleOffset", {})
    return {
        "name": info["name"],
        "metaname": info["name"],  # 7B QNN bin already uses underscored names
        "shape": list(info["dimensions"]),
        "dtype": _QNN_DTYPE_TO_SHORT[info["dataType"]],
        "scale": qp.get("scale"),
        "offset": qp.get("offset"),
    }


def build_part_cfg(ar: int = 1, ctx: int = CTX_LEN) -> dict:
    """Read the cached `qnn-context-binary-utility` JSONs for parts 1..6
    and return a dict in the same shape as the 4B oracle's
    `build_part_cfg` output. Filters each .bin's two graphs (token AR1
    + prompt AR128) by the requested `ar`.

    `ctx` accepts only 4096 — the only tier this bundle ships.
    """
    if ctx != CTX_LEN:
        raise ValueError(f"7B bundle only ships cl={CTX_LEN}; got {ctx}")
    if ar not in (1, 128):
        raise ValueError(f"ar must be 1 or 128, got {ar}")
    cfg = {}
    prefix = "token" if ar == 1 else "prompt"
    for part in range(1, NUM_PARTS + 1):
        json_path = QNN_METADATA_DIR / f"part{part}.json"
        meta = json.loads(json_path.read_text())
        target_name = f"{prefix}_ar{ar}_cl{ctx}_{part}_of_{NUM_PARTS}"
        graph = next(
            g for g in meta["info"]["graphs"]
            if g["info"]["graphName"] == target_name
        )
        cfg[part] = {
            "bin": f"qwen2_5_7b_instruct_w8a16_part_{part}_of_{NUM_PARTS}.bin",
            "graph_name": target_name,
            "ar": ar,
            "ctx": ctx,
            "inputs": [_io_spec_from_qnn(t) for t in graph["info"]["graphInputs"]],
            "outputs": [_io_spec_from_qnn(t) for t in graph["info"]["graphOutputs"]],
        }
    return cfg


def wrapper_path(bundle_dir: Path, part_idx: int, suffix: str = "", ctx: int = CTX_LEN) -> Path:
    """7B bundle has only cl=4096; suffix is "" (AR1) or "_ar128"."""
    return bundle_dir / f"oracle_7b_part{part_idx}{suffix}_cl{ctx}.wrapper.onnx"


class KVStore:
    """Persistent uint8 KV cache for the 7B layout.

    Per-layer shapes:
      keys[layer]:   [num_kv_heads, 1, head_dim, past_len]   = [4, 1, 128, 4095]
      values[layer]: [num_kv_heads, 1, past_len, head_dim]   = [4, 1, 4095, 128]

    The "zero code" is q=128 (offset=-128 → dequant=0) for every layer
    we've inspected; consistent with the 4B convention.
    """

    def __init__(
        self,
        num_layers: int,
        with_ar128_input: bool = False,
        ctx_len: int = CTX_LEN,
    ):
        self.num_layers = num_layers
        self.ctx_len = ctx_len
        self.past_len = ctx_len - 1
        self.past_len_ar128 = ctx_len - AR128_BATCH
        self.keys: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, HEAD_DIM, self.past_len), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        self.values: list[np.ndarray] = [
            np.full((NUM_KV_HEADS, 1, self.past_len, HEAD_DIM), 128, dtype=np.uint8)
            for _ in range(num_layers)
        ]
        self.has_ar128_in = with_ar128_input
        if with_ar128_input:
            self.keys_ar128_in: list[np.ndarray] = [
                np.full(
                    (NUM_KV_HEADS, 1, HEAD_DIM, self.past_len_ar128), 128, dtype=np.uint8
                )
                for _ in range(num_layers)
            ]
            self.values_ar128_in: list[np.ndarray] = [
                np.full(
                    (NUM_KV_HEADS, 1, self.past_len_ar128, HEAD_DIM), 128, dtype=np.uint8
                )
                for _ in range(num_layers)
            ]
        self.t = 0

    def stitch_step(self, k_outs: list[np.ndarray], v_outs: list[np.ndarray]) -> None:
        if self.t >= self.past_len:
            raise RuntimeError(f"KV cache full at t={self.t}; ctx={self.ctx_len}")
        for i in range(self.num_layers):
            self.keys[i][..., self.t : self.t + 1] = k_outs[i]
            self.values[i][:, :, self.t : self.t + 1, :] = v_outs[i]
        self.t += 1

    def stitch_batch(
        self,
        p_base: int,
        k_outs: list[np.ndarray],
        v_outs: list[np.ndarray],
    ) -> None:
        end = p_base + AR128_BATCH
        if end > self.past_len:
            raise RuntimeError(
                f"AR128 batch ending at {end} exceeds buffer {self.past_len}"
            )
        mirror = self.has_ar128_in and end <= self.past_len_ar128
        for i in range(self.num_layers):
            self.keys[i][..., p_base:end] = k_outs[i]
            self.values[i][:, :, p_base:end, :] = v_outs[i]
            if mirror:
                self.keys_ar128_in[i][..., p_base:end] = k_outs[i]
                self.values_ar128_in[i][:, :, p_base:end, :] = v_outs[i]
        self.t = end


def layer_input_name(kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_in"


def layer_output_name(kv: str, layer: int) -> str:
    return f"past_{kv}_{layer}_out"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "results" / "qualcomm_qwen2_5_7b_oracle"))
    args = parser.parse_args()

    print(f"=== Qualcomm Qwen2.5-7B w8a16 oracle (AR1 / CL{CTX_LEN}) ===")
    print(f"bundle: {BUNDLE_DIR}")
    print(f"prompt: {args.prompt!r}")

    parts = build_part_cfg(ar=1, ctx=CTX_LEN)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt).ids
    print(f"prompt: {len(args.prompt)} chars, {len(prompt_ids)} tokens")
    print(f"prefill positions 0..{len(prompt_ids) - 1}, "
          f"generate positions {len(prompt_ids)}..{len(prompt_ids) + args.gen_steps - 1}")

    # Build wrappers + load 6 sessions.
    print(f"\n--- building wrappers and loading {NUM_PARTS} sessions ---")
    sessions: dict[int, ort.InferenceSession] = {}
    t0 = time.perf_counter()
    for part_idx in range(1, NUM_PARTS + 1):
        wp = wrapper_path(BUNDLE_DIR, part_idx, "", CTX_LEN)
        if not wp.exists():
            build_wrapper(parts[part_idx], wp)
        t_load = time.perf_counter()
        sessions[part_idx] = load_session(wp, backend_path=QAIRT_2_45_BACKEND)
        print(f"  part {part_idx}: wrapper={wp.name}, "
              f"loaded in {time.perf_counter() - t_load:.1f} s")
    print(f"all {NUM_PARTS} parts loaded in {time.perf_counter() - t0:.1f} s")

    # Capture per-input scales. Use part 2 (first transformer part) for
    # cos/mask scales — they're identical across parts 2..6.
    cos_scale = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["scale"]
    cos_offset = next(io for io in parts[2]["inputs"] if io["name"] == "position_ids_cos")["offset"]
    mask_scale = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["scale"]
    mask_offset = next(io for io in parts[2]["inputs"] if io["name"] == "attention_mask")["offset"]
    logits_io = next(io for io in parts[NUM_PARTS]["outputs"] if io["name"] == "logits")
    logits_scale = logits_io["scale"]
    logits_offset = logits_io["offset"]

    kv = KVStore(NUM_LAYERS)

    all_logits_uint16: list[np.ndarray] = []
    all_logits_fp32: list[np.ndarray] = []
    all_argmax: list[int] = []
    all_step_tokens: list[int] = []
    all_decoded: list[str] = []
    step_latency_ms: list[float] = []

    total_steps = len(prompt_ids) + args.gen_steps
    next_token = prompt_ids[0]
    print(f"\n--- decoding {total_steps} steps ---")
    for step in range(total_steps):
        is_prefill = step < len(prompt_ids)
        token_in = prompt_ids[step] if is_prefill else next_token
        all_step_tokens.append(token_in)
        position = step

        cos_q, sin_q = half_dim_rope_quantized(position, cos_scale, cos_offset)
        mask_q = attention_mask_quantized(position, mask_scale, mask_offset)

        t_step = time.perf_counter()

        # Part 1: embed
        feed1 = {"input_ids": np.array([[token_in]], dtype=np.int32)}
        emb_out = sessions[1].run([HIDDEN_FROM_PART1], feed1)[0]

        # Parts 2..6: transformer layers (and lm_head on part 6)
        new_keys_all: list[np.ndarray] = [None] * NUM_LAYERS  # type: ignore
        new_vals_all: list[np.ndarray] = [None] * NUM_LAYERS  # type: ignore
        hidden = emb_out
        logits_uint16 = None
        for part_idx in range(2, NUM_PARTS + 1):
            layer_lo, layer_hi = PART_LAYER_RANGES[part_idx]
            feed = {
                PART_HIDDEN_IN[part_idx]: hidden,
                "attention_mask": mask_q,
                "position_ids_cos": cos_q,
                "position_ids_sin": sin_q,
            }
            for layer in range(layer_lo, layer_hi):
                feed[layer_input_name("key", layer)] = kv.keys[layer]
                feed[layer_input_name("value", layer)] = kv.values[layer]
            if part_idx == NUM_PARTS:
                # Last part emits logits + KV (no next-hidden)
                out_names = ["logits"] + [
                    layer_output_name(kvtype, layer)
                    for layer in range(layer_lo, layer_hi)
                    for kvtype in ("key", "value")
                ]
                outs = sessions[part_idx].run(out_names, feed)
                logits_uint16 = outs[0]
                for i, layer in enumerate(range(layer_lo, layer_hi)):
                    new_keys_all[layer] = outs[1 + 2 * i]
                    new_vals_all[layer] = outs[1 + 2 * i + 1]
            else:
                out_names = [PART_HIDDEN_OUT[part_idx]] + [
                    layer_output_name(kvtype, layer)
                    for layer in range(layer_lo, layer_hi)
                    for kvtype in ("key", "value")
                ]
                outs = sessions[part_idx].run(out_names, feed)
                hidden = outs[0]
                for i, layer in enumerate(range(layer_lo, layer_hi)):
                    new_keys_all[layer] = outs[1 + 2 * i]
                    new_vals_all[layer] = outs[1 + 2 * i + 1]

        kv.stitch_step(new_keys_all, new_vals_all)

        step_latency_ms.append((time.perf_counter() - t_step) * 1000)

        logits_fp32 = dequant_uint16(logits_uint16, logits_scale, logits_offset)
        argmax_id = int(np.argmax(logits_fp32))
        all_logits_uint16.append(logits_uint16.squeeze())
        all_logits_fp32.append(logits_fp32.squeeze())
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
                  f"in={token_in:6d}  (next argmax preview={argmax_id:6d})  "
                  f"{step_latency_ms[-1]:.1f} ms")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.out).with_suffix(".npz")
    md_path = Path(args.out).with_suffix(".md")

    np.savez_compressed(
        npz_path,
        logits_uint16=np.stack(all_logits_uint16),
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
        "# Qualcomm Qwen2.5-7B w8a16 oracle trace\n\n"
        f"- bundle: `{BUNDLE_DIR.relative_to(REPO_ROOT)}`\n"
        f"- mode: AR1 / CL{CTX_LEN}, greedy argmax\n"
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
    raise SystemExit(main())
