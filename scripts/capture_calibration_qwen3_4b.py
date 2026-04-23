"""Capture w4a16 PTQ calibration samples for Qwen3-4B at ctx=512.

Runs the SOURCE (optimum, inline rotary) Qwen3-4B ONNX on CPU-ORT,
prefills each prompt, then snapshots the inputs that PATHB expects:

    input_ids               [1, 1]                int64
    position_ids            [1, 1]                int64
    attention_bias          [1, 1, 1, 512]        float32
    position_ids_cos        [1, 1, 128]           float32
    position_ids_sin        [1, 1, 128]           float32
    past_key_values.{0..35}.key    [1, 8, 511, 128]  float32
    past_key_values.{0..35}.value  [1, 8, 511, 128]  float32

Saved as one npz with each tensor stacked along a sample-leading axis.

Source ONNX: models/qwen3-4b-arm-optimum/model.onnx (37 hf inputs:
input_ids + position_ids + attention_mask + 36 layers x 2 past_kv).
Pathb ONNX: same logical model, just rotary hoisted + attention_bias
splice — so KV state captured here is directly usable for pathb PTQ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REPO = Path(__file__).resolve().parents[1]

NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 2560
ROPE_THETA = 1_000_000.0
PAST_LEN = 511
CTX_LEN = 512


def rope_full_dim(position: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-dim cos/sin matching the optimum reference graph (NOT the
    half-dim Qualcomm convention — that's a Phase-4 lever)."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq                             # [64]
    emb = np.concatenate([freqs, freqs], axis=-1)           # [128]
    cos = np.cos(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    sin = np.sin(emb).astype(np.float32).reshape(1, 1, HEAD_DIM)
    return cos, sin


def attention_bias_at(position: int, ctx: int) -> np.ndarray:
    """Additive mask for AR=1 decode at `position`, KV layout
    [past_511 | current_1] = ctx slots. Past slots 0..position-1 are
    valid history; position..510 are empty; slot 511 is the current."""
    bias = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
    bias[..., :position] = 0.0  # valid past
    bias[..., -1] = 0.0          # current always attends
    return bias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path,
                        default=REPO / "models" / "qwen3-4b-arm-optimum" / "model.onnx")
    parser.add_argument("--tokenizer", type=Path,
                        default=REPO / "models" / "qwen3-4b-arm-optimum" / "tokenizer.json")
    parser.add_argument("--prompts", type=Path,
                        default=REPO / "prompts" / "humaneval_subset.jsonl")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Number of prompts to use (one calibration sample per prompt at the chosen capture position).")
    parser.add_argument("--capture-position", type=int, default=10,
                        help="Decode step at which to snapshot the model state. "
                             "10 = capture mid-prefill state for typical-length prompts.")
    parser.add_argument("--out", type=Path,
                        default=REPO / "models" / "calibration" / "qwen3_4b_ctx512_a.npz")
    args = parser.parse_args()

    print(f"loading source ONNX: {args.source_onnx}")
    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(args.source_onnx), sess_options=so,
                                providers=["CPUExecutionProvider"])
    print(f"  loaded in {time.perf_counter() - t0:.1f} s "
          f"({len(sess.get_inputs())} inputs / {len(sess.get_outputs())} outputs)")
    print(f"loading tokenizer: {args.tokenizer}")
    tok = Tokenizer.from_file(str(args.tokenizer))

    print(f"reading prompts: {args.prompts}")
    prompts: list[str] = []
    import json
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
            if len(prompts) >= args.n_prompts:
                break
    print(f"  using {len(prompts)} prompts")

    # Per-sample collected inputs.
    samples: dict[str, list[np.ndarray]] = {
        "input_ids": [],
        "position_ids": [],
        "attention_bias": [],
        "position_ids_cos": [],
        "position_ids_sin": [],
    }
    for li in range(NUM_LAYERS):
        samples[f"past_key_values.{li}.key"] = []
        samples[f"past_key_values.{li}.value"] = []

    for pi, prompt in enumerate(prompts):
        ids = tok.encode(prompt).ids
        if len(ids) <= args.capture_position:
            print(f"  skipping prompt {pi}: only {len(ids)} tokens (need > {args.capture_position})")
            continue
        # Run prefill up to capture_position; we want the past_kv state AT
        # that position (i.e. positions 0..position-1 already in cache),
        # and the input at position itself is what gets captured.
        # Build feed for a single forward of `capture_position+1` tokens
        # with empty past, so present_kv outputs give us positions 0..position.
        # Then sliced past_kv[..., :position, :] is what pathb expects at
        # the next decode step (position).
        run_ids = ids[: args.capture_position + 1]
        L = len(run_ids)
        feed = {
            "input_ids": np.array([run_ids], dtype=np.int64),
            "position_ids": np.array([list(range(L))], dtype=np.int64),
            "attention_mask": np.ones((1, L), dtype=np.int64),
        }
        for li in range(NUM_LAYERS):
            feed[f"past_key_values.{li}.key"] = np.zeros(
                (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            feed[f"past_key_values.{li}.value"] = np.zeros(
                (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
        t0 = time.perf_counter()
        outs = sess.run(None, feed)
        elapsed = time.perf_counter() - t0
        # Map outputs by name.
        out_names = [o.name for o in sess.get_outputs()]
        out_map = dict(zip(out_names, outs))

        # Build the calibration sample for the NEXT decode step (position).
        position = args.capture_position
        # KV cache after the prefill: present.{li}.key has shape
        # [1, 8, position+1, 128]. pathb's pinned shape is [1, 8, 511, 128]
        # — we right-pad with zeros (future slots are masked anyway).
        pad_to = PAST_LEN
        for li in range(NUM_LAYERS):
            kv_k_full = out_map[f"present.{li}.key"]      # [1,8,L,128]
            kv_v_full = out_map[f"present.{li}.value"]    # [1,8,L,128]
            assert kv_k_full.shape == (1, NUM_KV_HEADS, L, HEAD_DIM)
            # We use the FIRST `position` positions (0..position-1) as past.
            kv_k_past = kv_k_full[:, :, :position, :]
            kv_v_past = kv_v_full[:, :, :position, :]
            # Pad to PAST_LEN with zeros.
            pad_k = np.zeros((1, NUM_KV_HEADS, pad_to - position, HEAD_DIM), dtype=np.float32)
            samples[f"past_key_values.{li}.key"].append(
                np.concatenate([kv_k_past, pad_k], axis=2).astype(np.float32)
            )
            samples[f"past_key_values.{li}.value"].append(
                np.concatenate([kv_v_past, pad_k], axis=2).astype(np.float32)
            )

        # The token at this position = ids[position]
        samples["input_ids"].append(np.array([[ids[position]]], dtype=np.int64))
        samples["position_ids"].append(np.array([[position]], dtype=np.int64))
        samples["attention_bias"].append(attention_bias_at(position, CTX_LEN))
        cos, sin = rope_full_dim(position)
        samples["position_ids_cos"].append(cos)
        samples["position_ids_sin"].append(sin)

        print(f"  sample {pi}: prompt_tokens={len(ids)}, prefill_len={L}, "
              f"capture_pos={position}, prefill {elapsed:.1f}s")

    # Stack all samples and save.
    n_kept = len(samples["input_ids"])
    print(f"\n captured {n_kept} samples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stacked = {k: np.stack(v) for k, v in samples.items()}
    np.savez_compressed(args.out, **stacked)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
