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

Snapshots a configurable set of decode positions per prompt (default
{0, 1, 5, 10, 20}) from a single forward pass covering max(positions)+1
tokens. Wide position coverage — especially position 0 (BOS + empty
past_kv) — exposes qairt-quantizer to the true first-decode-step
activation ranges. Prior to this, single-position-10 calibration
produced a ±11.5 encoding at the part2/3 seam while runtime step-0
activations hit ±16000, saturating ~99% of the signal (see
docs/qualcomm_reproduction_4b.md Phase 5g).

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
                        default=REPO / "prompts" / "calibration_chat.jsonl",
                        help="JSONL file with a 'prompt' field per line. Default is "
                             "chat-templated prompts whose position-0 token is 151644 "
                             "(<|im_start|>), matching the runtime prompt distribution. "
                             "Pass --prompts prompts/humaneval_subset.jsonl for the "
                             "old humaneval calibration.")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Number of prompts to use. One calibration sample is emitted "
                             "per (prompt, capture-position) pair.")
    parser.add_argument("--capture-positions", type=str, default="0,1,5,10,20",
                        help="Comma-separated decode positions to snapshot per prompt. "
                             "Default covers BOS (0), early-decode (1), short-prefill (5), "
                             "mid-prefill (10), long-prefill (20) so the quantizer sees "
                             "the full activation range at each pathb seam.")
    parser.add_argument("--out", type=Path,
                        default=REPO / "models" / "calibration" / "qwen3_4b_ctx512_a.npz")
    args = parser.parse_args()

    positions = sorted({int(p) for p in args.capture_positions.split(",") if p.strip()})
    if not positions or positions[0] < 0:
        print(f"FATAL: invalid --capture-positions {args.capture_positions!r}")
        return 2
    max_position = positions[-1]
    print(f"capture positions: {positions} (max={max_position})")

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
        if len(ids) <= max_position:
            print(f"  skipping prompt {pi}: only {len(ids)} tokens (need > {max_position})")
            continue
        # One forward of length max_position+1 produces present_kv for
        # positions 0..max_position; we then slice per target position.
        run_ids = ids[: max_position + 1]
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
        out_names = [o.name for o in sess.get_outputs()]
        out_map = dict(zip(out_names, outs))

        pad_to = PAST_LEN
        for position in positions:
            # past_kv = first `position` slots of present_kv, right-padded
            # to PAST_LEN with zeros (future slots masked by attention_bias).
            for li in range(NUM_LAYERS):
                kv_k_full = out_map[f"present.{li}.key"]      # [1,8,L,128]
                kv_v_full = out_map[f"present.{li}.value"]    # [1,8,L,128]
                kv_k_past = kv_k_full[:, :, :position, :]
                kv_v_past = kv_v_full[:, :, :position, :]
                pad_k = np.zeros((1, NUM_KV_HEADS, pad_to - position, HEAD_DIM), dtype=np.float32)
                samples[f"past_key_values.{li}.key"].append(
                    np.concatenate([kv_k_past, pad_k], axis=2).astype(np.float32)
                )
                samples[f"past_key_values.{li}.value"].append(
                    np.concatenate([kv_v_past, pad_k], axis=2).astype(np.float32)
                )

            samples["input_ids"].append(np.array([[ids[position]]], dtype=np.int64))
            samples["position_ids"].append(np.array([[position]], dtype=np.int64))
            samples["attention_bias"].append(attention_bias_at(position, CTX_LEN))
            cos, sin = rope_full_dim(position)
            samples["position_ids_cos"].append(cos)
            samples["position_ids_sin"].append(sin)

        print(f"  prompt {pi}: prompt_tokens={len(ids)}, prefill_len={L}, "
              f"captured {len(positions)} positions in {elapsed:.1f}s")

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
