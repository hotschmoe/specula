"""CPU-equivalence probe for the Path B (rotary-hoisted) ONNX.

Compares last-token logits between
  REF:  models/qwen3-0.6b-optimum/model.onnx       (reference, inline rotary)
  NEW:  models/qwen3-0.6b-pathb/model.onnx         (rotary hoisted as inputs)

Gate (per docs/phase5_export_on_x86.md, Path B section):
  - cosine(REF_logits, NEW_logits) >= 0.9999
  - argmax(REF_logits) == argmax(NEW_logits)
  - top-5 overlap == 5/5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "models" / "qwen3-0.6b-optimum" / "model.onnx"
NEW = REPO / "models" / "qwen3-0.6b-pathb" / "model.onnx"

# Qwen3-0.6B
HEAD_DIM = 128
NUM_LAYERS = 28
NUM_KV_HEADS = 8
ROPE_THETA = 1_000_000.0
BOS = 151643
CTX = 256          # arbitrary; pathb keeps decode-step shape symbolic


def rope_tables(position_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute cos/sin for a single decode step.

    Matches the optimum export's rotary_emb chain:
      inv_freq -> position * inv_freq -> Concat([f,f]) -> Cos/Sin
    Output shape: [1, 1, HEAD_DIM] (matches pathb's graph input).
    """
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position_id * inv_freq                  # [HEAD_DIM/2]
    emb = np.concatenate([freqs, freqs], axis=-1)   # [HEAD_DIM]
    cos = np.cos(emb)[None, None, :].astype(np.float32)
    sin = np.sin(emb)[None, None, :].astype(np.float32)
    return cos, sin


def make_zero_kv(past_len: int) -> dict[str, np.ndarray]:
    """Build all-zero past_key_values for empty-KV decode probe."""
    feed = {}
    for i in range(NUM_LAYERS):
        kv_shape = (1, NUM_KV_HEADS, past_len, HEAD_DIM)
        feed[f"past_key_values.{i}.key"] = np.zeros(kv_shape, dtype=np.float32)
        feed[f"past_key_values.{i}.value"] = np.zeros(kv_shape, dtype=np.float32)
    return feed


def make_attention_bias(seq_q: int, seq_k: int) -> np.ndarray:
    """All-valid additive mask: zeros everywhere (no positions masked)."""
    return np.zeros((1, 1, seq_q, seq_k), dtype=np.float32)


def make_attention_mask(total_len: int) -> np.ndarray:
    """All-valid bool/int mask for the optimum reference graph."""
    return np.ones((1, total_len), dtype=np.int64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-12) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    print(f"loading REF: {REF}")
    s_ref = ort.InferenceSession(str(REF), providers=["CPUExecutionProvider"])
    print(f"loading NEW: {NEW}")
    s_new = ort.InferenceSession(str(NEW), providers=["CPUExecutionProvider"])

    ref_inputs = {i.name for i in s_ref.get_inputs()}
    new_inputs = {i.name for i in s_new.get_inputs()}
    print(f"  REF inputs: {len(ref_inputs)}")
    print(f"  NEW inputs: {len(new_inputs)}")
    print(f"  NEW-only: {sorted(new_inputs - ref_inputs)}")
    print(f"  REF-only: {sorted(ref_inputs - new_inputs)}")

    # ------------------------------------------------------------------
    # Probe 1: position 0, BOS, zero KV.
    # ------------------------------------------------------------------
    print("\n=== probe 1: position=0, BOS=151643, zero KV ===")
    pos = 0
    input_ids = np.array([[BOS]], dtype=np.int64)
    position_ids = np.array([[pos]], dtype=np.int64)
    cos, sin = rope_tables(pos)

    past_len = 0   # zero-KV
    kv_zero = make_zero_kv(past_len)

    # REF feed (uses HF-style attention_mask covering total length = past + new = 1)
    ref_feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": make_attention_mask(past_len + 1),
        **kv_zero,
    }
    print(f"  running REF...")
    ref_out = s_ref.run(None, ref_feed)
    ref_logits = ref_out[0]
    print(f"  REF logits shape: {ref_logits.shape}")

    # NEW feed (additive mask + cos/sin)
    new_feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_bias": make_attention_bias(seq_q=1, seq_k=past_len + 1),
        "position_ids_cos": cos,
        "position_ids_sin": sin,
        **kv_zero,
    }
    print(f"  running NEW...")
    new_out = s_new.run(None, new_feed)
    new_logits = new_out[0]
    print(f"  NEW logits shape: {new_logits.shape}")

    last_ref = ref_logits[0, -1]
    last_new = new_logits[0, -1]
    cs = cosine(last_ref, last_new)
    argmax_ref = int(np.argmax(last_ref))
    argmax_new = int(np.argmax(last_new))
    top5_ref = set(np.argsort(last_ref)[-5:].tolist())
    top5_new = set(np.argsort(last_new)[-5:].tolist())
    overlap = len(top5_ref & top5_new)
    print(f"  cosine = {cs:.6f}")
    print(f"  argmax: REF={argmax_ref}  NEW={argmax_new}  match={argmax_ref == argmax_new}")
    print(f"  top-5 overlap: {overlap}/5")
    print(f"  top-5 REF: {sorted(top5_ref)}")
    print(f"  top-5 NEW: {sorted(top5_new)}")

    gate1 = cs >= 0.9999 and argmax_ref == argmax_new and overlap == 5

    # ------------------------------------------------------------------
    # Probe 2: position=5 with a small synthetic past_kv to exercise
    # rotary at a non-zero offset (where wrong cos/sin would diverge).
    # ------------------------------------------------------------------
    print("\n=== probe 2: position=5 with tiny synthetic past_kv ===")
    pos = 5
    past_len = 5
    rng = np.random.default_rng(0)
    kv_small = {}
    for i in range(NUM_LAYERS):
        kv_shape = (1, NUM_KV_HEADS, past_len, HEAD_DIM)
        kv_small[f"past_key_values.{i}.key"] = rng.standard_normal(kv_shape).astype(np.float32) * 0.01
        kv_small[f"past_key_values.{i}.value"] = rng.standard_normal(kv_shape).astype(np.float32) * 0.01

    input_ids = np.array([[100]], dtype=np.int64)   # arbitrary token
    position_ids = np.array([[pos]], dtype=np.int64)
    cos, sin = rope_tables(pos)

    ref_feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": make_attention_mask(past_len + 1),
        **kv_small,
    }
    ref_out = s_ref.run(None, ref_feed)
    ref_logits = ref_out[0]

    new_feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_bias": make_attention_bias(seq_q=1, seq_k=past_len + 1),
        "position_ids_cos": cos,
        "position_ids_sin": sin,
        **kv_small,
    }
    new_out = s_new.run(None, new_feed)
    new_logits = new_out[0]

    last_ref = ref_logits[0, -1]
    last_new = new_logits[0, -1]
    cs2 = cosine(last_ref, last_new)
    argmax_ref = int(np.argmax(last_ref))
    argmax_new = int(np.argmax(last_new))
    top5_ref = set(np.argsort(last_ref)[-5:].tolist())
    top5_new = set(np.argsort(last_new)[-5:].tolist())
    overlap2 = len(top5_ref & top5_new)
    print(f"  cosine = {cs2:.6f}")
    print(f"  argmax: REF={argmax_ref}  NEW={argmax_new}  match={argmax_ref == argmax_new}")
    print(f"  top-5 overlap: {overlap2}/5")

    gate2 = cs2 >= 0.9999 and argmax_ref == argmax_new and overlap2 == 5

    print("\n=== summary ===")
    print(f"  probe 1 (pos=0): cos={cs:.6f}  PASS={gate1}")
    print(f"  probe 2 (pos=5): cos={cs2:.6f} PASS={gate2}")

    if gate1 and gate2:
        print("\nALL GATES PASSED")
        return
    print("\nGATE FAILURE — investigate before handoff")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
