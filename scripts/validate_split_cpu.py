"""Round-trip validation of the 4-part pathb split against the
monolithic pathb-ctx512 graph on CPU-ORT.

Drives both paths on a fixed synthetic input and reports cosine
similarity on `logits` plus per-layer KV tensors. Gate: cos >= 0.9999
(this is a pure topology split, no quant/lowering — anything less
than bit-for-bit on fp32 ORT indicates a splitting bug).

Run:
    .venv/Scripts/python.exe scripts/validate_split_cpu.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
MONO_DIR = MODELS / "qwen3-4b-arm-pathb-ctx512"
PART_DIRS = [MODELS / f"qwen3-4b-arm-pathb-ctx512-part{i}" for i in (1, 2, 3, 4)]

NUM_LAYERS = 36
LAYERS_PER_PART = 12
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 2560
CTX = 512
PAST = CTX - 1
ROPE_THETA = 1_000_000.0

EMBED_HIDDEN = "/model/embed_tokens/Gather_output_0"
L11_HIDDEN = "/model/layers.11/Add_1_output_0"
L23_HIDDEN = "/model/layers.23/Add_1_output_0"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a_f))
    nb = float(np.linalg.norm(b_f))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a_f, b_f) / (na * nb))


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


def load_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path / "model.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])


def synth_inputs(rng: np.random.Generator, position: int = 10) -> dict[str, np.ndarray]:
    cos, sin = rope_full_dim(position)
    feed: dict[str, np.ndarray] = {
        "input_ids": np.array([[12345]], dtype=np.int64),
        "position_ids": np.array([[position]], dtype=np.int64),
        "attention_bias": attention_bias_at(position),
        "position_ids_cos": cos,
        "position_ids_sin": sin,
    }
    for li in range(NUM_LAYERS):
        # Non-zero KV so any mis-plumbing between parts actually diverges.
        feed[f"past_key_values.{li}.key"] = rng.standard_normal(
            (1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32
        ) * 0.1
        feed[f"past_key_values.{li}.value"] = rng.standard_normal(
            (1, NUM_KV_HEADS, PAST, HEAD_DIM), dtype=np.float32
        ) * 0.1
    return feed


def run_monolithic(feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # The ctx512 pinned graph still has position_ids as a declared input
    # even though rotary was hoisted out, so keep it in the feed.
    print(f"  loading monolithic {MONO_DIR} ...")
    t0 = time.perf_counter()
    sess = load_session(MONO_DIR)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s "
          f"({len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs)")
    input_names = {i.name for i in sess.get_inputs()}
    # Prune feed to declared inputs (monolithic may or may not have position_ids after Phase 3 --remove_unused_inputs).
    mono_feed = {k: v for k, v in feed.items() if k in input_names}
    missing = input_names - set(mono_feed)
    if missing:
        raise RuntimeError(f"monolithic missing inputs: {missing}")
    t0 = time.perf_counter()
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, mono_feed)
    print(f"  monolithic forward: {time.perf_counter() - t0:.1f}s")
    return dict(zip(out_names, outs))


def run_split(feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sessions = []
    for p in PART_DIRS:
        print(f"  loading {p.name} ...")
        t0 = time.perf_counter()
        sessions.append(load_session(p))
        print(f"    loaded in {time.perf_counter() - t0:.1f}s")

    # Part 1: embed
    p1 = sessions[0]
    p1_out = p1.run(None, {"input_ids": feed["input_ids"]})[0]

    # Shared tensors for decode parts.
    shared = {
        "attention_bias": feed["attention_bias"],
        "position_ids_cos": feed["position_ids_cos"],
        "position_ids_sin": feed["position_ids_sin"],
    }

    def decode_part(sess: ort.InferenceSession, hidden_in_name: str,
                    hidden_in: np.ndarray, layer_start: int, layer_end: int,
                    hidden_out_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        f: dict[str, np.ndarray] = {hidden_in_name: hidden_in, **shared}
        for li in range(layer_start, layer_end + 1):
            f[f"past_key_values.{li}.key"] = feed[f"past_key_values.{li}.key"]
            f[f"past_key_values.{li}.value"] = feed[f"past_key_values.{li}.value"]
        out_names = [hidden_out_name]
        for li in range(layer_start, layer_end + 1):
            out_names.append(f"present.{li}.key")
            out_names.append(f"present.{li}.value")
        outs = sess.run(out_names, f)
        hidden_out = outs[0]
        kv_map = dict(zip(out_names[1:], outs[1:]))
        return hidden_out, kv_map

    t0 = time.perf_counter()
    p2_hidden, p2_kv = decode_part(sessions[1], EMBED_HIDDEN, p1_out, 0, 11, L11_HIDDEN)
    print(f"  part2 forward: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    p3_hidden, p3_kv = decode_part(sessions[2], L11_HIDDEN, p2_hidden, 12, 23, L23_HIDDEN)
    print(f"  part3 forward: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    # Part 4's hidden output is `logits`, not a layer hidden — reuse the
    # helper with hidden_out_name="logits".
    logits, p4_kv = decode_part(sessions[3], L23_HIDDEN, p3_hidden, 24, 35, "logits")
    print(f"  part4 forward: {time.perf_counter() - t0:.1f}s")

    merged = {"logits": logits, **p2_kv, **p3_kv, **p4_kv}
    return merged


def main() -> int:
    print("=== 4-part pathb split CPU round-trip validation ===")
    rng = np.random.default_rng(seed=42)
    feed = synth_inputs(rng)

    print("\n-- monolithic run --")
    mono = run_monolithic(feed)

    print("\n-- split run --")
    split = run_split(feed)

    print("\n-- comparison --")
    # logits
    cos = cosine(mono["logits"], split["logits"])
    max_abs = float(np.max(np.abs(mono["logits"] - split["logits"])))
    print(f"  logits            cos={cos:.9f}  max_abs_diff={max_abs:.3e}")
    ok = cos >= 0.9999
    # per-layer KV spot-check (layers 0, 11, 12, 23, 24, 35)
    min_cos = cos
    for li in (0, 11, 12, 23, 24, 35):
        for kv in ("key", "value"):
            name = f"present.{li}.{kv}"
            c = cosine(mono[name], split[name])
            m = float(np.max(np.abs(mono[name] - split[name])))
            print(f"  {name:22s} cos={c:.9f}  max_abs_diff={m:.3e}")
            if c < min_cos:
                min_cos = c
    if min_cos < 0.9999:
        print(f"\nFAIL: min cos across probed tensors = {min_cos:.9f} (< 0.9999)")
        return 1
    print(f"\nPASS: min cos = {min_cos:.9f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
