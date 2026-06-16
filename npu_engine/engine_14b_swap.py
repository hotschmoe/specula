"""Streaming 2-group decode for the calibrated 14B bundle (fits the ~10 GB HTP wall).

The full w8a16 bundle (~16 GB) exceeds the HTP context-memory ceiling (~6
contexts / ~10 GB) — ORT-QNN can't hold all 10 parts at once and doesn't honor
genie's use-mmap. So we stream: load parts 1..K (≤6 contexts, separate sessions
with host-bridged fp32 seams + KV), run them, unload, load K+1..10, run. The
engine's host-side fp32 seam + KV stores carry state across the swap. Seam/KV
encodings differ per part, so every boundary is dequant→fp32→requant in host
(exact) — that's why each part stays its own session.

Prefill is batched: run all prompt tokens through group A, save per-token
boundary seams, then run them all through group B (one swap for prefill).
Decode is autoregressive (one swap per token).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe npu_engine/engine_14b_swap.py \
        --bundle models/qwen3_14b-w8a16-specula-x2e-calib --split 5 --gen 8 \
        --ref results/pathb_eval/ref_cpu_14b.npz
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_14b_q import (Part, Session, quant, dequant, rope_cache,  # noqa: E402
                          load_parts, _layer, EVAL_DIR)


def run_session(s: Session, fp32_seam, kv, tid, cb, sb, ab):
    """Run one part's session: fill its graph inputs from host stores (quant as
    needed), run, write outputs back to host stores (dequant as needed)."""
    b = s.buf

    def setq(name, fp32val, t):
        if t.quantized:
            b[name][:] = quant(fp32val, t.scale, t.offset, t.qmax)
        elif b[name].dtype == np.int64:
            b[name][:] = fp32val
        else:
            b[name][:] = fp32val.astype(b[name].dtype)

    for nm, t in s.gin.items():
        if nm == "input_ids":
            b[nm][0, 0] = tid
        elif nm == "position_ids_cos":
            setq(nm, cb.reshape(b[nm].shape), t)
        elif nm == "position_ids_sin":
            setq(nm, sb.reshape(b[nm].shape), t)
        elif nm == "attention_bias":
            setq(nm, ab.reshape(b[nm].shape), t)
        elif nm.startswith("past_key_values"):
            L = _layer(nm)
            setq(nm, kv[(L, "key" if nm.endswith("key") else "value")], t)
        else:
            setq(nm, fp32_seam[nm], t)
    s.run()
    logits = None
    for nm, t in s.gout.items():
        val = dequant(b[nm], t.scale, t.offset) if t.quantized else b[nm].astype(np.float32)
        if nm == "logits":
            logits = val.reshape(-1).copy()
        elif nm.startswith("present"):
            L = _layer(nm)
            kv[(L, "key" if nm.endswith("key") else "value")][:] = val[:, :, 1:, :]
        else:
            fp32_seam[nm] = val.copy()
    return logits


class Group:
    """A contiguous set of parts as separate sessions, loaded/unloaded together."""
    def __init__(self, parts, bundle):
        self.parts, self.bundle = parts, bundle
        self.sessions = None

    def load(self):
        self.sessions = [Session([p], self.bundle) for p in self.parts]
        t0 = time.perf_counter()
        for s in self.sessions:
            s.load()
            s.alloc()
            s.bind()
        return time.perf_counter() - t0

    def unload(self):
        for s in self.sessions:
            s.s = None
            s.io = None
            s.buf = None
        self.sessions = None
        gc.collect()

    def run(self, fp32_seam, kv, tid, cb, sb, ab):
        logits = None
        for s in self.sessions:
            lg = run_session(s, fp32_seam, kv, tid, cb, sb, ab)
            if lg is not None:
                logits = lg
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--split", type=int, default=5, help="parts 1..split in group A")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--tag", default="w8a16_14b_swap")
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    cfg = json.loads((bundle / "config.json").read_text())
    ctx, n_layers = int(meta["ctx"]), int(cfg["num_hidden_layers"])
    head_dim, theta = int(cfg["head_dim"]), float(cfg["rope_theta"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    prompt_ids = tok.encode((bundle / "sample_prompt.txt").read_text("utf-8")).ids[:args.max_prompt]
    cos, sin = rope_cache(theta, head_dim, ctx + 64)

    parts = load_parts(bundle)
    gA = Group(parts[:args.split], bundle)
    gB = Group(parts[args.split:], bundle)
    boundary = parts[args.split - 1].seam_out.name  # group A -> B seam
    print(f"=== engine_14b_swap: {bundle.name} | {len(prompt_ids)} prompt tok | "
          f"split A=parts1-{args.split} B=parts{args.split+1}-{len(parts)} ===")
    print(f"  boundary seam: {boundary}")

    fp32_seam = {}
    kv = {(L, k): np.zeros((1, 8, ctx - 1, head_dim), np.float32)
          for L in range(n_layers) for k in ("key", "value")}

    def ab_at(pos):
        a = np.full((1, 1, 1, ctx), -65504.0, np.float32)
        a[..., ctx - 1 - pos:] = 0.0
        return a

    # ---- PREFILL (batched: 1 swap) ----
    t0 = time.perf_counter()
    print(f"\n--- prefill group A (parts 1-{args.split}) over {len(prompt_ids)} tokens ---")
    dt = gA.load()
    print(f"  group A loaded {dt:.1f}s")
    seam_saved = []
    for pos, tid in enumerate(prompt_ids):
        gA.run(fp32_seam, kv, tid, cos[pos], sin[pos], ab_at(pos))
        seam_saved.append(fp32_seam[boundary].copy())
    gA.unload()
    print(f"--- prefill group B (parts {args.split+1}-{len(parts)}) ---")
    dt = gB.load()
    print(f"  group B loaded {dt:.1f}s")
    last = None
    for pos, tid in enumerate(prompt_ids):
        fp32_seam[boundary] = seam_saved[pos]
        last = gB.run(fp32_seam, kv, tid, cos[pos], sin[pos], ab_at(pos))
    gB.unload()
    first_logits = last.copy()
    print(f"  prefill done {time.perf_counter()-t0:.1f}s")

    # ---- DECODE (autoregressive: 2 swaps/token) ----
    print(f"\n--- decode {args.gen} tokens (swap/token) ---")
    nxt = int(np.argmax(last))
    gen, tg = [], []
    for i in range(args.gen):
        pos = len(prompt_ids) + i
        t1 = time.perf_counter()
        gA.load()
        gA.run(fp32_seam, kv, nxt, cos[pos], sin[pos], ab_at(pos))
        gA.unload()
        gB.load()
        lg = gB.run(fp32_seam, kv, nxt, cos[pos], sin[pos], ab_at(pos))
        gB.unload()
        tg.append(time.perf_counter() - t1)
        gen.append(nxt)
        nxt = int(np.argmax(lg))
        print(f"  tok {i}: {tok.id_to_token(gen[-1])!r} ({tg[-1]:.0f}s)")

    txt = tok.decode(gen)
    top5 = np.argsort(first_logits)[-5:][::-1]
    print(f"\n=== RESULT ===")
    print(f"  first-decode top5: {[(int(t), tok.id_to_token(int(t))) for t in top5]}")
    print(f"  continuation: {txt!r}")
    if args.ref and args.ref.exists():
        ref = np.load(args.ref, allow_pickle=True)["first_decode_logits"].astype(np.float64)
        a = first_logits.astype(np.float64)
        cs = float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref)))
        print(f"  COS-SIM vs CPU ref: {cs:.5f}  argmax NPU={int(top5[0])} "
              f"ref={int(np.argmax(ref))} {'MATCH' if int(top5[0])==int(np.argmax(ref)) else 'DIFFER'}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL_DIR / f"{args.tag}.npz", first_decode_logits=first_logits,
                        gen_ids=np.array(gen, np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main())
