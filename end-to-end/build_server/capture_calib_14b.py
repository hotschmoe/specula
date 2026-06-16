"""Capture per-part calibration inputs for the 14B decoder parts (2-9).

Runs the fp32 06_split chain on CPU-ORT autoregressively over a few
calibration prompts, threading the KV ring exactly like the runtime engine,
and dumps each decoder part's real input feeds as raw files + an
`input_list.txt` in the layout `qairt-quantizer --input_list` expects.

This is what was MISSING in the original build: with no calibration the
quantizer never derives activation encodings, so the HTP compiles a float
graph and weights are stored fp16 (3.3 GB/part, unloadable). Feeding these
raws lets the quantizer commit to the int8-weight path (~1.65 GB/part).

IO is kept fp32 (the 06_split ONNX inputs are fp32; we do NOT change IO
dtype), so seams stay FLOAT_32 across every part boundary — part1/part10
bins are reused unchanged.

Run INSIDE specula-qairt:2.45 (has onnxruntime), SSD mounted at /workspace:
    docker run --rm -v /mnt/vm_8tb/specula-build:/workspace specula-qairt:2.45 \
      python3 /workspace/end-to-end/build_server/capture_calib_14b.py \
        --upto 9 --samples 24
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SPLIT = Path("/workspace/runs/qwen3_14b_w8a16/06_split")
OUT = Path("/workspace/runs/qwen3_14b_w8a16/cal_raw")
HEAD_DIM = 128
N_KV = 8
CTX = 512
THETA = 1_000_000.0

# Pre-tokenized (bundle tokenizer, computed on the X2E) calibration set —
# diverse instruction-style prompts. Token IDs baked in so the container
# needs no tokenizer/transformers, just onnxruntime.
PROMPT_IDS = [
    [3838, 374, 23249, 30, 13655, 279, 4226, 1212, 5779, 4244, 13],
    [7985, 264, 6386, 38242, 911, 279, 17951, 518, 38393, 13],
    [840, 20772, 3170, 279, 12884, 374, 6303, 311, 264, 4236, 1042, 2310, 13],
    [852, 2326, 5711, 369, 264, 5567, 7974, 13],
    [27473, 364, 18536, 6556, 6, 1119, 8585, 323, 15154, 13],
    [9190, 5612, 551, 279, 7089, 315, 70192, 323, 71024, 304, 825, 11652, 13],
]


def rope_cache(theta, head_dim, max_pos):
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_pos, dtype=np.float64)
    emb = np.concatenate([np.outer(pos, inv)] * 2, -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_").lstrip("_")


def load_sess(part_dir: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(part_dir / "model.onnx"), sess_options=so,
                               providers=["CPUExecutionProvider"])


def classify(sess):
    """Return (seam_in, seam_out, present_names, past_names, has_cos)."""
    ins = [i.name for i in sess.get_inputs()]
    outs = [o.name for o in sess.get_outputs()]
    past = [n for n in ins if n.startswith("past_key_values.")]
    seam_in = [n for n in ins if n not in past
               and n not in ("position_ids_cos", "position_ids_sin",
                             "attention_bias", "input_ids")]
    present = [n for n in outs if n.startswith("present.")]
    seam_out = [n for n in outs if not n.startswith("present.")]
    return (seam_in[0] if seam_in else None,
            seam_out[0] if seam_out else None, present, past,
            "position_ids_cos" in ins)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=9, help="load parts 1..upto")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--dump-from", type=int, default=2)
    args = ap.parse_args()

    cos, sin = rope_cache(THETA, HEAD_DIM, CTX + 64)

    print(f"loading parts 1..{args.upto} ...", flush=True)
    t0 = time.perf_counter()
    sess = {}
    meta = {}
    for k in range(1, args.upto + 1):
        s = load_sess(SPLIT / f"part{k}")
        sess[k] = s
        if k == 1:
            meta[1] = ("input_ids", s.get_outputs()[0].name, [], [], False)
        else:
            meta[k] = classify(s)
        print(f"  part{k} loaded ({time.perf_counter() - t0:.0f}s)", flush=True)

    dump_parts = list(range(args.dump_from, args.upto + 1))
    handles = {}
    for k in dump_parts:
        d = OUT / f"part{k}"
        d.mkdir(parents=True, exist_ok=True)
        handles[k] = (d, (d / "input_list.txt").open("w", encoding="utf-8"))

    sample = 0
    for ids in PROMPT_IDS:
        if sample >= args.samples:
            break
        ids = ids[:CTX - 1]
        # fresh KV ring per prompt
        past = {k: {n: np.zeros((1, N_KV, CTX - 1, HEAD_DIM), np.float32)
                    for n in meta[k][3]} for k in sess if k >= 2}
        for pos, tid in enumerate(ids):
            if sample >= args.samples:
                break
            cb = cos[pos:pos + 1][None, ...]
            sb = sin[pos:pos + 1][None, ...]
            ab = np.full((1, 1, 1, CTX), -65504.0, np.float32)
            ab[..., CTX - 1 - pos:] = 0.0

            seam = sess[1].run([meta[1][1]],
                               {"input_ids": np.array([[tid]], np.int64)})[0]
            for k in range(2, args.upto + 1):
                seam_in, seam_out, present, pn, _ = meta[k]
                feed = {seam_in: seam, "position_ids_cos": cb,
                        "position_ids_sin": sb, "attention_bias": ab}
                feed.update(past[k])
                if k in handles:
                    d, h = handles[k]
                    order = [i.name for i in sess[k].get_inputs()]
                    parts = []
                    sd = d / f"sample_{sample:03d}"
                    sd.mkdir(exist_ok=True)
                    for n in order:
                        p = sd / f"{sanitize(n)}.raw"
                        np.ascontiguousarray(feed[n]).tofile(str(p))
                        parts.append(f"{n}:={p}")
                    h.write(" ".join(parts) + "\n")
                outs = sess[k].run(present + [seam_out], feed)
                pres = outs[:len(present)]
                seam = outs[len(present)]
                # roll KV ring: past = present[:,:,1:,:]
                for i, p_out in enumerate(present):
                    li = "".join(c if c.isdigit() else " "
                                 for c in p_out).split()[0]
                    kind = "key" if p_out.endswith(".key") else "value"
                    past[k][f"past_key_values.{li}.{kind}"][:] = pres[i][:, :, 1:, :]
            sample += 1
            print(f"  sample {sample}/{args.samples} (pos {pos})", flush=True)

    for k, (d, h) in handles.items():
        h.close()
        nb = sum(p.stat().st_size for p in d.rglob("*.raw"))
        print(f"part{k}: {sample} samples, {nb / 1e6:.0f} MB -> {d/'input_list.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
