"""Quantized (uint16-IO) ORT-QNN runtime for the calibrated 14B bundle.

The calibrated decoder parts (2-9) have **uint16 IO** with per-tensor
`scaleOffset` encodings (QNN: `real = scale*(q + offset)`); part1 (embed) and
part10 (lm_head) stay fp16 with **fp32 IO**. This engine threads the chain
across the dtype boundary by keeping a canonical **fp32** seam + KV store and
quantizing/dequantizing at each part's IO using the encodings read from
`bin_info`. cos/sin/attention_bias are built fp32 per step and quantized to
each decoder part's encoding.

Correctness-first: the fp32 KV store is always exact. If a layer's past-in and
present-out encodings match (threaded calibration usually makes them equal,
like Qualcomm's bundles), we could roll raw uint16 — reported as `kv_raw_ok`
so a later pass can switch on the fast path.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe npu_engine/engine_14b_q.py \
        --bundle models/qwen3_14b-w8a16-specula-x2e-calib \
        --groups "1|2|3|4|5|6|7|8|9|10" --gen 32 \
        --ref results/pathb_eval/ref_cpu_14b.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

REPO = Path(__file__).resolve().parent.parent
SYS_QNN = Path(r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\QnnHtp.dll")
EVAL_DIR = REPO / "results" / "pathb_eval"

_QNN_NP = {"QNN_DATATYPE_INT_64": np.int64, "QNN_DATATYPE_FLOAT_32": np.float32,
           "QNN_DATATYPE_UFIXED_POINT_16": np.uint16, "QNN_DATATYPE_UFIXED_POINT_8": np.uint8}
_NP_PROTO = {np.int64: TensorProto.INT64, np.float32: TensorProto.FLOAT,
             np.uint16: TensorProto.UINT16, np.uint8: TensorProto.UINT8}


def quant(f, scale, offset, qmax):
    return np.clip(np.round(f / scale) - offset, 0, qmax).astype(
        np.uint16 if qmax == 65535 else np.uint8)


def dequant(q, scale, offset):
    return (q.astype(np.int32) + offset).astype(np.float32) * scale


def rope_cache(theta, head_dim, max_pos):
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    emb = np.concatenate([np.outer(np.arange(max_pos), inv)] * 2, -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _layer(name):
    return int("".join(c if c.isdigit() else " " for c in name).split()[0])


class IOT:
    """One graph IO tensor: name, numpy dtype, shape, optional scale/offset."""
    def __init__(self, t):
        i = t["info"]
        self.name = i["name"]
        self.np = _QNN_NP[i["dataType"]]
        self.shape = tuple(int(d) for d in i["dimensions"])
        so = i.get("quantizeParams", {}).get("scaleOffset")
        self.scale = float(so["scale"]) if so else None
        self.offset = int(so["offset"]) if so else None
        self.qmax = 65535 if self.np == np.uint16 else (255 if self.np == np.uint8 else None)

    @property
    def quantized(self):
        return self.scale is not None


class Part:
    def __init__(self, idx, bin_info, bin_name):
        g = bin_info["info"]["graphs"][0]["info"]
        self.idx, self.graph_name, self.bin_name = idx, g["graphName"], bin_name
        self.ins = [IOT(t) for t in g["graphInputs"]]
        self.outs = [IOT(t) for t in g["graphOutputs"]]
        self.in_by = {t.name: t for t in self.ins}
        self.out_by = {t.name: t for t in self.outs}
        # classify
        self.seam_in = self.seam_out = self.logits = self.token = None
        self.past, self.present = [], []
        for t in self.ins:
            if t.name == "input_ids":
                self.token = t
            elif t.name in ("position_ids_cos", "position_ids_sin", "attention_bias"):
                pass
            elif t.name.startswith("past_key_values"):
                self.past.append(t)
            else:
                self.seam_in = t
        for t in self.outs:
            if t.name == "logits":
                self.logits = t
            elif t.name.startswith("present"):
                self.present.append(t)
            else:
                self.seam_out = t


def load_parts(bundle):
    genie = json.loads((bundle / "genie_config.json").read_text())
    bins = genie["dialog"]["engine"]["model"]["binary"]["ctx-bins"]
    info = bundle / "bin_info"
    return [Part(i, json.loads((info / f"part_{i}_of_{len(bins)}.json").read_text()), b)
            for i, b in enumerate(bins, 1)]


def parse_groups(spec, n):
    if not spec:
        return [[i] for i in range(1, n + 1)]
    out = []
    for c in spec.split("|"):
        c = c.strip()
        out.append(list(range(int(c.split("-")[0]), int(c.split("-")[1]) + 1))
                   if "-" in c else [int(c)])
    return out


class Session:
    """ORT-QNN session over a contiguous group of parts (EPContext nodes)."""
    def __init__(self, parts, bundle):
        self.parts, self.bundle = parts, bundle
        self.tag = f"g{parts[0].idx}" + ("" if len(parts) == 1 else f"-{parts[-1].idx}")
        produced = {t.name for p in parts for t in p.outs}
        consumed = {t.name for p in parts for t in p.ins}
        self.gin, self.gout = {}, {}
        for p in parts:
            for t in p.ins:
                if t.name not in produced:
                    self.gin.setdefault(t.name, t)
            for t in p.outs:
                if t.name == "logits" or t.name.startswith("present") or t.name not in consumed:
                    self.gout[t.name] = t
        self.internal = {t.name: t for p in parts for t in p.outs
                         if t.name not in self.gout and t.name in consumed}

    def wrapper(self):
        return self.bundle / f"_engq_{self.tag}.wrapper.onnx"

    def build(self):
        def tvi(t):
            return helper.make_tensor_value_info(t.name, _NP_PROTO[t.np], list(t.shape))
        nodes = [helper.make_node("EPContext", inputs=[t.name for t in p.ins],
                 outputs=[t.name for t in p.outs], name=f"ctx{p.idx}",
                 domain="com.microsoft", embed_mode=0, ep_cache_context=p.bin_name,
                 source="Qnn") for p in self.parts]
        g = helper.make_graph(nodes, f"engq_{self.tag}",
                              [tvi(t) for t in self.gin.values()],
                              [tvi(t) for t in self.gout.values()],
                              value_info=[tvi(t) for t in self.internal.values()])
        m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 17),
                              helper.make_operatorsetid("com.microsoft", 1)])
        m.ir_version = 10
        onnx.save(m, str(self.wrapper()))

    def load(self):
        self.build()
        opts = {"backend_path": str(SYS_QNN), "htp_performance_mode": "burst",
                "soc_model": "88", "htp_arch": "81", "enable_htp_fp16_precision": "1"}
        import os
        if os.environ.get("HTP_SHARED_MEM") == "1":
            opts["enable_htp_shared_memory_allocator"] = "1"
        if os.environ.get("HTP_SPILL_FILL") == "1":
            opts["enable_htp_spill_fill_buffer"] = "1"
        so = ort.SessionOptions()
        so.log_severity_level = 3
        t0 = time.perf_counter()
        self.s = ort.InferenceSession(str(self.wrapper()), sess_options=so,
                                      providers=[("QNNExecutionProvider", opts)])
        if self.s.get_providers()[0] != "QNNExecutionProvider":
            raise RuntimeError(f"{self.tag} fell back")
        return time.perf_counter() - t0

    def alloc(self):
        """Per-session IO buffers (typed). Buffers are NOT shared across
        sessions by name — the same seam tensor is float32 in its producer part
        and uint16 in its consumer part, so each session needs its own."""
        self.buf = {n: np.zeros(t.shape, dtype=t.np)
                    for n, t in {**self.gin, **self.gout}.items()}

    def bind(self):
        self.io = self.s.io_binding()
        for n, t in self.gout.items():
            a = self.buf[n]
            self.io.bind_output(name=n, device_type="cpu", device_id=0,
                                element_type=t.np, shape=t.shape, buffer_ptr=a.ctypes.data)
        for n, t in self.gin.items():
            a = self.buf[n]
            self.io.bind_input(name=n, device_type="cpu", device_id=0,
                               element_type=a.dtype.type, shape=a.shape, buffer_ptr=a.ctypes.data)

    def run(self):
        self.s.run_with_iobinding(self.io)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--groups", default="")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--tag", default="w8a16_14b_calib")
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
    groups = parse_groups(args.groups, len(parts))
    sessions = [Session([parts[i - 1] for i in grp], bundle) for grp in groups]
    print(f"=== engine_14b_q: {bundle.name} | {len(prompt_ids)} prompt tok | "
          f"{len(sessions)} sessions ===")
    t0 = time.perf_counter()
    for s in sessions:
        dt = s.load()
        print(f"  {s.tag}: {dt:.1f}s")
    print(f"  load total {time.perf_counter() - t0:.1f}s")

    # Per-session typed IO buffers + canonical fp32 seam + KV host stores.
    for s in sessions:
        s.alloc()
        s.bind()
    fp32_seam = {}  # seam tensor name -> fp32 canonical [1,1,5120]
    kv = {(L, k): np.zeros((1, 8, ctx - 1, head_dim), np.float32)
          for L in range(n_layers) for k in ("key", "value")}

    def _set(dst, fp32_val, t):
        if t.quantized:
            dst[:] = quant(fp32_val, t.scale, t.offset, t.qmax)
        elif dst.dtype == np.int64:
            dst[:] = fp32_val
        else:
            dst[:] = fp32_val.astype(dst.dtype)

    def _get(src, t):
        return dequant(src, t.scale, t.offset) if t.quantized else src.astype(np.float32)

    def step(tid, pos):
        cb, sb = cos[pos], sin[pos]
        ab = np.full((1, 1, 1, ctx), -65504.0, np.float32)
        ab[..., ctx - 1 - pos:] = 0.0
        logits = None
        for s in sessions:          # sessions are in part order; seams flow forward
            b = s.buf
            for nm, t in s.gin.items():
                if nm == "input_ids":
                    b[nm][0, 0] = tid
                elif nm == "position_ids_cos":
                    _set(b[nm], cb.reshape(b[nm].shape), t)
                elif nm == "position_ids_sin":
                    _set(b[nm], sb.reshape(b[nm].shape), t)
                elif nm == "attention_bias":
                    _set(b[nm], ab.reshape(b[nm].shape), t)
                elif nm.startswith("past_key_values"):
                    L = _layer(nm)
                    _set(b[nm], kv[(L, "key" if nm.endswith("key") else "value")], t)
                else:               # seam_in (bridged via fp32 canonical)
                    _set(b[nm], fp32_seam[nm], t)
            s.run()
            for nm, t in s.gout.items():
                if nm == "logits":
                    logits = _get(b[nm], t).reshape(-1)
                elif nm.startswith("present"):
                    L = _layer(nm)
                    kv[(L, "key" if nm.endswith("key") else "value")][:] = \
                        _get(b[nm], t)[:, :, 1:, :]
                else:               # seam_out -> fp32 canonical for next session
                    fp32_seam[nm] = _get(b[nm], t).copy()
        return logits

    # warmup + reset
    print("--- warmup ---")
    step(prompt_ids[0], 0)
    for v in kv.values():
        v[:] = 0.0
    fp32_seam.clear()

    print(f"--- prefill {len(prompt_ids)} + decode {args.gen} ---")
    last = None
    tg = []
    for pos, tid in enumerate(prompt_ids):
        last = step(tid, pos).copy()
    first_logits = last.copy()
    nxt = int(np.argmax(last))
    gen = []
    for i in range(args.gen):
        t1 = time.perf_counter()
        lg = step(nxt, len(prompt_ids) + i)
        tg.append((time.perf_counter() - t1) * 1000)
        gen.append(nxt)
        nxt = int(np.argmax(lg))
    txt = tok.decode(gen)
    top5 = np.argsort(first_logits)[-5:][::-1]
    print(f"\n  TG ~{1000/np.median(tg):.2f} t/s (median {np.median(tg):.0f} ms/step)")
    print(f"  first-decode top5: {[(int(t), tok.id_to_token(int(t))) for t in top5]}")
    print(f"  continuation: {txt!r}")
    if args.ref and args.ref.exists():
        ref = np.load(args.ref, allow_pickle=True)["first_decode_logits"].astype(np.float64)
        a = first_logits.astype(np.float64)
        cs = float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref)))
        print(f"\n  COS-SIM vs CPU ref: {cs:.5f}  argmax NPU={int(top5[0])} "
              f"ref={int(np.argmax(ref))} {'MATCH' if int(top5[0])==int(np.argmax(ref)) else 'DIFFER'}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL_DIR / f"{args.tag}.npz", first_decode_logits=first_logits,
                        gen_ids=np.array(gen, np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main())
