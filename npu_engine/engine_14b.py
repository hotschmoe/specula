"""General ORT-QNN runtime for the specula multi-part NPU bundles.

Built for the 10-part Qwen3-14B w8a16 bundle but written general over part
count + topology (so it also serves the w4a16 14B and the 27B later). It
reads each part's IO contract from `bin_info/part_*.json` (generate with
`qnn-context-binary-utility` if missing) and wires:

  * the residual **seam** part k -> part k+1
  * a live **attention_bias[1,1,1,ctx]** input shared by every decoder part
    (this bundle uses transformers-4.57's live additive mask, NOT the 4B
    pathb folded ScatterND mask),
  * full-dim RoPE **position_ids_cos/sin[1,1,head_dim]** (theta from config),
  * a per-layer KV **ring buffer** (past[1,8,ctx-1,128] / present[1,8,ctx,128],
    roll `past[:] = present[:,:,1:,:]`).

The >7-session ceiling ([[reference_ortqnn_session_limit]]) is handled by
**grouping** parts: each ORT session can wrap multiple EPContext nodes (one
per `.bin`), with seam outputs flowing node->node as internal graph edges.
For this bundle the per-part input names are distinct (distinct seams,
distinct `past_key_values_{L}`) and the only shared inputs (cos/sin/bias) are
legitimately one tensor feeding several nodes — so the 4B combined-wrapper
name-collision failure does not apply. `--groups` controls the layout.

Binding is uniform regardless of grouping: every graph input/output is bound
by pointer to a persistent host buffer keyed by tensor name. Sessions run in
order; KV rolls once per step.

Usage:
    # Probe the real session ceiling (load parts 1..N as separate sessions):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe npu_engine/engine_14b.py \
        --bundle models/qwen3_14b-w8a16-specula-x2e --probe-ceiling

    # Full decode, one session per part (if the ceiling allows):
    ... --groups "1|2|3|4|5|6|7|8|9|10" --gen 32 --ref results/pathb_eval/ref_cpu_14b.npz

    # Full decode, 2 parts per session (5 sessions):
    ... --groups "1-2|3-4|5-6|7-8|9-10" --gen 32
"""
from __future__ import annotations

import argparse
import csv
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
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
SYS_QNN = Path(
    r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\QnnHtp.dll"
)
CSV_DIR = REPO / "results" / "csv"
EVAL_DIR = REPO / "results" / "pathb_eval"

_QNN_DTYPE = {"QNN_DATATYPE_INT_64": "int64", "QNN_DATATYPE_FLOAT_32": "float32"}
_NP = {"int64": np.int64, "float32": np.float32}
_PROTO = {"int64": TensorProto.INT64, "float32": TensorProto.FLOAT}
_MASK_NEG = np.float32(-65504.0)  # most-negative fp16 value (HTP runs fp16)


def build_rope_cache(theta: float, head_dim: int, max_pos: int):
    """Qwen3 full-dim RoPE cos/sin, [max_pos, head_dim] f32 (mirrors lib/rope.py)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_pos, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)
    emb = np.concatenate([freqs, freqs], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _kv_layer(name: str) -> tuple[int, str]:
    """`past_key_values_11_key` / `present_11_value` -> (11, 'key'|'value')."""
    kind = "key" if name.endswith("_key") else "value"
    digits = "".join(c if c.isdigit() else " " for c in name).split()
    return int(digits[0]), kind


class Part:
    """One bundle part: graph name + classified IO read from bin_info."""

    def __init__(self, idx: int, bin_info: dict, bin_name: str):
        g = bin_info["info"]["graphs"][0]["info"]
        self.idx = idx
        self.graph_name = g["graphName"]
        self.bin_name = bin_name
        self.inputs = [
            (t["info"]["name"], _QNN_DTYPE[t["info"]["dataType"]],
             [int(d) for d in t["info"]["dimensions"]])
            for t in g["graphInputs"]
        ]
        self.outputs = [
            (t["info"]["name"], _QNN_DTYPE[t["info"]["dataType"]],
             [int(d) for d in t["info"]["dimensions"]])
            for t in g["graphOutputs"]
        ]

    @property
    def is_embed(self) -> bool:
        return any(n == "input_ids" for n, _, _ in self.inputs)

    @property
    def is_head(self) -> bool:
        return any(n == "logits" for n, _, _ in self.outputs)


def load_parts(bundle: Path) -> list[Part]:
    """Read bin_info/part_*.json + the genie_config ctx-bins (ordered)."""
    genie = json.loads((bundle / "genie_config.json").read_text())
    bins = genie["dialog"]["engine"]["model"]["binary"]["ctx-bins"]
    info_dir = bundle / "bin_info"
    parts = []
    for i, bin_name in enumerate(bins, start=1):
        info_f = info_dir / f"part_{i}_of_{len(bins)}.json"
        parts.append(Part(i, json.loads(info_f.read_text()), bin_name))
    return parts


def parse_groups(spec: str, n_parts: int) -> list[list[int]]:
    """'1-2|3-4|5' -> [[1,2],[3,4],[5]] (1-based, inclusive)."""
    if not spec:
        return [[i] for i in range(1, n_parts + 1)]
    groups = []
    for chunk in spec.split("|"):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            groups.append(list(range(int(a), int(b) + 1)))
        else:
            groups.append([int(chunk)])
    return groups


class Session:
    """One ORT-QNN InferenceSession wrapping a contiguous group of parts as
    EPContext nodes. Internal seams flow node->node; boundary seams + KV +
    logits are graph IO bound by pointer."""

    def __init__(self, parts: list[Part], bundle: Path):
        self.parts = parts
        self.bundle = bundle
        self.idx0, self.idx1 = parts[0].idx, parts[-1].idx
        self.tag = f"g{self.idx0}" if len(parts) == 1 else f"g{self.idx0}-{self.idx1}"
        # Compute graph IO over the group.
        produced = {n: (dt, sh) for p in parts for n, dt, sh in p.outputs}
        consumed = {n for p in parts for n, _, _ in p.inputs}
        self.gin: dict[str, tuple] = {}
        for p in parts:
            for n, dt, sh in p.inputs:
                if n not in produced:
                    self.gin.setdefault(n, (dt, sh))
        self.gout: dict[str, tuple] = {}
        for p in parts:
            for n, dt, sh in p.outputs:
                if n == "logits" or n.startswith("present_") or n not in consumed:
                    self.gout[n] = (dt, sh)
        self.internal = {n: produced[n] for n in produced
                         if n in consumed and n not in self.gout}
        self.session: ort.InferenceSession | None = None
        self.io: ort.IOBinding | None = None

    def wrapper_path(self) -> Path:
        return self.bundle / f"_eng14b_{self.tag}.wrapper.onnx"

    def build_wrapper(self) -> None:
        nodes = [
            helper.make_node(
                "EPContext", inputs=[n for n, _, _ in p.inputs],
                outputs=[n for n, _, _ in p.outputs], name=f"ctx_part{p.idx}",
                domain="com.microsoft", embed_mode=0,
                ep_cache_context=p.bin_name, source="Qnn")
            for p in self.parts
        ]
        ins = [helper.make_tensor_value_info(n, _PROTO[dt], sh)
               for n, (dt, sh) in self.gin.items()]
        outs = [helper.make_tensor_value_info(n, _PROTO[dt], sh)
                for n, (dt, sh) in self.gout.items()]
        vinfo = [helper.make_tensor_value_info(n, _PROTO[dt], sh)
                 for n, (dt, sh) in self.internal.items()]
        g = helper.make_graph(nodes, f"eng14b_{self.tag}", ins, outs,
                              value_info=vinfo)
        m = helper.make_model(g, opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1)])
        m.ir_version = 10
        onnx.save(m, str(self.wrapper_path()))

    def load(self) -> float:
        self.build_wrapper()
        opts = {
            "backend_path": str(SYS_QNN),
            "htp_performance_mode": "burst",
            "soc_model": "88", "htp_arch": "81",
            "enable_htp_fp16_precision": "1",
        }
        so = ort.SessionOptions()
        so.log_severity_level = 3
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.wrapper_path()), sess_options=so,
            providers=[("QNNExecutionProvider", opts)])
        prov = self.session.get_providers()[0]
        if prov != "QNNExecutionProvider":
            raise RuntimeError(f"{self.tag} fell back to {prov}")
        return time.perf_counter() - t0

    def bind(self, buffers: dict[str, np.ndarray]) -> None:
        """Bind every graph input/output by pointer to a shared host buffer."""
        self.io = self.session.io_binding()
        for n, (dt, sh) in self.gout.items():
            arr = buffers[n]
            self.io.bind_output(name=n, device_type="cpu", device_id=0,
                                element_type=_NP[dt], shape=tuple(sh),
                                buffer_ptr=arr.ctypes.data)
        for n, (dt, sh) in self.gin.items():
            arr = buffers[n]
            self.io.bind_input(name=n, device_type="cpu", device_id=0,
                               element_type=arr.dtype.type, shape=tuple(arr.shape),
                               buffer_ptr=arr.ctypes.data)

    def run(self) -> None:
        self.session.run_with_iobinding(self.io)


def alloc_buffers(parts: list[Part], head_dim: int) -> dict[str, np.ndarray]:
    """One persistent host buffer per distinct tensor name across all parts."""
    buf: dict[str, np.ndarray] = {}
    for p in parts:
        for n, dt, sh in list(p.inputs) + list(p.outputs):
            if n not in buf:
                buf[n] = np.zeros(tuple(sh), dtype=_NP[dt])
    return buf


def roll_kv(buffers: dict[str, np.ndarray], n_layers: int) -> None:
    """past[:] = present[:,:,1:,:] for every layer (ring buffer)."""
    for L in range(n_layers):
        for kind in ("key", "value"):
            past = buffers.get(f"past_key_values_{L}_{kind}")
            pres = buffers.get(f"present_{L}_{kind}")
            if past is not None and pres is not None:
                past[:] = pres[:, :, 1:, :]


def set_step_inputs(buffers, token_id, pos, cos, sin, ctx):
    """Write the 3 shared per-step inputs in place (token, rope, mask)."""
    if "input_ids" in buffers:
        buffers["input_ids"][0, 0] = token_id
    if "position_ids_cos" in buffers:
        buffers["position_ids_cos"][0, 0, :] = cos[pos]
        buffers["position_ids_sin"][0, 0, :] = sin[pos]
    if "attention_bias" in buffers:
        b = buffers["attention_bias"]
        b[:] = _MASK_NEG
        b[..., ctx - 1 - pos:] = 0.0


def probe_ceiling(bundle: Path) -> int:
    """Load parts 1..N as SEPARATE sessions until one fails. Reports the
    real HTP session ceiling for this (mmap'd w8a16) bundle."""
    parts = load_parts(bundle)
    print(f"=== ceiling probe: {bundle.name} ({len(parts)} parts, separate sessions) ===")
    held = []
    for p in parts:
        s = Session([p], bundle)
        try:
            t = s.load()
        except Exception as e:
            print(f"  part{p.idx}: FAILED to load -> {type(e).__name__}: "
                  f"{str(e)[:300]}")
            print(f"\n  >>> session ceiling = {len(held)} parts "
                  f"(part{p.idx} was the {len(held) + 1}th)")
            return len(held)
        held.append(s)
        print(f"  part{p.idx} ({p.graph_name}): loaded OK in {t:.1f}s  "
              f"[{len(held)} sessions held]")
    print(f"\n  >>> ALL {len(held)} parts loaded as separate sessions. "
          f"No ceiling hit (mmap likely changed it).")
    return len(held)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--groups", default="",
                    help="e.g. '1-2|3-4|5-6|7-8|9-10'; default each part separate")
    ap.add_argument("--probe-ceiling", action="store_true")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--ref", type=Path, default=None,
                    help="npz with first_decode_logits for cos-sim")
    ap.add_argument("--tag", default="w8a16_14b")
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    if args.probe_ceiling:
        probe_ceiling(bundle)
        return 0

    meta = json.loads((bundle / "metadata.json").read_text())
    cfg = json.loads((bundle / "config.json").read_text())
    ctx = int(meta["ctx"])
    n_layers = int(cfg["num_hidden_layers"])
    head_dim = int(cfg["head_dim"])
    theta = float(cfg["rope_theta"])
    print(f"=== engine_14b: {bundle.name} ===")
    print(f"  precision={meta['precision']} ctx={ctx} layers={n_layers} "
          f"head_dim={head_dim} theta={theta:g}")

    parts = load_parts(bundle)
    groups = parse_groups(args.groups, len(parts))
    print(f"  groups: {groups}  ({len(groups)} sessions)")

    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    prompt = (bundle / "sample_prompt.txt").read_text(encoding="utf-8")
    prompt_ids = tok.encode(prompt).ids[:args.max_prompt]
    print(f"  prompt: {len(prompt_ids)} tokens")
    cos, sin = build_rope_cache(theta, head_dim, ctx + 64)

    # Build sessions.
    sessions = [Session([parts[i - 1] for i in grp], bundle) for grp in groups]
    print(f"\n--- loading {len(sessions)} sessions (QAIRT 2.45 DLL) ---")
    t_load = time.perf_counter()
    for s in sessions:
        dt = s.load()
        print(f"  {s.tag}: {dt:.1f}s  (in={list(s.gin)[:3]}... out={len(s.gout)})")
    load_s = time.perf_counter() - t_load
    print(f"  total load: {load_s:.1f}s")

    buffers = alloc_buffers(parts, head_dim)
    for s in sessions:
        s.bind(buffers)
    logits = buffers["logits"]

    def step(token_id: int, pos: int) -> np.ndarray:
        set_step_inputs(buffers, token_id, pos, cos, sin, ctx)
        for s in sessions:
            s.run()
        roll_kv(buffers, n_layers)
        return logits.reshape(-1)

    # warmup
    print("\n--- warmup (1 step) ---")
    step(prompt_ids[0], 0)
    for b in buffers.values():
        if b.ndim == 4:
            b[:] = 0.0  # reset KV

    # prefill (AR1)
    print(f"--- prefill: {len(prompt_ids)} AR1 steps ---")
    pp_lat = []
    last = None
    t_pp = time.perf_counter()
    for pos, tid in enumerate(prompt_ids):
        t0 = time.perf_counter()
        last = step(tid, pos).copy()
        pp_lat.append((time.perf_counter() - t0) * 1000)
    pp_wall = time.perf_counter() - t_pp
    first_logits = last.copy()

    # decode (AR1 greedy)
    print(f"--- decode: {args.gen} AR1 greedy steps ---")
    tg_lat, gen_ids = [], []
    nxt = int(np.argmax(last))
    t_tg = time.perf_counter()
    for i in range(args.gen):
        pos = len(prompt_ids) + i
        t0 = time.perf_counter()
        lg = step(nxt, pos)
        tg_lat.append((time.perf_counter() - t0) * 1000)
        gen_ids.append(nxt)
        nxt = int(np.argmax(lg))
    tg_wall = time.perf_counter() - t_tg

    gen_text = tok.decode(gen_ids)
    top5 = np.argsort(first_logits)[-5:][::-1]
    print(f"\n=== {bundle.name} ===")
    print(f"  PP: {len(prompt_ids) / pp_wall:7.2f} t/s  "
          f"(median {np.median(pp_lat):.1f} ms/step)")
    print(f"  TG: {args.gen / tg_wall:7.2f} t/s  "
          f"(median {np.median(tg_lat):.1f} ms/step)")
    print(f"  first-decode top-5: "
          f"{[(int(t), tok.id_to_token(int(t))) for t in top5]}")
    print(f"  continuation: {gen_text!r}")

    cos_sim = None
    if args.ref and args.ref.exists():
        ref = np.load(args.ref, allow_pickle=True)["first_decode_logits"].astype(
            np.float64)
        a = first_logits.astype(np.float64)
        cos_sim = float(np.dot(a, ref) / (np.linalg.norm(a) * np.linalg.norm(ref)))
        ref_argmax = int(np.argmax(ref))
        print(f"\n  cos-sim vs CPU ref: {cos_sim:.5f}")
        print(f"  argmax NPU={int(top5[0])} ref={ref_argmax} "
              f"{'MATCH' if int(top5[0]) == ref_argmax else 'DIFFER'}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL_DIR / f"{args.tag}.npz",
                        first_decode_logits=first_logits,
                        gen_ids=np.array(gen_ids, dtype=np.int64),
                        prompt_ids=np.array(prompt_ids, dtype=np.int64))
    row = dict(bundle=bundle.name, tag=args.tag, precision=meta["precision"],
               ctx=ctx, groups=args.groups or "separate", n_sessions=len(sessions),
               pp_tokens=len(prompt_ids), tg_tokens=args.gen,
               pp_tps=round(len(prompt_ids) / pp_wall, 3),
               tg_tps=round(args.gen / tg_wall, 3),
               tg_median_ms=round(float(np.median(tg_lat)), 2),
               load_s=round(load_s, 1),
               first_decode_argmax=int(top5[0]),
               cos_sim=round(cos_sim, 5) if cos_sim is not None else "",
               gen_text=gen_text.replace("\n", "\\n"))
    csv_path = CSV_DIR / f"engine_14b_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
