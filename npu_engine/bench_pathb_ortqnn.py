"""Bench a specula `pathb` NPU bundle through our ORT-QNN runtime.

The pathb bundles produced by the cloud-GPU pipeline
(`end-to-end/compile_split_bundle.py`) are NOT Genie-loadable — the
`fold-pathbmask` rewrite removes the `attention_mask` graph input that
Genie's KV-cache manager requires (see memory
`reference_pathb_not_genie_loadable`). They CAN be driven directly by
ORT-QNN, which is what this harness does.

Differences from the Qualcomm bundle harness (`qualcomm_qwen3_4b_oracle`):
  * FP32 IO everywhere (no uint16/uint8 quant on the IO tensors —
    quantization is internal to the graph).
  * No `attention_mask` input. The folded causal mask is produced as a
    cross-part tensor (`_model_ScatterND_output_0`, shape [1,1,1,ctx])
    by the first decoder part and threaded into every later decoder
    part by this harness.
  * Full-dim RoPE: `position_ids_cos/sin` are [1,1,head_dim], not the
    half-dim [1,1,1,head_dim/2] Qualcomm uses.
  * One graph per `.bin` (no multi-AR / multi-ctx graphs). AR1 only —
    prefill is token-by-token.
  * Bundle compiled with QAIRT 2.45 ⇒ ORT-QNN must be pointed at the
    system QAIRT 2.45 `QnnHtp.dll` (the venv-bundled 2.42 DLL fails
    with QNN error 5000).

The IO contract is read from each part's `bin_info/part_*.json`, so the
same harness drives the 4-part ctx512 bundles without per-bundle edits.

Usage:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \\
        npu_engine/bench_pathb_ortqnn.py \\
        --bundle models/specula-qwen3-4b-ref/qwen3-4b_w4a16_pathb_ctx512_x2e_v81 \\
        --pp-tokens 256 --tg-tokens 128 --tag w4a16_ctx512

Outputs:
    results/csv/pathb_ortqnn_<tag>.csv
    results/pathb_eval/<tag>.npz   (first-decode logits + generated ids)
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
PROMPT_PATH = REPO / "results" / "qwen3_4b_baseline" / "pp512_prompt.txt"
CSV_DIR = REPO / "results" / "csv"
EVAL_DIR = REPO / "results" / "pathb_eval"

_QNN_DTYPE = {"QNN_DATATYPE_INT_64": "int64", "QNN_DATATYPE_FLOAT_32": "float32"}
_NP = {"int64": np.int64, "float32": np.float32}
_PROTO = {"int64": TensorProto.INT64, "float32": TensorProto.FLOAT}
MASK_TENSOR = "_model_ScatterND_output_0"


def build_rope_cache(theta: float, head_dim: int, max_pos: int):
    """Qwen3 full-dim RoPE cos/sin, shape [max_pos, head_dim] FP32.
    Mirrors end-to-end/lib/rope.py."""
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
    """One bundle part: its EPContext-wrapped QNN session + IO classification."""

    def __init__(self, idx: int, bin_info: dict, bin_name: str, bundle: Path):
        g = bin_info["graphs"][0]
        self.idx = idx
        self.graph_name = g["graphName"]
        self.bin_name = bin_name
        self.bundle = bundle
        self.inputs = [
            (t["name"], _QNN_DTYPE[t["dataType"]],
             [int(d) for d in t["dimensions"]])
            for t in g["graphInputs"]
        ]
        self.outputs = [
            (t["name"], _QNN_DTYPE[t["dataType"]],
             [int(d) for d in t["dimensions"]])
            for t in g["graphOutputs"]
        ]
        # Classify inputs.
        self.in_token = self.in_seam = None
        self.in_cos = self.in_sin = self.in_mask = None
        self.in_kv: list[str] = []
        for n, _dt, _sh in self.inputs:
            if n == "input_ids":
                self.in_token = n
            elif n == "position_ids_cos":
                self.in_cos = n
            elif n == "position_ids_sin":
                self.in_sin = n
            elif n.startswith("past_key_values_"):
                self.in_kv.append(n)
            elif "ScatterND" in n:
                self.in_mask = n
            else:
                self.in_seam = n
        # Classify outputs.
        self.out_logits = self.out_seam = self.out_mask = None
        self.out_kv: list[str] = []
        for n, _dt, _sh in self.outputs:
            if n == "logits":
                self.out_logits = n
            elif n.startswith("present_"):
                self.out_kv.append(n)
            elif "ScatterND" in n:
                self.out_mask = n
            else:
                self.out_seam = n
        self.session: ort.InferenceSession | None = None

    def wrapper_path(self) -> Path:
        return self.bundle / f"_ortqnn_part{self.idx}.wrapper.onnx"

    def build_wrapper(self) -> None:
        ins = [helper.make_tensor_value_info(n, _PROTO[dt], sh)
               for n, dt, sh in self.inputs]
        outs = [helper.make_tensor_value_info(n, _PROTO[dt], sh)
                for n, dt, sh in self.outputs]
        node = helper.make_node(
            "EPContext", inputs=[v.name for v in ins],
            outputs=[v.name for v in outs], name=self.graph_name,
            domain="com.microsoft", embed_mode=0,
            ep_cache_context=self.bin_name, source="Qnn",
        )
        g = helper.make_graph([node], f"pathb_{self.graph_name}", ins, outs)
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
        if self.session.get_providers()[0] != "QNNExecutionProvider":
            raise RuntimeError(
                f"part{self.idx} fell back to {self.session.get_providers()[0]}")
        return time.perf_counter() - t0


def load_parts(bundle: Path) -> list[Part]:
    """Read every bin_info/part_*.json + the genie_config ctx-bins list."""
    genie = json.loads((bundle / "genie_config.json").read_text())
    bins = genie["dialog"]["engine"]["model"]["binary"]["ctx-bins"]
    info_files = sorted(
        (bundle / "bin_info").glob("part_*.json"),
        key=lambda p: int(p.stem.split("part_")[1].split("_")[0]))
    if len(info_files) != len(bins):
        raise RuntimeError(
            f"part-count mismatch: {len(info_files)} bin_info vs {len(bins)} ctx-bins")
    parts = []
    for i, (info_f, bin_name) in enumerate(zip(info_files, bins), start=1):
        parts.append(Part(i, json.loads(info_f.read_text()), bin_name, bundle))
    return parts


class KV:
    """Per-layer rolling FP32 KV cache. `past` feeds the graph
    ([1,8,past,128]); after a step the graph returns `present`
    ([1,8,ctx,128]) and the next step's past is present[..., 1:, :]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, ctx: int):
        self.past = ctx - 1
        shp = (1, n_kv_heads, self.past, head_dim)
        self.key = [np.zeros(shp, dtype=np.float32) for _ in range(n_layers)]
        self.value = [np.zeros(shp, dtype=np.float32) for _ in range(n_layers)]

    def get(self, layer: int, kind: str) -> np.ndarray:
        return (self.key if kind == "key" else self.value)[layer]

    def roll(self, layer: int, kind: str, present: np.ndarray) -> None:
        buf = (self.key if kind == "key" else self.value)
        buf[layer] = np.ascontiguousarray(present[:, :, 1:, :])


def forward(parts: list[Part], kv: KV, token_id: int, pos: int,
            cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, float]:
    """One AR1 step through all parts. Returns (logits[vocab], wall_ms)."""
    cos_q = cos[pos:pos + 1][None, ...]   # [1,1,head_dim]
    sin_q = sin[pos:pos + 1][None, ...]
    t0 = time.perf_counter()

    # part 1: embed
    p1 = parts[0]
    hidden = p1.session.run(
        None, {p1.in_token: np.array([[token_id]], dtype=np.int64)})[0]

    mask = None
    logits = None
    for part in parts[1:]:
        feed = {part.in_seam: hidden, part.in_cos: cos_q, part.in_sin: sin_q}
        for kv_name in part.in_kv:
            layer, kind = _kv_layer(kv_name)
            feed[kv_name] = kv.get(layer, kind)
        if part.in_mask is not None:
            feed[part.in_mask] = mask
        out_names = [o[0] for o in part.outputs]
        results = dict(zip(out_names, part.session.run(out_names, feed)))
        if part.out_mask is not None:
            mask = results[part.out_mask]
        if part.out_seam is not None:
            hidden = results[part.out_seam]
        if part.out_logits is not None:
            logits = results[part.out_logits]
        for kv_name in part.out_kv:
            layer, kind = _kv_layer(kv_name)
            kv.roll(layer, kind, results[kv_name])
    wall_ms = (time.perf_counter() - t0) * 1000
    return logits.reshape(-1), wall_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pp-tokens", type=int, default=256)
    ap.add_argument("--tg-tokens", type=int, default=128)
    ap.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    ctx = int(meta["ctx"])
    cap = ctx - 1
    if args.pp_tokens + args.tg_tokens > cap:
        print(f"ERROR: pp+tg={args.pp_tokens + args.tg_tokens} > cap {cap}")
        return 2

    cfg = json.loads((bundle / "config.json").read_text())
    n_layers = int(cfg["num_hidden_layers"])
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    theta = float(cfg["rope_theta"])

    print(f"=== pathb ORT-QNN bench: {bundle.name} ===")
    print(f"  precision={meta['precision']}  ctx={ctx}  parts={meta['num_parts']}"
          f"  layers={n_layers}")

    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    prompt_ids = tok.encode(args.prompt.read_text(encoding="utf-8")).ids[:args.pp_tokens]
    print(f"  prompt: {len(prompt_ids)} tokens (target {args.pp_tokens})")

    cos, sin = build_rope_cache(theta, head_dim, ctx + 64)

    parts = load_parts(bundle)
    print(f"\n--- loading {len(parts)} QNN sessions (QAIRT 2.45 DLL) ---")
    t_load = time.perf_counter()
    per_part = []
    for part in parts:
        dt = part.load()
        per_part.append(dt)
        print(f"  part{part.idx} ({part.graph_name}): {dt:.1f} s")
    load_s = time.perf_counter() - t_load
    print(f"  total load: {load_s:.1f} s")

    kv = KV(n_layers, n_kv, head_dim, ctx)

    # warmup (1 step, discarded) — first HTP call pays HMX init.
    print("\n--- warmup (1 step, discarded) ---")
    forward(parts, KV(n_layers, n_kv, head_dim, ctx), prompt_ids[0], 0, cos, sin)

    # ---- prefill (AR1) ----
    print(f"\n--- prefill: {len(prompt_ids)} AR1 steps ---")
    pp_lat = []
    last_logits = None
    t_pp = time.perf_counter()
    for pos, tid in enumerate(prompt_ids):
        last_logits, ms = forward(parts, kv, tid, pos, cos, sin)
        pp_lat.append(ms)
        if pos % 32 == 0 or pos == len(prompt_ids) - 1:
            print(f"  prefill step {pos:3d}  {ms:.1f} ms")
    pp_wall = time.perf_counter() - t_pp
    pp_tps = len(prompt_ids) / pp_wall
    first_decode_logits = last_logits.copy()

    # ---- decode (AR1 greedy) ----
    print(f"\n--- decode: {args.tg_tokens} AR1 greedy steps ---")
    tg_lat = []
    gen_ids = []
    next_tok = int(np.argmax(last_logits))
    t_tg = time.perf_counter()
    for i in range(args.tg_tokens):
        pos = len(prompt_ids) + i
        logits, ms = forward(parts, kv, next_tok, pos, cos, sin)
        tg_lat.append(ms)
        gen_ids.append(next_tok)
        next_tok = int(np.argmax(logits))
        if i % 16 == 0 or i == args.tg_tokens - 1:
            print(f"  decode step {i:3d}  {ms:.1f} ms  "
                  f"(median {np.median(tg_lat):.1f} ms)")
    tg_wall = time.perf_counter() - t_tg
    tg_tps = args.tg_tokens / tg_wall

    gen_text = tok.decode(gen_ids)
    top5 = np.argsort(first_decode_logits)[-5:][::-1]
    print(f"\n=== {bundle.name} ===")
    print(f"  PP : {pp_tps:7.2f} t/s  ({len(prompt_ids)} tok, "
          f"median {np.median(pp_lat):.1f} ms/step)")
    print(f"  TG : {tg_tps:7.2f} t/s  ({args.tg_tokens} tok, "
          f"median {np.median(tg_lat):.1f} ms/step)")
    print(f"  first-decode top-5: "
          f"{[(int(t), repr(tok.id_to_token(int(t)))) for t in top5]}")
    print(f"  generated continuation:\n    {gen_text!r}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EVAL_DIR / f"{args.tag}.npz",
        first_decode_logits=first_decode_logits,
        gen_ids=np.array(gen_ids, dtype=np.int64),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        pp_lat_ms=np.array(pp_lat), tg_lat_ms=np.array(tg_lat))
    row = dict(
        bundle=bundle.name, tag=args.tag, precision=meta["precision"], ctx=ctx,
        num_parts=meta["num_parts"], pp_tokens=len(prompt_ids),
        tg_tokens=args.tg_tokens, pp_tps=round(pp_tps, 3), tg_tps=round(tg_tps, 3),
        pp_median_ms=round(float(np.median(pp_lat)), 2),
        tg_median_ms=round(float(np.median(tg_lat)), 2),
        load_s=round(load_s, 1),
        per_part_load_s=";".join(f"{x:.1f}" for x in per_part),
        first_decode_argmax=int(top5[0]),
        gen_text=gen_text.replace("\n", "\\n"),
        runtime="ort-qnn-1.24.4 + QAIRT-2.45 DLL")
    csv_path = CSV_DIR / f"pathb_ortqnn_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n  csv : {csv_path}")
    print(f"  npz : {EVAL_DIR / (args.tag + '.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
