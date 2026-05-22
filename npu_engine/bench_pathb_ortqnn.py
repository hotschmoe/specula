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

**Execution uses fully-static IOBinding.** Every part's outputs and
inputs are bound once, by pointer, to persistent numpy buffers:
  * Output buffers (hidden / present-KV / mask / logits) are bound with
    `bind_output(buffer_ptr=...)` — no per-step ~150 MB allocation.
  * The seam (hidden) and folded mask flow part→part by binding the
    producer's output buffer pointer straight into the consumer's
    input — zero copy.
  * The KV `past` buffers are bound by pointer and updated in place
    each step (`past[:] = present[..., 1:, :]`).
  * `input_ids` / `position_ids_cos` / `position_ids_sin` are tiny
    persistent buffers written in place each step.
A decode step is then just: write the 3 small input buffers, run each
session's `run_with_iobinding`, roll the KV. No per-step binding, no
per-step allocation.

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
    """One bundle part: its EPContext-wrapped QNN session, IO
    classification, persistent output buffers, and a pre-wired
    IOBinding."""

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
        self.io: ort.IOBinding | None = None
        self.out_buf: dict[str, np.ndarray] = {}

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

    def alloc_outputs(self) -> None:
        """Pre-allocate one persistent numpy buffer per graph output."""
        for n, dt, sh in self.outputs:
            self.out_buf[n] = np.zeros(tuple(sh), dtype=_NP[dt])

    def bind(self, *, seam_src: np.ndarray | None, mask_src: np.ndarray | None,
             token_buf: np.ndarray | None, cos_buf: np.ndarray,
             sin_buf: np.ndarray, kv) -> None:
        """Wire the IOBinding once. All inputs/outputs are bound by
        pointer to persistent buffers, so no per-step (re)binding is
        needed.

        seam_src  — the previous part's hidden output buffer (None for
                    part 1, which is fed `input_ids` instead).
        mask_src  — part 2's folded-mask output buffer (None if this
                    part neither consumes nor is part 2).
        token_buf — persistent [1,1] int64 buffer (part 1 only).
        """
        self.io = self.session.io_binding()
        # --- outputs ---
        for n, dt, sh in self.outputs:
            arr = self.out_buf[n]
            self.io.bind_output(
                name=n, device_type="cpu", device_id=0,
                element_type=_NP[dt], shape=tuple(sh),
                buffer_ptr=arr.ctypes.data)
        # --- inputs ---
        def bind_in(name: str, arr: np.ndarray) -> None:
            self.io.bind_input(
                name=name, device_type="cpu", device_id=0,
                element_type=arr.dtype.type, shape=tuple(arr.shape),
                buffer_ptr=arr.ctypes.data)

        if self.in_token is not None:
            bind_in(self.in_token, token_buf)
        if self.in_seam is not None:
            bind_in(self.in_seam, seam_src)
        if self.in_cos is not None:
            bind_in(self.in_cos, cos_buf)
        if self.in_sin is not None:
            bind_in(self.in_sin, sin_buf)
        if self.in_mask is not None:
            bind_in(self.in_mask, mask_src)
        for kv_name in self.in_kv:
            layer, kind = _kv_layer(kv_name)
            bind_in(kv_name, kv.past(layer, kind))


class KV:
    """Per-layer rolling FP32 KV cache. `past` buffers ([1,8,past,128])
    are bound once into the graph inputs by pointer; after each step the
    graph writes `present` ([1,8,ctx,128]) into a Part output buffer and
    `roll()` updates past in place: past[:] = present[..., 1:, :]."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, ctx: int):
        self.n_layers = n_layers
        shp = (1, n_kv_heads, ctx - 1, head_dim)
        self._key = [np.zeros(shp, dtype=np.float32) for _ in range(n_layers)]
        self._value = [np.zeros(shp, dtype=np.float32) for _ in range(n_layers)]

    def past(self, layer: int, kind: str) -> np.ndarray:
        return (self._key if kind == "key" else self._value)[layer]

    def roll(self, layer: int, kind: str, present: np.ndarray) -> None:
        buf = (self._key if kind == "key" else self._value)[layer]
        buf[:] = present[:, :, 1:, :]

    def reset(self) -> None:
        for a in self._key:
            a[:] = 0.0
        for a in self._value:
            a[:] = 0.0


def wire(parts: list[Part], kv: KV, token_buf: np.ndarray,
         cos_buf: np.ndarray, sin_buf: np.ndarray) -> None:
    """Allocate output buffers and pre-wire every part's IOBinding."""
    for part in parts:
        part.alloc_outputs()
    mask_src = None
    for part in parts:
        if part.out_mask is not None:
            mask_src = part.out_buf[part.out_mask]
    for i, part in enumerate(parts):
        seam_src = None
        if part.in_seam is not None:
            # seam input == the preceding part's hidden/seam output.
            prev = parts[i - 1]
            seam_src = prev.out_buf[prev.out_seam]
        part.bind(seam_src=seam_src,
                  mask_src=mask_src if part.in_mask is not None else None,
                  token_buf=token_buf, cos_buf=cos_buf, sin_buf=sin_buf, kv=kv)


def step(parts: list[Part], kv: KV, token_buf: np.ndarray, cos_buf: np.ndarray,
         sin_buf: np.ndarray, token_id: int, pos: int,
         cos: np.ndarray, sin: np.ndarray,
         prof: dict | None = None) -> tuple[np.ndarray, float]:
    """One AR1 step. Inputs are already bound by pointer; this just
    refreshes the 3 small position-dependent buffers, runs each
    session, and rolls the KV. Returns (logits[vocab], wall_ms).

    If `prof` is given it must hold `part` (list of N lists) and `roll`
    (list); per-partition `run_with_iobinding` times and the host-side
    KV-roll time are appended — this separates on-device cost from host
    overhead."""
    token_buf[0, 0] = token_id
    cos_buf[0, 0, :] = cos[pos]
    sin_buf[0, 0, :] = sin[pos]
    t0 = time.perf_counter()
    logits = None
    for i, part in enumerate(parts):
        tp = time.perf_counter()
        part.session.run_with_iobinding(part.io)
        if prof is not None:
            prof["part"][i].append((time.perf_counter() - tp) * 1000)
        if part.out_logits is not None:
            logits = part.out_buf[part.out_logits]
    t_roll = time.perf_counter()
    for part in parts:
        for kv_name in part.out_kv:
            layer, kind = _kv_layer(kv_name)
            kv.roll(layer, kind, part.out_buf[kv_name])
    if prof is not None:
        prof["roll"].append((time.perf_counter() - t_roll) * 1000)
    wall_ms = (time.perf_counter() - t0) * 1000
    return logits.reshape(-1).copy(), wall_ms


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

    print(f"=== pathb ORT-QNN bench (IOBinding): {bundle.name} ===")
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

    # Persistent position-dependent input buffers + KV, then wire all
    # IOBindings once (static, by pointer).
    token_buf = np.zeros((1, 1), dtype=np.int64)
    cos_buf = np.zeros((1, 1, head_dim), dtype=np.float32)
    sin_buf = np.zeros((1, 1, head_dim), dtype=np.float32)
    kv = KV(n_layers, n_kv, head_dim, ctx)
    wire(parts, kv, token_buf, cos_buf, sin_buf)

    # warmup (1 step, discarded) — first HTP call pays HMX init.
    print("\n--- warmup (1 step, discarded) ---")
    step(parts, kv, token_buf, cos_buf, sin_buf, prompt_ids[0], 0, cos, sin)
    kv.reset()

    # ---- prefill (AR1) ----
    print(f"\n--- prefill: {len(prompt_ids)} AR1 steps ---")
    pp_lat = []
    last_logits = None
    t_pp = time.perf_counter()
    for pos, tid in enumerate(prompt_ids):
        last_logits, ms = step(parts, kv, token_buf, cos_buf, sin_buf,
                                tid, pos, cos, sin)
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
    prof = {"part": [[] for _ in parts], "roll": []}
    next_tok = int(np.argmax(last_logits))
    t_tg = time.perf_counter()
    for i in range(args.tg_tokens):
        pos = len(prompt_ids) + i
        logits, ms = step(parts, kv, token_buf, cos_buf, sin_buf,
                          next_tok, pos, cos, sin, prof=prof)
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
    part_med = [float(np.median(p)) for p in prof["part"]]
    roll_med = float(np.median(prof["roll"]))
    print(f"  per-step breakdown (median ms): "
          f"parts={[round(x, 1) for x in part_med]}  "
          f"sum_parts={sum(part_med):.1f}  kv_roll={roll_med:.1f}")
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
        part_run_median_ms=";".join(f"{x:.1f}" for x in part_med),
        kv_roll_median_ms=round(roll_med, 2),
        gen_text=gen_text.replace("\n", "\\n"),
        runtime="ort-qnn-1.24.4 + QAIRT-2.45 DLL + static IOBinding")
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
