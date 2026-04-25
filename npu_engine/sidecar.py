"""NPU engine sidecar — long-lived process that holds NPU sessions.

The 4-partition Qwen3-4B w4a16 bundle takes ~15 s to load 4 ORT-QNN
sessions on the X2E NPU (HTP context init dominates, not file I/O —
see `docs/npu_engine_prefill_sidequest.md` for the per-partition
profile). Standalone bench runs pay this cost on every invocation.
This sidecar pays it ONCE at startup and amortizes across many
inference requests, mirroring vLLM's server pattern.

State machine: at any moment one chain is loaded — AR1 (decode +
short-prompt prefill) or AR128 (long-prompt prefill). The 8-session
HTP context-memory ceiling means both can't coexist
(`reference_ortqnn_session_limit.md`). When a request needs the
other mode, the sidecar swaps:

  current_mode != target_mode  ->  tear down current + load target
  current_mode == target_mode  ->  no swap; reuse loaded sessions

So sequential requests that share a mode (e.g. 5 short prompts in a
row) pay the load cost only once. Mode boundaries pay one swap each.

Two modes of operation:
  --demo       : run a fixed mixed-request sequence in-process and
                 report amortized cost vs N standalone bench runs.
  --serve      : read newline-delimited JSON requests from stdin,
                 emit JSON responses to stdout. Used by external
                 drivers (bench_sidecar.py, future spec-decode glue).

Request schema (--serve):
    {"op": "infer", "id": "<str>", "pp_tokens": <int>,
     "tg_tokens": <int>, "force_ar128": <bool, optional>}
    {"op": "shutdown"}

Response schema (one JSON object per line):
    {"event": "ready", "startup_s": <float>, "ar1_per_part_s": [...]}
    {"id": "<str>", "ok": true, "route": "ar1"|"ar128",
     "swap_s": <float>, "pp_compute_s": <float>, "tg_compute_s": <float>,
     "pp_tps": <float>, "tg_tps": <float>}
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so JSON containing tokenizer special tokens
# (Qwen's "Ġ" space marker, etc.) doesn't crash on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import yaml
from tokenizers import Tokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    AR128_BATCH,
    BUNDLE_DIR,
    NUM_LAYERS,
    PAST_LEN,
    KVStore,
    build_part_cfg,
    build_wrapper,
    dequant_uint16,
    load_session,
)
from bench_qwen3_4b_ortqnn import (  # noqa: E402
    _step,
    _step_ar128,
    make_bound_chain,
    PROMPT_PATH,
)


REPO_ROOT = _HERE.parent


def _scales_tuple(part_cfg):
    """cos/mask/logits scale+offset extracted from a parts_cfg dict."""
    def find(side, part_idx, name):
        return next(io for io in part_cfg[part_idx][side] if io["name"] == name)
    cos = find("inputs", 2, "position_ids_cos")
    mask = find("inputs", 2, "attention_mask")
    logits = find("outputs", 4, "logits")
    return (cos["scale"], cos["offset"],
            mask["scale"], mask["offset"],
            logits["scale"], logits["offset"])


class EngineState:
    """Holds the currently-loaded ORT-QNN chain and IO bindings.
    `mode` is one of ("ar1", "ar128", None). Swap by calling
    `ensure_mode(target)` — returns the swap wall time (0 if hit)."""

    def __init__(self, parts_cfg_ar1, parts_cfg_ar128):
        self.parts_cfg_ar1 = parts_cfg_ar1
        self.parts_cfg_ar128 = parts_cfg_ar128
        self.scales_ar1 = _scales_tuple(parts_cfg_ar1)
        self.scales_ar128 = _scales_tuple(parts_cfg_ar128)
        self.mode = None
        self.sessions = None
        self.bindings = None
        self.out_bufs = None
        self.warmup_done = False  # one HMX warmup per loaded chain

    def _load(self, target):
        if target == "ar1":
            parts_cfg, suffix = self.parts_cfg_ar1, ""
        else:
            parts_cfg, suffix = self.parts_cfg_ar128, "_ar128"
        sessions = {}
        per_part_s = []
        for part_idx in (1, 2, 3, 4):
            wrapper = BUNDLE_DIR / f"oracle_part{part_idx}{suffix}.wrapper.onnx"
            if not wrapper.exists():
                build_wrapper(parts_cfg[part_idx], wrapper)
            t = time.perf_counter()
            sessions[part_idx] = load_session(wrapper)
            per_part_s.append(time.perf_counter() - t)
        bindings, out_bufs = make_bound_chain(sessions, parts_cfg)
        return sessions, bindings, out_bufs, per_part_s

    def ensure_mode(self, target):
        """Switch to target mode if not already loaded. Returns
        (swap_wall_s, per_part_load_s | None). swap_wall_s is 0
        when the right mode is already loaded."""
        if self.mode == target:
            return 0.0, None
        t_swap = time.perf_counter()
        if self.sessions is not None:
            # Drop refs and force-collect so QNN context releases HTP
            # memory before the new chain tries to allocate.
            self.sessions = None
            self.bindings = None
            self.out_bufs = None
            gc.collect()
        sessions, bindings, out_bufs, per_part_s = self._load(target)
        self.sessions = sessions
        self.bindings = bindings
        self.out_bufs = out_bufs
        self.mode = target
        self.warmup_done = False
        return time.perf_counter() - t_swap, per_part_s

    def shutdown(self):
        self.sessions = None
        self.bindings = None
        self.out_bufs = None
        self.mode = None
        gc.collect()


def _maybe_warmup(state, prompt_ids):
    """One HMX warmup call after every load — first NPU op pays a
    ~1 s context-init cost we don't want hitting the measured run."""
    if state.warmup_done:
        return
    kv_w = KVStore(NUM_LAYERS, with_ar128_input=(state.mode == "ar128"))
    if state.mode == "ar128":
        warmup_batch = list(prompt_ids[:AR128_BATCH])
        if len(warmup_batch) < AR128_BATCH:
            warmup_batch = (warmup_batch * (AR128_BATCH // len(warmup_batch) + 1))[:AR128_BATCH]
        _, _ = _step_ar128(state.sessions, state.bindings, state.out_bufs,
                           kv_w, 0, warmup_batch, state.scales_ar128)
    else:
        _, _ = _step(state.sessions, state.bindings, state.out_bufs,
                     kv_w, 0, prompt_ids[0], state.scales_ar1)
    state.warmup_done = True


def serve_request(state, prompt_ids, tg_tokens, ar128_min_tokens, force_ar128):
    """Execute one full inference request.

    Returns dict with route, swap_s, pp_compute_s, tg_compute_s,
    pp_tps, tg_tps, ar128_per_part_s, ar1_per_part_s.
    """
    if len(prompt_ids) + tg_tokens > PAST_LEN:
        return {"ok": False, "error":
                f"pp+tg = {len(prompt_ids) + tg_tokens} exceeds CL-512 cap {PAST_LEN}"}

    use_ar128 = (
        force_ar128 or (
            len(prompt_ids) >= AR128_BATCH
            and len(prompt_ids) >= ar128_min_tokens
        )
    )
    n_ar128_calls = (len(prompt_ids) // AR128_BATCH) if use_ar128 else 0
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH
    swap_total_s = 0.0
    ar128_per_part_s = None
    ar1_per_part_s = None

    pp_compute_s = 0.0
    pp_ar128_compute_s = 0.0
    pp_ar1_compute_s = 0.0
    last_logits = None
    kv = KVStore(NUM_LAYERS, with_ar128_input=use_ar128)

    # ----- AR128 prefill phase (if used) -----
    if use_ar128:
        swap_s, per_part = state.ensure_mode("ar128")
        swap_total_s += swap_s
        if per_part:
            ar128_per_part_s = per_part
        _maybe_warmup(state, prompt_ids)
        kv = KVStore(NUM_LAYERS, with_ar128_input=True)
        t = time.perf_counter()
        p = 0
        for _ in range(n_ar128_calls):
            batch = list(prompt_ids[p:p + AR128_BATCH])
            last_logits, _ = _step_ar128(
                state.sessions, state.bindings, state.out_bufs,
                kv, p, batch, state.scales_ar128,
            )
            p += AR128_BATCH
        pp_ar128_compute_s = time.perf_counter() - t

    # ----- AR1 chain for tail prefill + decode -----
    swap_s, per_part = state.ensure_mode("ar1")
    swap_total_s += swap_s
    if per_part:
        ar1_per_part_s = per_part
    _maybe_warmup(state, prompt_ids)

    # AR1 tail prefill
    if not use_ar128 or n_ar1_tail > 0:
        p = n_ar128_calls * AR128_BATCH
        t = time.perf_counter()
        while p < len(prompt_ids):
            last_logits, _ = _step(
                state.sessions, state.bindings, state.out_bufs,
                kv, p, prompt_ids[p], state.scales_ar1,
            )
            p += 1
        pp_ar1_compute_s = time.perf_counter() - t

    pp_compute_s = pp_ar128_compute_s + pp_ar1_compute_s
    pp_tps = len(prompt_ids) / pp_compute_s if pp_compute_s > 0 else 0.0

    # ----- AR1 decode -----
    next_token = int(np.argmax(last_logits))
    t = time.perf_counter()
    for i in range(tg_tokens):
        position = len(prompt_ids) + i
        logits_fp32, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            kv, position, next_token, state.scales_ar1,
        )
        next_token = int(np.argmax(logits_fp32))
    tg_compute_s = time.perf_counter() - t
    tg_tps = tg_tokens / tg_compute_s

    return {
        "ok": True,
        "route": "ar128+ar1tail" if (use_ar128 and n_ar1_tail) else ("ar128" if use_ar128 else "ar1"),
        "pp_tokens": len(prompt_ids),
        "tg_tokens": tg_tokens,
        "swap_s": swap_total_s,
        "pp_compute_s": pp_compute_s,
        "pp_ar128_compute_s": pp_ar128_compute_s,
        "pp_ar1_compute_s": pp_ar1_compute_s,
        "tg_compute_s": tg_compute_s,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ar128_per_part_s": ar128_per_part_s,
        "ar1_per_part_s": ar1_per_part_s,
    }


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_engine(args):
    """Build EngineState and warm to the args.start_mode. Returns
    (state, base_tokens, startup_s, per_part_s)."""
    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts_cfg_ar1 = build_part_cfg(metadata, ar=1)
    parts_cfg_ar128 = build_part_cfg(metadata, ar=AR128_BATCH)
    state = EngineState(parts_cfg_ar1, parts_cfg_ar128)

    tokenizer = Tokenizer.from_file(str(BUNDLE_DIR / "tokenizer.json"))
    base_tokens = tokenizer.encode(PROMPT_PATH.read_text(encoding="utf-8")).ids

    t = time.perf_counter()
    swap_s, per_part = state.ensure_mode(args.start_mode)
    startup_s = time.perf_counter() - t
    return state, base_tokens, startup_s, per_part


def synth_prompt(base_tokens, n):
    """Take n tokens from base_tokens, repeating if needed."""
    if n <= len(base_tokens):
        return list(base_tokens[:n])
    out = []
    while len(out) < n:
        out.extend(base_tokens)
    return out[:n]


def cmd_serve(args):
    state, base_tokens, startup_s, per_part = _load_engine(args)
    emit({"event": "ready", "startup_s": startup_s,
          "start_mode": args.start_mode,
          "start_per_part_s": [round(x, 2) for x in per_part]})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"ok": False, "error": f"json: {e}"})
            continue
        op = req.get("op")
        if op == "shutdown":
            emit({"event": "shutdown"})
            break
        if op != "infer":
            emit({"id": req.get("id"), "ok": False, "error": f"unknown op: {op}"})
            continue
        prompt_ids = synth_prompt(base_tokens, int(req["pp_tokens"]))
        result = serve_request(
            state, prompt_ids,
            tg_tokens=int(req["tg_tokens"]),
            ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
            force_ar128=bool(req.get("force_ar128", False)),
        )
        result["id"] = req.get("id")
        emit(result)
    state.shutdown()


# Default demo schedule — designed to exercise the state machine:
# 3 AR1 in a row (amortize over single warm chain), 1 AR128 (pays
# the swap), 1 AR128 in a row (NO additional swap — same mode), 2
# AR1 (pays one swap back). Total: 7 requests, only 2 swap events.
DEMO_SCHEDULE = [
    {"id": "a1", "pp": 128, "tg": 64},   # ar1
    {"id": "a2", "pp": 128, "tg": 64},   # ar1 (no swap)
    {"id": "a3", "pp": 256, "tg": 64},   # ar1 (no swap; below threshold)
    {"id": "b1", "pp": 384, "tg": 64, "force_ar128": True},  # swap → ar128
    {"id": "b2", "pp": 384, "tg": 64, "force_ar128": True},  # ar128 (no swap)
    {"id": "c1", "pp": 128, "tg": 64},   # swap back → ar1
    {"id": "c2", "pp": 128, "tg": 64},   # ar1 (no swap)
]


def cmd_demo(args):
    state, base_tokens, startup_s, per_part = _load_engine(args)
    print(f"=== sidecar demo ===")
    print(f"startup ({args.start_mode}) : {startup_s:.1f} s   "
          f"per-part: {[round(x, 1) for x in per_part]}")
    print()
    print(f"{'id':>3}  {'pp':>4}  {'tg':>3}  {'force':>5}  {'route':>14}  "
          f"{'swap_s':>7}  {'pp_compute':>11}  {'tg_compute':>11}  "
          f"{'pp_tps':>9}  {'tg_tps':>7}")
    print("-" * 100)
    t_total = time.perf_counter()
    n_swaps = 0
    cum_swap_s = 0.0
    cum_compute_s = 0.0
    for r in DEMO_SCHEDULE:
        prompt_ids = synth_prompt(base_tokens, r["pp"])
        result = serve_request(
            state, prompt_ids,
            tg_tokens=r["tg"],
            ar128_min_tokens=args.ar128_min_tokens,
            force_ar128=r.get("force_ar128", False),
        )
        if not result["ok"]:
            print(f"  {r['id']} FAILED: {result.get('error')}")
            continue
        if result["swap_s"] > 0.5:
            n_swaps += 1
        cum_swap_s += result["swap_s"]
        cum_compute_s += result["pp_compute_s"] + result["tg_compute_s"]
        print(f"  {r['id']:>3}  {r['pp']:>4}  {r['tg']:>3}  "
              f"{'Y' if r.get('force_ar128') else '':>5}  "
              f"{result['route']:>14}  "
              f"{result['swap_s']:>7.1f}  "
              f"{result['pp_compute_s']:>11.2f}  "
              f"{result['tg_compute_s']:>11.2f}  "
              f"{result['pp_tps']:>9.1f}  "
              f"{result['tg_tps']:>7.2f}")
    t_total = time.perf_counter() - t_total
    print()
    print(f"  N requests           : {len(DEMO_SCHEDULE)}")
    print(f"  swap events (>0.5 s) : {n_swaps}  "
          f"(would be {len(DEMO_SCHEDULE)} for N standalone bench runs)")
    print(f"  cum swap_s           : {cum_swap_s:.1f} s")
    print(f"  cum compute_s        : {cum_compute_s:.1f} s")
    print(f"  total wall (sidecar) : {t_total:.1f} s   "
          f"(startup {startup_s:.1f} + serve {t_total:.1f})")
    print(f"  total wall projected : "
          f"{startup_s + t_total:.1f} s including startup")

    # Project standalone: each request would pay one full ~14.8 s AR1 load,
    # plus AR128 requests would also pay a full swap_s like in the bench.
    standalone_load_s = 14.8  # measured AR1 cold load
    ar128_swap_s = 21.1       # measured AR128 swap (load 14.8 + teardown 6.3)
    standalone_total = 0.0
    for r in DEMO_SCHEDULE:
        standalone_total += standalone_load_s + r["pp"] * 0.0372 + r["tg"] * 0.0376
        if r.get("force_ar128"):
            # standalone bench always reaches AR1 at the end too
            standalone_total += ar128_swap_s
    print(f"  total wall (standalone, projected): {standalone_total:.1f} s")
    print(f"  ===> sidecar saves ~{standalone_total - (startup_s + t_total):.0f} s "
          f"over {len(DEMO_SCHEDULE)} standalone bench runs")
    state.shutdown()


def main():
    p = argparse.ArgumentParser(description="NPU engine sidecar")
    p.add_argument("--mode", choices=("demo", "serve"), default="demo")
    p.add_argument("--start-mode", choices=("ar1", "ar128"), default="ar1",
                   help="which chain to load at startup. ar1 is the steady-"
                        "state for decode + short prompts.")
    p.add_argument("--ar128-min-tokens", type=int, default=512,
                   help="prompt-token threshold above which inference uses "
                        "AR128 prefill (and pays a swap if needed).")
    args = p.parse_args()
    if args.mode == "demo":
        cmd_demo(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
