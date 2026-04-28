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
    CTX_LEN,
    NUM_LAYERS,
    KVStore,
    build_part_cfg,
    build_wrapper,
    dequant_uint16,
    load_session,
    wrapper_path,
)
from bench_qwen3_4b_ortqnn import (  # noqa: E402
    CTX_TIERS,
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
    `ensure_mode(target)` — returns the swap wall time (0 if hit).

    `ctx_len` (default 512) selects the bundle's context-tier graphs
    that this engine will load. The two parts_cfg dicts must be built
    against the same ctx_len so AR1 ⇄ AR128 swaps share KV layout."""

    def __init__(self, parts_cfg_ar1, parts_cfg_ar128, ctx_len: int = CTX_LEN):
        self.parts_cfg_ar1 = parts_cfg_ar1
        self.parts_cfg_ar128 = parts_cfg_ar128
        self.scales_ar1 = _scales_tuple(parts_cfg_ar1)
        self.scales_ar128 = _scales_tuple(parts_cfg_ar128)
        self.ctx_len = ctx_len
        self.past_len = ctx_len - 1
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
            wrapper = wrapper_path(BUNDLE_DIR, part_idx, suffix, self.ctx_len)
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
    kv_w = KVStore(
        NUM_LAYERS,
        with_ar128_input=(state.mode == "ar128"),
        ctx_len=state.ctx_len,
    )
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


class Stream:
    """Per-stream inference state — KV cache + decode position +
    last logits. The engine's mode is shared; each Stream just owns
    its own KV. Lets phase-batched workloads run N prefills against
    N independent KVs in one AR128 mode-batch, then drain N decodes
    in one AR1 mode-batch."""

    __slots__ = ("id", "prompt_ids", "kv", "last_logits", "next_token",
                 "position", "decoded")

    def __init__(self, stream_id, prompt_ids, kv, last_logits):
        self.id = stream_id
        self.prompt_ids = prompt_ids
        self.kv = kv
        self.last_logits = last_logits
        self.next_token = int(np.argmax(last_logits)) if last_logits is not None else None
        self.position = len(prompt_ids)
        self.decoded: list[int] = []


def prefill_only(state, stream_id, prompt_ids, ar128_min_tokens, force_ar128):
    """Run prefill (no decode). Returns a Stream with KV ready for
    later decode_only() calls.

    Caller is responsible for ordering — to get the phase-batching
    win, batch all AR128-eligible prefills together so they share
    one ensure_mode("ar128") swap. AR1-mode prefills can interleave
    freely with AR1 decodes (same chain).

    NOTE: prefills with an AR1 tail (pp_tokens not a multiple of
    AR128_BATCH) currently force a swap to AR1 inside this call,
    breaking the batch. To stay in AR128 across N batched prefills,
    use multiple-of-128 prompt sizes.
    """
    if len(prompt_ids) > state.past_len:
        raise ValueError(
            f"prompt {len(prompt_ids)} exceeds CL-{state.ctx_len} cap {state.past_len}"
        )
    use_ar128 = force_ar128 or (
        len(prompt_ids) >= AR128_BATCH
        and len(prompt_ids) >= ar128_min_tokens
    )
    n_ar128_calls = (len(prompt_ids) // AR128_BATCH) if use_ar128 else 0
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH

    kv = KVStore(NUM_LAYERS, with_ar128_input=use_ar128, ctx_len=state.ctx_len)
    last_logits = None

    if use_ar128:
        state.ensure_mode("ar128")
        _maybe_warmup(state, prompt_ids)
        # Re-init kv after warmup mutated it
        kv = KVStore(NUM_LAYERS, with_ar128_input=True, ctx_len=state.ctx_len)
        p = 0
        for _ in range(n_ar128_calls):
            batch = list(prompt_ids[p:p + AR128_BATCH])
            last_logits, _ = _step_ar128(
                state.sessions, state.bindings, state.out_bufs,
                kv, p, batch, state.scales_ar128,
            )
            p += AR128_BATCH

    if not use_ar128 or n_ar1_tail > 0:
        state.ensure_mode("ar1")
        _maybe_warmup(state, prompt_ids)
        if not use_ar128:
            kv = KVStore(NUM_LAYERS, ctx_len=state.ctx_len)
        p = n_ar128_calls * AR128_BATCH if use_ar128 else 0
        while p < len(prompt_ids):
            last_logits, _ = _step(
                state.sessions, state.bindings, state.out_bufs,
                kv, p, prompt_ids[p], state.scales_ar1,
            )
            p += 1

    return Stream(stream_id, prompt_ids, kv, last_logits)


def decode_only(state, stream, n_tokens):
    """Run n_tokens decode steps against `stream`. Mutates the stream
    in place. Requires AR1 mode; ensure_mode is idempotent so calling
    decode_only across multiple streams in a row pays only ONE swap
    when transitioning from AR128 prefill mode."""
    if stream.position + n_tokens > state.past_len:
        raise ValueError(
            f"stream {stream.id}: decode would overrun KV ({stream.position} + "
            f"{n_tokens} > {state.past_len})"
        )
    state.ensure_mode("ar1")
    _maybe_warmup(state, stream.prompt_ids)
    for _ in range(n_tokens):
        logits, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            stream.kv, stream.position, stream.next_token, state.scales_ar1,
        )
        stream.decoded.append(stream.next_token)
        stream.next_token = int(np.argmax(logits))
        stream.position += 1
        stream.last_logits = logits


def serve_request(state, prompt_ids, tg_tokens, ar128_min_tokens, force_ar128):
    """Execute one full inference request.

    Returns dict with route, swap_s, pp_compute_s, tg_compute_s,
    pp_tps, tg_tps, ar128_per_part_s, ar1_per_part_s.
    """
    if len(prompt_ids) + tg_tokens > state.past_len:
        return {"ok": False, "error":
                f"pp+tg = {len(prompt_ids) + tg_tokens} exceeds CL-{state.ctx_len} cap {state.past_len}"}

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
    kv = KVStore(NUM_LAYERS, with_ar128_input=use_ar128, ctx_len=state.ctx_len)

    # ----- AR128 prefill phase (if used) -----
    if use_ar128:
        swap_s, per_part = state.ensure_mode("ar128")
        swap_total_s += swap_s
        if per_part:
            ar128_per_part_s = per_part
        _maybe_warmup(state, prompt_ids)
        kv = KVStore(NUM_LAYERS, with_ar128_input=True, ctx_len=state.ctx_len)
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
    ctx_tier = getattr(args, "ctx_tier", CTX_LEN)
    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    parts_cfg_ar1 = build_part_cfg(metadata, ar=1, ctx=ctx_tier)
    parts_cfg_ar128 = build_part_cfg(metadata, ar=AR128_BATCH, ctx=ctx_tier)
    state = EngineState(parts_cfg_ar1, parts_cfg_ar128, ctx_len=ctx_tier)

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


def cmd_demo_phase_batch(args):
    """Head-to-head: NAIVE sequential (each request is full prefill+decode)
    vs PHASE-BATCHED (all prefills in one AR128 mode-batch, then all
    decodes in one AR1 mode-batch). Same workload, two execution orders.

    The win comes from removing per-request swap round-trips. With N
    AR128 requests, naive pays N round-trips (~42 s each); phase-batched
    pays exactly two (~21 s each). Compute time is identical.
    """
    state, base_tokens, startup_s, per_part = _load_engine(args)
    print(f"=== sidecar phase-batched A/B demo ===")
    print(f"startup ({args.start_mode}) : {startup_s:.1f} s   "
          f"per-part: {[round(x, 1) for x in per_part]}")
    print()

    # N requests of pp=384 (3 AR128 calls each, no AR1 tail), tg=64.
    # Choosing pp=384 / tg=64 keeps each request well under CL-512
    # (384+64=448) and exercises real AR128 batching.
    n_requests = args.n_phase_batch
    pp = 384
    tg = 64

    print(f"workload: {n_requests} × (pp={pp} forced AR128, tg={tg})")
    print()

    # ---------- Pass 1: NAIVE sequential ----------
    print("--- NAIVE sequential (current sidecar default) ---")
    print(f"  each request runs full prefill+decode, swapping AR1 ⇄ AR128")
    naive_swap = naive_compute = 0.0
    t = time.perf_counter()
    for i in range(n_requests):
        prompt_ids = synth_prompt(base_tokens, pp)
        result = serve_request(
            state, prompt_ids, tg_tokens=tg,
            ar128_min_tokens=args.ar128_min_tokens, force_ar128=True,
        )
        if not result["ok"]:
            print(f"  req {i}: FAILED {result.get('error')}")
            continue
        naive_swap += result["swap_s"]
        naive_compute += result["pp_compute_s"] + result["tg_compute_s"]
        print(f"  req {i}: route={result['route']:>13}  "
              f"swap={result['swap_s']:5.1f}s  "
              f"pp={result['pp_compute_s']:5.2f}s  tg={result['tg_compute_s']:5.2f}s")
    naive_total = time.perf_counter() - t
    print(f"  totals: swap={naive_swap:.1f}s  compute={naive_compute:.1f}s  "
          f"wall={naive_total:.1f}s")
    print()

    # ---------- Pass 2: PHASE-BATCHED ----------
    print("--- PHASE-BATCHED (prefill all → decode all) ---")
    print(f"  one swap to AR128 for all prefills, one swap to AR1 for all decodes")
    pb_swap = pb_compute_pp = pb_compute_tg = 0.0
    t = time.perf_counter()

    # Phase 1: batch all prefills in AR128
    streams: list[Stream] = []
    pre_mode = state.mode
    t_pre = time.perf_counter()
    for i in range(n_requests):
        prompt_ids = synth_prompt(base_tokens, pp)
        # Detect if this prefill_only call triggers a swap by sampling
        # state.mode before and after; we add the swap time post hoc.
        before_mode = state.mode
        t_call = time.perf_counter()
        stream = prefill_only(
            state, stream_id=i, prompt_ids=prompt_ids,
            ar128_min_tokens=args.ar128_min_tokens, force_ar128=True,
        )
        dt = time.perf_counter() - t_call
        # If mode changed at this call, attribute the leading slack to swap.
        # Simpler: just measure compute as the per-call AR128 latency budget
        # (n_ar128_calls × ~60 ms) and let the rest be swap. Approximate.
        n_ar128 = pp // AR128_BATCH
        approx_compute = n_ar128 * 0.060
        approx_swap = max(0.0, dt - approx_compute)
        if before_mode != state.mode:
            pb_swap += approx_swap
            pb_compute_pp += approx_compute
        else:
            pb_compute_pp += dt
        streams.append(stream)
        print(f"  prefill stream {i}: {dt:.2f}s  "
              f"(mode now={state.mode}, swap_approx={approx_swap:.1f}s)")

    # Phase 2: batch all decodes in AR1
    for s in streams:
        before_mode = state.mode
        t_call = time.perf_counter()
        decode_only(state, s, tg)
        dt = time.perf_counter() - t_call
        approx_compute = tg * 0.038
        approx_swap = max(0.0, dt - approx_compute)
        if before_mode != state.mode:
            pb_swap += approx_swap
            pb_compute_tg += approx_compute
        else:
            pb_compute_tg += dt
        print(f"  decode stream {s.id}: {dt:.2f}s  "
              f"(mode now={state.mode}, swap_approx={approx_swap:.1f}s)")

    pb_total = time.perf_counter() - t
    print(f"  totals: swap={pb_swap:.1f}s  pp={pb_compute_pp:.1f}s  "
          f"tg={pb_compute_tg:.1f}s  wall={pb_total:.1f}s")
    print()

    # ---------- Comparison ----------
    speedup = naive_total / pb_total if pb_total > 0 else 0
    saved = naive_total - pb_total
    print("--- comparison ---")
    print(f"  NAIVE     : {naive_total:.1f} s ({n_requests} swap round-trips)")
    print(f"  PHASE-BAT : {pb_total:.1f} s (~2 swap round-trips total)")
    print(f"  speedup   : {speedup:.2f}×   saved {saved:.0f} s on {n_requests} requests")
    print(f"  per-request avg: NAIVE {naive_total/n_requests:.1f}s vs "
          f"PHASE-BAT {pb_total/n_requests:.1f}s")
    state.shutdown()


def main():
    p = argparse.ArgumentParser(description="NPU engine sidecar")
    p.add_argument("--mode", choices=("demo", "demo-phase-batch", "serve"), default="demo")
    p.add_argument("--n-phase-batch", type=int, default=3,
                   help="number of AR128 requests in the phase-batched A/B demo "
                        "(default 3). Larger N exaggerates the speedup since "
                        "naive pays per-request swap and phase-batched doesn't.")
    p.add_argument("--start-mode", choices=("ar1", "ar128"), default="ar1",
                   help="which chain to load at startup. ar1 is the steady-"
                        "state for decode + short prompts.")
    p.add_argument("--ar128-min-tokens", type=int, default=512,
                   help="prompt-token threshold above which inference uses "
                        "AR128 prefill (and pays a swap if needed).")
    p.add_argument("--ctx-tier", type=int, default=CTX_LEN, choices=CTX_TIERS,
                   help="bundle context-tier graphs to load. Bundle ships {512,"
                        " 1024, 2048, 3072, 4096}. KV memory grows linearly per"
                        " session; the HTP simultaneous-session ceiling shrinks"
                        " as ctx grows.")
    args = p.parse_args()
    if args.mode == "demo":
        cmd_demo(args)
    elif args.mode == "demo-phase-batch":
        cmd_demo_phase_batch(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
