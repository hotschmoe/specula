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

# Model module pluggability — the sidecar is single-tenant per process,
# so a `--model` flag selects which oracle/chain modules to bind to the
# names used throughout this file. Both modules expose the same surface
# (BUNDLE_DIR, NUM_LAYERS, NUM_PARTS, KVStore, load_parts_cfg, ...).
import qualcomm_qwen3_4b_oracle as _model_4b  # noqa: E402
import qualcomm_qwen2_5_7b_oracle as _model_7b  # noqa: E402
import bench_qwen3_4b_ortqnn as _chain_4b  # noqa: E402
import bench_qwen2_5_7b_ortqnn as _chain_7b  # noqa: E402

# Default to 4B for backward-compat with existing callers / scripts.
_model = _model_4b
_chain = _chain_4b


def _bind_model(model_name: str) -> None:
    """Rebind module-level names from the selected model module pair.
    Call once before any sidecar function that touches the NPU.
    Functions defined in this file resolve these names via globals at
    call time, so rebinding here propagates through the whole module."""
    global _model, _chain
    global AR128_BATCH, BUNDLE_DIR, CTX_LEN, NUM_LAYERS, NUM_PARTS, KVStore
    global BACKEND_PATH, build_wrapper, dequant_uint16, load_session
    global load_parts_cfg, wrapper_path
    global CTX_TIERS, _step, _step_ar128, make_bound_chain, PROMPT_PATH
    if model_name == "qwen2_5-7b":
        _model = _model_7b
        _chain = _chain_7b
    elif model_name == "qwen3-4b":
        _model = _model_4b
        _chain = _chain_4b
    else:
        raise ValueError(f"unknown --model {model_name!r}; "
                         "expected 'qwen3-4b' or 'qwen2_5-7b'")
    AR128_BATCH = _model.AR128_BATCH
    BUNDLE_DIR = _model.BUNDLE_DIR
    CTX_LEN = _model.CTX_LEN
    NUM_LAYERS = _model.NUM_LAYERS
    NUM_PARTS = _model.NUM_PARTS
    KVStore = _model.KVStore
    BACKEND_PATH = _model.BACKEND_PATH
    build_wrapper = _model.build_wrapper
    dequant_uint16 = _model.dequant_uint16
    load_session = _model.load_session
    load_parts_cfg = _model.load_parts_cfg
    wrapper_path = _model.wrapper_path
    CTX_TIERS = _chain.CTX_TIERS
    _step = _chain._step
    _step_ar128 = _chain._step_ar128
    make_bound_chain = _chain.make_bound_chain
    PROMPT_PATH = _chain.PROMPT_PATH


# Initialize globals from the default model so module-level functions
# referencing these names resolve correctly even if main() runs --model
# qwen3-4b (the default).
_bind_model("qwen3-4b")


REPO_ROOT = _HERE.parent


def _scales_tuple(part_cfg):
    """cos/mask/logits scale+offset extracted from a parts_cfg dict.
    cos/mask come from part 2 (the first transformer partition — same
    in 4B and 7B). logits come from the last partition, which differs
    by model (4B: part 4, 7B: part 6) — read NUM_PARTS at call time."""
    def find(side, part_idx, name):
        return next(io for io in part_cfg[part_idx][side] if io["name"] == name)
    cos = find("inputs", 2, "position_ids_cos")
    mask = find("inputs", 2, "attention_mask")
    logits = find("outputs", NUM_PARTS, "logits")
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
        # Persistent streams keyed by stream_id. Holds Stream objects
        # that survive across requests so chat-server / spec-decode can
        # ingest only the delta tokens between rounds rather than re-
        # prefilling the full context every time.
        self.streams: dict[str, "Stream"] = {}

    def _load(self, target):
        if target == "ar1":
            parts_cfg, suffix = self.parts_cfg_ar1, ""
        else:
            parts_cfg, suffix = self.parts_cfg_ar128, "_ar128"
        sessions = {}
        per_part_s = []
        for part_idx in range(1, NUM_PARTS + 1):
            wrapper = wrapper_path(BUNDLE_DIR, part_idx, suffix, self.ctx_len)
            if not wrapper.exists():
                build_wrapper(parts_cfg[part_idx], wrapper)
            t = time.perf_counter()
            sessions[part_idx] = load_session(wrapper, backend_path=BACKEND_PATH)
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


def serve_draft_request(state, prompt_ids, n_draft, ar128_min_tokens, force_ar128):
    """Prefill the supplied prompt_ids and draft `n_draft` tokens; return
    the actual token IDs produced.

    Differs from `serve_request` in two ways:
      1. Caller supplies prompt_ids (driver controls tokenization).
      2. Returns `draft_ids` — the K decoded tokens, not just throughput.

    Used by SQ1's heterogeneous demo driver (NPU draft, CPU target verify).
    Composes prefill_only + decode_only so the existing Stream-based
    machinery does the work.
    """
    if not isinstance(prompt_ids, list) or not prompt_ids:
        return {"ok": False, "error": "prompt_ids must be a non-empty list[int]"}
    if len(prompt_ids) + n_draft > state.past_len:
        return {"ok": False, "error":
                f"pp+draft = {len(prompt_ids) + n_draft} exceeds CL-{state.ctx_len} cap {state.past_len}"}

    # Pre-mode-swap snapshot so the timing reflects the work this request
    # actually did (independent of which mode the engine was in before).
    pre_mode = state.mode
    swap_total_s = 0.0

    t_pp = time.perf_counter()
    # Capture mode after prefill_only so we can attribute its swap cost.
    if pre_mode != ("ar128" if (force_ar128 or
                                (len(prompt_ids) >= AR128_BATCH and
                                 len(prompt_ids) >= ar128_min_tokens)) else "ar1"):
        # Will swap; track via state.ensure_mode return values inside prefill_only.
        # (prefill_only itself doesn't return swap_s — we approximate via wall.)
        pass
    stream = prefill_only(
        state, stream_id="sq1-draft",
        prompt_ids=list(prompt_ids),
        ar128_min_tokens=ar128_min_tokens,
        force_ar128=force_ar128,
    )
    pp_wall_s = time.perf_counter() - t_pp

    t_tg = time.perf_counter()
    decode_only(state, stream, n_draft)
    tg_wall_s = time.perf_counter() - t_tg

    # The first decoded token is the prefill-predicted one (Stream
    # constructor stores next_token = argmax(last_logits)). Each decode
    # step appends to stream.decoded BEFORE refreshing next_token, so
    # stream.decoded[0:n_draft] is exactly the draft. Per the existing
    # decode_only contract.
    draft_ids = list(stream.decoded[-n_draft:])

    return {
        "ok": True,
        "draft_ids": draft_ids,
        "n_draft": n_draft,
        "prompt_tokens": len(prompt_ids),
        "pp_wall_s": pp_wall_s,
        "tg_wall_s": tg_wall_s,
        "swap_pre_mode": pre_mode,
        "swap_post_mode": state.mode,
        # NB: "swap" wall is folded into pp_wall_s when prefill_only triggered
        # ensure_mode("ar128") or ensure_mode("ar1") inside this call.
    }


def serve_chat_request(state, prompt_ids, max_new_tokens, eos_ids,
                       stop_token_seqs, ar128_min_tokens, force_ar128):
    """Greedy single-shot chat completion: prefill + decode with early-stop
    on EOS / stop-sequence / max_new_tokens. Returns the generated token
    IDs (not just timing), so an HTTP wrapper can decode them to text.

    Same prefill phase shape as `serve_request` (AR128 batches + AR1 tail).
    Decode loop differs: walks one step at a time, checking EOS set and
    stop-sequence suffix-match on `generated` after each token. Stateless
    — every call re-prefills the full prompt; that's the Phase A MVP shape.
    Phase B will replace this with stream_open/append/decode primitives
    that skip re-prefill across turns.

    Args:
      prompt_ids: list[int] — already chat-templated and tokenized
      max_new_tokens: hard cap on decode iterations
      eos_ids: list[int] — token IDs that terminate generation
      stop_token_seqs: list[list[int]] — token-ID sequences; if any matches
        as a suffix of `generated`, terminate (matched seq IS included in
        `generated`; HTTP wrapper strips before returning text)
      ar128_min_tokens, force_ar128: same semantics as serve_request

    Returns dict with: ok, generated_ids, stop_reason ("eos"/"stop"/"max_new_tokens"),
        n_prompt, n_generated, swap_s, pp_compute_s, tg_compute_s, pp_tps, tg_tps.
    """
    if len(prompt_ids) + max_new_tokens > state.past_len:
        return {"ok": False, "error":
                f"prompt+max_new = {len(prompt_ids) + max_new_tokens} > CL-{state.ctx_len} cap {state.past_len}"}

    use_ar128 = (
        force_ar128 or (
            len(prompt_ids) >= AR128_BATCH
            and len(prompt_ids) >= ar128_min_tokens
        )
    )
    n_ar128_calls = (len(prompt_ids) // AR128_BATCH) if use_ar128 else 0
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH
    swap_total_s = 0.0
    pp_ar128_compute_s = 0.0
    pp_ar1_compute_s = 0.0
    last_logits = None
    kv = KVStore(NUM_LAYERS, with_ar128_input=use_ar128, ctx_len=state.ctx_len)

    # ----- AR128 prefill phase -----
    if use_ar128:
        swap_s, _ = state.ensure_mode("ar128")
        swap_total_s += swap_s
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
    swap_s, _ = state.ensure_mode("ar1")
    swap_total_s += swap_s
    _maybe_warmup(state, prompt_ids)

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

    # ----- AR1 decode with early-stop -----
    eos_set = set(eos_ids or [])
    stop_seqs = [list(s) for s in (stop_token_seqs or []) if s]
    generated: list[int] = []
    next_token = int(np.argmax(last_logits))
    stop_reason = "max_new_tokens"

    t = time.perf_counter()
    for i in range(max_new_tokens):
        position = len(prompt_ids) + i
        logits, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            kv, position, next_token, state.scales_ar1,
        )
        generated.append(next_token)

        if next_token in eos_set:
            stop_reason = "eos"
            break
        for seq in stop_seqs:
            if len(generated) >= len(seq) and generated[-len(seq):] == seq:
                stop_reason = "stop"
                break
        if stop_reason == "stop":
            break

        next_token = int(np.argmax(logits))
    tg_compute_s = time.perf_counter() - t
    tg_tps = len(generated) / tg_compute_s if tg_compute_s > 0 else 0.0

    return {
        "ok": True,
        "generated_ids": generated,
        "stop_reason": stop_reason,
        "n_prompt": len(prompt_ids),
        "n_generated": len(generated),
        "swap_s": swap_total_s,
        "pp_compute_s": pp_compute_s,
        "tg_compute_s": tg_compute_s,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
    }


# ---------- Stateful stream API ----------
#
# These ops keep a Stream's KV cache live between requests so the next
# round only ingests the delta tokens since the last interaction. Two
# headline use cases:
#
#   1. SQ1 spec-decode (NPU rewind op):
#        stream_open(prompt) → stream_decode(K) → ...verify on target...
#        → stream_truncate(L+i*) + stream_append([target_token])
#        → stream_decode(K) → ... (next round)
#
#   2. SQ6 chat server:
#        stream_open(initial_prompt) → stream_decode_until_stop(...)
#        → ...user sends new message...
#        → stream_append(delta_tokens) → stream_decode_until_stop(...)
#
# Mode handling: streams are AR1-resident. stream_open prefills (AR128
# allowed for the initial prompt), but stream_append always uses AR1
# decode steps — large deltas (>128 tokens) are slow but functional.
# Future optimization: AR128 batched ingest for big deltas.


def _ensure_stream(state, stream_id: str):
    """Return (Stream, None) if found else (None, error_dict). Non-raising
    so the dispatch loop returns proper error JSON instead of crashing."""
    s = state.streams.get(stream_id)
    if s is None:
        return None, {"ok": False,
                      "error": f"unknown stream_id {stream_id!r}; call stream_open first"}
    return s, None


def serve_stream_open(state, stream_id, prompt_ids, ar128_min_tokens, force_ar128):
    """Run prefill for a fresh stream. Replaces any existing stream of
    the same id (idempotent for retries). Returns position + timings."""
    if not prompt_ids:
        return {"ok": False, "error": "prompt_ids must be non-empty"}
    if len(prompt_ids) > state.past_len:
        return {"ok": False, "error":
                f"prompt {len(prompt_ids)} > CL-{state.ctx_len} cap {state.past_len}"}
    if stream_id in state.streams:
        del state.streams[stream_id]

    t = time.perf_counter()
    stream = prefill_only(
        state, stream_id=stream_id, prompt_ids=list(prompt_ids),
        ar128_min_tokens=ar128_min_tokens, force_ar128=force_ar128,
    )
    pp_wall_s = time.perf_counter() - t
    state.streams[stream_id] = stream
    return {"ok": True, "stream_id": stream_id,
            "position": stream.position, "n_prompt": len(prompt_ids),
            "pp_wall_s": pp_wall_s}


def serve_stream_truncate(state, stream_id, new_position):
    """Drop KV slots at positions >= new_position. Slot data past .t is
    unattended (mask-controlled) so we don't bother zeroing it. After
    truncate, `next_token` and `last_logits` are stale; caller must
    `stream_append` before `stream_decode`."""
    stream, err = _ensure_stream(state, stream_id)
    if err:
        return err
    if new_position < 0 or new_position > stream.position:
        return {"ok": False, "error":
                f"new_position {new_position} out of range "
                f"(stream is at {stream.position})"}
    stream.kv.t = new_position
    stream.position = new_position
    stream.next_token = None
    stream.last_logits = None
    # Drop the corresponding tail of decoded[] so the stream's notion
    # of what's been decoded matches its KV state.
    n_keep = new_position - len(stream.prompt_ids)
    if n_keep < 0:
        n_keep = 0
    if len(stream.decoded) > n_keep:
        stream.decoded = stream.decoded[:n_keep]
    return {"ok": True, "stream_id": stream_id, "position": new_position}


def serve_stream_append(state, stream_id, append_ids, ar128_min_tokens, force_ar128):
    """Ingest each token in append_ids at the stream's current position.
    Uses AR1 decode steps (no AR128 batching today — TODO for big deltas).
    Updates stream.last_logits + stream.next_token after the final token."""
    stream, err = _ensure_stream(state, stream_id)
    if err:
        return err
    if not append_ids:
        return {"ok": True, "stream_id": stream_id,
                "position": stream.position, "n_appended": 0,
                "compute_s": 0.0}
    if stream.position + len(append_ids) > state.past_len:
        return {"ok": False, "error":
                f"append would overrun KV ({stream.position} + "
                f"{len(append_ids)} > {state.past_len})"}
    state.ensure_mode("ar1")
    _maybe_warmup(state, stream.prompt_ids)

    t = time.perf_counter()
    last_logits = None
    for tok in append_ids:
        logits, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            stream.kv, stream.position, int(tok), state.scales_ar1,
        )
        stream.decoded.append(int(tok))
        stream.position += 1
        last_logits = logits
    compute_s = time.perf_counter() - t

    stream.last_logits = last_logits
    stream.next_token = int(np.argmax(last_logits)) if last_logits is not None else None
    return {"ok": True, "stream_id": stream_id,
            "position": stream.position,
            "n_appended": len(append_ids),
            "compute_s": compute_s,
            "tps": len(append_ids) / compute_s if compute_s > 0 else 0.0}


def _decode_loop(state, stream, max_new, eos_ids, stop_token_seqs, on_token=None):
    """Shared decode loop for stream_decode + stream_decode_stream.

    `on_token`: optional callback invoked with each newly decoded token
    after it's committed to KV. Used by the streaming variant to emit
    per-token events.

    Returns (generated_ids, stop_reason, compute_s).
    """
    if stream.next_token is None:
        raise RuntimeError("stream has no next_token; stream_append before stream_decode")
    state.ensure_mode("ar1")
    _maybe_warmup(state, stream.prompt_ids)

    eos_set = set(eos_ids or [])
    stop_seqs = [list(s) for s in (stop_token_seqs or []) if s]
    generated: list[int] = []
    stop_reason = "max_new_tokens"

    t = time.perf_counter()
    for i in range(max_new):
        if stream.position >= state.past_len:
            stop_reason = "ctx_full"
            break
        logits, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            stream.kv, stream.position, stream.next_token, state.scales_ar1,
        )
        token = stream.next_token
        stream.decoded.append(token)
        stream.position += 1
        stream.last_logits = logits
        stream.next_token = int(np.argmax(logits))
        generated.append(token)
        if on_token is not None and token not in eos_set:
            on_token(token)

        if token in eos_set:
            stop_reason = "eos"
            break
        for seq in stop_seqs:
            if len(generated) >= len(seq) and generated[-len(seq):] == seq:
                stop_reason = "stop"
                break
        if stop_reason == "stop":
            break
    compute_s = time.perf_counter() - t
    return generated, stop_reason, compute_s


def serve_stream_decode(state, stream_id, max_new, eos_ids, stop_token_seqs):
    """Run up to max_new greedy decode steps with optional early-stop on
    EOS / stop-sequence. Returns the generated token IDs."""
    stream, err = _ensure_stream(state, stream_id)
    if err:
        return err
    if max_new <= 0:
        return {"ok": False, "error": "max_new must be > 0"}
    if stream.next_token is None:
        return {"ok": False, "error":
                "stream has no next_token; stream_append before stream_decode"}
    generated, stop_reason, compute_s = _decode_loop(
        state, stream, max_new, eos_ids, stop_token_seqs)
    return {
        "ok": True, "stream_id": stream_id,
        "generated_ids": generated,
        "stop_reason": stop_reason,
        "position": stream.position,
        "n_generated": len(generated),
        "compute_s": compute_s,
        "tps": len(generated) / compute_s if compute_s > 0 else 0.0,
    }


def serve_stream_decode_stream(state, req_id, stream_id, max_new,
                               eos_ids, stop_token_seqs):
    """Streaming variant: emits {event:"token",token_id} per decoded token,
    concludes with chat_done summary."""
    stream, err = _ensure_stream(state, stream_id)
    if err:
        emit({"id": req_id, "event": "chat_done", **err})
        return
    if stream.next_token is None:
        emit({"id": req_id, "event": "chat_done", "ok": False,
              "error": "stream has no next_token; stream_append before stream_decode"})
        return

    def on_token(tok):
        emit({"id": req_id, "event": "token", "token_id": tok})

    generated, stop_reason, compute_s = _decode_loop(
        state, stream, max_new, eos_ids, stop_token_seqs, on_token=on_token)
    emit({
        "id": req_id, "event": "chat_done", "ok": True,
        "stream_id": stream_id,
        "stop_reason": stop_reason,
        "position": stream.position,
        "n_generated": len(generated),
        # Includes EOS if present — caller mirrors this into its own
        # history tracker so longest-common-prefix between turns sees
        # the same prefix the NPU's KV holds.
        "generated_ids": generated,
        "compute_s": compute_s,
        "tps": len(generated) / compute_s if compute_s > 0 else 0.0,
    })


def serve_stream_close(state, stream_id):
    state.streams.pop(stream_id, None)
    return {"ok": True, "stream_id": stream_id}


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def serve_chat_stream_request(state, req_id, prompt_ids, max_new_tokens,
                              eos_ids, stop_token_seqs,
                              ar128_min_tokens, force_ar128):
    """Streaming variant of serve_chat_request. Same prefill phase, but
    after each decoded token emits a {"event":"token","token_id":T} line
    via `emit`. Concludes with {"event":"chat_done", ...summary...}. The
    caller (HTTP wrapper) reads lines until it sees the chat_done event.

    Returns nothing (`emit`s directly). The serve loop checks the return
    value to decide whether to also emit a final response — this op
    handles its own emission.
    """
    if len(prompt_ids) + max_new_tokens > state.past_len:
        emit({"id": req_id, "event": "chat_done", "ok": False,
              "error": f"prompt+max_new = {len(prompt_ids) + max_new_tokens} > "
                       f"CL-{state.ctx_len} cap {state.past_len}"})
        return

    use_ar128 = (
        force_ar128 or (
            len(prompt_ids) >= AR128_BATCH
            and len(prompt_ids) >= ar128_min_tokens
        )
    )
    n_ar128_calls = (len(prompt_ids) // AR128_BATCH) if use_ar128 else 0
    n_ar1_tail = len(prompt_ids) - n_ar128_calls * AR128_BATCH
    swap_total_s = 0.0
    pp_ar128_compute_s = 0.0
    pp_ar1_compute_s = 0.0
    last_logits = None
    kv = KVStore(NUM_LAYERS, with_ar128_input=use_ar128, ctx_len=state.ctx_len)

    if use_ar128:
        swap_s, _ = state.ensure_mode("ar128")
        swap_total_s += swap_s
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

    swap_s, _ = state.ensure_mode("ar1")
    swap_total_s += swap_s
    _maybe_warmup(state, prompt_ids)

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

    eos_set = set(eos_ids or [])
    stop_seqs = [list(s) for s in (stop_token_seqs or []) if s]
    generated: list[int] = []
    next_token = int(np.argmax(last_logits))
    stop_reason = "max_new_tokens"

    t = time.perf_counter()
    for i in range(max_new_tokens):
        position = len(prompt_ids) + i
        logits, _ = _step(
            state.sessions, state.bindings, state.out_bufs,
            kv, position, next_token, state.scales_ar1,
        )
        generated.append(next_token)
        # Emit token event (excluding final EOS — http_server strips it
        # from the visible text anyway, and clients prefer the EOS not
        # to leak into the SSE delta stream).
        if next_token not in eos_set:
            emit({"id": req_id, "event": "token", "token_id": next_token})

        if next_token in eos_set:
            stop_reason = "eos"
            break
        for seq in stop_seqs:
            if len(generated) >= len(seq) and generated[-len(seq):] == seq:
                stop_reason = "stop"
                break
        if stop_reason == "stop":
            break

        next_token = int(np.argmax(logits))
    tg_compute_s = time.perf_counter() - t
    tg_tps = len(generated) / tg_compute_s if tg_compute_s > 0 else 0.0

    emit({
        "id": req_id,
        "event": "chat_done",
        "ok": True,
        "stop_reason": stop_reason,
        "n_prompt": len(prompt_ids),
        "n_generated": len(generated),
        "swap_s": swap_total_s,
        "pp_compute_s": pp_compute_s,
        "tg_compute_s": tg_compute_s,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
    })


def _load_engine(args):
    """Build EngineState and warm to the args.start_mode. Returns
    (state, base_tokens, startup_s, per_part_s)."""
    ctx_tier = getattr(args, "ctx_tier", CTX_LEN)
    parts_cfg_ar1 = load_parts_cfg(ar=1, ctx=ctx_tier)
    parts_cfg_ar128 = load_parts_cfg(ar=AR128_BATCH, ctx=ctx_tier)
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
        if op == "infer":
            prompt_ids = synth_prompt(base_tokens, int(req["pp_tokens"]))
            result = serve_request(
                state, prompt_ids,
                tg_tokens=int(req["tg_tokens"]),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
        elif op == "draft":
            # Driver-supplied prompt; returns the actual K drafted tokens
            # so the caller can verify them against a target model.
            prompt_ids = list(req["prompt_ids"])
            result = serve_draft_request(
                state, prompt_ids,
                n_draft=int(req["n_draft"]),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
        elif op == "chat":
            # Single-shot chat completion. Caller supplies pre-tokenized
            # prompt + max_new + EOS/stop config; sidecar runs prefill +
            # greedy decode loop with early stop. Used by http_server.py.
            prompt_ids = list(req["prompt_ids"])
            result = serve_chat_request(
                state, prompt_ids,
                max_new_tokens=int(req["max_new_tokens"]),
                eos_ids=req.get("eos_ids", []),
                stop_token_seqs=req.get("stop_token_seqs", []),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
        elif op == "chat_stream":
            # Streaming variant: emits one {event:"token", token_id} per
            # decoded token, then a final {event:"chat_done", ...} summary.
            # Caller is responsible for reading lines until chat_done.
            prompt_ids = list(req["prompt_ids"])
            serve_chat_stream_request(
                state, req_id=req.get("id"),
                prompt_ids=prompt_ids,
                max_new_tokens=int(req["max_new_tokens"]),
                eos_ids=req.get("eos_ids", []),
                stop_token_seqs=req.get("stop_token_seqs", []),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
            # serve_chat_stream_request emits its own chat_done; skip the
            # generic emit at the bottom of the loop.
            continue
        elif op == "stream_open":
            result = serve_stream_open(
                state, stream_id=str(req["stream_id"]),
                prompt_ids=list(req["prompt_ids"]),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
        elif op == "stream_truncate":
            result = serve_stream_truncate(
                state, stream_id=str(req["stream_id"]),
                new_position=int(req["new_position"]),
            )
        elif op == "stream_append":
            result = serve_stream_append(
                state, stream_id=str(req["stream_id"]),
                append_ids=list(req["append_ids"]),
                ar128_min_tokens=int(req.get("ar128_min_tokens", args.ar128_min_tokens)),
                force_ar128=bool(req.get("force_ar128", False)),
            )
        elif op == "stream_decode":
            result = serve_stream_decode(
                state, stream_id=str(req["stream_id"]),
                max_new=int(req["max_new"]),
                eos_ids=req.get("eos_ids", []),
                stop_token_seqs=req.get("stop_token_seqs", []),
            )
        elif op == "stream_decode_stream":
            serve_stream_decode_stream(
                state, req_id=req.get("id"),
                stream_id=str(req["stream_id"]),
                max_new=int(req["max_new"]),
                eos_ids=req.get("eos_ids", []),
                stop_token_seqs=req.get("stop_token_seqs", []),
            )
            continue
        elif op == "stream_close":
            result = serve_stream_close(state, stream_id=str(req["stream_id"]))
        else:
            emit({"id": req.get("id"), "ok": False, "error": f"unknown op: {op}"})
            continue
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
    p.add_argument("--model", choices=("qwen3-4b", "qwen2_5-7b"),
                   default="qwen3-4b",
                   help="which Qualcomm NPU bundle to drive. qwen3-4b is "
                        "the original 4-partition w4a16 bundle "
                        "(metadata.yaml-based, multi-ctx-tier). qwen2_5-7b "
                        "is the 6-partition w8a16 bundle "
                        "(qnn-context-binary-utility introspection, "
                        "cl=4096 only, requires QAIRT 2.45 backend DLL).")
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
    p.add_argument("--ctx-tier", type=int, default=None,
                   help="bundle context-tier graphs to load. Per-model:"
                        " qwen3-4b ships {512, 1024, 2048, 3072, 4096};"
                        " qwen2_5-7b ships only {4096}. Defaults to the model's"
                        " CTX_LEN. KV memory grows linearly per session; the"
                        " HTP simultaneous-session ceiling shrinks as ctx grows.")
    args = p.parse_args()
    _bind_model(args.model)
    if args.ctx_tier is None:
        args.ctx_tier = CTX_LEN
    if args.ctx_tier not in CTX_TIERS:
        raise SystemExit(
            f"--ctx-tier {args.ctx_tier} not available for --model "
            f"{args.model}. Bundle ships {CTX_TIERS}."
        )
    if args.mode == "demo":
        cmd_demo(args)
    elif args.mode == "demo-phase-batch":
        cmd_demo_phase_batch(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
