"""OpenAI-compatible HTTP server backed by the NPU sidecar.

Spawns `npu_engine/sidecar.py --mode serve` as a long-lived subprocess
(holds the 4 ORT-QNN sessions and amortizes the ~15 s startup cost),
then exposes `/v1/chat/completions` so any agent CLI (pi/hermes/opencode)
can drive the NPU just like a local llama-server.

Phase A (this file) is the **stateless** MVP: every request re-prefills
the full message history. Correct but slow — multi-turn dialogue at
TG ~22 t/s with a re-prefill of growing context per turn. Phase B will
add stateful streams (`stream_open`/`stream_append`/`stream_decode`/
`stream_truncate`) so subsequent turns ingest only the delta tokens.

Run:
    .venv\\Scripts\\python.exe -m uvicorn npu_engine.http_server:app \\
        --host 127.0.0.1 --port 8081 --no-access-log

Then point pi / opencode / hermes at:
    base URL: http://127.0.0.1:8081/v1
    api key : (anything)
    model   : qwen3-4b-npu

Or one-shot smoke test:
    curl http://127.0.0.1:8081/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"qwen3-4b-npu","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
BUNDLE_DIR = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"
)
TOKENIZER_PATH = BUNDLE_DIR / "tokenizer.json"

# Qwen3 ChatML special tokens. Stable across all Qwen3-* tokenizers
# (verified via probe_tokenizer_match.py for Qwen3-4B). Hardcoded so
# we don't have to round-trip through tokenizer.token_to_id() for
# every request.
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
DEFAULT_EOS_IDS = [IM_END_ID, ENDOFTEXT_ID]

MODEL_NAME = "qwen3-4b-npu"

# Phase A re-prefills every turn. We cap max_tokens conservatively so a
# 4K-ctx run can fit ~3.5K prompt + ~512 generated. Override via request.
DEFAULT_MAX_NEW_TOKENS = 512

# How long to wait for sidecar startup (per-part NPU load × 4). The 4B
# bundle reports ~15 s typical from cold per `bench_qwen3_4b_ortqnn.py`.
SIDECAR_READY_TIMEOUT_S = 60.0


# ---------- Sidecar subprocess wrapper ----------

class SidecarClient:
    """Talks to npu_engine/sidecar.py via JSON-over-stdio.

    Single asyncio lock around request/response so we don't interleave
    requests on the NPU (which is unstable past 3-way concurrency per
    docs/qwen2_5_7b_baseline_all_backends.md). Caller of `request` is
    expected to be inside the lock — see `chat_completion`.
    """

    def __init__(self, ctx_tier: int = 2048):
        self.ctx_tier = ctx_tier
        self.proc: subprocess.Popen | None = None
        self.startup_info: dict | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        cmd = [
            sys.executable, str(_HERE / "sidecar.py"),
            "--mode", "serve",
            "--ctx-tier", str(self.ctx_tier),
            "--start-mode", "ar1",
        ]
        # Inherit env but strip Python-bootstrap vars that might leak
        # from a wrapper venv (Path C demo learned this the hard way).
        child_env = {k: v for k, v in os.environ.items()
                     if k not in ("PYTHONHOME", "PYTHONPATH",
                                  "PYTHONSTARTUP", "VIRTUAL_ENV")}
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8",
            env=child_env,
        )
        # Wait for "ready" event (blocks the event loop briefly during
        # startup; that's fine — the server isn't accepting requests yet).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SIDECAR_READY_TIMEOUT_S
        while loop.time() < deadline:
            line = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not line:
                stderr_tail = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"sidecar died before ready: {stderr_tail}")
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                # Sidecar prints non-JSON debug occasionally; skip
                continue
            if evt.get("event") == "ready":
                self.startup_info = evt
                return
        raise TimeoutError(f"sidecar did not emit 'ready' within {SIDECAR_READY_TIMEOUT_S}s")

    async def request(self, op: str, **kwargs) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("sidecar is not running")
        req = {"op": op, "id": kwargs.pop("id", "http"), **kwargs}
        loop = asyncio.get_running_loop()
        line = json.dumps(req) + "\n"
        await loop.run_in_executor(None, self.proc.stdin.write, line)
        await loop.run_in_executor(None, self.proc.stdin.flush)
        rsp_line = await loop.run_in_executor(None, self.proc.stdout.readline)
        if not rsp_line:
            raise RuntimeError("sidecar closed stdout mid-request")
        return json.loads(rsp_line.strip())

    async def stream_request(self, op: str, **kwargs):
        """Async generator yielding event dicts until a `chat_done` event
        arrives. Caller is responsible for holding self.lock."""
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("sidecar is not running")
        req = {"op": op, "id": kwargs.pop("id", "http-stream"), **kwargs}
        loop = asyncio.get_running_loop()
        line = json.dumps(req) + "\n"
        await loop.run_in_executor(None, self.proc.stdin.write, line)
        await loop.run_in_executor(None, self.proc.stdin.flush)
        while True:
            rsp_line = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not rsp_line:
                raise RuntimeError("sidecar closed stdout mid-stream")
            evt = json.loads(rsp_line.strip())
            yield evt
            if evt.get("event") == "chat_done":
                return

    async def shutdown(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            await self.request("shutdown")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ---------- Chat template (Qwen3 ChatML) ----------

def render_chatml(messages: list[dict], enable_thinking: bool = True) -> str:
    """Render OpenAI-style messages to Qwen3's ChatML.

    Minimal template (no tool-calling). Lifted from upstream
    Qwen/Qwen3-4B/tokenizer_config.json:

        <|im_start|>{role}\\n{content}<|im_end|>\\n  ... per message
        <|im_start|>assistant\\n                     ... generation cue
        <think>\\n\\n</think>\\n\\n                  ... if !enable_thinking

    Qwen3 emits `<think>...</think>` blocks by default. For coding-agent
    use this is usually noise — set `enable_thinking=False` to inject an
    empty `<think></think>` so the model continues past it directly to
    the answer (matches upstream's `enable_thinking=False` template path).

    `add_generation_prompt=true` is implicit (always added — every
    chat-completion request expects a model turn next).
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            # OpenAI allows content to be a list of parts; flatten the
            # text parts for now (no multimodal on this NPU).
            text_parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
            content = "".join(text_parts)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


# ---------- FastAPI app ----------

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = 0.0  # accepted but ignored (greedy only)
    top_p: float | None = 1.0  # accepted but ignored
    stop: list[str] | str | None = None
    stream: bool | None = False  # streaming arrives in Phase A.5
    n: int | None = 1
    # Qwen3-specific: inject empty <think></think> to suppress thinking
    # mode. Default True matches upstream Qwen3's default chat template;
    # coding agents that want terse output should set False.
    enable_thinking: bool | None = True


sidecar: SidecarClient | None = None
tokenizer: Tokenizer | None = None


class ConversationState:
    """Tracks the active server-side stream and its known token history.

    Single-tenant: one stream for the whole server. `history` is the full
    token-id sequence currently sitting in the NPU's KV cache (prompt
    tokens for the first turn + everything the model has decoded since,
    including any EOS tokens). When a new chat-completion request comes
    in, we tokenize its rendered ChatML and find the longest common
    prefix with `history`:
      * lcp == len(history) and len(new) >  lcp  → stream_append delta
      * lcp == len(history) and len(new) == lcp  → no ingest needed
      * lcp <  len(history)                      → truncate then append
      * stream_id is None (fresh server)          → stream_open(new)
    """

    def __init__(self, stream_id: str = "http"):
        self.stream_id = stream_id
        self.opened = False
        self.history: list[int] = []

    def reset(self):
        self.opened = False
        self.history = []


conv_state: ConversationState | None = None


def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


async def _stream_reopen(prompt_ids: list[int]) -> tuple[str, float]:
    """Close any existing stream, then stream_open with the full prompt.
    Used for fresh conversations and lcp=0 divergences — AR128 prefill is
    much faster than AR1 append for big ingests."""
    if conv_state.opened:
        await sidecar.request("stream_close", stream_id=conv_state.stream_id)
        conv_state.reset()
    t = time.perf_counter()
    rsp = await sidecar.request(
        "stream_open", stream_id=conv_state.stream_id,
        prompt_ids=prompt_ids,
        force_ar128=(len(prompt_ids) >= 128),
    )
    if not rsp.get("ok"):
        raise HTTPException(500, detail=f"stream_open: {rsp.get('error')}")
    elapsed = time.perf_counter() - t
    conv_state.history = list(prompt_ids)
    conv_state.opened = True
    return ("open", elapsed)


async def _sync_stream_to_prompt(prompt_ids: list[int]) -> dict:
    """Bring the server-side persistent stream in sync with `prompt_ids`.
    Returns dict with `stream_op`, `lcp`, `n_new`, and timings.
    Caller must hold `sidecar.lock`."""
    info = {"stream_op": None, "lcp": None, "n_new": 0,
            "prefill_s": 0.0, "truncate_s": 0.0, "append_s": 0.0}

    if not conv_state.opened:
        op, elapsed = await _stream_reopen(prompt_ids)
        info["stream_op"] = op
        info["prefill_s"] = elapsed
        info["n_new"] = len(prompt_ids)
        return info

    lcp = _longest_common_prefix(conv_state.history, prompt_ids)
    info["lcp"] = lcp
    delta_len = len(prompt_ids) - lcp

    if lcp == len(conv_state.history) and delta_len == 0:
        info["stream_op"] = "nochange"
        return info

    # Crossover analysis (Qwen3-4B NPU CL=2048):
    #   AR1 append cost   ≈ delta_len / 24 t/s
    #   AR128 reopen cost ≈ 22 s mode swap + delta_len / 1500 t/s
    # → AR128 wins only when delta > ~540 tokens. For typical chat
    # deltas (tens-to-hundreds of tokens), AR1-append always wins.
    # Threshold raised to 1024 to be safe; below this, prefer AR1.
    REOPEN_DELTA_THRESHOLD = 1024
    history_useful = lcp / max(len(conv_state.history), 1)
    if history_useful < 0.25 and delta_len >= REOPEN_DELTA_THRESHOLD:
        op, elapsed = await _stream_reopen(prompt_ids)
        info["stream_op"] = "reopen"
        info["prefill_s"] = elapsed
        info["n_new"] = len(prompt_ids)
        return info

    if lcp < len(conv_state.history):
        # Edge: client re-sent a strict prefix of the cached state (lcp ==
        # len(prompt_ids) but < len(history)). After a plain truncate
        # the stream has no `next_token` (unknown post-step prediction),
        # so the next stream_decode would fail. Truncate to lcp-1 then
        # re-feed the last common token: net effect is position back at
        # lcp with fresh logits available for decoding.
        empty_delta = (lcp == len(prompt_ids))
        truncate_to = (lcp - 1) if (empty_delta and lcp >= 1) else lcp
        t = time.perf_counter()
        rsp = await sidecar.request(
            "stream_truncate", stream_id=conv_state.stream_id,
            new_position=truncate_to,
        )
        if not rsp.get("ok"):
            raise HTTPException(500, detail=f"stream_truncate: {rsp.get('error')}")
        info["truncate_s"] = time.perf_counter() - t
        conv_state.history = conv_state.history[:truncate_to]
        if empty_delta and lcp >= 1:
            t = time.perf_counter()
            rsp = await sidecar.request(
                "stream_append", stream_id=conv_state.stream_id,
                append_ids=[prompt_ids[lcp - 1]],
            )
            if not rsp.get("ok"):
                raise HTTPException(500, detail=f"stream_append: {rsp.get('error')}")
            info["append_s"] = time.perf_counter() - t
            conv_state.history.append(prompt_ids[lcp - 1])
            info["n_new"] = 1
            info["stream_op"] = "truncate+refeed"
            return info

    delta = prompt_ids[lcp:]
    if delta:
        t = time.perf_counter()
        rsp = await sidecar.request(
            "stream_append", stream_id=conv_state.stream_id,
            append_ids=delta,
        )
        if not rsp.get("ok"):
            raise HTTPException(500, detail=f"stream_append: {rsp.get('error')}")
        info["append_s"] = time.perf_counter() - t
        conv_state.history.extend(delta)
        info["n_new"] = len(delta)
        info["stream_op"] = "truncate+append" if info["truncate_s"] > 0 else "append"
    else:
        info["stream_op"] = "truncate"

    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sidecar, tokenizer, conv_state
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"bundle tokenizer not found: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    sidecar = SidecarClient(ctx_tier=int(os.environ.get("NPU_CTX_TIER", "2048")))
    conv_state = ConversationState(stream_id="http")
    print(f"[http_server] starting NPU sidecar (ctx_tier={sidecar.ctx_tier})...",
          flush=True)
    t0 = time.perf_counter()
    await sidecar.start()
    print(f"[http_server] NPU sidecar ready in {time.perf_counter() - t0:.1f}s "
          f"(per-part {sidecar.startup_info.get('start_per_part_s')})",
          flush=True)
    yield
    print("[http_server] shutting down sidecar...", flush=True)
    await sidecar.shutdown()


app = FastAPI(title="specula-npu", lifespan=lifespan)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "specula",
            "created": int(time.time()),
        }],
    }


@app.get("/health")
async def health():
    if sidecar is None or sidecar.proc is None or sidecar.proc.poll() is not None:
        raise HTTPException(503, detail="sidecar not running")
    return {
        "status": "ok",
        "ctx_tier": sidecar.ctx_tier,
        "startup": sidecar.startup_info,
        "stream_opened": conv_state.opened if conv_state else False,
        "history_len": len(conv_state.history) if conv_state else 0,
    }


@app.post("/debug/reset_stream")
async def reset_stream():
    """Drop the active conversation stream. Next chat-completion request
    will open a fresh one. Useful for testing — most clients will just
    rely on LCP-divergence to do this implicitly."""
    if conv_state is None:
        raise HTTPException(503, detail="server not initialized")
    if sidecar.lock.locked():
        raise HTTPException(503, detail="busy")
    async with sidecar.lock:
        if conv_state.opened:
            await sidecar.request(
                "stream_close", stream_id=conv_state.stream_id)
        conv_state.reset()
    return {"status": "reset"}


def _resolve_stop_token_seqs(stop: list[str] | str | None) -> list[list[int]]:
    """Tokenize each stop string. BPE merges depend on context — `"five"`
    standalone is one token but ` five` is a different token (with `Ġ`
    space marker). To catch both forms we tokenize each stop with and
    without a leading space; the sidecar's suffix-match against generated
    IDs then catches whichever variant the model actually emits.

    Imperfect: e.g. `"\nfive"` won't match either variant. Phase B should
    move to text-side suffix matching for robustness.
    """
    if stop is None:
        return []
    if isinstance(stop, str):
        stop_strs = [stop]
    else:
        stop_strs = list(stop)
    seqs: list[list[int]] = []
    for s in stop_strs:
        if not s:
            continue
        for variant in (s, " " + s):
            ids = tokenizer.encode(variant, add_special_tokens=False).ids
            if ids and ids not in seqs:
                seqs.append(ids)
    return seqs


def _strip_trailing_eos(token_ids: list[int]) -> list[int]:
    """Drop trailing EOS markers; OpenAI clients don't expect to see them
    in the response text."""
    while token_ids and token_ids[-1] in (IM_END_ID, ENDOFTEXT_ID):
        token_ids = token_ids[:-1]
    return token_ids


_OPENAI_FINISH = {"eos": "stop", "stop": "stop", "max_new_tokens": "length"}


async def _do_chat_streaming(prompt_ids, max_new, stop_seqs, model: str):
    """Async generator yielding SSE-formatted lines for /v1/chat/completions.

    Uses the stateful stream API: brings the persistent stream in sync
    with `prompt_ids` (truncate+append delta, or full re-open if the
    history diverges) before invoking stream_decode_stream. The KV
    state on the NPU is preserved across requests so multi-turn chats
    only ingest new tokens between turns.
    """
    created = int(time.time())
    completion_id = f"chatcmpl-npu-{created}"

    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        }
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    yield chunk({"role": "assistant", "content": ""})

    async with sidecar.lock:
        # Bring stream in sync with the new prompt
        sync_info = await _sync_stream_to_prompt(prompt_ids)

        buffer_ids: list[int] = []
        emitted_text = ""
        async for evt in sidecar.stream_request(
            "stream_decode_stream",
            stream_id=conv_state.stream_id,
            max_new=max_new,
            eos_ids=DEFAULT_EOS_IDS,
            stop_token_seqs=stop_seqs,
        ):
            if evt.get("event") == "token":
                tok = evt["token_id"]
                buffer_ids.append(tok)
                # Decode-the-whole-buffer is O(N) per step but N is
                # small (≤ max_new_tokens). Robust to multi-byte UTF-8
                # chars split across tokens.
                full_text = tokenizer.decode(buffer_ids)
                if len(full_text) > len(emitted_text):
                    delta_text = full_text[len(emitted_text):]
                    yield chunk({"content": delta_text})
                    emitted_text = full_text
            elif evt.get("event") == "chat_done":
                if not evt.get("ok"):
                    yield chunk({}, finish="stop")
                    yield "data: [DONE]\n\n"
                    return
                # Authoritative generated ids INCLUDE any final EOS;
                # mirror them into history so next-turn LCP matches the
                # NPU's KV state.
                conv_state.history.extend(evt.get("generated_ids", []))
                stop_reason = evt.get("stop_reason", "stop")
                finish = _OPENAI_FINISH.get(stop_reason, "stop")
                yield chunk({}, finish=finish)
                yield "data: [DONE]\n\n"
                return


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if sidecar is None or tokenizer is None:
        raise HTTPException(503, detail="server not initialized")
    if req.n and req.n != 1:
        raise HTTPException(400, detail="n != 1 not supported (greedy)")

    msgs = [m.model_dump() for m in req.messages]
    prompt_text = render_chatml(
        msgs, enable_thinking=bool(req.enable_thinking)
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    max_new = req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    stop_seqs = _resolve_stop_token_seqs(req.stop)

    if sidecar.lock.locked():
        raise HTTPException(503, detail="busy — single-tenant NPU server, retry after current request")

    if req.stream:
        return StreamingResponse(
            _do_chat_streaming(prompt_ids, max_new, stop_seqs, MODEL_NAME),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path
    async with sidecar.lock:
        t0 = time.perf_counter()
        sync_info = await _sync_stream_to_prompt(prompt_ids)
        rsp = await sidecar.request(
            "stream_decode",
            stream_id=conv_state.stream_id,
            max_new=max_new,
            eos_ids=DEFAULT_EOS_IDS,
            stop_token_seqs=stop_seqs,
        )
        wall_s = time.perf_counter() - t0

    if not rsp.get("ok"):
        raise HTTPException(500, detail=f"sidecar: {rsp.get('error')}")

    raw_ids: list[int] = rsp["generated_ids"]
    conv_state.history.extend(raw_ids)
    visible_ids = _strip_trailing_eos(raw_ids)
    text = tokenizer.decode(visible_ids)

    finish_reason = _OPENAI_FINISH.get(rsp.get("stop_reason"), "stop")
    created = int(time.time())
    response = {
        "id": f"chatcmpl-npu-{created}",
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(raw_ids),
            "total_tokens": len(prompt_ids) + len(raw_ids),
        },
        "specula_npu": {
            "wall_s": wall_s,
            "stream_op": sync_info["stream_op"],
            "lcp": sync_info["lcp"],
            "n_new_ingested": sync_info["n_new"],
            "prefill_s": sync_info["prefill_s"],
            "truncate_s": sync_info["truncate_s"],
            "append_s": sync_info["append_s"],
            "decode_s": rsp.get("compute_s"),
            "decode_tps": rsp.get("tps"),
            "stop_reason": rsp.get("stop_reason"),
            "stream_position": rsp.get("position"),
            "history_len": len(conv_state.history),
        },
    }
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")
