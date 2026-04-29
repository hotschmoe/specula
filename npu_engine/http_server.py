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
from fastapi.responses import JSONResponse
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sidecar, tokenizer
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"bundle tokenizer not found: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    sidecar = SidecarClient(ctx_tier=int(os.environ.get("NPU_CTX_TIER", "2048")))
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
    return {"status": "ok", "ctx_tier": sidecar.ctx_tier,
            "startup": sidecar.startup_info}


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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if sidecar is None or tokenizer is None:
        raise HTTPException(503, detail="server not initialized")
    if req.stream:
        raise HTTPException(501, detail="streaming arrives in Phase A.5")
    if req.n and req.n != 1:
        raise HTTPException(400, detail="n != 1 not supported (greedy)")

    msgs = [m.model_dump() for m in req.messages]
    prompt_text = render_chatml(
        msgs, enable_thinking=bool(req.enable_thinking)
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    max_new = req.max_tokens or DEFAULT_MAX_NEW_TOKENS
    stop_seqs = _resolve_stop_token_seqs(req.stop)

    # Single-tenant: hold the lock for the entire request.
    if sidecar.lock.locked():
        raise HTTPException(503, detail="busy — single-tenant NPU server, retry after current request")

    async with sidecar.lock:
        t0 = time.perf_counter()
        rsp = await sidecar.request(
            "chat",
            prompt_ids=prompt_ids,
            max_new_tokens=max_new,
            eos_ids=DEFAULT_EOS_IDS,
            stop_token_seqs=stop_seqs,
            force_ar128=(len(prompt_ids) >= 128),
        )
        wall_s = time.perf_counter() - t0

    if not rsp.get("ok"):
        raise HTTPException(500, detail=f"sidecar: {rsp.get('error')}")

    raw_ids: list[int] = rsp["generated_ids"]
    # Strip trailing EOS so completion text is clean
    visible_ids = _strip_trailing_eos(raw_ids)
    text = tokenizer.decode(visible_ids)

    finish_reason = {
        "eos": "stop",
        "stop": "stop",
        "max_new_tokens": "length",
    }.get(rsp.get("stop_reason"), "stop")

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
            "prompt_tokens": rsp.get("n_prompt", len(prompt_ids)),
            "completion_tokens": rsp.get("n_generated", len(raw_ids)),
            "total_tokens": rsp.get("n_prompt", len(prompt_ids)) + rsp.get("n_generated", len(raw_ids)),
        },
        # Non-standard but useful for the SQ6 writeup
        "specula_npu": {
            "wall_s": wall_s,
            "swap_s": rsp.get("swap_s"),
            "pp_compute_s": rsp.get("pp_compute_s"),
            "tg_compute_s": rsp.get("tg_compute_s"),
            "pp_tps": rsp.get("pp_tps"),
            "tg_tps": rsp.get("tg_tps"),
            "stop_reason": rsp.get("stop_reason"),
        },
    }
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")
