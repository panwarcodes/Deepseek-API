"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.

Cline-style clients (which resend the whole history and never send a
conversation_id) are transparently mapped back onto the same DeepSeek chat
session using server/session_cache.py.
"""

from __future__ import annotations

import threading
import time
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
)
from .openai_format import completion_response, messages_to_prompt, stream_chunks
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, ChatMessage
from .session_cache import SESSION_CACHE, TaskSession

load_dotenv()

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


def _new_messages_since_last_assistant(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Return only the messages after the last assistant turn.

    On a resumed conversation DeepSeek already holds every earlier message
    (including the replies it generated), so we only forward what's new —
    typically the latest user message (a tool result or a fresh prompt).
    If there is no assistant message yet (first turn), send everything.
    """
    last_assistant = -1
    for i, m in enumerate(messages):
        if m.role == "assistant":
            last_assistant = i
    if last_assistant == -1:
        return list(messages)
    return list(messages[last_assistant + 1:])


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400,
                      err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # --- Figure out which DeepSeek conversation to continue -----------------
    #
    # Priority:
    #   1. Explicit `conversation_id` from the client (advanced users / our
    #      own examples).
    #   2. A cached mapping from this Cline task → the DeepSeek chat we already
    #      opened for it. This is the fix that stops Cline from spawning a new
    #      dashboard chat on every prompt: same first user message ⇒ same task
    #      ⇒ reuse the existing conversation.
    conversation_id = req.conversation_id
    task_key: str | None = None
    cached: TaskSession | None = None

    if conversation_id is None:
        task_key = SESSION_CACHE.task_key(req.messages)
        if task_key:
            cached = SESSION_CACHE.get(task_key)
            if cached is not None:
                conversation_id = cached.conversation_id

    # --- Decide what text to actually send to DeepSeek ----------------------
    #
    # Fresh conversation → send the whole flattened history.
    # Resumed conversation → send only the messages DeepSeek hasn't seen yet,
    # otherwise the thread accumulates duplicated context every turn.
    if conversation_id and cached is not None:
        messages_to_send = _new_messages_since_last_assistant(req.messages)
        if not messages_to_send:
            return _error(
                "No new messages to send on this conversation.",
                status=400, err_type="invalid_request_error",
            )
    else:
        messages_to_send = req.messages

    prompt = messages_to_prompt(messages_to_send)

    # A thread's model is fixed when it's created, so on resume we ignore
    # `model` (the OpenAI SDK always sends one) and let the existing model stand.
    model_type = None if conversation_id else resolve_model_type(req.model)

    try:
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    # --- Streaming path -----------------------------------------------------
    if req.stream:
        def gen():
            stream = client.stream(
                prompt,
                conversation_id=conversation_id,
                model=model_type,
                thinking=req.thinking,
                search=req.search,
            )
            yield from stream_chunks(req.model, stream)
            # After the stream is exhausted we know the (possibly brand-new)
            # conversation_id; remember it so the next turn of this task reuses it.
            if task_key:
                SESSION_CACHE.put(task_key, stream.conversation_id)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # --- Non-streaming path -------------------------------------------------
    try:
        reply = await run_in_threadpool(
            client.chat, prompt, conversation_id,
            model_type, req.thinking, req.search,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    if task_key:
        SESSION_CACHE.put(task_key, reply.conversation_id)

    return completion_response(req.model, reply.text, prompt, reply.conversation_id)
