"""
Server-side task → DeepSeek-conversation cache.

OpenAI clients like Cline are stateless: they resend the full message history
on every turn and have no idea about our `conversation_id` extension. Without
help, every request looks brand new and the server opens a fresh DeepSeek chat
in the dashboard — once per prompt. Annoying.

Fix: hash the first user message of the conversation. Within one Cline task
that message never changes, so the hash is stable and we can look up the
DeepSeek conversation_id we used last time. Clicking the dustbin / starting a
new task in Cline changes the first message, so the hash changes and a fresh
DeepSeek chat is created — which is what we want.

A time-to-live guards the edge case where two separate tasks happen to share
the same first message (e.g. you typed "hi" twice in a row on different days):
if we haven't seen this hash in TTL seconds, treat it as a brand new task.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

from .schemas import ChatMessage

# After this many seconds of silence for a given task-hash, a new request with
# the same hash is treated as a fresh task rather than a continuation.
# 15 minutes is comfortably longer than any normal Cline turn gap and short
# enough that "hi" in a new task tomorrow won't leak into today's session.
TASK_TTL_SECONDS = 15 * 60


@dataclass
class TaskSession:
    conversation_id: str
    last_seen: float


class TaskSessionCache:
    """Thread-safe LRU + TTL mapping of task-hash → TaskSession."""

    def __init__(self, max_size: int = 500, ttl: float = TASK_TTL_SECONDS):
        self._entries: "OrderedDict[str, TaskSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl

    @staticmethod
    def task_key(messages: List[ChatMessage]) -> Optional[str]:
        """Stable key for a task: sha256 of the FIRST user message.

        Returns None if there is no user message (nothing to key on).
        """
        for m in messages:
            if m.role != "user":
                continue
            content = m.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if not content:
                continue
            return hashlib.sha256(str(content).encode("utf-8")).hexdigest()
        return None

    def get(self, key: str) -> Optional[TaskSession]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.time() - entry.last_seen > self._ttl:
                # Stale: same hash, but no activity in TTL → new task.
                self._entries.pop(key, None)
                return None
            entry.last_seen = time.time()
            self._entries.move_to_end(key)
            return entry

    def put(self, key: str, conversation_id: str) -> None:
        with self._lock:
            self._entries[key] = TaskSession(
                conversation_id=conversation_id, last_seen=time.time()
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)


# Module-level singleton — one cache per server process.
SESSION_CACHE = TaskSessionCache()
