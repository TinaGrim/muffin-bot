"""Conversation thread state: pending questions, topics, and short answers."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from storage import atomic_write_text

YES = re.compile(r"^\s*(yes|yeah|yep|yup|ok(ay)?|okiee?|👍|🙏)\b", re.I)
NO = re.compile(r"^\s*(no|nope|nah|ort|ot)\b", re.I)
QUESTION = re.compile(
    r"\?|^\s*(what|why|how|where|when|who|which|can|could|do|did|are|is|will)\b"
    r"|\b(pong|tvr ey|ey|ng|mic|tv na)\b",
    re.I,
)
TOPICS = {
    "food": r"\b(nham|bay|eat|food|hungry|kteav|trey|moan|prek)\b",
    "sleep": r"\b(dek|sleep|keng|lok|night|gn)\b",
    "takmao": r"\btakmao\b",
    "school": r"\b(rean|school|class|teach|sala|meeting|work|code|study)\b",
    "love": r"\b(slnh|love|miss|kit bong|kiss|baby)\b",
    "music": r"\b(music|song|youtube|youtu\.be|link|video)\b|https?://",
}


def infer_topic(text: str) -> str:
    for topic, pattern in TOPICS.items():
        if re.search(pattern, text, re.I):
            return topic
    return ""


@dataclass
class ThreadState:
    topic: str = ""
    pending_question: str = ""
    pending_topic: str = ""
    pending_ts: float = 0.0
    last_user_question: str = ""
    last_user_question_ts: float = 0.0


class ThreadStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, ThreadState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            known = set(ThreadState.__dataclass_fields__)
            for chat_id, raw in data.get("chats", {}).items():
                self.chats[str(chat_id)] = ThreadState(
                    **{k: v for k, v in raw.items() if k in known}
                )
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps({"chats": {k: asdict(v) for k, v in self.chats.items()}}),
        )

    def _get(self, chat_id: int) -> ThreadState:
        return self.chats.setdefault(str(chat_id), ThreadState())

    def observe_user(self, chat_id: int, text: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self.lock:
            state = self._get(chat_id)
            topic = infer_topic(text)
            if topic:
                state.topic = topic
            if QUESTION.search(text) and not (YES.match(text) or NO.match(text)):
                state.last_user_question = text
                state.last_user_question_ts = now
            self._save()

    def pending_for_answer(self, chat_id: int, text: str, now: float | None = None) -> str | None:
        now = now if now is not None else time.time()
        if not (YES.match(text) or NO.match(text)):
            return None
        with self.lock:
            state = self._get(chat_id)
            if not state.pending_question or now - state.pending_ts > 1800:
                return None
            current = infer_topic(text) or state.topic
            if state.pending_topic and current and state.pending_topic != current:
                return None
            return state.pending_question

    def observe_reply(
        self, chat_id: int, user_text: str, reply: str, now: float | None = None
    ) -> None:
        now = now if now is not None else time.time()
        with self.lock:
            state = self._get(chat_id)
            if reply.rstrip().endswith("?"):
                state.pending_question = reply.strip()
                state.pending_topic = infer_topic(user_text) or state.topic
                state.pending_ts = now
            elif (YES.match(user_text) or NO.match(user_text)) and state.pending_question:
                state.pending_question = ""
                state.pending_topic = ""
                state.pending_ts = 0.0
            if state.last_user_question and not reply.rstrip().endswith("?"):
                state.last_user_question = ""
            self._save()
