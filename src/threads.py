"""Conversation thread state: pending questions, topics, and short answers."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
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
    "exam": r"\b(exam|test|quiz|midterm|final)\b",
    "school": r"\b(rean|school|class|teach|sala|meeting|work|code|study)\b",
    "love": r"\b(slnh|love|miss|kit bong|kiss|baby)\b",
    "music": r"\b(music|song|youtube|youtu\.be|link|video)\b|https?://",
}


def infer_topic(text: str) -> str:
    for topic, pattern in TOPICS.items():
        if re.search(pattern, text, re.I):
            return topic
    return ""


# A claim is something she said about herself that the next turns must not
# silently contradict (going to sleep, being awake, being at school).
CLAIMS = (
    ("sleep", re.compile(
        r"\b(?:pg\s+)?dek\s+(?:hz|nv|tv)\b|\bgood\s*night\b|"
        r"\babout\s+to\s+sleep\b|\bin\s+bed\b",
        re.I,
    )),
    ("awake", re.compile(r"\b(ngerb\s+hx|ngerb\s+yu\s+hx|i'?m\s+up|awake)\b", re.I)),
    ("studying", re.compile(
        r"\b(rean\s+nv|at\s+school|in\s+class|nv\s+sala|nv\s+pteas)\b", re.I
    )),
    ("not_studying", re.compile(r"\b(tvr\s+lomhat|pg\s+tvr\s+ka)\b", re.I)),
    ("home", re.compile(r"\b(nv\s+home|at\s+home)\b", re.I)),
)


def claim_from_reply(reply: str) -> str:
    """The last self-claim she made in a statement, ignoring her questions."""
    lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.endswith("?"):
            continue
        for name, pattern in CLAIMS:
            if pattern.search(line):
                return name
    return ""


@dataclass
class ThreadState:
    topic: str = ""
    pending_question: str = ""
    pending_topic: str = ""
    pending_ts: float = 0.0
    last_user_question: str = ""
    last_user_question_ts: float = 0.0
    last_bot_claim: str = ""
    last_bot_claim_ts: float = 0.0
    recent_questions: list[str] = field(default_factory=list)
    recent_question_ts: list[float] = field(default_factory=list)


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
                state = ThreadState(
                    **{k: v for k, v in raw.items() if k in known}
                )
                if len(state.recent_question_ts) != len(state.recent_questions):
                    state.recent_question_ts = [0.0] * len(state.recent_questions)
                self.chats[str(chat_id)] = state
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps({"chats": {k: asdict(v) for k, v in self.chats.items()}}),
        )

    def _get(self, chat_id: int) -> ThreadState:
        return self.chats.setdefault(str(chat_id), ThreadState())

    def describe(self, chat_id: int, now: float | None = None) -> str:
        """One short line of thread context for the system prompt."""
        now = now if now is not None else time.time()
        with self.lock:
            state = self._get(chat_id)
            parts: list[str] = []
            claim = state.last_bot_claim
            if claim and now - state.last_bot_claim_ts <= 8 * 3600:
                said = {
                    "sleep": "you said you were going to sleep",
                    "awake": "you said you were awake",
                    "studying": "you said you were studying",
                    "not_studying": "you said you were not studying",
                    "home": "you said you were at home",
                }.get(claim, f"you said you were {claim}")
                parts.append(said)
            if state.topic:
                parts.append(f"the topic is {state.topic}")
            if state.pending_question and now - state.pending_ts <= 1800:
                parts.append(f'you asked "{state.pending_question}" and he has not answered')
            elif (
                state.last_user_question
                and now - state.last_user_question_ts <= 3600
            ):
                parts.append(f'he asked "{state.last_user_question}"')
            elif (
                state.recent_questions
                and len(state.recent_question_ts) == len(state.recent_questions)
                and now - state.recent_question_ts[-1] <= 3600
            ):
                parts.append(
                    f'you already asked "{state.recent_questions[-1]}" — do not repeat it'
                )
            return "; ".join(parts)

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

    def bot_claim(self, chat_id: int) -> tuple[str, float]:
        """What she last said about herself, and when she said it."""
        with self.lock:
            state = self._get(chat_id)
            return state.last_bot_claim, state.last_bot_claim_ts

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
            claim = claim_from_reply(reply)
            if claim:
                state.last_bot_claim = claim
                state.last_bot_claim_ts = now
            if reply.rstrip().endswith("?"):
                question = reply.strip()
                state.pending_question = question
                state.pending_topic = infer_topic(user_text) or state.topic
                state.pending_ts = now
                if question not in state.recent_questions:
                    state.recent_questions.append(question)
                    state.recent_question_ts.append(now)
                    del state.recent_questions[:-3]
                    del state.recent_question_ts[:-3]
            elif (YES.match(user_text) or NO.match(user_text)) and state.pending_question:
                state.pending_question = ""
                state.pending_topic = ""
                state.pending_ts = 0.0
            if state.last_user_question and not reply.rstrip().endswith("?"):
                state.last_user_question = ""
            self._save()
