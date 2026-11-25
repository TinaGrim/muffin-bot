"""Durable things Bong told the bot, used as long-term chat context.

Recent turns live in `HistoryStore`; this store keeps the few facts that stay
true longer than the history window: exams, plans, family, work, preferences.
It never invents anything. It only keeps the sentence Bong actually wrote,
labelled with a topic, and injects it back when the topic comes up again.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from storage import atomic_write_text

URL = re.compile(r"https?://|www\.", re.I)
QUESTION = re.compile(
    r"^\s*(what|why|how|where|when|who|which|do|did|are|is|can|could|will|"
    r"pong|tvr|ey|ng|mic)\b",
    re.I,
)
EXPLICIT = re.compile(r"\b(remember|jmuy|note that|keep in mind)\b", re.I)

TOPICS: dict[str, re.Pattern[str]] = {
    "exam": re.compile(r"\b(exam|test|quiz|midterm|final|score|grade)\b", re.I),
    "school": re.compile(r"\b(school|class|rean|sala|teacher|university|major)\b", re.I),
    "work": re.compile(
        r"\b(work|job|interview|meeting|deadline|project|presentation|boss|office)\b",
        re.I,
    ),
    "plan": re.compile(
        r"\b(tomorrow|next week|weekend|tonight|going to|will go|plan|trip|"
        r"travel|visit|birthday|holiday|vacation)\b",
        re.I,
    ),
    "family": re.compile(
        r"\b(family|mom|mother|dad|father|sister|brother|parents|grandma|"
        r"grandpa|aunt|uncle|cousin|baby)\b",
        re.I,
    ),
    "health": re.compile(r"\b(sick|ill|fever|hospital|doctor|medicine|hurt|pain)\b", re.I),
    "preference": re.compile(
        r"\b(favorite|favourite|i like|i love|i hate|i prefer|allergic)\b", re.I
    ),
    "profile": re.compile(
        r"\b(my name is|call me|i am \d+|i'm \d+|i live|i'm from|i am from|"
        r"my birthday|chhmoh)\b",
        re.I,
    ),
    "money": re.compile(r"\b(money|luy|salary|pay|buy|order|price|cost|borrow)\b", re.I),
}


@dataclass
class Fact:
    topic: str
    text: str
    ts: float
    explicit: bool = False


class FactStore:
    def __init__(self, path: Path, limit_per_chat: int = 24):
        self.path = path
        self.limit = limit_per_chat
        self.lock = threading.Lock()
        self.chats: dict[str, list[Fact]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for chat_id, rows in data.get("chats", {}).items():
                self.chats[str(chat_id)] = [
                    Fact(
                        topic=row.get("topic", ""),
                        text=row.get("text", ""),
                        ts=float(row.get("ts", 0)),
                        explicit=bool(row.get("explicit", False)),
                    )
                    for row in rows
                    if row.get("text")
                ]
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        payload = {
            "chats": {
                chat_id: [asdict(fact) for fact in facts]
                for chat_id, facts in self.chats.items()
            }
        }
        atomic_write_text(self.path, json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def classify(text: str) -> tuple[str, bool]:
        explicit = bool(EXPLICIT.search(text))
        for topic, pattern in TOPICS.items():
            if pattern.search(text):
                return topic, explicit
        return ("note", True) if explicit else ("", False)

    def observe(self, chat_id: int, text: str, now: float | None = None) -> Fact | None:
        """Store a sentence only when it states something durable."""
        now = now if now is not None else time.time()
        text = (text or "").strip()
        if not (8 <= len(text) <= 180) or URL.search(text):
            return None
        # A question is not a fact, even when it says "remember".
        if QUESTION.match(text) or text.rstrip().endswith("?"):
            return None
        topic, explicit = self.classify(text)
        if not topic:
            return None
        # Strip a leading "remember that" so the stored line reads naturally.
        stored = EXPLICIT.sub("", text).strip(" ,.:-") or text
        fact = Fact(topic=topic, text=stored[:180], ts=now, explicit=explicit)
        with self.lock:
            facts = self.chats.setdefault(str(chat_id), [])
            key = stored.casefold()
            facts[:] = [f for f in facts if f.text.casefold() != key]
            facts.append(fact)
            if len(facts) > self.limit:
                del facts[: len(facts) - self.limit]
            self._save()
        return fact

    def recent(self, chat_id: int, now: float | None = None, days: float = 30) -> list[Fact]:
        now = now if now is not None else time.time()
        with self.lock:
            facts = list(self.chats.get(str(chat_id), []))
        cutoff = now - days * 86400
        return [f for f in facts if f.ts >= cutoff or f.explicit]

    def relevant(
        self,
        chat_id: int,
        text: str = "",
        now: float | None = None,
        limit: int = 5,
    ) -> list[Fact]:
        """Prefer facts about the current topic, then the newest ones."""
        now = now if now is not None else time.time()
        current, _ = self.classify(text)
        scored: list[tuple[float, Fact]] = []
        for fact in self.recent(chat_id, now):
            score = 0.0
            if current and fact.topic == current:
                score += 3.0
            if fact.explicit:
                score += 2.0
            age_days = max(0.0, (now - fact.ts) / 86400)
            score += max(0.0, 1.0 - age_days / 30)
            scored.append((score, fact))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    def describe(self, chat_id: int, text: str = "", now: float | None = None) -> list[str]:
        lines = []
        current = (text or "").strip().casefold()
        for fact in self.relevant(chat_id, text, now):
            if current and fact.text.strip().casefold() == current:
                continue  # it is already the latest user message
            label = fact.topic if fact.topic != "note" else "told you"
            lines.append(f"- {label}: {fact.text}")
        return lines

    def reset(self, chat_id: int) -> None:
        with self.lock:
            self.chats.pop(str(chat_id), None)
            self._save()
