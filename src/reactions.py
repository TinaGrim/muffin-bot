"""Rituals and emotional reactions: mornings, jealousy, apologies, being ignored."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from storage import atomic_write_text

GREETING = re.compile(r"^(good\s*morning|morning|bart)\b", re.I)
JEALOUS = re.compile(
    r"\b(srey|girl|ex|tina|vivian|neary|khat|jealous|another girl|other girl)\b", re.I
)
APOLOGY = re.compile(r"\b(sorry|khos|my bad)\b", re.I)
GOODNIGHT = re.compile(r"\b(good\s*night|gn|dek hz)\b", re.I)


@dataclass
class ReactionState:
    last_user_date: str = ""
    morning_greeted: str = ""
    morning_prompted: str = ""
    morning_followup: str = ""
    goodnight: str = ""
    last_user_ts: float = 0.0
    last_proactive_ts: float = 0.0
    ignored_followup_ts: float = 0.0


@dataclass
class Reaction:
    first_today: bool = False
    missed_morning: bool = False
    greeted: bool = False
    jealous: bool = False
    apology: bool = False
    ignored: bool = False


class ReactionStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, ReactionState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            known = set(ReactionState.__dataclass_fields__)
            for chat_id, raw in data.get("chats", {}).items():
                self.chats[str(chat_id)] = ReactionState(
                    **{k: v for k, v in raw.items() if k in known}
                )
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps({"chats": {k: asdict(v) for k, v in self.chats.items()}}),
        )

    def _get(self, chat_id: int) -> ReactionState:
        return self.chats.setdefault(str(chat_id), ReactionState())

    def observe(self, chat_id: int, text: str, now: float | None = None) -> Reaction:
        now = now if now is not None else time.time()
        dt = datetime.fromtimestamp(now)
        today = dt.date().isoformat()
        with self.lock:
            state = self._get(chat_id)
            first_today = state.last_user_date != today
            morning = 5 <= dt.hour < 11
            greeting = bool(GREETING.match(text))
            missed = bool(first_today and morning and not greeting)
            # She texted first and he stayed away for hours: that is a grudge,
            # not a missed morning. Ignore the very first exchange.
            ignored = bool(
                state.last_user_ts > 0
                and state.last_proactive_ts > state.last_user_ts
                and now - state.last_proactive_ts >= 2 * 3600
            )
            reaction = Reaction(
                first_today=first_today,
                missed_morning=missed,
                greeted=greeting and morning,
                jealous=bool(JEALOUS.search(text)),
                apology=bool(APOLOGY.search(text)),
                ignored=ignored,
            )
            state.last_user_date = today
            state.last_user_ts = now
            state.ignored_followup_ts = 0.0
            if reaction.greeted:
                state.morning_greeted = today
            if GOODNIGHT.search(text):
                state.goodnight = today
            self._save()
            return reaction

    @staticmethod
    def cold_for(reaction: Reaction) -> str | None:
        if reaction.missed_morning:
            return random.choice(["Ng", "Huh", "Ort", "Morning?", "Bong?"])
        if reaction.jealous:
            return random.choice(["Ke ney?", "Srey ng?", "Ort", "Huh", "Mic"])
        if reaction.ignored:
            return random.choice(["Hoy", "Bong?", "Ort reply teh?", "Hmm"])
        return None

    def note_proactive(self, chat_id: int, now: float | None = None) -> None:
        with self.lock:
            state = self._get(chat_id)
            state.last_proactive_ts = now if now is not None else time.time()
            self._save()

    def proactive(self, chat_id: int, now: float | None = None) -> str | None:
        now = now if now is not None else time.time()
        dt = datetime.fromtimestamp(now)
        today = dt.date().isoformat()
        hour = dt.hour
        with self.lock:
            state = self._get(chat_id)
            message = None

            morning_window = 7 <= hour < 9
            if (
                morning_window
                and state.morning_prompted != today
                and state.morning_greeted != today
                and state.last_user_date != today
            ):
                state.morning_prompted = today
                message = random.choice(["Morning?", "Bart?", "Morning bong?"])

            followup_window = 9 <= hour < 11
            if (
                not message
                and followup_window
                and state.morning_prompted == today
                and state.morning_followup != today
                and state.morning_greeted != today
                and state.last_user_date != today
            ):
                state.morning_followup = today
                message = random.choice(["Ort reply teh?", "Bong?", "Huh"])

            if (
                not message
                and 22 <= hour < 23
                and state.last_user_date == today
                and state.goodnight != today
            ):
                state.goodnight = today
                message = random.choice(["Good night bong", "Dek hz", "Good night"])

            if (
                not message
                and state.last_proactive_ts > state.last_user_ts
                and not state.ignored_followup_ts
                and 2 * 3600 <= now - state.last_proactive_ts <= 6 * 3600
            ):
                state.ignored_followup_ts = now
                message = random.choice(["Bong?", "Tv na?", "Ort reply teh?"])

            if message:
                self._save()
            return message
