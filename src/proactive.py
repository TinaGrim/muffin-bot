"""Proactive messages and small human follow-ups.

The bot should not wait forever for Bong, and sometimes a person sends a second
tiny message after the first. This module owns both behaviors and persists the
per-chat clocks so a restart does not spam anyone.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from storage import atomic_write_text
from typing import Any

log = logging.getLogger("muffin")

MORNING = ["Morning bong", "Bart bong", "Tngaii nih tv na?"]
BREAKFAST = ["Nham bay prek nv", "Nham bay nv?", "Nham hz?"]
LUNCH = ["Nham bay nv?", "Nham ey bong", "Nham hz?"]
DINNER = ["Nham bay nv?", "Nham hz?", "Nham ey?"]
DAY = ["Bong tv na?", "Hoy", "Bong?", "Kit bong"]
NIGHT = ["Bong?", "Kit bong", "Ort reply teh"]
GENERIC = re.compile(
    r"^(bong\?|hoy|hmm|kit bong|ort reply teh\??|tv na\??)$", re.I
)


@dataclass
class ChatState:
    last_user: float = 0.0
    last_bot: float = 0.0
    last_proactive: float = 0.0
    next_delay: float = 0.0
    enabled: bool = True
    unanswered: int = 0
    last_text: str = ""
    last_kind: str = ""
    generic_day: str = ""
    generic_count: int = 0


class ChatRegistry:
    def __init__(self, path: Path, min_hours: float, max_hours: float):
        self.path = path
        self.min_hours = min_hours
        self.max_hours = max_hours
        self.lock = threading.Lock()
        self.chats: dict[str, ChatState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for chat_id, raw in data.get("chats", {}).items():
                known = {f for f in ChatState.__dataclass_fields__}
                self.chats[str(chat_id)] = ChatState(
                    **{k: v for k, v in raw.items() if k in known}
                )
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chats": {k: asdict(v) for k, v in self.chats.items()}}
        atomic_write_text(self.path, json.dumps(payload))

    def _new_delay(self) -> float:
        return random.uniform(self.min_hours, self.max_hours) * 3600

    def seed(self, chat_id: int) -> None:
        with self.lock:
            key = str(chat_id)
            if key not in self.chats:
                now = time.time()
                self.chats[key] = ChatState(
                    last_user=now,
                    last_bot=now,
                    next_delay=self._new_delay(),
                )
                self._save()

    def note_user(self, chat_id: int) -> None:
        with self.lock:
            key = str(chat_id)
            state = self.chats.setdefault(key, ChatState())
            state.last_user = time.time()
            state.next_delay = self._new_delay()
            state.unanswered = 0
            self._save()

    def note_bot(self, chat_id: int) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            state.last_bot = time.time()
            self._save()

    def mark_proactive(
        self, chat_id: int, text: str = "", kind: str = ""
    ) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            bot_was_last = state.last_proactive >= state.last_user
            state.unanswered = state.unanswered + 1 if bot_was_last else 1
            state.last_proactive = time.time()
            state.next_delay = self._new_delay()
            clean = text.strip()
            if clean:
                state.last_text = clean
            state.last_kind = kind or (
                "question" if clean.endswith("?") else "other"
            )
            today = datetime.now().date().isoformat()
            if state.generic_day != today:
                state.generic_day = today
                state.generic_count = 0
            if clean and GENERIC.match(clean):
                state.generic_count += 1
            self._save()

    def generic_count(self, chat_id: int) -> int:
        with self.lock:
            state = self.chats.get(str(chat_id), ChatState())
            if state.generic_day != datetime.now().date().isoformat():
                return 0
            return state.generic_count

    def allow_generic(self, chat_id: int) -> bool:
        """One empty 'Bong?' per day is enough."""
        return self.generic_count(chat_id) < 1

    def last_kind(self, chat_id: int) -> str:
        with self.lock:
            return self.chats.get(str(chat_id), ChatState()).last_kind

    def unanswered(self, chat_id: int) -> int:
        with self.lock:
            return self.chats.get(str(chat_id), ChatState()).unanswered

    def last_text(self, chat_id: int) -> str:
        with self.lock:
            return self.chats.get(str(chat_id), ChatState()).last_text

    def set_enabled(self, chat_id: int, enabled: bool) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            state.enabled = enabled
            self._save()

    def enabled(self, chat_id: int) -> bool:
        with self.lock:
            return self.chats.get(str(chat_id), ChatState()).enabled

    def idle_for(self, chat_id: int, now: float | None = None) -> float:
        """Seconds since the last message in either direction."""
        now = now if now is not None else time.time()
        with self.lock:
            state = self.chats.get(str(chat_id))
            if not state:
                return 0.0
            return max(
                0.0,
                now - max(state.last_user, state.last_bot, state.last_proactive),
            )

    def due(self, now: float) -> list[int]:
        due = []
        with self.lock:
            for key, state in self.chats.items():
                if not state.enabled or state.next_delay <= 0:
                    continue
                if state.unanswered >= 2:
                    # Two texts with no answer: back off instead of spamming.
                    continue
                idle_since = max(state.last_user, state.last_bot, state.last_proactive)
                if now - idle_since >= state.next_delay:
                    due.append(int(key))
        return due

    def enabled_chats(self) -> list[int]:
        with self.lock:
            return [int(k) for k, v in self.chats.items() if v.enabled]


class Proactive(threading.Thread):
    def __init__(
        self,
        tg: Any,
        muffin: Any,
        registry: ChatRegistry,
        quiet_start: int = 23,
        quiet_end: int = 8,
        poll_seconds: int = 60,
    ):
        super().__init__(daemon=True)
        self.tg = tg
        self.muffin = muffin
        self.registry = registry
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end
        self.poll_seconds = poll_seconds

    def _quiet(self, now: datetime | None = None) -> bool:
        hour = (now or datetime.now()).hour
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= hour < self.quiet_end
        return hour >= self.quiet_start or hour < self.quiet_end

    def run(self) -> None:
        while True:
            time.sleep(self.poll_seconds)
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                log.exception("proactive tick failed: %s", exc)

    def tick(self) -> None:
        if self._quiet():
            return
        now = time.time()
        for chat_id in self.registry.enabled_chats():
            if self.registry.unanswered(chat_id) >= 2:
                continue
            message = self.muffin.ritual(chat_id)
            if message:
                log.info("ritual chat=%s -> %r", chat_id, message)
                self.tg.typing(chat_id)
                self.tg.send(chat_id, message)
                kind = (
                    "greeting"
                    if message.lower().startswith(("morning", "bart"))
                    else "question"
                    if message.strip().endswith("?")
                    else "other"
                )
                self.registry.mark_proactive(chat_id, message, kind=kind)
                self.muffin.reactions.note_proactive(chat_id, kind=kind)
        due = set(self.registry.due(now))
        # A grudge needs a nudge sooner than the normal 1.5-3 h cadence.
        for chat_id in self.registry.enabled_chats():
            mood = self.muffin.mood.snapshot(chat_id)
            if (
                mood.upset
                and self.registry.unanswered(chat_id) < 2
                and self.registry.idle_for(chat_id, now) >= 3600
            ):
                due.add(chat_id)
        for chat_id in due:
            self.send(chat_id)

    def send(self, chat_id: int, force: bool = False) -> None:
        if not force and not self.registry.enabled(chat_id):
            return
        parts = self.muffin.proactive(chat_id)
        if not parts:
            parts = [self._fallback()]
        if parts[0].strip() == self.registry.last_text(chat_id):
            parts[0] = self._fallback_specific()
        if GENERIC.match(parts[0].strip()) and not self.registry.allow_generic(
            chat_id
        ):
            parts[0] = self._fallback_specific()
        log.info("proactive chat=%s -> %r", chat_id, " | ".join(parts))
        for i, part in enumerate(parts[:2]):
            if i:
                time.sleep(random.uniform(1.2, 2.5))
            self.tg.typing(chat_id)
            self.tg.send(chat_id, part)
        kind = "question" if parts and parts[0].strip().endswith("?") else "other"
        self.registry.mark_proactive(chat_id, parts[0] if parts else "", kind=kind)
        self.muffin.reactions.note_proactive(chat_id, kind=kind)

    @staticmethod
    def _pool_for_hour(hour: int) -> list[str]:
        if 6 <= hour < 10:
            return BREAKFAST
        if 11 <= hour < 14:
            return LUNCH
        if 17 <= hour < 20:
            return DINNER
        if 5 <= hour < 11:
            return MORNING
        if hour >= 22 or hour < 5:
            return NIGHT
        return DAY

    def _fallback(self) -> str:
        return random.choice(self._pool_for_hour(datetime.now().hour))

    def _fallback_specific(self) -> str:
        pool = [
            line
            for line in self._pool_for_hour(datetime.now().hour)
            if not GENERIC.match(line)
        ]
        return random.choice(pool or ["Bong tv na?"])

    def _fallback_short(self) -> str:
        return random.choice(["Bong?", "Hoy", "Ort reply teh", "😭", "Kit bong"])
