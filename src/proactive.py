"""Proactive messages and small human follow-ups.

The bot should not wait forever for Bong, and sometimes a person sends a second
tiny message after the first. This module owns both behaviors and persists the
per-chat clocks so a restart does not spam anyone.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from storage import atomic_write_text
from typing import Any

log = logging.getLogger("muffin")

MORNING = ["Morning bong", "Bart bong", "Tngaii nih tv na?"]
DAY = ["Bong tv na?", "Nham bay nv?", "Hoy", "Bong?", "Kit bong"]
NIGHT = ["Bong?", "Kit bong", "Ort reply teh"]


@dataclass
class ChatState:
    last_user: float = 0.0
    last_bot: float = 0.0
    last_proactive: float = 0.0
    next_delay: float = 0.0
    enabled: bool = True


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
            self._save()

    def note_bot(self, chat_id: int) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            state.last_bot = time.time()
            self._save()

    def mark_proactive(self, chat_id: int) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            state.last_proactive = time.time()
            state.next_delay = self._new_delay()
            self._save()

    def set_enabled(self, chat_id: int, enabled: bool) -> None:
        with self.lock:
            state = self.chats.setdefault(str(chat_id), ChatState())
            state.enabled = enabled
            self._save()

    def enabled(self, chat_id: int) -> bool:
        with self.lock:
            return self.chats.get(str(chat_id), ChatState()).enabled

    def due(self, now: float) -> list[int]:
        due = []
        with self.lock:
            for key, state in self.chats.items():
                if not state.enabled or state.next_delay <= 0:
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
            message = self.muffin.ritual(chat_id)
            if message:
                log.info("ritual chat=%s -> %r", chat_id, message)
                self.tg.typing(chat_id)
                self.tg.send(chat_id, message)
                self.registry.mark_proactive(chat_id)
                self.muffin.reactions.note_proactive(chat_id)
        for chat_id in self.registry.due(now):
            self.send(chat_id)

    def send(self, chat_id: int, force: bool = False) -> None:
        if not force and not self.registry.enabled(chat_id):
            return
        parts = self.muffin.proactive(chat_id)
        if not parts:
            parts = [self._fallback()]
        log.info("proactive chat=%s -> %r", chat_id, " | ".join(parts))
        for i, part in enumerate(parts[:2]):
            if i:
                time.sleep(random.uniform(1.2, 2.5))
            self.tg.typing(chat_id)
            self.tg.send(chat_id, part)
        self.registry.mark_proactive(chat_id)
        self.muffin.reactions.note_proactive(chat_id)

    def _fallback(self) -> str:
        hour = datetime.now().hour
        pool = MORNING if 5 <= hour < 11 else NIGHT if (hour >= 22 or hour < 5) else DAY
        return random.choice(pool)

    def _fallback_short(self) -> str:
        return random.choice(["Bong?", "Hoy", "Ort reply teh", "😭", "Kit bong"])
