"""A persistent 'what is she doing right now' state.

The schedule follows Bong's description: school at 07:00, home in the middle of
the day, teaching at 17:30, back at 19:30, sometimes a 30-minute meeting until
20:00, then home. Weekends stay mostly at home. Replies that contradict the
current block are rejected.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from storage import atomic_write_text

HOME = "home"
OUT = "out"


@dataclass(frozen=True)
class ActivityDef:
    key: str
    label: str
    location: str


DEFS: dict[str, ActivityDef] = {
    "sleep": ActivityDef("sleep", "sleeping at home", HOME),
    "breakfast": ActivityDef("breakfast", "eating breakfast at home", HOME),
    "school": ActivityDef("school", "at school", OUT),
    "lunch": ActivityDef("lunch", "eating lunch at home", HOME),
    "chill": ActivityDef("chill", "at home, lying down and scrolling the phone", HOME),
    "work": ActivityDef("work", "working at home on the laptop", HOME),
    "commute": ActivityDef("commute", "on the way to teaching", OUT),
    "teach": ActivityDef("teach", "teaching at school", OUT),
    "meeting": ActivityDef("meeting", "in a short meeting", OUT),
    "dinner": ActivityDef("dinner", "eating dinner at home", HOME),
    "night": ActivityDef("night", "at home, in bed", HOME),
    "out_food": ActivityDef("out_food", "out buying food near home", OUT),
}

# minutes since midnight
WEEKDAY = [
    (0, 390, "sleep"),
    (390, 420, "breakfast"),
    (420, 690, "school"),
    (690, 780, "lunch"),
    (780, 1020, "chill"),
    (1020, 1050, "commute"),
    (1050, 1170, "teach"),
    (1170, 1200, "meeting"),
    (1200, 1380, "dinner"),
    (1380, 1440, "night"),
]

WEEKEND = [
    (0, 480, "sleep"),
    (480, 540, "breakfast"),
    (540, 720, "chill"),
    (720, 780, "lunch"),
    (780, 1080, "chill"),
    (1080, 1260, "dinner"),
    (1260, 1380, "chill"),
    (1380, 1440, "night"),
]

BAD_WHEN_HOME = {
    "vacation", "vacationing", "travel", "travelling", "traveling", "beach",
    "hotel", "airport", "flight", "srok", "cafe", "school", "outside",
}
BAD_WHEN_OUT = {"home", "house", "pteas", "bed", "sleep", "sleeping", "dek"}
BAD_WHEN_EATING = {"sleep", "sleeping", "dek", "vacation", "travel", "beach", "hotel"}
BAD_WHEN_SLEEPING = {
    "nham", "eat", "eating", "food", "fried", "fish", "bay", "coffee", "tea",
    "out", "work", "working", "study", "rean",
}
BAD_WHEN_STUDY = {"sleep", "sleeping", "vacation", "beach", "hotel"}

FALLBACK = {
    "eating": ["Nham bay nv", "Nham bay prek nv", "Bong nham nv?"],
    "sleep": ["Dek hz", "Hmm", "Good night"],
    "school": ["Rean nv", "Nv school", "Bong tv na?"],
    "teach": ["Pg teach nv", "Nv sala", "Bong tv na?"],
    "meeting": ["Meeting nv", "Jam tt", "Nv meeting"],
    "commute": ["Tv teach tv", "Nv ler", "Bong tv na?"],
    "study": ["Rean nv", "Pg reab pteas", "Nv home"],
    "work": ["Nv home", "Pg tvr work nv", "Bong tv na?"],
    "out": ["Nv psar", "Tv nham tv", "Bong tv na?"],
    "home": ["Nv home bong", "Nv ler", "Bong tv na?"],
}


@dataclass
class Activity:
    key: str = "chill"
    started: float = 0.0
    ends: float = 0.0


class LifeStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, Activity] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for chat_id, raw in data.get("chats", {}).items():
                self.chats[str(chat_id)] = Activity(
                    key=raw.get("key", "chill"),
                    started=float(raw.get("started", 0)),
                    ends=float(raw.get("ends", 0)),
                )
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chats": {k: asdict(v) for k, v in self.chats.items()}}
        atomic_write_text(self.path, json.dumps(payload))

    def _scheduled(self, now: datetime) -> Activity:
        table = WEEKDAY if now.weekday() < 5 else WEEKEND
        minute = now.hour * 60 + now.minute
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for start, end, key in table:
            if start <= minute < end:
                if key == "meeting" and random.random() >= 0.5:
                    key = "dinner"
                if key not in DEFS:
                    key = "chill"
                return Activity(
                    key=key,
                    started=(midnight + timedelta(minutes=start)).timestamp(),
                    ends=(midnight + timedelta(minutes=end)).timestamp(),
                )
        return Activity(
            key="chill",
            started=now.timestamp(),
            ends=(now + timedelta(minutes=60)).timestamp(),
        )

    def observe(self, chat_id: int) -> Activity:
        now = time.time()
        with self.lock:
            key = str(chat_id)
            state = self.chats.get(key)
            if state is None or now >= state.ends or state.key not in DEFS:
                state = self._scheduled(datetime.now())
                self.chats[key] = state
                self._save()
            return state

    def reset(self, chat_id: int) -> None:
        with self.lock:
            self.chats.pop(str(chat_id), None)
            self._save()

    @staticmethod
    def definition(state: Activity) -> ActivityDef:
        return DEFS.get(state.key, DEFS["chill"])

    def describe(self, state: Activity) -> str:
        d = self.definition(state)
        minutes = max(0, int((time.time() - state.started) / 60))
        return (
            f"You are currently {d.label}. You started {minutes} minutes ago. "
            "Stories must match this: do not claim you are somewhere else, "
            "travelling, or on vacation."
        )

    def conflicts(self, state: Activity, reply: str) -> bool:
        """Reject only replies that actually claim a current location/activity.

        Talking about food, school or work while at home is normal. Blocking
        those topics made the bot answer unrelated lines.
        """
        low = reply.lower()
        d = self.definition(state)
        markers = re.compile(
            r"\b(i'?m|i am|oun|nv)\b.{0,35}\b(at|in|to|nv|na)\b|"
            r"\b(at home|in bed|at school|teaching|working|studying|"
            r"in class|in a meeting|on vacation|at the beach|at the airport|"
            r"at a hotel|sleeping|working|studying|teaching|in a meeting)\b|"
            r"\bnv\s+(home|house|school|class|sala|work|bed|pteas|psar|"
            r"meeting|vacation|beach|hotel|airport)\b",
            re.I,
        )
        if not markers.search(low):
            return False
        if d.location == HOME and any(word in low for word in BAD_WHEN_HOME):
            return True
        if d.location == OUT and any(word in low for word in BAD_WHEN_OUT):
            return True
        if d.key in {"breakfast", "lunch", "dinner"} and re.search(
            r"\b(sleep|sleeping|dek|bed|at school|teaching|working)\b", low
        ):
            return True
        if d.key == "sleep" and re.search(
            r"\b(eat|eating|nham|bay|work|working|study|rean)\b", low
        ):
            return True
        return False

    def fallback(self, state: Activity) -> str:
        key = state.key
        if key in {"breakfast", "lunch", "dinner"}:
            key = "eating"
        elif key not in {"sleep", "school", "teach", "meeting", "commute", "study", "work", "out_food", "chill"}:
            key = "home"
        if key == "out_food":
            key = "out"
        if key == "chill":
            key = "home"
        return random.choice(FALLBACK.get(key, FALLBACK["home"]))
