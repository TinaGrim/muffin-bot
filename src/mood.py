"""A small persistent mood/energy model.

The bot is not pretending to be sentient; this just gives the persona a
consistent internal state so replies have continuity: it gets tired at night,
loses patience when pinged repeatedly, warms up when Bong is affectionate, and
remembers recent topics for a while.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

from storage import atomic_write_text

COMPLIMENT = re.compile(
    r"\b(cute|handsome|smart|sweet|best|good boy|love you|miss you|slnh|😘|❤️|🥰)\b",
    re.I,
)
INSULT = re.compile(r"\b(stupid|idiot|hate you|shut up|annoying|boring)\b", re.I)
APOLOGY = re.compile(r"\b(sorry|khos|my bad|apolog\w*)\b", re.I)
PAMPER = re.compile(
    r"\b(sorry|khos|my bad|apolog\w*|slnh|love you|miss you|cute|handsome|"
    r"sweet|best|pamper|kom khot|kom jom|oy mun|som|kiss|hug|plov)\b"
    r"|😘|❤️|🥰|🤗|🙏",
    re.I,
)
REPAIR = re.compile(
    r"\b(kit bong|miss bong|miss oun|bong tv na|tv na bong|bong nham|bong dek|"
    r"bong rean|busy|meeting|phone|charge|late|ort reply|ot reply)\b",
    re.I,
)
WHY_COLD = re.compile(
    r"\b(hoy ey|ey hoy|(mic|mix)\s+ban(\s+hoy)?|hoy mic|mic hoy|"
    r"why (are )?(you )?(like this|cold|angry)|why hoy)\b",
    re.I,
)
SAD = re.compile(r"\b(sad|cry|stress|hurt|tired|hort|depress|lonely|scared)\b", re.I)
FOOD = re.compile(r"\b(nham bay|nham ey|hungry|eat|food|bay)\b", re.I)
WORK = re.compile(r"\b(homework|work|class|rean|job|exam|study)\b", re.I)
LOVE = re.compile(r"\b(love|slnh|miss)\b", re.I)
GREETING = re.compile(r"^(hi|hello|hey|bart|good morning|morning)\b", re.I)
QUESTION = re.compile(r"\?|^(what|why|how|where|when|who|pong|tver|tvr)\b", re.I)

HINTS = {
    "playful": "You are in a good mood: tease him, be a little bratty, keep it light.",
    "warm": "You feel soft toward him. You can be sweeter than usual, but still brief.",
    "tired": "You are tired. Keep it very short and low-effort; do not write paragraphs.",
    "sleepy": "It is late and you are sleepy. Short, slightly slow replies. Maybe a yawn.",
    "annoyed": "You are a little annoyed. Be blunt and short. Do not explain why.",
    "soft": "He seems down. Drop the teasing, be gentle, and keep it to one short line.",
    "angry": (
        "You are still angry with him and he has not made up for it. Stay cold, "
        "short and distant. Do not tease, do not act cute, do not forgive. Only "
        "an apology or real pampering can soften you."
    ),
    "neutral": "You are calm. Just answer like yourself.",
}

TOPIC_LABELS = {
    "food": "he was hungry or talking about food",
    "sad": "he seemed down",
    "work": "he mentioned work, class or homework",
    "love": "he was being affectionate",
}


@dataclass
class Mood:
    energy: int = 75
    affection: int = 60
    patience: int = 70
    mood: str = "neutral"
    last_user: float = 0.0
    last_bot: float = 0.0
    soft_until: float = 0.0
    upset: str = ""
    upset_since: float = 0.0
    comfort_needed: int = 0
    topics: dict[str, float] = field(default_factory=dict)


class MoodStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, Mood] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            repaired = False
            for chat_id, raw in data.get("chats", {}).items():
                known = {f for f in Mood.__dataclass_fields__}
                mood = Mood(**{k: v for k, v in raw.items() if k in known})
                # Repair the old death spiral: a grudge should not drain every
                # good feeling to zero and keep her cold forever.
                if mood.upset and mood.affection < 20:
                    mood.affection = 20
                    repaired = True
                self.chats[str(chat_id)] = mood
            if repaired:
                self._save()
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chats": {k: asdict(v) for k, v in self.chats.items()}}
        atomic_write_text(self.path, json.dumps(payload))

    def _get(self, chat_id: int) -> Mood:
        return self.chats.setdefault(str(chat_id), Mood())

    def snapshot(self, chat_id: int) -> Mood:
        with self.lock:
            return replace(self._get(chat_id))

    def reset(self, chat_id: int) -> None:
        with self.lock:
            self.chats.pop(str(chat_id), None)
            self._save()

    def observe(self, chat_id: int, text: str) -> Mood:
        now = time.time()
        low = text.lower()
        with self.lock:
            st = self._get(chat_id)
            self._decay(st, now)

            if st.last_user and now - st.last_user < 90:
                st.patience = max(0, st.patience - 1)
                st.energy = max(0, st.energy - 1)

            if COMPLIMENT.search(low):
                st.affection = min(100, st.affection + 6)
                st.energy = min(100, st.energy + 2)
            if INSULT.search(low):
                st.patience = max(0, st.patience - 8)
                st.affection = max(0, st.affection - 3)
                self._hurt(st, "insulted", 45, now)
            if WHY_COLD.search(low):
                # Asking why she is cold is a repair attempt: answer it and
                # let the grudge ease a little.
                self._comfort(st, 5, now)
            if PAMPER.search(low):
                self._comfort(st, 30 if APOLOGY.search(low) else 22, now)
            elif APOLOGY.search(low):
                self._comfort(st, 30, now)
            elif REPAIR.search(low):
                # He came back, explained himself, or asked about her. That is
                # not full pampering, but it is not nothing either.
                self._comfort(st, 7, now)
            if SAD.search(low):
                st.affection = min(100, st.affection + 4)
                st.energy = max(0, st.energy - 2)
                st.soft_until = now + 3600
            if FOOD.search(low):
                st.topics["food"] = now
            if WORK.search(low):
                st.topics["work"] = now
            if LOVE.search(low):
                st.topics["love"] = now
            if SAD.search(low):
                st.topics["sad"] = now

            st.last_user = now
            st.mood = self._pick(st, now)
            self._save()
            return replace(st)

    @staticmethod
    def _hurt(st: Mood, reason: str, amount: int, now: float) -> None:
        """Start or deepen a grudge; only comfort can clear it."""
        if not st.upset:
            st.upset = reason
            st.upset_since = now
        # Do not let one fight drain every good feeling; otherwise the cold
        # replies keep her cold forever.
        st.affection = max(st.affection, 20)
        st.comfort_needed = min(100, st.comfort_needed + amount)

    @staticmethod
    def _comfort(st: Mood, amount: int, now: float) -> None:
        """Apology or pampering chips at the grudge; zero clears it."""
        if not st.upset:
            st.patience = min(100, st.patience + 4)
            st.affection = min(100, st.affection + 2)
            return
        st.comfort_needed = max(0, st.comfort_needed - amount)
        st.patience = min(100, st.patience + 3)
        st.affection = min(100, st.affection + 2)
        if st.comfort_needed == 0:
            st.upset = ""
            st.upset_since = 0.0
            st.soft_until = now + 1200
            st.mood = "soft"

    def hurt(self, chat_id: int, reason: str, amount: int = 30) -> Mood:
        """External trigger: ignored, jealousy, or an insult."""
        now = time.time()
        with self.lock:
            st = self._get(chat_id)
            self._hurt(st, reason, amount, now)
            st.mood = self._pick(st, now)
            self._save()
            return replace(st)

    def comforted(self, chat_id: int, amount: int = 30) -> Mood:
        now = time.time()
        with self.lock:
            st = self._get(chat_id)
            self._comfort(st, amount, now)
            st.mood = self._pick(st, now)
            self._save()
            return replace(st)

    def after_reply(self, chat_id: int) -> None:
        with self.lock:
            st = self._get(chat_id)
            st.last_bot = time.time()
            st.energy = max(0, st.energy - 1)
            self._save()

    def adjust(
        self,
        chat_id: int,
        energy: int = 0,
        affection: int = 0,
        patience: int = 0,
    ) -> None:
        with self.lock:
            st = self._get(chat_id)
            st.energy = max(0, min(100, st.energy + energy))
            st.affection = max(0, min(100, st.affection + affection))
            st.patience = max(0, min(100, st.patience + patience))
            st.mood = self._pick(st, time.time())
            self._save()

    def _decay(self, st: Mood, now: float) -> None:
        if not st.last_user:
            return
        hours = min((now - st.last_user) / 3600, 24)
        if hours > 0.5:
            st.energy = min(100, st.energy + int(hours * 4))
            # Time alone does not fix a fight. Patience recovers only when she
            # is not upset; a grudge waits for an apology or pampering.
            if not st.upset:
                st.patience = min(100, st.patience + int(hours * 4))
                st.affection = max(0, st.affection - int(hours * 0.5))
        if 0 <= datetime.now().hour < 6:
            st.energy = max(10, st.energy - 8)

    def _pick(self, st: Mood, now: float) -> str:
        if st.upset and st.comfort_needed > 0:
            return "angry"
        if now < st.soft_until:
            return "soft"
        if st.energy <= 25:
            return "tired"
        if st.patience <= 20:
            return "annoyed"
        if st.affection >= 75:
            return "warm"
        if st.energy >= 65 and st.affection >= 50:
            return "playful"
        if 0 <= datetime.now().hour < 6:
            return "sleepy"
        return "neutral"

    def describe(self, st: Mood) -> str:
        lines = [
            f"How you feel right now: {st.mood}. "
            f"Energy {st.energy}/100, patience {st.patience}/100, affection {st.affection}/100.",
            HINTS.get(st.mood, HINTS["neutral"]),
            "Show the mood through how you talk; never announce it or mention numbers.",
        ]
        if st.upset:
            lines.append(
                f"You are still upset about being {st.upset}. He has not made up "
                f"for it yet (comfort {st.comfort_needed}/100 needed). Keep the "
                "grudge across turns. Do not become warm just because time passed."
            )
        recent = sorted(
            ((topic, ts) for topic, ts in st.topics.items() if time.time() - ts < 8 * 3600),
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        for topic, ts in recent:
            label = TOPIC_LABELS.get(topic)
            if not label:
                continue
            hours = (time.time() - ts) / 3600
            when = "just now" if hours < 1 else f"{int(hours)}h ago"
            lines.append(f"Recent context: {label} ({when}). You may bring it up if it fits.")
        return "\n".join(lines)
