"""Persistent per-chat conversation history."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from storage import atomic_write_text


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            chats = data.get("chats", {})
            for chat_id, history in chats.items():
                if isinstance(history, list):
                    self.chats[str(chat_id)] = [
                        m for m in history if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
                    ]
        except (ValueError, TypeError):
            self.chats = {}

    def _save(self) -> None:
        atomic_write_text(
            self.path, json.dumps({"chats": self.chats}, ensure_ascii=False)
        )

    def get(self, chat_id: int) -> list[dict]:
        with self.lock:
            return list(self.chats.get(str(chat_id), []))

    def set(self, chat_id: int, history: list[dict]) -> None:
        with self.lock:
            self.chats[str(chat_id)] = list(history)
            self._save()

    def reset(self, chat_id: int) -> None:
        with self.lock:
            self.chats.pop(str(chat_id), None)
            self._save()
