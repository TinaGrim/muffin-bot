"""Human-in-the-loop review for every reply the bot sends.

Each sent reply gets an inline keyboard in Telegram: OK, Regenerate, or
Recommend reply. The chosen answer is stored so the bot can learn from it:

* OK             -> the reply was good (kept in the log)
* Regenerate     -> a new answer is generated for the same message
* Recommend reply-> Bong types the line she should have said; it is saved as a
                    correction and wins for that exact message from then on.

Nothing here invents Khmerlish: corrections are exactly what Bong typed.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from storage import atomic_write_text

from retrieval import normalize

MAX_RECORDS = 400
MAX_LOG_LINES = 5000


@dataclass
class SentReply:
    id: str
    chat_id: int
    user: str
    parts: list[str]
    prior: list[str]
    history_before: list[dict] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    status: str = "pending"  # pending | ok | fixed | regenerated
    corrected: str = ""
    note: str = ""
    original: str = ""
    attempts: list[str] = field(default_factory=list)
    message_id: int | None = None
    message_ids: list[int] = field(default_factory=list)

    @property
    def reply(self) -> str:
        return "\n".join(self.parts)


class Corrections:
    """Exact messages Bong corrected, keyed by normalized text.

    A correction always wins: if he took the time to type the right line, the
    bot must not go back to the archive or the model for that message.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self.lock = threading.Lock()
        self.map: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            self.map = {str(k): str(v) for k, v in raw.items() if v}

    def get(self, text: str) -> str | None:
        with self.lock:
            return self.map.get(normalize(text))

    def set(self, text: str, reply: str) -> None:
        key = normalize(text)
        if not key or not reply.strip():
            return
        with self.lock:
            self.map[key] = reply.strip()
            if self.path is None:
                return
            atomic_write_text(
                self.path,
                json.dumps(self.map, ensure_ascii=False, indent=1, sort_keys=True),
            )


class FeedbackStore:
    def __init__(self, data_dir: Path, default_training: bool = True):
        self.data_dir = data_dir
        self.records_path = data_dir / "sent_replies.json"
        self.log_path = data_dir / "feedback.jsonl"
        self.train_path = data_dir / "train.json"
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        self.training: dict[str, bool] = {}
        self.pending: dict[str, str] = {}
        self.default_training = default_training
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.records_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.records = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            pass
        try:
            raw = json.loads(self.train_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.training = {str(k): bool(v) for k, v in raw.items()}
        except (OSError, ValueError):
            pass

    def training_on(self, chat_id: int) -> bool:
        with self.lock:
            return self.training.get(str(chat_id), self.default_training)

    def toggle_training(self, chat_id: int) -> bool:
        with self.lock:
            value = not self.training.get(str(chat_id), self.default_training)
            self.training[str(chat_id)] = value
            atomic_write_text(
                self.train_path, json.dumps(self.training, indent=1, sort_keys=True)
            )
            return value

    def add(self, record: SentReply) -> SentReply:
        with self.lock:
            self.records[record.id] = asdict(record)
            if len(self.records) > MAX_RECORDS:
                for old in list(self.records)[: len(self.records) - MAX_RECORDS]:
                    self.records.pop(old, None)
            self._save_locked()
        return record

    def update(self, record: SentReply) -> None:
        with self.lock:
            self.records[record.id] = asdict(record)
            self._save_locked()

    def get(self, record_id: str) -> SentReply | None:
        with self.lock:
            raw = self.records.get(record_id)
            if not raw:
                return None
            try:
                return SentReply(**raw)
            except TypeError:
                return None

    def _save_locked(self) -> None:
        atomic_write_text(
            self.records_path, json.dumps(self.records, ensure_ascii=False)
        )

    def log(self, record: SentReply, action: str, extra: str = "") -> None:
        entry = {
            "ts": round(time.time(), 3),
            "id": record.id,
            "chat_id": record.chat_id,
            "user": record.user,
            "reply": record.reply,
            "action": action,
        }
        if extra:
            entry["extra"] = extra
        with self.lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if self._log_too_long():
                    self._rotate_log_locked()
            except OSError:
                pass

    def _log_too_long(self) -> bool:
        try:
            with self.log_path.open("rb") as f:
                return sum(1 for _ in f) > MAX_LOG_LINES
        except OSError:
            return False

    def _rotate_log_locked(self) -> None:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            atomic_write_text(
                self.log_path, "\n".join(lines[-MAX_LOG_LINES // 2 :]) + "\n"
            )
        except OSError:
            pass

    def outcome(
        self,
        record_id: str,
        status: str,
        corrected: str = "",
        note: str = "",
    ) -> SentReply | None:
        record = self.get(record_id)
        if record is None:
            return None
        record.status = status
        if corrected:
            record.corrected = corrected
        if note:
            record.note = note
        self.update(record)
        self.log(record, status, note or corrected)
        return record

    def replies_for(self, user: str) -> list[str]:
        """Every line already tried for this message."""
        with self.lock:
            seen: list[str] = []
            for raw in self.records.values():
                if raw.get("user") != user:
                    continue
                seen.extend(raw.get("attempts") or [])
                if raw.get("parts"):
                    seen.append("\n".join(raw["parts"]))
                elif raw.get("reply"):
                    seen.append(str(raw["reply"]))
            return [line for line in seen if line.strip()]

    def set_pending(self, chat_id: int, record_id: str) -> None:
        with self.lock:
            self.pending[str(chat_id)] = record_id

    def take_pending(self, chat_id: int) -> str | None:
        with self.lock:
            return self.pending.pop(str(chat_id), None)


def new_id() -> str:
    return uuid.uuid4().hex[:12]
