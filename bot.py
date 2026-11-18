#!/usr/bin/env python3
"""Run Muffin as a Telegram bot, or chat with the same pipeline in the terminal."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feedback import Corrections, FeedbackStore, SentReply, new_id  # noqa: E402
from life import LifeStore  # noqa: E402
from history import HistoryStore  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from threads import ThreadStore  # noqa: E402
from muffin import Chat, Muffin, load_config  # noqa: E402
from mood import MoodStore  # noqa: E402
from proactive import ChatRegistry, Proactive  # noqa: E402
from retrieval import CachedRetriever, normalize  # noqa: E402
from storage import atomic_write_text  # noqa: E402

ROOT = Path(__file__).parent
log = logging.getLogger("muffin")


def make_muffin(env: dict[str, str]) -> Muffin:
    config = load_config(env, ROOT / "persona.md")
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl",
        ROOT / "data" / "retrieval.pkl",
        embeddings=env.get("USE_EMBEDDINGS", "true").lower()
        not in {"0", "false", "no"},
        embed_model=env.get("EMBED_MODEL", "bge-m3"),
        embed_base=env.get("EMBED_BASE_URL", "http://localhost:11434"),
        embed_path=ROOT / "data" / env.get("EMBED_INDEX", "embeddings_bge_m3.npz"),
    ).get()
    mood = MoodStore(ROOT / "data" / "mood.json")
    life = LifeStore(ROOT / "data" / "life.json")
    history = HistoryStore(ROOT / "data" / "history.json")
    reactions = ReactionStore(ROOT / "data" / "reactions.json")
    threads = ThreadStore(ROOT / "data" / "threads.json")
    corrections = Corrections(ROOT / "data" / "corrections.json")
    log.info(
        "loaded %d exchanges, model=%s base_url=%s",
        len(retriever.pairs),
        config.model,
        config.base_url,
    )
    return Muffin(
        config, retriever, mood, life, history, reactions, threads, corrections
    )


class Telegram:
    def __init__(self, token: str):
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.client = httpx.Client(timeout=60)

    def call(self, method: str, **params):
        for attempt in range(3):
            try:
                response = self.client.post(f"{self.base}/{method}", json=params)
                data = response.json()
                if response.status_code == 429:
                    retry = data.get("parameters", {}).get("retry_after", 3)
                    log.warning("rate limited, sleeping %ss", retry)
                    time.sleep(retry)
                    continue
                if not data.get("ok"):
                    raise RuntimeError(f"{method}: {data}")
                return data["result"]
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 2:
                    raise
                log.warning("%s failed (%s), retrying", method, exc)
                time.sleep(2)
        return None

    def get_updates(self, offset: int | None):
        params = {
            "timeout": 30,
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if offset is not None:
            params["offset"] = offset
        return self.call("getUpdates", **params) or []

    def send(
        self,
        chat_id: int,
        text: str,
        reply_to: int | None = None,
        markup: dict | None = None,
    ) -> int | None:
        is_code = "```" in text
        if is_code:
            text = re.sub(r"^```[A-Za-z0-9_+-]*\n", "```\n", text)
        params = {"chat_id": chat_id, "text": text[:4096]}
        if is_code:
            params["parse_mode"] = "Markdown"
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        if markup:
            params["reply_markup"] = markup
        try:
            result = self.call("sendMessage", **params)
            return result.get("message_id") if isinstance(result, dict) else None
        except Exception as exc:  # noqa: BLE001
            log.warning("send failed (%s), retrying once", exc)
            time.sleep(1.0)
            try:
                params.pop("parse_mode", None)
                result = self.call("sendMessage", **params)
                return result.get("message_id") if isinstance(result, dict) else None
            except Exception as exc2:  # noqa: BLE001
                log.error("send failed twice: %s", exc2)
                return None

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        params = {"callback_query_id": callback_id}
        if text:
            params["text"] = text
        try:
            self.call("answerCallbackQuery", **params)
        except Exception as exc:  # noqa: BLE001
            log.debug("answerCallbackQuery failed: %s", exc)

    def edit_markup(self, chat_id: int, message_id: int, markup: dict) -> None:
        try:
            self.call(
                "editMessageReplyMarkup",
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=markup,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("editMessageReplyMarkup failed: %s", exc)

    def edit_text(
        self, chat_id: int, message_id: int, text: str, markup: dict | None = None
    ) -> bool:
        params = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text[:4096],
        }
        if markup:
            params["reply_markup"] = markup
        try:
            self.call("editMessageText", **params)
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("editMessageText failed: %s", exc)
            return False

    def delete(self, chat_id: int, message_id: int) -> None:
        try:
            self.call("deleteMessage", chat_id=chat_id, message_id=message_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("deleteMessage failed: %s", exc)

    def typing(self, chat_id: int):
        try:
            self.call("sendChatAction", chat_id=chat_id, action="typing")
        except Exception as exc:  # noqa: BLE001
            log.debug("typing indicator failed: %s", exc)

    def send_sticker(self, chat_id: int, path: str) -> None:
        try:
            with open(path, "rb") as f:
                response = self.client.post(
                    f"{self.base}/sendSticker",
                    data={"chat_id": chat_id},
                    files={"sticker": (Path(path).name, f, "image/webp")},
                )
            data = response.json()
            if not data.get("ok"):
                log.warning("sendSticker failed: %s", data.get("description"))
        except Exception as exc:  # noqa: BLE001
            log.warning("sendSticker error: %s", exc)


def allowed(env: dict[str, str], chat: dict, user_id: int, text: str, username: str) -> tuple[bool, str]:
    allowed_users = {
        int(x) for x in env.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x
    }
    allowed_chats = {
        int(x) for x in env.get("ALLOWED_CHAT_IDS", "").replace(" ", "").split(",") if x
    }
    if allowed_users and user_id not in allowed_users:
        return False, text

    chat_id = chat.get("id")
    if chat.get("type") == "private":
        return True, text

    if allowed_chats and chat_id not in allowed_chats:
        return False, text

    mention = f"@{username}".lower() if username else None
    if mention and mention in text.lower():
        text = re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.I).strip()
        return True, text
    return False, text


def feedback_keyboard(record_id: str) -> dict:
    """Review buttons attached to the last bubble of each answer."""
    return {
        "inline_keyboard": [
            [
                {"text": "✅ OK", "callback_data": f"fb:ok:{record_id}"},
                {"text": "🔄 Regenerate", "callback_data": f"fb:regen:{record_id}"},
            ],
            [
                {
                    "text": "✍️ Recommend reply",
                    "callback_data": f"fb:fix:{record_id}",
                }
            ],
        ]
    }


USED_MARKUP = {"inline_keyboard": [[{"text": "✅", "callback_data": "fb:noop:0"}]]}
REGENERATED_MARKUP = {
    "inline_keyboard": [[{"text": "🔄 Regenerated", "callback_data": "fb:noop:0"}]]
}


class ChatWorker(threading.Thread):
    """Collect a burst of messages, then answer once after the user stops typing."""

    def __init__(
        self,
        tg: "Telegram",
        muffin: Muffin,
        chat_id: int,
        registry: ChatRegistry | None = None,
        delay: float = 1.2,
        feedback: FeedbackStore | None = None,
    ):
        super().__init__(daemon=True)
        self.tg = tg
        self.muffin = muffin
        self.chat_id = chat_id
        self.registry = registry
        self.delay = delay
        self.feedback = feedback
        self.inbox: queue.Queue[tuple[str, object, int | None, bool]] = queue.Queue()

    def submit(self, text: str, message_id: int | None, specific: bool = False) -> None:
        self.inbox.put(("text", text, message_id, specific))

    def submit_regen(self, record: SentReply) -> None:
        self.inbox.put(("regen", record, None, False))

    def submit_note(self, record: SentReply, note: str) -> None:
        self.inbox.put(("note", (record, note), None, False))

    def run(self) -> None:
        while True:
            kind, payload, message_id, specific = self.inbox.get()
            if kind == "regen":
                self._regenerate(payload)  # type: ignore[arg-type]
                continue
            if kind == "note":
                record, note = payload  # type: ignore[misc]
                self._note(record, note)
                continue
            burst = [str(payload)]
            deadline = time.time() + self.delay
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    more_kind, more, more_id, more_specific = self.inbox.get(
                        timeout=remaining
                    )
                except queue.Empty:
                    break
                if more_kind in {"regen", "note"}:
                    # Answer the burst first, then handle the correction.
                    self.inbox.put((more_kind, more, more_id, more_specific))
                    break
                burst.append(str(more))
                specific = specific or more_specific
                deadline = time.time() + self.delay
            self._answer(burst, message_id, specific)

    def _answer(self, burst: list[str], message_id: int | None, specific: bool) -> None:
        try:
            self._answer_inner(burst, message_id, specific)
        except Exception as exc:  # noqa: BLE001
            log.exception("worker answer failed: %s", exc)

    def _answer_inner(
        self, burst: list[str], message_id: int | None, specific: bool
    ) -> None:
        text = burst[-1]
        # Keep a human beat, but never add a long pause on top of a slow model.
        time.sleep(min(self.muffin.human_delay(self.chat_id), 1.5))
        self.tg.typing(self.chat_id)
        started = time.time()
        chat = self.muffin.chats.setdefault(self.chat_id, Chat())
        history_start = list(chat.history)
        try:
            parts = self.muffin.reply(self.chat_id, text, prior=burst[:-1])
        except Exception as exc:  # noqa: BLE001
            log.exception("reply failed: %s", exc)
            # Even a failure answer gets the review buttons, so Bong can press
            # Regenerate once the cause is gone instead of losing the message.
            history_before = history_start + [
                {"role": "user", "content": message}
                for message in burst[:-1]
                if message.strip()
            ]
            record = self._new_record(text, ["Ort"], burst[:-1], history_before)
            self._deliver(["Ort"], message_id, specific, record)
            return

        # Real people do not answer a one-word line instantly. Let the typing
        # indicator run for a human-ish minimum, but never add lag to slow LLM
        # replies that already took their time.
        elapsed = time.time() - started
        target = min(3.5, 0.7 + sum(len(part) for part in parts) * 0.03)
        if elapsed < target:
            time.sleep(target - elapsed)

        history_before = history_start + [
            {"role": "user", "content": message}
            for message in burst[:-1]
            if message.strip()
        ]
        record = self._new_record(text, parts, burst[:-1], history_before)

        log.info(
            "chat=%s %.1fs %r -> %r",
            self.chat_id,
            time.time() - started,
            " / ".join(burst)[:100],
            " | ".join(parts)[:160],
        )
        self._deliver(parts, message_id, specific, record)

        if "```" not in "".join(parts) and not "".join(parts).rstrip().endswith("?"):
            sticker = self.muffin.sticker(self.chat_id)
            if sticker:
                time.sleep(random.uniform(0.8, 1.8))
                self.tg.send_sticker(self.chat_id, sticker)

        if self.registry:
            self.registry.note_bot(self.chat_id)

    def _new_record(
        self,
        text: str,
        parts: list[str],
        prior: list[str],
        history_before: list[dict],
    ) -> SentReply | None:
        if not self.feedback or not self.feedback.training_on(self.chat_id):
            return None
        record = SentReply(
            id=new_id(),
            chat_id=self.chat_id,
            user=text,
            parts=list(parts),
            prior=[message for message in prior if message.strip()],
            history_before=history_before,
        )
        return self.feedback.add(record)

    def _deliver(
        self,
        parts: list[str],
        reply_to: int | None,
        specific: bool,
        record: SentReply | None,
    ) -> None:
        last_index = min(len(parts), 4) - 1
        sent_ids: list[int] = []
        for i, part in enumerate(parts[:4]):
            if i:
                time.sleep(random.uniform(0.5, 1.2))
            self.tg.typing(self.chat_id)
            markup = feedback_keyboard(record.id) if record and i == last_index else None
            sent_id = self.tg.send(
                self.chat_id,
                part,
                reply_to=reply_to if specific and i == 0 else None,
                markup=markup,
            )
            if sent_id:
                sent_ids.append(sent_id)
        if record and sent_ids:
            record.message_ids = list(record.message_ids) + sent_ids
            record.message_id = sent_ids[-1]
            self.feedback.update(record)

    def _regenerate(self, record: SentReply) -> None:
        """Answer again in place, avoiding every line already tried."""
        try:
            chat = self.muffin.chats.get(self.chat_id)
            current = (
                next(
                    (
                        message["content"]
                        for message in reversed(chat.history)
                        if message["role"] == "assistant"
                    ),
                    "",
                )
                if chat
                else ""
            )
            # Do not rewind a conversation that has moved past this answer.
            if current and current != record.reply:
                log.info("regenerate ignored: chat moved on")
                self.tg.send(self.chat_id, "Ort")
                return

            avoid = [record.reply]
            if self.feedback:
                for line in self.feedback.replies_for(record.user):
                    if line not in avoid:
                        avoid.append(line)
            self.muffin.restore_history(self.chat_id, record.history_before)
            self.tg.typing(self.chat_id)
            parts = self.muffin.reply(
                self.chat_id, record.user, prior=(), avoid=avoid
            )
            answer = "\n".join(parts).strip()
            blocked = {normalize(line) for line in avoid if line.strip()}
            if not answer or normalize(answer) in blocked:
                # Archive replay keeps winning; ask the model directly for a
                # line that is none of the rejected attempts.
                rewritten = self.muffin.rewrite_with_note(
                    self.chat_id,
                    record.user,
                    "Give a different answer. These lines were already rejected.",
                    record.history_before,
                    "\n".join(avoid),
                )
                rewritten = [part for part in rewritten if part.strip()]
                if rewritten and normalize("\n".join(rewritten)) not in blocked:
                    parts = self.muffin.style_fix(rewritten, record.user)
                    answer = "\n".join(parts).strip()
                    self.muffin.apply_correction(self.chat_id, answer)
            if not answer:
                raise RuntimeError("empty regenerated reply")
        except Exception as exc:  # noqa: BLE001
            log.exception("regenerate failed: %s", exc)
            self.tg.send(self.chat_id, "Ort")
            return

        old_reply = record.reply
        record.attempts = list(record.attempts) + [old_reply]
        record.parts = list(parts)
        record.status = "regenerated"

        old_ids = list(record.message_ids)
        if not old_ids and record.message_id:
            old_ids = [record.message_id]
        keep = old_ids[0] if old_ids else None
        edited = False
        if keep:
            if answer == old_reply:
                self.tg.edit_markup(
                    self.chat_id, keep, feedback_keyboard(record.id)
                )
                edited = True
            else:
                edited = self.tg.edit_text(
                    self.chat_id, keep, answer, feedback_keyboard(record.id)
                )
        if not edited:
            sent = self.tg.send(
                self.chat_id, answer, markup=feedback_keyboard(record.id)
            )
            if sent:
                keep = sent
                old_ids.append(sent)
        for message_id in old_ids[1:]:
            if message_id != keep:
                self.tg.delete(self.chat_id, message_id)
        if keep:
            record.message_id = keep
            record.message_ids = [keep]
        if self.feedback:
            self.feedback.update(record)
            self.feedback.log(record, "regenerated", answer)
        log.info(
            "chat=%s regenerated %r -> %r",
            self.chat_id,
            record.user[:80],
            answer[:160],
        )
        if self.registry:
            self.registry.note_bot(self.chat_id)

    def _note(self, record: SentReply, note: str) -> None:
        """Rewrite the rejected bubble in place, following Bong's note."""
        try:
            self.tg.typing(self.chat_id)
            parts = self.muffin.rewrite_with_note(
                self.chat_id,
                record.user,
                note,
                record.history_before,
                record.reply,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("note rewrite failed: %s", exc)
            self.tg.send(self.chat_id, "Ort")
            return

        parts = self.muffin.style_fix(parts, record.user)
        answer = "\n".join(parts).strip()
        if not answer:
            self.tg.send(self.chat_id, "Ort")
            return

        # The corrected line replaces this answer in the live conversation and
        # is logged with Bong's note. It is deliberately not saved as a global
        # override: a note like "she is at school now" is context, and the same
        # question tomorrow must follow her real schedule again.
        chat = self.muffin.chats.get(self.chat_id)
        last = (
            next(
                (
                    message["content"]
                    for message in reversed(chat.history)
                    if message["role"] == "assistant"
                ),
                "",
            )
            if chat
            else ""
        )
        if last == record.reply:
            self.muffin.apply_correction(self.chat_id, answer)

        old_ids = list(record.message_ids)
        if not old_ids and record.message_id:
            old_ids = [record.message_id]

        old_reply = record.reply
        record.parts = list(parts)
        record.note = note
        record.corrected = answer
        record.status = "fixed"

        keep = old_ids[0] if old_ids else None
        already = answer == old_reply
        if already and keep:
            # Telegram rejects an edit that does not change the text; keep the
            # bubble and just refresh its buttons.
            self.tg.edit_markup(
                self.chat_id, keep, feedback_keyboard(record.id)
            )
            edited = True
        else:
            edited = (
                self.tg.edit_text(
                    self.chat_id, keep, answer, feedback_keyboard(record.id)
                )
                if keep
                else False
            )
        if not edited:
            sent = self.tg.send(
                self.chat_id, answer, markup=feedback_keyboard(record.id)
            )
            if sent:
                keep = sent
                old_ids.append(sent)
        for message_id in old_ids[1:]:
            if message_id != keep:
                self.tg.delete(self.chat_id, message_id)

        if keep:
            record.message_id = keep
            record.message_ids = [keep]
        if self.feedback:
            self.feedback.update(record)
            self.feedback.log(record, "fixed", note)
        if self.registry:
            self.registry.note_bot(self.chat_id)
        log.info(
            "chat=%s note corrected %r note=%r -> %r",
            self.chat_id,
            record.user[:80],
            note[:80],
            answer[:160],
        )




def get_worker(
    tg: "Telegram",
    muffin: Muffin,
    feedback: FeedbackStore,
    workers: dict[int, "ChatWorker"],
    registry: ChatRegistry | None,
    chat_id: int,
) -> "ChatWorker":
    worker = workers.get(chat_id)
    if worker is None or not worker.is_alive():
        worker = ChatWorker(tg, muffin, chat_id, registry, feedback=feedback)
        worker.start()
        workers[chat_id] = worker
    return worker


def handle_callback(
    tg: "Telegram",
    muffin: Muffin,
    feedback: FeedbackStore,
    workers: dict[int, "ChatWorker"],
    registry: ChatRegistry,
    callback: dict,
) -> None:
    """Handle the OK / Regenerate / Recommend buttons."""
    data = callback.get("data") or ""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "fb":
        return
    action, record_id = parts[1], parts[2]
    callback_id = callback.get("id", "")
    message = callback.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if not chat_id:
        return

    if action == "noop":
        tg.answer_callback(callback_id)
        return

    record = feedback.get(record_id)
    if record is None:
        tg.answer_callback(callback_id, "Too old")
        return

    if action == "ok":
        feedback.outcome(record_id, "ok")
        tg.answer_callback(callback_id, "Saved")
        if message_id:
            tg.edit_markup(chat_id, message_id, USED_MARKUP)
        return

    if action == "fix":
        feedback.set_pending(chat_id, record_id)
        tg.answer_callback(callback_id)
        tg.send(chat_id, "What was wrong? Send a note for her.")
        return

    if action == "regen":
        tg.answer_callback(callback_id, "Again")
        worker = get_worker(tg, muffin, feedback, workers, registry, chat_id)
        worker.submit_regen(record)
        return

    tg.answer_callback(callback_id)


def run_telegram(muffin: Muffin, env: dict[str, str]) -> None:
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "1. Open @BotFather in Telegram\n"
            "2. Send /newbot and follow the prompts\n"
            "3. Put the token in .env as TELEGRAM_BOT_TOKEN=..."
        )

    tg = Telegram(token)
    me = tg.call("getMe") or {}
    username = me.get("username", "")
    log.info("connected as @%s", username)

    offset_file = ROOT / "data" / "offset.txt"
    offset = None
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip())
        except ValueError:
            offset = None

    max_age = int(env.get("MAX_MESSAGE_AGE_SECONDS", 300))
    min_hours = float(env.get("PROACTIVE_MIN_HOURS", 3))
    max_hours = float(env.get("PROACTIVE_MAX_HOURS", 6))
    registry = ChatRegistry(ROOT / "data" / "chats.json", min_hours, max_hours)
    allowed_ids = [
        int(x)
        for x in env.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",")
        if x
    ]
    for chat_id in allowed_ids:
        registry.seed(chat_id)

    proactive: Proactive | None = None
    if env.get("PROACTIVE", "true").lower() in {"1", "true", "yes"}:
        proactive = Proactive(
            tg,
            muffin,
            registry,
            quiet_start=int(env.get("PROACTIVE_QUIET_START", 23)),
            quiet_end=int(env.get("PROACTIVE_QUIET_END", 8)),
        )
        proactive.start()

    feedback = FeedbackStore(
        ROOT / "data",
        default_training=env.get("FEEDBACK_BUTTONS", "true").lower()
        not in {"0", "false", "no"},
    )

    workers: dict[int, ChatWorker] = {}
    while True:
        try:
            updates = tg.get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                offset_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(offset_file, str(offset))

                callback = update.get("callback_query")
                if callback:
                    if allowed_ids and (callback.get("from") or {}).get("id") not in allowed_ids:
                        continue
                    handle_callback(tg, muffin, feedback, workers, registry, callback)
                    continue

                message = update.get("message")
                if not message or not message.get("text"):
                    continue
                if max_age and time.time() - message.get("date", time.time()) > max_age:
                    log.info("skipping stale message %s", message.get("message_id"))
                    continue

                sender = message.get("from") or {}
                if sender.get("is_bot"):
                    continue
                chat = message.get("chat") or {}
                text = message["text"].strip()
                ok, text = allowed(env, chat, sender.get("id", 0), text, username)
                if not ok or not text:
                    continue
                registry.note_user(chat["id"])

                chat_id = chat["id"]

                # A pending "Recommend reply" turns the next message into the
                # corrected line, exactly as Bong typed it.
                pending_id = feedback.take_pending(chat_id)
                if pending_id:
                    record = feedback.get(pending_id)
                    if text.startswith("/cancel") or record is None:
                        tg.send(chat_id, "Okay")
                        continue
                    # The text is a note for the model, not the literal reply.
                    worker = get_worker(
                        tg, muffin, feedback, workers, registry, chat_id
                    )
                    worker.submit_note(record, text)
                    continue

                if text.startswith("/start"):
                    tg.send(chat["id"], "Bart bong")
                    continue
                if text.startswith("/reset"):
                    muffin.reset(chat["id"])
                    tg.send(chat["id"], "Ort")
                    continue
                if text.startswith("/help"):
                    tg.send(
                        chat["id"],
                        "Muffin clone. /reset clears memory, /mood shows how it feels, "
                        "/life shows what it is doing, /resetlife rerolls that, "
                        "/resetmood calms it down, /poke makes it text you first, "
                        "/proactive turns spontaneous texts on or off. "
                        "/train toggles the OK / Regenerate / Recommend buttons.",
                    )
                    continue
                if text.startswith("/train"):
                    on = feedback.toggle_training(chat["id"])
                    tg.send(
                        chat["id"],
                        "Train mode on: each reply has OK / Regenerate / Recommend."
                        if on
                        else "Train mode off",
                    )
                    continue
                if text.startswith("/stats"):
                    stats = json.loads((ROOT / "data" / "stats.json").read_text())
                    tg.send(
                        chat["id"],
                        f"{stats['pairs']:,} real exchanges, "
                        f"model {muffin.config.model}",
                    )
                    continue
                if text.startswith("/mood"):
                    st = muffin.mood.snapshot(chat["id"])
                    tg.send(
                        chat["id"],
                        f"{st.mood} · energy {st.energy}/100 · "
                        f"affection {st.affection}/100 · patience {st.patience}/100",
                    )
                    continue
                if text.startswith("/life"):
                    st = muffin.life.observe(chat["id"])
                    d = muffin.life.definition(st)
                    minutes = max(0, int((time.time() - st.started) / 60))
                    tg.send(chat["id"], f"{d.label} · {minutes} min")
                    continue
                if text.startswith("/resetlife"):
                    muffin.life.reset(chat["id"])
                    st = muffin.life.observe(chat["id"])
                    tg.send(chat["id"], muffin.life.definition(st).label)
                    continue
                if text.startswith("/resetmood"):
                    muffin.mood.reset(chat["id"])
                    tg.send(chat["id"], "Ort")
                    continue
                if text.startswith("/poke"):
                    if proactive:
                        proactive.send(chat["id"], force=True)
                    continue
                if text.startswith("/proactive"):
                    enabled = registry.enabled(chat["id"])
                    registry.set_enabled(chat["id"], not enabled)
                    tg.send(chat["id"], "Okay" if not enabled else "Ort")
                    continue
                if text.startswith("/"):
                    continue

                worker = workers.get(chat_id)
                if worker is None or not worker.is_alive():
                    worker = ChatWorker(tg, muffin, chat_id, registry, feedback=feedback)
                    worker.start()
                    workers[chat_id] = worker
                worker.submit(
                    text,
                    message.get("message_id"),
                    specific=bool(message.get("reply_to_message")),
                )
        except KeyboardInterrupt:
            log.info("stopped")
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("polling error: %s", exc)
            time.sleep(3)


def run_cli(muffin: Muffin) -> None:
    print("Muffin clone. Ctrl+D to quit.\n")
    chat_id = 1
    while True:
        try:
            text = input("Bong> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        if text in {"/reset", "/quit", "/exit"}:
            muffin.reset(chat_id)
            print("Muffin> (memory cleared)")
            continue
        try:
            parts = muffin.reply(chat_id, text)
        except Exception as exc:  # noqa: BLE001
            print(f"! {exc}")
            continue
        for part in parts:
            print(f"Muffin> {part}")


def run_retrieve(muffin: Muffin, query: str, k: int = 8) -> None:
    for match in muffin.retriever.search(query, k=k):
        print(f"{match['score']:.3f}  {match['user']!r}\n       -> {match['reply']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="terminal chat")
    parser.add_argument("--retrieve", metavar="TEXT", help="show nearest real exchanges")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # httpx logs the full request URL at INFO, which contains the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    muffin = make_muffin(env)

    if args.retrieve:
        run_retrieve(muffin, args.retrieve)
    elif args.chat:
        run_cli(muffin)
    else:
        run_telegram(muffin, env)


if __name__ == "__main__":
    main()
