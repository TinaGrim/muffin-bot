"""Per-chat burst worker, reply delivery and feedback button handling."""

from __future__ import annotations

import logging
import queue
import random
import threading
import time

from feedback import FeedbackStore, SentReply, new_id
from metrics import Metrics
from muffin import Chat, Muffin
from proactive import ChatRegistry
from retrieval import normalize
from telegram_client import Telegram

log = logging.getLogger("muffin")


def pick_target(burst: list[str]) -> str:
    """Answer the message that matters, not just the last one in a burst."""
    for text in reversed(burst):
        if "?" in text:
            return text
    greetings = {
        "hello",
        "hi",
        "hey",
        "bart",
        "hmm",
        "hello oun",
        "morning",
        "morning oun",
        "morning bong",
        "good night",
    }
    for text in reversed(burst):
        if text.strip().lower() not in greetings:
            return text
    return burst[-1]


def feedback_keyboard(record_id: str) -> dict:
    """Review buttons attached to the last bubble of each answer."""
    return {
        "inline_keyboard": [
            [
                {"text": "✅ OK", "callback_data": f"fb:ok:{record_id}"},
                {"text": "👎 Bad", "callback_data": f"fb:bad:{record_id}"},
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
        metrics: Metrics | None = None,
    ):
        super().__init__(daemon=True)
        self.tg = tg
        self.muffin = muffin
        self.chat_id = chat_id
        self.registry = registry
        self.delay = delay
        self.feedback = feedback
        self.metrics = metrics
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
        target = pick_target(burst)
        text = target
        # Keep a human beat, but never add a long pause on top of a slow model.
        time.sleep(min(self.muffin.human_delay(self.chat_id), 1.5))
        self.tg.typing(self.chat_id)
        started = time.time()
        chat = self.muffin.chats.setdefault(self.chat_id, Chat())
        history_start = list(chat.history)
        target_index = burst.index(target)
        prior = burst[:target_index] + burst[target_index + 1 :]
        try:
            parts = self.muffin.reply(self.chat_id, text, prior=prior)
        except Exception as exc:  # noqa: BLE001
            log.exception("reply failed: %s", exc)
            # Even a failure answer gets the review buttons, so Bong can press
            # Regenerate once the cause is gone instead of losing the message.
            history_before = history_start + [
                {"role": "user", "content": message}
                for message in prior
                if message.strip()
            ]
            record = self._new_record(text, ["Ort"], prior, history_before)
            self._deliver(["Ort"], message_id, specific, record)
            return

        # Real people do not answer a one-word line instantly. Let the typing
        # indicator run for a human-ish minimum, but never add lag to slow LLM
        # replies that already took their time.
        elapsed = time.time() - started
        if self.metrics:
            self.metrics.record(
                self.chat_id,
                elapsed,
                self.muffin.last_provider,
                self.muffin.last_fallbacks,
                self.muffin.last_style_issues,
                " ".join(parts),
            )
        target = min(3.5, 0.7 + sum(len(part) for part in parts) * 0.03)
        if elapsed < target:
            time.sleep(target - elapsed)

        history_before = history_start + [
            {"role": "user", "content": message}
            for message in prior
            if message.strip()
        ]
        record = self._new_record(text, parts, prior, history_before)

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
        started = time.time()
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

        if self.metrics:
            self.metrics.record(
                self.chat_id,
                time.time() - started,
                self.muffin.last_provider,
                self.muffin.last_fallbacks,
                self.muffin.last_style_issues,
                " ".join(parts),
            )
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
        started = time.time()
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

        if self.metrics:
            self.metrics.record(
                self.chat_id,
                time.time() - started,
                self.muffin.last_provider,
                self.muffin.last_fallbacks,
                self.muffin.last_style_issues,
                " ".join(parts),
            )
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
    metrics: Metrics | None = None,
) -> "ChatWorker":
    worker = workers.get(chat_id)
    if worker is None or not worker.is_alive():
        worker = ChatWorker(
            tg, muffin, chat_id, registry, feedback=feedback, metrics=metrics
        )
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
    metrics: Metrics | None = None,
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

    if action in {"regen", "bad"}:
        if action == "bad":
            feedback.log(record, "bad", "rejected")
            tg.answer_callback(callback_id, "Bad")
        else:
            tg.answer_callback(callback_id, "Again")
        worker = get_worker(
            tg, muffin, feedback, workers, registry, chat_id, metrics
        )
        worker.submit_regen(record)
        return

    tg.answer_callback(callback_id)
