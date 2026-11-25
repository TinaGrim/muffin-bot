"""Telegram command handling and the polling loop."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from feedback import FeedbackStore
from metrics import Metrics
from muffin import Muffin
from proactive import ChatRegistry, Proactive
from storage import atomic_write_text
from telegram_client import Telegram
from worker import ChatWorker, get_worker, handle_callback

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("muffin")


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
    min_hours = float(env.get("PROACTIVE_MIN_HOURS", 1.5))
    max_hours = float(env.get("PROACTIVE_MAX_HOURS", 3.0))
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
    metrics = Metrics(ROOT / "data" / "metrics.jsonl")

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
                    handle_callback(tg, muffin, feedback, workers, registry, callback, metrics)
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
                        tg, muffin, feedback, workers, registry, chat_id, metrics
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
                        "/train toggles the OK / Bad / Recommend buttons. "
                        "/metrics shows reply latency and fallback rate.",
                    )
                    continue
                if text.startswith("/train"):
                    on = feedback.toggle_training(chat["id"])
                    tg.send(
                        chat["id"],
                        "Train mode on: each reply has OK / Bad / Recommend."
                        if on
                        else "Train mode off",
                    )
                    continue
                if text.startswith("/metrics"):
                    summary = metrics.summary()
                    if not summary.get("count"):
                        tg.send(chat["id"], "No replies recorded yet")
                    else:
                        top = ", ".join(
                            f"{model} {count}"
                            for model, count in summary["providers"].most_common(3)
                        )
                        tg.send(
                            chat["id"],
                            f"{summary['count']} replies · "
                            f"p50 {summary['p50']:.1f}s · "
                            f"p95 {summary['p95']:.1f}s · "
                            f"fallback {summary['fallback_rate'] * 100:.0f}% · "
                            f"style {summary['style_fixes']}\n{top}",
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
                    worker = ChatWorker(
                        tg, muffin, chat_id, registry, feedback=feedback, metrics=metrics
                    )
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
