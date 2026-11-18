"""Minimal Telegram Bot API client used by the app and tests."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

log = logging.getLogger("muffin")


class Telegram:
    def __init__(self, token: str, base_url: str | None = None):
        self.token = token
        self.base = (base_url or f"https://api.telegram.org/bot{token}").rstrip("/")
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
