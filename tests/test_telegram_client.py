import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from telegram_client import Telegram


class Handler(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict]] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append((self.path, body))
        result = {"message_id": 77} if self.path.endswith("sendMessage") else True
        payload = json.dumps({"ok": True, "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture()
def api():
    Handler.calls = []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_telegram_client_wire_format(api):
    tg = Telegram("test-token", base_url=api)
    markup = {"inline_keyboard": [[{"text": "OK", "callback_data": "fb:ok:1"}]]}

    assert tg.send(1, "hi", markup=markup) == 77
    tg.answer_callback("cb1", "Saved")
    assert tg.edit_text(1, 77, "new text", markup) is True
    tg.delete(1, 77)

    methods = [path.rsplit("/", 1)[-1] for path, _body in Handler.calls]
    assert methods == [
        "sendMessage",
        "answerCallbackQuery",
        "editMessageText",
        "deleteMessage",
    ]
    assert Handler.calls[0][1]["reply_markup"] == markup
    assert Handler.calls[1][1]["callback_query_id"] == "cb1"
    assert Handler.calls[2][1]["message_id"] == 77
    assert Handler.calls[3][1]["message_id"] == 77
