import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feedback import Corrections, FeedbackStore, SentReply  # noqa: E402
from history import HistoryStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Muffin, load_config  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadStore  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()


def make_muffin(tmp_path, retriever, corrections=None) -> Muffin:
    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    return Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
        corrections,
    )


def test_feedback_store_roundtrip(tmp_path):
    store = FeedbackStore(tmp_path, default_training=True)
    record = SentReply(
        id="abc123",
        chat_id=1,
        user="Pong tvr ey",
        parts=["Pg dek merl tiktok"],
        prior=[],
        history_before=[{"role": "user", "content": "Pong tvr ey"}],
    )
    store.add(record)
    loaded = store.get("abc123")
    assert loaded is not None
    assert loaded.reply == "Pg dek merl tiktok"

    assert store.training_on(1) is True
    assert store.toggle_training(1) is False
    assert FeedbackStore(tmp_path).training_on(1) is False

    store.set_pending(1, "abc123")
    assert store.take_pending(1) == "abc123"
    assert store.take_pending(1) is None

    fixed = store.outcome("abc123", "fixed", "Pg nham bay hx")
    assert fixed is not None and fixed.status == "fixed"
    assert fixed.corrected == "Pg nham bay hx"
    assert (tmp_path / "feedback.jsonl").exists()


def test_correction_wins_and_persists(tmp_path, retriever):
    path = tmp_path / "corrections.json"
    corrections = Corrections(path)
    muffin = make_muffin(tmp_path, retriever, corrections)

    corrections.set("Pong tvr ey", "Pg nham bay hx")

    # The doing-question handler would normally answer from the schedule.
    assert muffin.reply(1, "Pong tvr ey") == ["Pg nham bay hx"]
    # Reloading the bot keeps Bong's line.
    assert Corrections(path).get("Pong tvr ey") == "Pg nham bay hx"


def test_apply_correction_replaces_history(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    muffin.reply(1, "Pong tvr ey")
    muffin.apply_correction(1, "Pg nham bay hx")
    last = [
        message["content"]
        for message in muffin.chats[1].history
        if message["role"] == "assistant"
    ][-1]
    assert last == "Pg nham bay hx"
    assert muffin.history.chats["1"][-1]["content"] == "Pg nham bay hx"


def test_regenerate_avoids_the_rejected_line(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    first = muffin.reply(1, "Pong tvr ey")
    second = muffin.reply(1, "Pong tvr ey", avoid=first)
    assert second != first


class FakeTelegram:
    def __init__(self):
        self.edits = []
        self.sent = []
        self.deleted = []
        self.answers = []

    def typing(self, chat_id):
        pass

    def send(self, chat_id, text, reply_to=None, markup=None):
        self.sent.append((text, markup))
        return 900 + len(self.sent)

    def send_sticker(self, chat_id, path):
        pass

    def edit_text(self, chat_id, message_id, text, markup=None):
        self.edits.append((message_id, text, markup))
        return True

    def delete(self, chat_id, message_id):
        self.deleted.append(message_id)

    def edit_markup(self, chat_id, message_id, markup):
        pass

    def answer_callback(self, callback_id, text=""):
        self.answers.append((callback_id, text))


def test_note_rewrites_in_place(tmp_path, retriever, monkeypatch):
    from bot import ChatWorker

    muffin = make_muffin(tmp_path, retriever)
    feedback = FeedbackStore(tmp_path)
    record = SentReply(
        id="rid",
        chat_id=1,
        user="Pong tvr ey",
        parts=["Pg nv pteas"],
        prior=[],
        history_before=[],
        message_id=77,
        message_ids=[77],
    )
    feedback.add(record)
    monkeypatch.setattr(
        muffin,
        "rewrite_with_note",
        lambda chat_id, text, note, history, rejected: ["Pg dek merl tiktok"],
    )
    tg = FakeTelegram()
    worker = ChatWorker(tg, muffin, 1, feedback=feedback)
    worker._note(record, "she is at school now")

    assert tg.edits[0][0] == 77
    assert tg.edits[0][1] == "Pg dek merl tiktok"
    assert tg.deleted == []
    saved = feedback.get("rid")
    assert saved is not None and saved.status == "fixed"
    assert saved.note == "she is at school now"
    # A note is context, not a permanent override for that message.
    assert muffin.corrections.get("Pong tvr ey") is None


def test_note_prompt_carries_note_and_rejected_line(tmp_path, retriever):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured["system"] = body["messages"][0]["content"]
            payload = json.dumps(
                {"choices": [{"message": {"content": "Pg dek merl tiktok"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 18090), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = load_config(
            {"LLM_BASE_URL": "http://127.0.0.1:18090/v1", "LLM_MODEL": "mock"},
            ROOT / "persona.md",
        )
        muffin = Muffin(
            config,
            retriever,
            MoodStore(tmp_path / "m.json"),
            LifeStore(tmp_path / "l.json"),
            HistoryStore(tmp_path / "h.json"),
            ReactionStore(tmp_path / "r.json"),
            ThreadStore(tmp_path / "t.json"),
        )
        parts = muffin.rewrite_with_note(
            1, "Pong tvr ey", "she is at school now", [], "Pg nv pteas"
        )
    finally:
        server.shutdown()
        server.server_close()

    assert parts == ["Pg dek merl tiktok"]
    assert "she is at school now" in captured["system"]
    assert "Pg nv pteas" in captured["system"]
    assert 1 not in muffin.chats  # rewrites do not touch live history


def test_error_reply_gets_review_keyboard(tmp_path, retriever, monkeypatch):
    from bot import ChatWorker

    muffin = make_muffin(tmp_path, retriever)
    monkeypatch.setattr(muffin, "human_delay", lambda chat_id: 0.0)

    def boom(*args, **kwargs):
        raise RuntimeError("pipeline broke")

    monkeypatch.setattr(muffin, "reply", boom)
    feedback = FeedbackStore(tmp_path)
    tg = FakeTelegram()
    worker = ChatWorker(tg, muffin, 1, feedback=feedback)
    worker._answer_inner(["Pong tvr ey"], None, False)

    text, markup = tg.sent[-1]
    assert text == "Ort"
    assert markup is not None
    assert any("fb:regen:" in button["callback_data"] for row in markup["inline_keyboard"] for button in row)
    record = next(iter(feedback.records.values()))
    assert record["parts"] == ["Ort"]


def test_regenerate_replaces_in_place(tmp_path, retriever, monkeypatch):
    from bot import ChatWorker

    muffin = make_muffin(tmp_path, retriever)
    feedback = FeedbackStore(tmp_path)
    record = SentReply(
        id="r1",
        chat_id=1,
        user="Pong tvr ey",
        parts=["Pg nv pteas"],
        prior=[],
        history_before=[],
        message_id=77,
        message_ids=[77],
    )
    feedback.add(record)
    seen = {}

    def fake_reply(chat_id, text, prior=(), avoid=()):
        seen["avoid"] = list(avoid)
        return ["Pg dek merl tiktok"]

    monkeypatch.setattr(muffin, "reply", fake_reply)
    monkeypatch.setattr(muffin, "restore_history", lambda chat_id, history: None)
    monkeypatch.setattr(muffin, "apply_correction", lambda chat_id, text: None)
    tg = FakeTelegram()
    worker = ChatWorker(tg, muffin, 1, feedback=feedback)
    worker._regenerate(record)

    assert tg.edits[0][0] == 77
    assert tg.edits[0][1] == "Pg dek merl tiktok"
    assert tg.deleted == []
    assert record.status == "regenerated"
    assert record.parts == ["Pg dek merl tiktok"]
    assert record.attempts == ["Pg nv pteas"]
    assert "Pg nv pteas" in seen["avoid"]


def test_style_guard_detects_and_fixes(tmp_path, retriever):
    from muffin import Chat, looks_english

    muffin = make_muffin(tmp_path, retriever)
    khmer = "Kjel tv \u1781\u17d2\u1798\u17c2\u179a"
    assert "khmer" in muffin.style_problems(khmer, "Pong tvr ey")
    assert "comma" in muffin.style_problems("Ort, nv teh", "Pong tvr ey")
    assert "english" in muffin.style_problems(
        "My favorite is maybe Spider-Man but I forget", "Pong tvr ey"
    )
    assert "foreign" in muffin.style_problems("Ng haii \u05d4\u05e0\u05d4", "Pong tvr ey")
    assert not looks_english("Mi hel maybe")
    assert looks_english("Good night my friend")

    fixed = muffin.style_fix(["Nv, Kjel tv na\u178e\u17b6\u179f\u17cb"], "Pong tvr ey")
    assert fixed == ["Nv Kjel tv na nas"]
    assert muffin.style_fix(["Ng haii \u05d4\u05e0\u05d4"], "Pong tvr ey") == ["Ng haii"]

    chat = Chat()
    muffin._remember(1, chat, "Pong tvr ey", "Nv, Kjel tv na\u178e\u17b6\u179f\u17cb")
    stored = chat.history[-1]["content"]
    assert "," not in stored
    assert not any("\u1780" <= c <= "\u17ff" for c in stored)


def test_persona_has_life_facts():
    persona = (ROOT / "persona.md").read_text(encoding="utf-8")
    assert "17:30" in persona
    assert "responsibility" in persona
    assert "ask him what he means" in persona


def test_handle_callback_ok_fix_and_too_old(tmp_path, retriever):
    from worker import handle_callback

    muffin = make_muffin(tmp_path, retriever)
    feedback = FeedbackStore(tmp_path)
    feedback.add(
        SentReply(
            id="r1",
            chat_id=1,
            user="Pong tvr ey",
            parts=["Pg nv pteas"],
            prior=[],
        )
    )
    tg = FakeTelegram()

    def callback(data):
        return {
            "id": "cb1",
            "data": data,
            "message": {"message_id": 77, "chat": {"id": 1}},
        }

    handle_callback(tg, muffin, feedback, {}, None, callback("fb:ok:r1"))
    assert feedback.get("r1").status == "ok"
    assert tg.answers[-1] == ("cb1", "Saved")

    handle_callback(tg, muffin, feedback, {}, None, callback("fb:fix:r1"))
    assert feedback.pending["1"] == "r1"
    assert "wrong" in tg.sent[-1][0].lower()

    handle_callback(tg, muffin, feedback, {}, None, callback("fb:ok:missing"))
    assert tg.answers[-1][1] == "Too old"


def test_handle_callback_regenerate_submits_job(tmp_path, retriever, monkeypatch):
    import worker
    from worker import handle_callback

    muffin = make_muffin(tmp_path, retriever)
    feedback = FeedbackStore(tmp_path)
    record = SentReply(
        id="r2", chat_id=1, user="Pong tvr ey", parts=["Pg nv pteas"], prior=[]
    )
    feedback.add(record)

    class FakeWorker:
        def __init__(self):
            self.record = None

        def submit_regen(self, submitted):
            self.record = submitted

    fake = FakeWorker()
    monkeypatch.setattr(worker, "get_worker", lambda *args, **kwargs: fake)
    tg = FakeTelegram()
    callback = {
        "id": "cb2",
        "data": "fb:regen:r2",
        "message": {"message_id": 77, "chat": {"id": 1}},
    }
    handle_callback(tg, muffin, feedback, {}, None, callback)
    assert fake.record is not None and fake.record.id == "r2"


def test_metrics_summary(tmp_path):
    from metrics import Metrics

    metrics = Metrics(tmp_path / "metrics.jsonl")
    for index, latency in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        metrics.record(
            1,
            latency,
            "primary" if index < 4 else "fallback-model",
            [] if index < 4 else ["fallback-model"],
            ["comma"] if index == 0 else [],
            "reply",
        )
    summary = metrics.summary()
    assert summary["count"] == 5
    assert summary["p50"] == 3.0
    assert abs(summary["fallback_rate"] - 0.2) < 1e-9
    assert summary["style_fixes"] == 1
    assert summary["providers"]["primary"] == 4


def test_relevance_junk_and_emotion_guards(tmp_path, retriever):
    from muffin import is_neutral_reply, junk_reply, note_emotion

    muffin = make_muffin(tmp_path, retriever)
    assert junk_reply("Bajjjj")
    assert not junk_reply("Bart")
    assert muffin.relevance_problems("Kiss kiss?", "Ort / Nham bay sin") == ["food"]
    assert muffin.relevance_problems("Nham bay nv", "Nham saj phat") == []
    assert muffin.relevance_problems("Pong tvr ey", "Pg nham bay") == []
    assert note_emotion("She should be angry to me") == "angry"
    assert note_emotion("be soft with him") == "soft"
    assert is_neutral_reply(["Hmm"])
    assert not is_neutral_reply(["Khg bong eg nv"])


def test_typo_message_asks_instead_of_guessing(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    assert muffin.reply(1, "Der kg jit nis ter") == ["Mean ey bong?"]
