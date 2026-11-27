"""Regression tests for the live-conversation failures of Nov 26-27.

These are properties, not exact strings: the bot may pick any real line, but it
must not trap itself in a grudge, ignore a content message, or keep texting.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from history import HistoryStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Chat, EXPLAIN_LINES, Muffin, load_config  # noqa: E402
from proactive import ChatRegistry, Proactive  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadStore  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()


def make_muffin(tmp_path, retriever) -> Muffin:
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
    )


def test_why_cold_gets_an_explanation(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    muffin.mood.hurt(1, "ignored", 35)

    parts = muffin.reply(1, "Hoy ey oun")

    assert parts[0] in EXPLAIN_LINES
    assert muffin.mood.snapshot(1).comfort_needed < 35


def test_two_cold_lines_then_a_real_explanation(tmp_path, retriever, monkeypatch):
    muffin = make_muffin(tmp_path, retriever)
    chat = muffin.chats.setdefault(1, Chat())
    chat.history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "Hoy", "kind": "reply"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "Bong?", "kind": "reply"},
    ]
    muffin.mood.hurt(1, "ignored", 35)
    monkeypatch.setattr("muffin.random.random", lambda: 0.0)

    parts = muffin.reply(1, "abc")

    assert parts[0] in EXPLAIN_LINES
    assert muffin._cold_streak(chat) == 0


def test_engaged_reply_and_apology_clear_the_grudge(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    muffin.mood.hurt(1, "ignored", 35)

    muffin.reply(1, "sorry bong")
    assert muffin.mood.snapshot(1).comfort_needed < 35

    muffin.reply(1, "bong tv na")
    state = muffin.mood.snapshot(1)
    assert state.comfort_needed == 0
    assert state.upset == ""


def test_note_correction_keeps_its_note_in_history(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    chat = muffin.chats.setdefault(1, Chat())
    chat.history = [{"role": "assistant", "content": "bad", "ts": 1.0}]

    muffin.apply_correction(1, "Ort", note="she should be angry")

    assert chat.history[-1]["content"] == "Ort"
    assert chat.history[-1]["kind"] == "note"
    assert chat.history[-1]["note"] == "she should be angry"
    assert muffin._clean_history(chat.history)[-1]["note"] == "she should be angry"


def test_proactive_and_ritual_are_tagged(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)

    muffin.remember_assistant(1, "Morning?", kind="ritual")
    muffin.remember_assistant(1, "Bong?", kind="proactive")

    kinds = [m.get("kind") for m in muffin.chats[1].history]
    assert kinds == ["ritual", "proactive"]


def test_one_empty_ping_per_day(tmp_path):
    registry = ChatRegistry(tmp_path / "chats.json", 2.0, 4.0)
    registry.seed(1)

    registry.mark_proactive(1, "Bong?")
    assert not registry.allow_generic(1)

    registry.mark_proactive(1, "Bong tv na?")
    assert registry.generic_count(1) == 1


def test_ritual_stops_after_two_unanswered(tmp_path, monkeypatch):
    class FakeMood:
        upset = ""

    class FakeMoodStore:
        def snapshot(self, chat_id):
            return FakeMood()

    class FakeReactions:
        def proactive(self, chat_id):
            return None

        def note_proactive(self, chat_id, **kwargs):
            pass

    class FakeMuffin:
        mood = FakeMoodStore()
        reactions = FakeReactions()

        def ritual(self, chat_id):
            return "Bong?"

        def proactive(self, chat_id):
            return ["Bong tv na?"]

    class FakeTG:
        def __init__(self):
            self.sent = []

        def typing(self, chat_id):
            pass

        def send(self, chat_id, text, markup=None):
            self.sent.append(text)

    tg = FakeTG()
    registry = ChatRegistry(tmp_path / "chats.json", 2.0, 4.0)
    registry.seed(1)
    registry.chats["1"].next_delay = 10**9
    proactive = Proactive(tg, FakeMuffin(), registry)
    monkeypatch.setattr(proactive, "_quiet", lambda: False)

    proactive.tick()
    proactive.tick()
    proactive.tick()

    assert tg.sent == ["Bong?", "Bong?"]


def test_grudge_needs_a_direct_question_and_three_hours(tmp_path):
    store = ReactionStore(tmp_path / "reactions.json")
    start = datetime(2026, 11, 17, 12, 0).timestamp()

    store.observe(1, "hi", now=start)
    store.note_proactive(1, now=start + 3600, kind="question")
    assert not store.observe(1, "hello", now=start + 2 * 3600).ignored

    store.observe(1, "hi", now=start + 3 * 3600)
    store.note_proactive(1, now=start + 4 * 3600, kind="question")
    assert store.observe(1, "hello", now=start + 7 * 3600).ignored
