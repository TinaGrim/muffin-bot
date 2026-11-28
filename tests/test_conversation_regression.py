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
from life import Activity as LifeActivity  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Chat, EXPLAIN_LINES, Muffin, load_config  # noqa: E402
from proactive import ChatRegistry, Proactive  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadState, ThreadStore  # noqa: E402
from turn import (  # noqa: E402
    ACT_ANSWER,
    ACT_CALLOUT,
    ACT_GREETING,
    ACT_META,
    ACT_QUESTION,
    classify,
    claim_allows_wake,
)


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


def test_turn_classification_keeps_vocatives_and_references():
    assert classify("Bart").act == ACT_GREETING
    assert classify("Ort").act == ACT_ANSWER
    assert classify("Ngerb nv").act == ACT_QUESTION

    study = classify("Bart ort rean ey phg")
    assert study.act == ACT_QUESTION
    assert study.refers_to("school")
    assert study.has_vocative

    meta = classify("jos min merl tv dg hz")
    assert meta.act == ACT_META

    correction = classify("Hark lerng lngeac hz")
    assert correction.act == ACT_CALLOUT
    assert correction.refers_to("time")


def test_sleep_claim_holds_until_morning():
    bed = datetime(2026, 11, 27, 22, 0)

    assert not claim_allows_wake("sleep", bed.timestamp(), datetime(2026, 11, 27, 22, 10))
    assert not claim_allows_wake("sleep", bed.timestamp(), datetime(2026, 11, 28, 3, 0))
    assert claim_allows_wake("sleep", bed.timestamp(), datetime(2026, 11, 28, 7, 30))
    assert claim_allows_wake("awake", bed.timestamp(), datetime(2026, 11, 27, 22, 10))


def test_going_to_sleep_is_not_reversed_ten_minutes_later(
    tmp_path, retriever, monkeypatch
):
    import turn

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 27, 22, 10)

    monkeypatch.setattr(turn, "datetime", FrozenDatetime)

    muffin = make_muffin(tmp_path, retriever)
    night = FrozenDatetime.now()
    muffin.threads.chats["1"] = ThreadState(
        last_bot_claim="sleep", last_bot_claim_ts=night.timestamp()
    )
    now = night.timestamp()
    muffin.life.chats["1"] = LifeActivity(key="night", started=now, ends=now + 3600)

    parts = muffin.reply(1, "Ngerb nv bong? Men ngerb ot teh?")
    assert parts[0] in {"Dek nv", "Ngerb ot rouch teh", "Ngerb ot rouch"}

    # The judge's 22:00 failure: "Lov pov nguy nv" is about being sleepy.
    assert classify("Lov pov nguy nv").refers_to("sleep")
    parts = muffin.reply(1, "Lov pov nguy nv")
    assert parts[0] in {"Dek nv", "Ngerb ot rouch teh", "Ngerb ot rouch"}


def test_ordinary_statement_gets_a_reaction_even_when_cold(
    tmp_path, retriever, monkeypatch
):
    muffin = make_muffin(tmp_path, retriever)
    muffin.mood.hurt(1, "ignored", 35)
    monkeypatch.setattr("muffin.random.random", lambda: 0.0)

    parts = muffin.reply(1, "oun ngerb hx lu 7-8")

    assert parts[0] in {"Ohh", "Hxhx", "Hmm", "Jah"}
    assert parts[0] not in {"Ort", "Ot", "No"}


def test_study_question_with_a_vocative_gets_a_study_answer(
    tmp_path, retriever
):
    muffin = make_muffin(tmp_path, retriever)
    now = datetime.now().timestamp()
    muffin.life.chats["1"] = LifeActivity(key="chill", started=now, ends=now + 3600)

    parts = muffin.reply(1, "Bart ort rean ey phg")

    assert parts[0] in {"Tvr lomhat nv", "Pg tvr ka", "Tvr lomhat hx"}


def test_only_real_goodnight_lines_create_a_sleep_claim():
    from threads import claim_from_reply

    assert claim_from_reply("Dek hz") == "sleep"
    assert claim_from_reply("Pg dek hz") == "sleep"
    assert claim_from_reply("Good night") == "sleep"
    # Lying down to scroll TikTok is not a promise to be asleep.
    assert claim_from_reply("Pg dek merl tiktok") == ""
    assert claim_from_reply("Dek hz?") == ""
    assert claim_from_reply("Dek hz\nBong?") == "sleep"


def test_context_shows_the_commitment_she_made(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    now = datetime.now().timestamp()
    muffin.threads.chats["1"] = ThreadState(
        last_bot_claim="sleep", last_bot_claim_ts=now
    )

    summary = muffin.context_summary(1)

    assert "going to sleep" in summary


def test_time_callout_without_food_still_gets_an_answer(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)

    parts = muffin.reply(1, "Lngeac hz")

    assert parts[0] in {"Hxhx oun kit", "Oun kit teh", "Hxhx"}


def test_cold_food_question_gets_a_real_answer(tmp_path, retriever, monkeypatch):
    muffin = make_muffin(tmp_path, retriever)
    muffin.mood.hurt(1, "ignored", 35)
    monkeypatch.setattr("muffin.random.random", lambda: 0.0)

    parts = muffin.reply(1, "ban ey nham bay nv jos")

    assert parts[0] in {"Nham bay muy chicken", "Bay cha", "Nham eggs"}
    assert "?" not in parts[0]


def test_cold_morning_greeting_says_morning_back(
    tmp_path, retriever, monkeypatch
):
    muffin = make_muffin(tmp_path, retriever)
    muffin.mood.hurt(1, "ignored", 35)
    monkeypatch.setattr("muffin.random.random", lambda: 0.0)

    parts = muffin.reply(1, "morning")

    assert parts[0] in {"Morning", "Morning bong", "Ngob"}


def test_reach_out_dedupe_ignores_punctuation(tmp_path):
    registry = ChatRegistry(tmp_path / "chats.json", 3.0, 5.0)
    registry.seed(1)

    registry.mark_proactive(1, "Bat tt hz men")

    assert registry.already_sent(1, "Bat tt hz men?")
    assert registry.already_sent(1, "bat tt hz men")
    assert not registry.already_sent(1, "Bong tv na?")


def test_no_grudge_text_after_she_brushed_him_off(tmp_path, retriever):
    muffin = make_muffin(tmp_path, retriever)
    chat = muffin.chats.setdefault(1, Chat())
    chat.history = [
        {"role": "user", "content": "Bart rvol nas jg mic"},
        {"role": "assistant", "content": "Ort mic teh\nRvol tv jg"},
    ]
    muffin.mood.hurt(1, "ignored", 35)

    parts = muffin.proactive(1)

    assert parts[0] in {"Bong tv na?", "Hmm", "Hoy"}
    assert parts[0] != "Bong ort reply oun"


def test_judge_html_escapes_the_transcript():
    from judge_conversation import render_html

    data = {
        "overall": 5,
        "summary": "summary",
        "turns": [{"n": 1, "score": 2, "issue": "bad"}],
        "top_fixes": ["fix <x>"],
    }
    messages = [
        {"role": "user", "content": "<b>hi</b>"},
        {"role": "assistant", "content": "Ort"},
    ]

    page = render_html(data, messages)

    assert "<b>hi</b>" not in page
    assert "&lt;b&gt;hi&lt;/b&gt;" in page
    assert "fix &lt;x&gt;" in page
    assert 'class="note"' in page
