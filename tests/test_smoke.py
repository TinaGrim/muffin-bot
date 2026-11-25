import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bot import allowed  # noqa: E402
from history import HistoryStore  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from threads import ThreadStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import (  # noqa: E402
    Chat,
    Memory,
    Muffin,
    clean_reply,
    is_stock_echo,
    load_config,
    only_echoes,
    split_messages,
)
from collections import Counter  # noqa: E402
from datetime import date  # noqa: E402
from reactions import ReactionState  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from rules import rule_reply  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system"
        assert any(m["role"] == "user" for m in body["messages"])
        payload = json.dumps(
            {"choices": [{"message": {"content": "Bart\nOrt"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def mock_llm():
    server = HTTPServer(("127.0.0.1", 18089), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:18089/v1"
    server.shutdown()
    server.server_close()


def test_retrieval_exact_match(retriever):
    hits = retriever.search("Good morning", k=5)
    assert hits[0]["score"] > 0.9
    assert hits[0]["reply"] in {"Morning", "Good bye"}


def test_clean_and_split():
    assert clean_reply('Muffin:  *Ort*  ') == "Ort"
    assert clean_reply("Bong: No probs") == "No probs"
    assert clean_reply("Meach yan mles") == "Mix yu mles"
    assert split_messages("Bart\nOrt\nTV na?") == ["Bart", "Ort", "TV na?"]
    long_text = "word " * 80
    assert split_messages(long_text) == [long_text]


def test_intent_rules():
    assert rule_reply("good morning bong") in {
        "Morning",
        "Morning bong",
        "Morning oun hz",
    }
    assert rule_reply("hello") == "Bart"
    assert rule_reply("Bye") in {"Bye", "Bye bong"}
    assert rule_reply("I'm sad today") == "Kom sad teh bong, oun nv nh"
    assert rule_reply("kiss kiss?") in {"Ort Ot jg", "Tic tt", "Ort"}
    assert rule_reply("Bong slnh oun?") in {
        "Oun sur bong jrern",
        "Slnh bong",
        "Kit bong",
    }
    assert rule_reply("sg ng ey ke pov") in {
        "Sg = sangsar bong",
        "Sg sangsar ng",
        "Sg = srey ng bong",
    }
    assert rule_reply("ort nk morning ban ort?") is None
    assert rule_reply("tnaii ng ban ey nham bay nv") in {
        "Nham bay prek nv",
        "Nham bay kteav",
        "Nham bay trey",
        "Maybe fried fish",
    }
    assert rule_reply("pong tvr ey vinh") in {
        "Pg dek merl tiktok",
        "Dek merl tiktok",
        "Pg scroll",
    }
    assert rule_reply("tv na?") in {"Nv ler", "Tv ler"}
    assert rule_reply("yepp") in {"Yep", "Yes", "Okay"}


def test_pipeline_with_mock_llm(retriever, mock_llm, tmp_path, monkeypatch):
    # Keep the test deterministic: no random morning-cold reaction.
    monkeypatch.setattr("muffin.random.random", lambda: 1.0)
    config = load_config(
        {
            "LLM_BASE_URL": mock_llm,
            "LLM_MODEL": "mock",
            "DIRECT_REPLY_THRESHOLD": "0",
        },
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    assert muffin.reply(1, "how was your weekend") == ["Bart", "Ort"]


def test_mood_moves(tmp_path):
    store = MoodStore(tmp_path / "mood.json")
    first = store.observe(1, "you are so cute")
    assert first.affection > 55
    second = store.observe(1, "stupid")
    assert second.patience < first.patience
    assert second.mood == "angry"
    # Sadness does not cancel an insult; the grudge waits for comfort.
    assert store.observe(1, "i feel lonely today").mood == "angry"
    soft = store.observe(2, "i feel lonely today")
    assert soft.mood == "soft"
    assert "gentle" in store.describe(soft).lower()
    assert (tmp_path / "mood.json").exists()


def test_anger_persists_until_pampered(tmp_path, monkeypatch):
    import time

    import mood as mood_mod

    store = MoodStore(tmp_path / "mood.json")
    store.observe(1, "you stupid")
    assert store.snapshot(1).mood == "angry"

    later = time.time() + 8 * 3600
    monkeypatch.setattr(mood_mod.time, "time", lambda: later)
    # Time alone never fixes it.
    assert store.observe(1, "hello").mood == "angry"
    # One sorry is not enough for a real insult.
    halfway = store.observe(1, "sorry bong")
    assert halfway.mood == "angry"
    assert halfway.comfort_needed > 0
    # Real pampering clears the grudge and softens her.
    cleared = store.observe(1, "bong slnh oun cute")
    assert cleared.upset == ""
    assert cleared.mood == "soft"


def test_being_ignored_starts_a_grudge(tmp_path):
    store = ReactionStore(tmp_path / "reactions.json")
    now = 1_000_000.0
    store.observe(1, "hi", now=now)
    store.note_proactive(1, now=now + 3600)
    reaction = store.observe(1, "hello", now=now + 4 * 3600)
    assert reaction.ignored
    assert store.cold_for(reaction) is not None


def test_life_activity(tmp_path):
    import time

    from life import Activity

    store = LifeStore(tmp_path / "life.json")
    store.chats["1"] = Activity(key="dinner", started=time.time(), ends=time.time() + 600)
    state = store.observe(1)
    assert "eating dinner" in store.describe(state)
    assert store.conflicts(state, "i am on vacation at the beach")
    assert store.conflicts(state, "i am sleeping now")
    assert not store.conflicts(state, "Nham bay nv")


def test_morning_ritual(tmp_path):
    from datetime import datetime

    store = ReactionStore(tmp_path / "reactions.json")
    morning = datetime(2026, 11, 17, 8, 0).timestamp()
    first = store.observe(1, "tv na?", now=morning)
    assert first.missed_morning
    assert store.cold_for(first)
    greeting = store.observe(1, "Morning oun", now=morning + 60)
    assert greeting.greeted
    assert not greeting.missed_morning

    store2 = ReactionStore(tmp_path / "reactions2.json")
    early = datetime(2026, 11, 17, 7, 30).timestamp()
    assert store2.proactive(1, now=early) in {"Morning?", "Bart?", "Morning bong?"}


def test_access_control():
    env = {"ALLOWED_USER_IDS": "42"}
    assert allowed(env, {"id": 1, "type": "private"}, 42, "hi", "muffin_bot")[0]
    assert not allowed(env, {"id": 1, "type": "private"}, 7, "hi", "muffin_bot")[0]

    ok, text = allowed({}, {"id": -10, "type": "group"}, 42, "hey @Muffin_bot", "Muffin_bot")
    assert ok and text == "hey"

    assert not allowed({}, {"id": -10, "type": "group"}, 42, "hey", "Muffin_bot")[0]


def test_partial_khmer_is_not_mangled():
    from muffin import normalize_script

    khmer = "ដល់ហើយ ត្រូវទឹកភ្លៀងទៀតហើយ"
    assert normalize_script(khmer, "Mean ban khoj na") == khmer
    assert normalize_script("ហើយ", "hello") == "hz"


def test_short_ambiguous_memory_is_not_replayed():
    from muffin import Memory

    pairs = [
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Ot nham teh", "context": []},
        {"user": "nham bay nv", "reply": "Nv teh", "context": []},
        {"user": "nham bay nv", "reply": "Hmm", "context": []},
    ]
    assert Memory(pairs).lookup("nham bay nv", "") is None

    stable = [
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Ot nham teh", "context": []},
    ]
    assert Memory(stable).lookup("nham bay nv", "") == "Nham hz"


def test_emoji_reply_is_not_deduplicated():
    from muffin import Chat

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()
    tmp = Path(tempfile.mkdtemp())
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp / "m.json"),
        LifeStore(tmp / "l.json"),
        HistoryStore(tmp / "h.json"),
        ReactionStore(tmp / "r.json"),
        ThreadStore(tmp / "t.json"),
    )
    assert muffin._dedupe(1, Chat(history=[]), "😒") == "😒"


def test_doing_question_uses_schedule(tmp_path, retriever):
    from life import Activity

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.reactions.chats["1"] = ReactionState(
        last_user_date=date.today().isoformat(),
        morning_greeted=date.today().isoformat(),
    )
    now = time.time()
    muffin.life.chats["1"] = Activity(key="chill", started=now, ends=now + 3600)

    first = muffin.reply(1, "Pong tvr ey")
    assert first[0] in {"Pg dek merl tiktok", "Dek merl tiktok", "Pg scroll"}
    # "Kuy nv" is a question ("are you sitting down?"), never an answer.
    assert first[0] != "Kuy nv"
    # The same question again must not reply with the exact same line.
    second = muffin.reply(1, "I mean what are you doing")
    assert second[0] in {"Just chilling", "Nothing much", "On my phone"}
    assert second[0] != first[0]

    # At home, "still not studying?" answers with what she is doing instead.
    assert muffin.reply(1, "Hak min men nv rean teh?")[0] in {
        "Tvr lomhat nv",
        "Pg tvr ka",
        "Tvr lomhat hx",
    }
    # "I thought you were at school" confirms she is really at home.
    assert muffin.reply(1, "Jos minh tha nv ptes")[0] == "Nv home"
    assert muffin.reply(1, "Jenh nv")[0] in {
        "Jenh mk nv",
        "Jenh mk",
        "Jenh mk hx",
    }
    not_study = muffin.reply(1, "Ort rean heh")
    assert not_study[0] in {"Tvr lomhat nv", "Pg tvr ka", "Tvr lomhat hx"}
    # Bong says he is eating, so she names her own food.
    assert muffin.reply(1, "Nham bay nv")[0] in {
        "Pg nham eggs",
        "Nham bay muy chicken",
        "Nham saj phat",
        "Pg nham bay",
    }


def test_time_of_day_disambiguation():
    from muffin import Memory

    pairs = [
        {
            "user": "na",
            "reply": "Nv sala",
            "date": "2024-12-23T08:00",
            "context": [],
        },
        {
            "user": "na",
            "reply": "Nv home",
            "date": "2024-12-23T20:00",
            "context": [],
        },
    ]
    memory = Memory(pairs)
    assert memory.lookup("na", "") is None  # no dominant answer
    assert memory.lookup_time("na", "school") == "Nv sala"
    assert memory.lookup_time("na", "dinner") == "Nv home"


def test_stock_glossary_echo_is_rejected():
    assert is_stock_echo("Ng haii ??")
    assert is_stock_echo("Lern mk tos.")
    assert is_stock_echo("Mix dea?")
    assert only_echoes("Ngaii pov?\nSlnh bong.")
    assert is_stock_echo("Mix dea bong?")
    assert is_stock_echo("Nv ler")
    assert is_stock_echo("Hurs?")
    assert not only_echoes("Ort")
    assert not is_stock_echo("Ort")
    assert not is_stock_echo("Nham bay nv")


def test_one_off_question_replays_but_one_word_does_not():
    pairs = [{"user": "Ey tne klas?", "reply": "Ort tang os", "context": []}]
    assert Memory(pairs).lookup("Ey tne klas?", "") == "Ort tang os"

    # A distinctive one-off line (even a short strange token) has one real
    # answer; a generic filler does not.
    short = [{"user": "Luy?", "reply": "Focus on good one", "context": []}]
    assert Memory(short).lookup("Luy?", "") == "Focus on good one"

    one_word = [{"user": "who", "reply": "Focus on good one", "context": []}]
    assert Memory(one_word).lookup("who", "") is None


def test_exact_question_answers_before_llm(tmp_path, retriever):
    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.reactions.chats["1"] = ReactionState(
        last_user_date=date.today().isoformat(),
        morning_greeted=date.today().isoformat(),
    )
    answer = "why not you blood pressure lerng dl 208 tea me nv ort pheii turs load dl 520 kor besdong bong tah mean oun"
    muffin.memory.by_user.clear()
    muffin.memory.by_user["maybe i see it"] = Counter({answer: 1})
    assert muffin.reply(1, "Maybe I see it?") == [answer]


def test_fallback_prefers_real_answer_over_activity_line(tmp_path, retriever):
    from life import Activity

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.memory.by_user.clear()
    answer = "why not you blood pressure lerng dl 208 tea me nv"
    muffin.memory.by_user["maybe i see it"] = Counter({answer: 1})
    assert muffin._safe_fallback(
        1, "Maybe I see it?", "", Activity(key="dinner", started=1, ends=2)
    ) == answer


def test_safe_fallback_never_crashes(tmp_path, retriever):
    from life import Activity

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    reply = muffin._safe_fallback(
        1, "unseen sentence", "", Activity(key="chill", started=1, ends=2)
    )
    assert reply


def test_burst_keeps_prior_messages(tmp_path, retriever):
    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.reply(1, "nham bay nv", prior=["tv school hz"])
    roles = [(m["role"], m["content"]) for m in muffin.chats[1].history]
    assert roles[0] == ("user", "tv school hz")
    assert roles[1] == ("user", "nham bay nv")
    assert roles[2][0] == "assistant"


def test_hurry_come_go_phrase():
    assert rule_reply("lern mk tos") in {"Tv tv", "Nv teh", "Where"}
    assert rule_reply("lern mk") in {"Tv tv", "Nv teh", "Where"}
    assert rule_reply("mk lern") in {"Tv tv", "Nv teh", "Where"}


def test_mles_intensifier():
    assert rule_reply("lern mles") in {"Jam tt", "Vinh tvr", "Ort hz"}
    assert rule_reply("mles lern") in {"Jam tt", "Vinh tvr", "Ort hz"}
    assert rule_reply("jrern mles") is None


def test_awake_question_uses_schedule(tmp_path, retriever):
    import time
    from datetime import date

    from life import Activity
    from reactions import ReactionState

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )

    def make(key: str) -> Muffin:
        muffin = Muffin(
            config,
            retriever,
            MoodStore(tmp_path / f"mood-{key}.json"),
            LifeStore(tmp_path / f"life-{key}.json"),
            HistoryStore(tmp_path / f"history-{key}.json"),
            ReactionStore(tmp_path / f"reactions-{key}.json"),
            ThreadStore(tmp_path / f"threads-{key}.json"),
        )
        now = time.time()
        muffin.life.chats["1"] = Activity(key=key, started=now, ends=now + 3600)
        muffin.reactions.chats["1"] = ReactionState(
            last_user_date=date.today().isoformat(),
            morning_greeted=date.today().isoformat(),
        )
        return muffin

    awake = {make("chill").reply(1, "Ngerb nv")[0] for _ in range(5)}
    asleep = {make("sleep").reply(1, "Ngerb nv")[0] for _ in range(5)}
    assert awake <= {"Ngerb hx", "Ngerb yu hx", "Hxhx"}
    assert asleep <= {"Ngerb ot rouch teh", "Ngerb ot rouch", "Ot", "Dek nv"}


def test_school_time_question(tmp_path, retriever):
    import time
    from datetime import date

    from life import Activity
    from reactions import ReactionState

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )

    def make(key: str) -> Muffin:
        muffin = Muffin(
            config,
            retriever,
            MoodStore(tmp_path / f"mood-s-{key}.json"),
            LifeStore(tmp_path / f"life-s-{key}.json"),
            HistoryStore(tmp_path / f"history-s-{key}.json"),
            ReactionStore(tmp_path / f"reactions-s-{key}.json"),
            ThreadStore(tmp_path / f"threads-s-{key}.json"),
        )
        now = time.time()
        muffin.life.chats["1"] = Activity(key=key, started=now, ends=now + 3600)
        muffin.reactions.chats["1"] = ReactionState(
            last_user_date=date.today().isoformat(),
            morning_greeted=date.today().isoformat(),
        )
        return muffin

    times = {make("breakfast").reply(1, "Tv sala kal")[0] for _ in range(6)}
    at_school = {make("school").reply(1, "Tv sala nv")[0] for _ in range(6)}
    assert times <= {"Mg 7", "7 jeang", "Tv 7h"}
    assert at_school <= {"Nv sala", "Rean nv", "Nv school"}


def test_system_prompt_carries_real_examples(tmp_path, retriever):
    """Few-shot examples come from the archive, not from invented lines."""
    from life import Activity
    from mood import Mood

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    prompt = muffin._system_prompt(
        "i miss you so much",
        has_history=False,
        mood=Mood(),
        life=Activity(key="chill", started=1, ends=2),
    )
    assert "These are real messages from the same conversation." in prompt
    assert "Bong: " in prompt
    assert "Muffin: " in prompt
    # The block belongs to this message, not to a fixed script.
    assert "How was your day kon pov" not in prompt


def test_fallback_uses_retrieval_before_activity_line(tmp_path, retriever, monkeypatch):
    """An unknown message gets the closest real reply, not a generic line."""
    from life import Activity

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.memory.by_user.clear()
    muffin.memory.by_context.clear()
    monkeypatch.setattr(muffin.life, "fallback", lambda _life: "ACTIVITY LINE")
    monkeypatch.setattr(
        retriever,
        "search",
        lambda text, k=5: [
            {"score": 0.81, "user": "i miss you", "reply": "Slnh bong", "date": None}
        ],
    )
    assert (
        muffin._safe_fallback(
            1, "zzqx frobnicate womp", "", Activity(key="chill", started=1, ends=2)
        )
        == "Slnh bong"
    )


def test_fallback_ignores_weak_retrieval_hits(tmp_path, retriever, monkeypatch):
    from life import Activity

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "none"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.memory.by_user.clear()
    muffin.memory.by_context.clear()
    monkeypatch.setattr(muffin.life, "fallback", lambda _life: "ACTIVITY LINE")
    monkeypatch.setattr(
        retriever,
        "search",
        lambda text, k=5: [
            {"score": 0.12, "user": "unrelated", "reply": "Slnh bong", "date": None}
        ],
    )
    assert (
        muffin._safe_fallback(
            1, "zzqx frobnicate womp", "", Activity(key="chill", started=1, ends=2)
        )
        == "ACTIVITY LINE"
    )


def test_fallback_chain_from_env():
    base = {
        "LLM_BASE_URL": "https://opencode.ai/inference/go/v1",
        "LLM_MODEL": "gemini-4-argon",
        "LLM_API_KEY": "go-key",
        "FALLBACK_BASE_URL": "https://opencode.ai/inference/go/v1",
        "FALLBACK_MODEL": "longcat-2.5-free",
        "FALLBACK_API_KEY": "go-key",
        "FALLBACK2_BASE_URL": "https://openrouter.ai/api/v1",
        "FALLBACK2_MODEL": "openrouter/free",
        "FALLBACK2_API_KEY": "",
        "FALLBACK3_BASE_URL": "http://localhost:11434/v1",
        "FALLBACK3_MODEL": "qwen2.5:3b-instruct",
        "FALLBACK3_API_KEY": "ollama",
    }
    # No OpenRouter key: that provider is skipped, the rest still works.
    config = load_config(dict(base), ROOT / "persona.md")
    assert [model for _base, _key, model in config.fallbacks] == [
        "longcat-2.5-free",
        "qwen2.5:3b-instruct",
    ]

    with_key = dict(base, OPENROUTER_API_KEY="or-key")
    config = load_config(with_key, ROOT / "persona.md")
    assert [model for _base, _key, model in config.fallbacks] == [
        "longcat-2.5-free",
        "openrouter/free",
        "qwen2.5:3b-instruct",
    ]


def test_retrieval_falls_back_when_embeddings_fail():
    from retrieval import ResilientRetriever

    class Down:
        pairs: list = []

        def search(self, query, k=8, min_score=0.0):
            raise RuntimeError("ollama is stopped")

    class Ngram:
        def search(self, query, k=8):
            return [{"score": 0.7, "user": "hi", "reply": "Bart", "date": None}]

    retriever = ResilientRetriever(Down(), Ngram())
    hits = retriever.search("hi", k=5)
    assert hits[0]["reply"] == "Bart"
    assert retriever.pairs == []


def test_memory_generic_replay_can_be_disabled():
    from muffin import Memory

    pairs = [
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
        {"user": "nham bay nv", "reply": "Nham hz", "context": []},
    ]
    memory = Memory(pairs)
    assert memory.lookup("nham bay nv", "") == "Nham hz"
    assert memory.lookup("nham bay nv", "", allow_generic=False) is None

    # A real (message, previous reply) pair is still safe in a live chat.
    contextual = Memory(
        [
            {
                "user": "nham bay nv",
                "reply": "Nham hz",
                "context": [{"reply": "Tv nham tv"}],
            }
        ]
    )
    assert (
        contextual.lookup("nham bay nv", "Tv nham tv", allow_generic=False)
        == "Nham hz"
    )


def test_ongoing_chat_uses_model_not_archive_replay(tmp_path, retriever, monkeypatch):
    from collections import Counter
    from datetime import date

    from muffin import Chat
    from reactions import ReactionState

    config = load_config(
        {"LLM_BASE_URL": "http://127.0.0.1:1/v1", "LLM_MODEL": "mock"},
        ROOT / "persona.md",
    )
    muffin = Muffin(
        config,
        retriever,
        MoodStore(tmp_path / "mood.json"),
        LifeStore(tmp_path / "life.json"),
        HistoryStore(tmp_path / "history.json"),
        ReactionStore(tmp_path / "reactions.json"),
        ThreadStore(tmp_path / "threads.json"),
    )
    muffin.reactions.chats["1"] = ReactionState(
        last_user_date=date.today().isoformat(),
        morning_greeted=date.today().isoformat(),
    )
    muffin.memory.by_user.clear()
    muffin.memory.by_user["maybe i see it"] = Counter(
        {"Archive answer from another chat": 1}
    )

    # No history: the one-off real answer is replayed.
    assert muffin.reply(1, "Maybe I see it?") == ["Archive answer from another chat"]

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return "Flow line"

    monkeypatch.setattr(muffin.llm, "chat", fake_chat)
    muffin.chats[1] = Chat(
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Bart"},
        ]
    )

    # With history: no unrelated archive line, the model continues the chat.
    assert muffin.reply(1, "Maybe I see it?") == ["Flow line"]
    assert calls
    flat = " ".join(message["content"] for message in calls[0])
    assert "hello" in flat and "Bart" in flat


def test_upset_grudge_triggers_a_nudge_after_an_hour(tmp_path):
    import time

    from proactive import ChatRegistry, Proactive

    class FakeMood:
        upset = "ignored"
        comfort_needed = 35

    class FakeMoodStore:
        def snapshot(self, chat_id):
            return FakeMood()

    class FakeReactions:
        def proactive(self, chat_id):
            return None

        def note_proactive(self, chat_id):
            pass

    class FakeMuffin:
        mood = FakeMoodStore()
        reactions = FakeReactions()

        def ritual(self, chat_id):
            return None

        def proactive(self, chat_id):
            return ["Bong?"]

    class FakeTG:
        def __init__(self):
            self.sent = []

        def typing(self, chat_id):
            pass

        def send(self, chat_id, text):
            self.sent.append((chat_id, text))

    registry = ChatRegistry(tmp_path / "chats.json", 1.5, 3.0)
    registry.seed(1)
    state = registry.chats["1"]
    state.last_user = state.last_bot = state.last_proactive = time.time() - 2 * 3600

    tg = FakeTG()
    proactive = Proactive(tg, FakeMuffin(), registry)
    proactive._quiet = lambda: False
    proactive.tick()

    assert tg.sent == [(1, "Bong?")]


def _tmp_muffin(tmp_path, retriever):
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


def test_fact_store_keeps_durable_context(tmp_path):
    from facts import FactStore

    store = FactStore(tmp_path / "facts.json")
    assert store.observe(1, "I have an exam on Friday") is not None
    assert store.observe(1, "what should I eat?") is None
    assert store.observe(1, "do you remember my exam?") is None
    assert store.observe(1, "my mom is sick") is not None
    lines = store.describe(1, "my exam is tomorrow")
    assert any("exam on Friday" in line for line in lines)


def test_prompt_injects_facts_and_thread(tmp_path, retriever):
    muffin = _tmp_muffin(tmp_path, retriever)
    muffin.facts.observe(1, "my exam is on Friday")
    muffin.threads.observe_user(1, "I have an exam tomorrow")
    prompt = muffin._system_prompt(
        "when is my exam",
        True,
        muffin.mood.snapshot(1),
        muffin.life.observe(1),
        chat_id=1,
    )
    assert "exam is on Friday" in prompt
    assert "topic is exam" in prompt


def test_history_budget_and_cleaning(tmp_path, retriever):
    muffin = _tmp_muffin(tmp_path, retriever)
    muffin.config.history_messages = 5
    muffin.config.history_chars = 10_000
    chat = muffin.chats.setdefault(1, Chat())
    for index in range(20):
        muffin._remember(1, chat, f"m{index}", "Ort")
    history = muffin._history(1)
    assert len(history) == 5
    assert history[-1]["content"] == "Ort"

    chat.history.append({"role": "assistant", "content": "Bitch Jg bork oun"})
    cleaned = muffin._history(1)
    assert all("Bitch" not in message["content"] for message in cleaned)

    muffin.config.history_chars = 30
    tiny = muffin._history(1)
    assert len(tiny) >= 1
    assert sum(len(m["content"]) for m in tiny) <= 60


def test_thread_describe(tmp_path):
    store = ThreadStore(tmp_path / "threads.json")
    store.observe_user(1, "I have an exam tomorrow")
    store.observe_reply(1, "x", "Ort?")
    text = store.describe(1)
    assert "exam" in text.lower()
    assert "you asked" in text


def test_context_summary_shows_real_memory(tmp_path, retriever):
    muffin = _tmp_muffin(tmp_path, retriever)
    muffin.facts.observe(1, "I have an exam on Friday")
    muffin.threads.observe_user(1, "I have an exam tomorrow")
    summary = muffin.context_summary(1)
    assert "exam" in summary.lower()
    assert "remember" in summary
    assert "No context yet" != summary
