"""Muffin persona pipeline: retrieval + OpenAI-compatible chat completion."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import httpx

from history import HistoryStore
from life import Activity, LifeStore
from mood import Mood, MoodStore, WHY_COLD
from reactions import Reaction, ReactionStore
from retrieval import Retriever, normalize
from threads import ThreadStore
from turn import ACT_CALLOUT, ACT_META, classify, claim_allows_wake
from facts import FactStore
from feedback import Corrections
from rules import intent_reply, override_reply

log = logging.getLogger("muffin")

PREFIXES = re.compile(r"^\s*(muffin|oun|bong)\s*:\s*", re.I)
MARKDOWN = re.compile(r"[*_`~]+")
EMOTIONAL = re.compile(r"\b(sad|cry|stress|hurt|tired|miss|love|sorry|scared)\b", re.I)
CODE_REQUEST = re.compile(
    r"\b(write|make|give|show|create|generate|sample|example)\b[^.?!\n]{0,50}"
    r"\b(code|c\+\+|cpp|python|javascript|js|java|html|css|sql|function|program|script)\b"
    r"|\b(code|c\+\+|cpp|python|javascript|js|java|html|css|sql)\b[^.?!\n]{0,30}"
    r"\b(for me|sample|example)\b",
    re.I,
)
ASSISTANT_INTRO = re.compile(
    r"^(sure|certainly|of course|here('s| is)|okay|no problem|absolutely)",
    re.I,
)
SCHOOL_QUESTION = re.compile(
    r"^\s*(oun\s+)?(tv|jg\s+tv)\s+sala\b"
    r"(?:\s+(kal|mong\s*marn|nv|hz|te|men|jg|oun|bong|brb|phg|ha|hy|jos|lg|das|tt|tvr|rean|sala|pon|man|marn))*"
    r"\s*[?.!]*$",
    re.I,
)
AWAKE_QUESTION = re.compile(
    r"^\s*(oun|bong)?\s*ngerb\b"
    r"(?:\s+(nv|hz|has|ha|haii|men|ban|yu|ruc|ort|ot|teh|jos|ng|pov|oun|bong|ah|la|der|lg))*"
    r"\s*[?.!]*$",
    re.I,
)
SLEEP_QUESTION = re.compile(
    r"\b(kg|keng|kern)\s*(lok|leak|lk)\b"
    r"|គេងលក់"
    r"|\b(are|r)\s+(you|u)\s+(a)?sleep(ing)?\b"
    r"|\basleep\b",
    re.I,
)
DOING_QUESTION = re.compile(
    r"^\s*(?:(?:pong|oun|bong|jg)\s+)?(?:tvr|tver|tv)\s+ey\b"
    r"|\bwhat\s+(?:are|r)\s+(?:you|u)\s+doing\b"
    r"|\bwhatcha\s+doing\b"
    r"|\bwhat\s+you\s+doing\b"
    r"|\bdoing\s+what\b",
    re.I,
)
ENGLISH_DOING = re.compile(
    r"\b(what\s+(?:are|r)\s+(?:you|u)|whatcha|what\s+you)\b", re.I
)
EATING_STATEMENT = re.compile(
    r"^\s*(?:oun\s+|bong\s+)?nham\s+(?:bay|ey|mi|ruc|hz|hx)\b", re.I
)
# Every line here is copied from the real chat, not invented. "Kuy nv" is a
# question ("are you sitting down?"), so it can never answer "what are you
# doing". Bong confirmed "Pg" = right now, so it can lead a live line.
DOING_KH = {
    "sleep": ["Dek hz", "Pg dek hz"],
    "breakfast": ["Pg nham koung", "Nham koung"],
    "school": ["Pg nv pteas", "Nv pteas"],
    "lunch": ["Pg nham bay", "Nham bay nv"],
    "chill": ["Pg dek merl tiktok", "Dek merl tiktok", "Pg scroll"],
    "work": ["Pg tvr code", "Pg nv pteas"],
    "commute": ["Pg tv pteas", "Nv ler"],
    "teach": ["Pg teach", "Pg bang"],
    "meeting": ["Pg nv meeting", "Meeting nv"],
    "dinner": ["Pg nham bay", "Nham bay nv"],
    "night": ["Dek hz", "Pg dek hz"],
    "out_food": ["Pg tv nham tv", "Nv psar"],
}
# She answers "I'm eating" by naming her own food, the way she does in the chat.
EATING_KH = [
    "Pg nham eggs",
    "Nham bay muy chicken",
    "Nham saj phat",
    "Pg nham bay",
]
# "Ort rean heh" asks why she is not studying, so answer with what she is
# actually doing: exercises and tasks, not exam prep.
NOT_STUDY_KH = ["Tvr lomhat nv", "Pg tvr ka", "Tvr lomhat hx"]
# "still not studying?" / "you're not studying yet?" — answered from schedule.
STUDY_STATUS = re.compile(
    r"\b(?:hak|min|ot|ort|mun)\b.{0,20}\brean\b"
    r"|\brean\s+(?:ot|ort|min)\b",
    re.I,
)
# "I thought you were at school" — confirm where she actually is.
THOUGHT_AT = re.compile(
    r"\b(?:jos|kit|tha)\b.{0,28}\b(?:nv|tv)\s+(?:ptes|pteas|sala)\b", re.I
)
DOING_EN = {
    "sleep": ["Sleeping", "In bed"],
    "breakfast": ["Eating breakfast", "Having breakfast"],
    "school": ["At school", "In class"],
    "lunch": ["Eating lunch", "Having lunch"],
    "chill": ["Just chilling", "Nothing much", "On my phone"],
    "work": ["Working", "On my laptop"],
    "commute": ["On the way to teach", "Heading to teach"],
    "teach": ["Teaching", "In class"],
    "meeting": ["In a meeting", "At a meeting"],
    "dinner": ["Eating dinner", "Having dinner"],
    "night": ["In bed", "About to sleep"],
    "out_food": ["Out buying food", "Getting food"],
}
HUNGER_YES = re.compile(
    r"^\s*(yes|yeah|yep|yup|ok(ay)?|alittle|a little|bit|hz|still|men)\b",
    re.I,
)
BANNED = re.compile(
    r"(let'?s do it|how can i help|what subject|as an ai|i'?m here to help|"
    r"is there anything else|right now it is|you are muffin|"
    r"i'?m sorry to hear|i hope you|let me know|happy to help|"
    r"here to listen|that sounds|is there anything)",
    re.I,
)
AFFIRM = re.compile(r"^\s*(yes|yepp?|yep|ok(ay)?|okiee?|hmm+|hxhx|haha+|lol|👍+|🙏+)\s*[.!]*\s*$", re.I)
YES = re.compile(r"^\s*(yes|yeah|yep|yup|ok(ay)?|okiee?|👍|🙏)\b", re.I)
NO = re.compile(r"^\s*(no|nope|nah|ort|ot)\b", re.I)
MORNING_GREETING = re.compile(r"^\s*(good\s*)?morning\b", re.I)
COLD_CONTEXT = re.compile(
    r"\b(morning|slnh|love|miss|srey|girl|jealous|bong|oun)\b", re.I
)
RETURN_QUESTION = re.compile(
    r"\b(tas pov|jos oun|jos bong|and you|what about you)\b", re.I
)
# One-off generic fillers: their single archived reply belongs to one
# conversation, so replaying it out of context is nonsense.
GENERIC_REPLAY_BLOCK = {
    "yes", "yeah", "yep", "no", "nope", "nah", "ok", "okay", "oki", "okei",
    "ort", "ot", "huh", "hurs", "mic", "hmm", "idk", "na", "who", "what",
    "why", "where", "when", "how", "hi", "hey", "hello", "bye", "gn",
    "now", "or", "i", "us", "up", "me", "pls", "plz", "haa", "hoh", "wow",
    "hug", "love", "meh", "sus", "now", "der", "tt", "pa", "me", "haii",
    "nuh", "nor", "due", "oy",
}
# Phrases that only answer one specific kind of message, or that the archive
# never uses at all. Small local models copy them from GLOSSARY or invent
# variants ("Ngaii?", "Mix dea bong?", "Hurs?"), so an LLM reply that is just
# one of these is treated as a non-answer. Rules and real-chat memory may still
# use them: there they have a trigger.
STOCK_ECHOES = {
    normalize(phrase)
    for phrase in (
        "Ng haii?",
        "Ngaii?",
        "Ngaii",
        "Ngaii pov?",
        "Ngaii bong?",
        "Slnh bong.",
        "Hurs?",
        "Lern mk tos.",
        "Lern mles.",
        "Pov jg tv lern mk tos.",
        "Mix dea?",
        "Mix dea bong?",
        "Mix dea hurs?",
        "Takmao mix dea?",
        "Jg mic haa kon pov?",
        "Jg mic haa kon bong?",
        "Nv ler",
        "Tv ler",
        "Slnh",
        "Tv tv",
        "Nv teh",
        "Where",
    )
}


def is_stock_echo(text: str) -> bool:
    return normalize(text) in STOCK_ECHOES


def only_echoes(text: str) -> bool:
    """True when a reply is just stock echoes plus filler.

    "Ngaii pov? | Slnh bong." looks like an answer but is two glossary lines
    pasted together; the archive never does that on its own.
    """
    chunks = [normalize(part) for part in text.split("\n") if part.strip()]
    if not chunks:
        return False
    return all(chunk in STOCK_ECHOES or not chunk for chunk in chunks)
GLOSSARY = (
    "Khmerlish: tngaii = day/today, nham bay = eat, pong/tvr ey = what doing, "
    "ort = no, ban = past, jg = want, tv = go, hz = already, nv = now/here, "
    "vinh = right now, bong = you, oun = me, pov = me/Muffin, slnh = love, "
    "kjel/kjil = lazy, men = like, hurs = huh, merl = look/see, hak sur = ask, "
    "jg mic haa kon pov = what are you gonna do, my little? (teasing), "
    "ng haii = what now? (aggressive), ort = don't want to, ot = no/not, "
    "mix dea = how was it?, smarn = guessing, "
    "ngerb = awake / wake up, "
    "lern = hurry, mk = come, tos = go, lern mk tos = hurry up come on let's go, "
    "mles = degree particle after a verb/adjective: so / too / a lot, "
    "jrern = many/much, jrern mles = too many, tha jrern = that much, "
    "lern mles = hurry up already / too slow, "
    "mix yu mles = why so slow / still not done?"
)

MOOD_STYLE = {
    "playful": (0.9, 24, 90),
    "warm": (0.8, 30, 120),
    "tired": (0.7, 16, 70),
    "sleepy": (0.7, 16, 70),
    "annoyed": (0.6, 14, 50),
    "angry": (0.6, 12, 55),
    "soft": (0.8, 40, 160),
    "neutral": (0.75, 24, 90),
}

# Real distribution from the archive: 50.9% one message, 28.5% two,
# 12.6% three. Mood shifts the odds.
BURST_PROBS = {
    "playful": (0.45, 0.36, 0.19),
    "warm": (0.50, 0.33, 0.17),
    "neutral": (0.51, 0.33, 0.16),
    "soft": (0.60, 0.28, 0.12),
    "tired": (0.75, 0.22, 0.03),
    "sleepy": (0.75, 0.22, 0.03),
    "annoyed": (0.85, 0.15, 0.00),
    "angry": (0.92, 0.08, 0.00),
}

# Short cold lines for a grudge. Time passing does not reset these; only an
# apology or pampering clears the mood.
UPSET_LINES = {
    "ignored": [
        "Ort",
        "Hmm",
        "Ort mean ey",
        "Kit bong",
        "Bong ort reply oun",
        "Ort kit oun teh?",
    ],
    "jealous": ["Ke ney?", "Srey ng?", "Ort", "Huh"],
    "insulted": ["Ort", "Huh", "Mic", "Kom near oun"],
}

# After two cold one-word replies she has to say what is actually wrong. A
# human girlfriend does not repeat "Hoy" forever.
EXPLAIN_LINES = [
    "Bong ort reply oun",
    "Bong tv na?",
    "Ort kit oun teh?",
    "Bong slnh oun teh?",
]
COLD_LINE = re.compile(
    r"^(hoy|bong\?|hmm|kit bong|ort reply teh\?|ort|ng)\s*[.!?]*$", re.I
)


def trim_repeats(text: str, keep: int = 3) -> str:
    """Cut degenerate loops like 'jg bong jg bong jg bong ...' from small models."""
    words = text.split()
    for n in (1, 2, 3):
        i = 0
        while i + n * (keep + 1) <= len(words):
            seq = words[i : i + n]
            count = 1
            j = i + n
            while j + n <= len(words) and words[j : j + n] == seq:
                count += 1
                j += n
            if count > keep:
                return " ".join(words[: i + n * keep]).rstrip(" ,;.")
            i += 1
    return text


@dataclass
class Config:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.8
    max_tokens: int = 120
    top_k: int = 8
    example_min_score: float = 0.35
    fallback_min_score: float = 0.5
    direct_threshold: float = 0.0
    timeout: float = 90.0
    max_reply_seconds: float = 15.0
    history_messages: int = 12
    history_chars: int = 1800
    history_keep: int = 200
    max_reply_chars: int = 160
    remote_min_tokens: int = 300
    # Ordered fallback chain: (base_url, api_key, model).
    fallbacks: list[tuple[str, str, str]] = field(default_factory=list)
    double_text_chance: float = 0.25
    sticker_chance: float = 0.10
    use_rules: bool = True
    use_intents: bool = False
    examples_mode: str = "always"  # always | first | never
    adapter_mode: bool = False
    temperature_override: float | None = None
    persona_path: Path = Path("persona.md")
    compact_persona_file: str = "persona_compact.md"


@dataclass
class Chat:
    history: list[dict] = field(default_factory=list)


def load_persona(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


REASONING_CUT = re.compile(
    r"(思考过程|推理过程|分析过程|分析：|思考：|chain\s*of\s*thought|"
    r"reasoning\s*:|let me think)",
    re.I,
)


def clean_reply(text: str) -> str:
    text = text.strip()
    # A provider can leak hidden reasoning. Drop it before anything else and
    # reject any foreign script (CJK, Arabic, ...) that slipped through.
    text = REASONING_CUT.split(text, maxsplit=1)[0].strip()
    if FOREIGN_SCRIPT.search(text):
        return ""
    text = PREFIXES.sub("", text)
    text = PREFIXES.sub("", text)
    text = MARKDOWN.sub("", text)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\bmeach yan mles\b", "Mix yu mles", text, flags=re.I)
    text = re.sub(r"\bOrth\b", "Ort", text)
    text = re.sub(r",\s*pov\b", ", bong", text, flags=re.I)
    text = re.sub(r"\bpov\b(?=\s*[?.!]?\s*$)", "bong", text, flags=re.I)
    return trim_repeats(text)


KHMER_SCRIPT = re.compile(r"[\u1780-\u17ff]")
FOREIGN_SCRIPT = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)
KHMER_TO_ROMAN = [
    ("ណាស់", "nas"),
    ("អត់", "ot"),
    ("ទេ", "teh"),
    ("ហើយ", "hz"),
    ("មែន", "men"),
    ("គេង", "keng"),
    ("ដេក", "dek"),
    ("ញ៉ាំ", "nham"),
    ("បាយ", "bay"),
    ("ស្រលាញ់", "slnh"),
    ("អូន", "oun"),
    ("បង", "bong"),
    ("គេ", "ke"),
    ("អី", "ey"),
    ("ម៉េច", "mic"),
    ("ណា", "na"),
    ("ចង់", "jg"),
    ("ទៅ", "tv"),
    ("តើ", "te"),
    ("ចុះ", "chuh"),
    ("នៅ", "nv"),
    ("មក", "mk"),
    ("បាន", "ban"),
    ("មិន", "min"),
    ("គ្នា", "knea"),
    ("នេះ", "nih"),
    ("នោះ", "noh"),
    ("ខ្ញុំ", "knhom"),
    ("សិន", "sen"),
    ("ហ៎", "hah"),
    ("អើ", "ah"),
    ("គឺ", "ki"),
    ("ព្រោះ", "prus"),
]


def normalize_script(text: str, user_text: str = "") -> str:
    """Keep Khmerlish replies in Khmerlish when Bong writes Khmerlish."""
    if KHMER_SCRIPT.search(user_text or ""):
        return text
    if not KHMER_SCRIPT.search(text):
        return text
    converted = text
    for khmer, roman in KHMER_TO_ROMAN:
        converted = converted.replace(khmer, f" {roman} ")
    converted = re.sub(r"[ \t]{2,}", " ", converted)
    converted = re.sub(r"\s+([,.!?])", r"\1", converted).strip()
    # The glossary only covers a few words. A half-converted line looks like
    # broken Khmerlish, so keep the real Khmer script instead of mangling it.
    if KHMER_SCRIPT.search(converted):
        return text.strip()
    return converted


CODE_FENCE = re.compile(r"```.*?```", re.S)

ENGLISH_FUNCTION = {
    "the", "is", "are", "was", "were", "you", "your", "i", "my", "me", "we",
    "they", "he", "she", "it", "and", "but", "or", "about", "think", "maybe",
    "forget", "remember", "good", "night", "morning", "love", "miss", "what",
    "doing", "just", "chilling", "here", "there", "this", "that", "for",
    "with", "from", "have", "has", "had", "do", "does", "did", "can", "will",
    "would", "should", "not", "no", "yes", "answer", "question", "because",
    "really", "very", "much", "like", "want", "need", "know", "feel",
}

KHMERLISH_MARKERS = {
    "nv", "tvr", "bong", "oun", "jg", "hz", "nham", "dek", "ot", "teh",
    "ter", "men", "na", "ey", "tv", "mk", "pteas", "prean", "kjel", "jlom",
    "der", "lg", "srey", "luy", "ort", "ban", "min", "nas", "vinh", "hx",
    "ah", "oh", "ming", "hah", "bay", "tirk", "sala", "rean", "chill",
    "mic", "kert", "smarn", "tngaii", "tngai", "pel", "mok", "ler", "kroy",
    "khae", "srol", "knea", "knhom", "pov",
}


def strip_code(text: str) -> str:
    return CODE_FENCE.sub(" ", text or "")


def looks_english(text: str) -> bool:
    """True only for a real English sentence, not romanized Khmer."""
    words = re.findall(r"[a-z']+", strip_code(text).lower())
    if len(words) < 3:
        return False
    if any(word in KHMERLISH_MARKERS for word in words):
        return False
    hits = sum(1 for word in words if word in ENGLISH_FUNCTION)
    return hits >= 3 and hits / len(words) >= 0.45


FOOD_WORDS = re.compile(
    r"\b(nham|bay|prek|kteav|trey|saj|phat|eggs?|chicken|dinner|lunch|"
    r"breakfast|eat|eating|food|hungry|nguy)\b",
    re.I,
)
FOOD_ASK = re.compile(
    r"\b(nham|bay|eat|eating|food|hungry|nguy|prek|kteav|trey|"
    r"breakfast|lunch|dinner|tirk)\b",
    re.I,
)
DOING_ASK = re.compile(
    r"(pong|bong|oun)\s*tvr\s*ey|what are you doing|tvr ey|tvr ey\b", re.I
)
TYPO_QUESTION = re.compile(r"\bder\s+(kg|lg)\b|\bkg\s+ler\b", re.I)
MEAL_WORD = re.compile(r"\b(nham|bay|prek|breakfast|morning)\b", re.I)
# Lines that end the exchange by sending him away. Complaining "you did not
# reply" right after this is what made the reach-out feel unfair.
BRUSH_OFF = re.compile(
    r"^(?:rvol|rean|tv|jenh)\s+tv(?:\s+jg)?$|^tv\s+rean(?:\s+tv)?$|"
    r"^ort\s+mean\s+ey$",
    re.I,
)


# Insults that are only funny in their original fight, never as a random
# replay to a greeting or a normal question.
PROFANE = re.compile(
    r"\b(bitch|bitches|slut|whore|fuck(?:ing|ed)?|shit|dick|cunt|"
    r"bastard|asshole|retard(?:ed)?)\b",
    re.I,
)

# Mood-consistency guards: angry must not turn affectionate/romantic
AFFECTIONATE = re.compile(
    r"\b(love|miss|slnh|kiss|hug|cuddle|pamper|sweet|cutie|baby|dear|honey)\b",
    re.I,
)
ROMANTIC = re.compile(
    r"\b(kiss|make\s+out|hard\s+and\s+burning|passion|intimate|romantic)\b",
    re.I,
)
COLD_WORDS = re.compile(
    r"\b(ort|ot|hoy|bong\?|ignore|go\s+away|leave\s+me|fuck\s+off)\b",
    re.I,
)


def too_long_for_chat(text: str, max_words: int = 12) -> bool:
    """A chat line this long is a drift, not a reply."""
    return len((text or "").split()) > max_words


def junk_reply(text: str) -> bool:
    """One-off noise that should never be replayed as a chat line."""
    text = (text or "").strip()
    if not text:
        return True
    if re.search(r"(.)\1{3,}", text):  # Bajjjj, nooooo
        return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    if PROFANE.search(text):
        return True
    return False


def note_emotion(note: str) -> str:
    lowered = (note or "").lower()
    for word, emotion in (
        ("angry", "angry"),
        ("mad", "angry"),
        ("annoyed", "annoyed"),
        ("upset", "upset"),
        ("sad", "sad"),
        ("soft", "soft"),
        ("gentle", "gentle"),
        ("jealous", "jealous"),
    ):
        if word in lowered:
            return emotion
    return ""


def is_neutral_reply(parts: list[str]) -> bool:
    text = " ".join(parts).strip().lower()
    return text in {"hmm", "hm", "ort", "ot", "okay", "ok", "mic", "no", "teh"}


def drop_commas(text: str) -> str:
    """Remove commas from chat lines while leaving fenced code untouched."""
    if "," not in (text or ""):
        return text
    pieces = re.split(r"(```.*?```)", text or "", flags=re.S)
    cleaned: list[str] = []
    for piece in pieces:
        if piece.startswith("```"):
            cleaned.append(piece)
            continue
        piece = re.sub(r"\s*,\s*", " ", piece)
        piece = re.sub(r"[ \t]{2,}", " ", piece)
        cleaned.append(piece.strip())
    return "\n".join(part for part in cleaned if part).strip()


def split_messages(
    text: str,
    user_text: str = "",
    mood: str = "neutral",
    generated: bool = False,
) -> list[str]:
    """Split at real message boundaries, not arbitrary commas.

    Newlines are real bursts (from memory or the model). A generated single
    block is split only at sentence boundaries, and only as often as the real
    archive does: ~51% one, ~29% two, ~13% three.
    """
    text = normalize_script(text, user_text)
    if not looks_english(user_text):
        text = drop_commas(text)
    if "```" in text:
        return [text]
    if "\n" not in text and text.count(",") == 1 and len(text) <= 90:
        left, right = (part.strip() for part in text.split(",", 1))
        if (
            left
            and right
            and len(left) <= 50
            and len(right) <= 50
            and not right[0].isdigit()
            and not KHMER_SCRIPT.search(user_text or "")
        ):
            return [left, right]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and len(lines) <= 4 and all(len(line) <= 160 for line in lines):
        return lines
    if generated and len(text) >= 15:
        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
        if 2 <= len(parts) <= 3:
            weights = BURST_PROBS.get(mood, BURST_PROBS["neutral"])
            count = random.choices([1, 2, 3], weights=weights, k=1)[0]
            if count >= 2:
                if count == 2:
                    return [parts[0], " ".join(parts[1:])]
                return parts[:3]
    return [text]


class LLM:
    def __init__(self, config: Config):
        self.config = config
        self._cooldown_until = 0.0
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "User-Agent": "muffin-bot/1.0",
            },
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stop: list[str] | None = None,
        timeout: float | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if stop is None:
            payload["stop"] = ["\n\n", "Bong:", "Bong："]
        elif stop:
            payload["stop"] = stop
        if self._cooldown_until > time.time():
            raise RuntimeError("primary LLM cooling down after 429")
        headers = {"x-opencode-session": session_id} if session_id else None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout or self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
            except httpx.ReadTimeout as exc:
                last_error = exc
                log.warning("LLM timed out after %.0fs", self.config.timeout)
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                body = exc.response.text[:300].replace("\n", " ")
                log.warning(
                    "LLM HTTP %s: %s", exc.response.status_code, body
                )
                if exc.response.status_code == 429:
                    self._cooldown_until = time.time() + 900
                    break
            except Exception as exc:  # noqa: BLE001 - report a useful message
                last_error = exc
                log.warning("LLM attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"LLM request failed: {last_error}")


def clip(text: str, limit: int = 240) -> str:
    text = text.replace("\n", " / ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Memory:
    """Exact replay of the real chat, preferring the same preceding reply."""

    def __init__(self, pairs: list[dict]):
        self.by_user: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_context: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.by_user_time: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for pair in pairs:
            user = normalize(pair["user"])
            self.by_user[user][pair["reply"]] += 1
            context = pair.get("context") or []
            previous = context[-1]["reply"] if context else ""
            self.by_context[(user, normalize(previous))][pair["reply"]] += 1
            # Same short line, different answer at school vs at dinner: the
            # real timestamp is a usable signal when there is no chat context.
            try:
                when = datetime.fromisoformat(pair["date"])
                key = LifeStore._scheduled(None, when).key
                self.by_user_time[(user, key)][pair["reply"]] += 1
            except (KeyError, ValueError, TypeError):
                pass

    def lookup_time(self, text: str, activity: str) -> str | None:
        """Disambiguate a short line using when Bong said it in the archive."""
        user = normalize(text)
        pool = self.by_user_time.get((user, activity))
        if not pool:
            return None
        for reply, _ in pool.most_common(1):
            if (
                reply
                and len(reply) <= 240
                and "http" not in reply.lower()
                and not junk_reply(reply)
            ):
                return reply
        return None

    def lookup_context(self, text: str, last_assistant: str) -> str | None:
        user = normalize(text)
        if not user or not last_assistant.strip():
            return None
        pool = self.by_context.get((user, normalize(last_assistant)))
        if not pool:
            return None
        for reply, _ in pool.most_common(5):
            if junk_reply(reply):
                continue
            return reply
        return None

    def lookup(
        self,
        text: str,
        last_assistant: str,
        max_len: int = 160,
        allow_generic: bool = True,
    ) -> str | None:
        """Replay a real reply only when the match is trustworthy.

        An exact (message, previous reply) match is always safe. Without that
        context a short phrase such as "nham bay nv" appears in many different
        conversations with unrelated answers, so replaying the most common one
        produces nonsense. In that case return None and let rules or the model
        answer instead.

        ``allow_generic=False`` is used once a conversation is already running:
        only the same (message, previous reply) pair may replay, otherwise the
        model continues with the real history instead of dropping in a line
        from an unrelated archive conversation.
        """
        user = normalize(text)
        if not user:
            return None
        context_reply = self.lookup_context(text, last_assistant)
        if context_reply:
            return context_reply
        if not allow_generic:
            return None
        pool = self.by_user.get(user)
        if not pool:
            return None
        total = sum(pool.values())
        words = len(re.findall(r"[A-Za-z]+", user))
        khmer = len(re.findall(r"[\u1780-\u17ff]", user))
        top_reply, top_count = pool.most_common(1)[0]
        # A short multi-sample line ("nham bay nv", "na") has no stable real
        # answer: the archive's most common reply is tied to one conversation
        # and replaying it produces nonsense. Only replay when it dominates.
        if total > 1 and words <= 3 and khmer < 4 and top_count / total <= 0.5:
            return None
        # A one-off generic filler seen once ("yes", "huh", "ort") still has no
        # stable answer. Strange one-off tokens ("qr", "vd", "pto") do: that is
        # the exact thing Bong said, and there is only one real reply to it.
        if total == 1 and (
            user in GENERIC_REPLAY_BLOCK
            or (words < 2 and khmer < 4 and len(user) < 2)
        ):
            return None
        for reply, _ in pool.most_common(5):
            if "http" in reply.lower() or len(reply) > max_len:
                continue
            if "nheb tinarith" in reply.lower() or junk_reply(reply):
                continue
            return reply
        return None


class Muffin:
    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        mood_store: MoodStore | None = None,
        life_store: LifeStore | None = None,
        history_store: HistoryStore | None = None,
        reaction_store: ReactionStore | None = None,
        thread_store: ThreadStore | None = None,
        corrections: Corrections | None = None,
        facts_store: FactStore | None = None,
    ):
        self.config = config
        self.retriever = retriever
        self.persona = load_persona(config.persona_path)
        compact = config.persona_path.with_name(config.compact_persona_file)
        self.compact_persona = (
            compact.read_text(encoding="utf-8").strip()
            if compact.exists()
            else self.persona
        )
        self.llm = LLM(config)
        self.memory = Memory(retriever.pairs)
        log.info("primary model: %s (%s)", config.model, config.base_url)
        self.fallbacks = [
            LLM(replace(config, base_url=base, api_key=key, model=model))
            for base, key, model in config.fallbacks
        ]
        if self.fallbacks:
            log.info(
                "fallback chain: %s",
                " -> ".join(llm.config.model for llm in self.fallbacks),
            )
        self.mood = mood_store or MoodStore(Path("data") / "mood.json")
        self.life = life_store or LifeStore(Path("data") / "life.json")
        self.history = history_store or HistoryStore(Path("data") / "history.json")
        self.facts = facts_store or FactStore(self.mood.path.parent / "facts.json")
        self.reactions = reaction_store or ReactionStore(Path("data") / "reactions.json")
        self.threads = thread_store or ThreadStore(Path("data") / "threads.json")
        self.corrections = corrections or Corrections()
        self.last_provider = config.model
        self.last_fallbacks: list[str] = []
        self.last_style_issues: list[str] = []
        self._deadline = 0.0
        self.chats: dict[int, Chat] = {}
        for chat_id, messages in self.history.chats.items():
            try:
                chat = Chat(history=list(messages))
            except ValueError:
                continue
            cleaned = self._clean_history(chat.history)
            if len(cleaned) != len(chat.history):
                chat.history = cleaned
                self.history.set(int(chat_id), cleaned)
            self.chats[int(chat_id)] = chat
        stickers = Path("data") / "stickers.json"
        self.stickers = (
            json.loads(stickers.read_text(encoding="utf-8")).get("muffin", [])
            if stickers.exists()
            else []
        )
        vocab = Path("data") / "vocab.json"
        self.vocab = set(json.loads(vocab.read_text(encoding="utf-8"))) if vocab.exists() else set()

    def _system_prompt(
        self,
        text: str,
        has_history: bool,
        mood: Mood,
        life: Activity,
        chat_id: int | None = None,
    ) -> str:
        now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
        lines = [
            self.persona,
            "",
            GLOSSARY,
            "",
            "The glossary is only for understanding Bong. Do not copy a glossary"
            " example as your answer; answer what he actually asked.",
            "",
            self.mood.describe(mood),
            "",
            self.life.describe(life),
            "",
            f"Right now it is {now}.",
        ]

        if chat_id is not None:
            topic = self.threads.describe(chat_id)
            if topic:
                lines.append(f"Conversation so far: {topic}.")
            gap = self._last_gap(chat_id)
            if gap:
                lines.append(f"The last message in this chat was {gap} ago.")
            facts = self.facts.describe(chat_id, text)
            if facts:
                lines += [
                    "",
                    "Bong told you these things earlier. Use one only if it fits "
                    "the latest message; never list them or recite them:",
                    *facts,
                ]

        if self.config.adapter_mode:
            return self.compact_persona

        if has_history:
            lines += [
                "",
                "This is an ongoing conversation. Continue it from the messages",
                "above. Do not greet again, do not repeat an earlier reply, and",
                "use what was just said (names, plans, food, feelings) when it",
                "matters. Answer only the latest message.",
            ]

        # Similar real exchanges teach the model slang and typical replies. They
        # are skipped for short confirmations, where they add noise instead.
        examples: list[dict] = []
        want_examples = self.config.examples_mode == "always" or (
            self.config.examples_mode == "first" and not has_history
        )
        if want_examples and not AFFIRM.match(text):
            seen: set[str] = set()
            try:
                hits = self.retriever.search(text, k=max(24, self.config.top_k * 3))
            except Exception as exc:  # noqa: BLE001 - embeddings may be offline
                log.warning("example retrieval failed: %s", exc)
                hits = []
            for ex in hits:
                if ex["score"] < self.config.example_min_score:
                    continue
                reply = ex["reply"]
                if "http" in reply.lower():
                    continue
                if len(reply) < 2 or len(reply) > 160:
                    continue
                if re.fullmatch(r"[\d\W_]+", reply):
                    continue
                key = ex["user"].casefold()
                if key in seen:
                    continue
                seen.add(key)
                examples.append(ex)
                if len(examples) >= self.config.top_k:
                    break

        if examples:
            lines += [
                "",
                "These are real messages from the same conversation. Study the",
                "words, spelling and length, then answer Bong's latest message in",
                "that exact voice. Do not copy an example that answers a different",
                "question; only borrow how she sounds.",
            ]
            for ex in examples:
                lines.append(f"Bong: {clip(ex['user'])}")
                lines.append(f"Muffin: {clip(ex['reply'])}")

        lines += [
            "",
            "Bong may ask several questions in a row. You do not have to answer",
            "all of them. You can answer one, ignore one, or change the subject,",
            "but keep it coherent.",
            "",
            "Now answer Bong's last message with one short line, or 2-3 tiny",
            "lines separated by newlines when it feels natural. Never repeat a phrase.",
        ]
        return "\n".join(lines)

    def _last_gap(self, chat_id: int) -> str:
        chat = self.chats.get(chat_id)
        stamps = [
            message.get("ts")
            for message in (chat.history if chat else [])
            if message.get("ts")
        ]
        if not stamps:
            return ""
        seconds = max(0.0, time.time() - max(stamps))
        if seconds < 900:
            return ""
        if seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        if seconds < 86400:
            return f"{int(seconds / 3600)} hours"
        return f"{int(seconds / 86400)} days"

    def _history(self, chat_id: int) -> list[dict]:
        """Recent turns, cleaned and bounded by a character budget.

        Short Khmerlish lines are cheap, so this keeps more real conversation
        than a fixed message count while still capping prompt size.
        """
        chat = self.chats.setdefault(chat_id, Chat())
        cleaned = self._clean_history(chat.history)
        budget = self.config.history_chars
        max_messages = max(4, self.config.history_messages)
        selected: list[dict] = []
        used = 0
        for message in reversed(cleaned):
            size = len(message.get("content", ""))
            if selected and (
                len(selected) >= max_messages or used + size > budget
            ):
                break
            selected.append(message)
            used += size
        return list(reversed(selected))

    @staticmethod
    def _clean_history(history: list[dict]) -> list[dict]:
        """Drop bot noise that should not become the next model's context."""
        cleaned: list[dict] = []
        for message in history:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if message.get("role") == "assistant":
                if junk_reply(content) or BANNED.search(content):
                    continue
                if FOREIGN_SCRIPT.search(content) or re.fullmatch(
                    r"[\d\W_]+", content
                ):
                    continue
                if cleaned and cleaned[-1]["role"] == "assistant":
                    if normalize(cleaned[-1]["content"]) == normalize(content):
                        continue
            item = {"role": message.get("role"), "content": content}
            if message.get("ts"):
                item["ts"] = message["ts"]
            if message.get("kind"):
                item["kind"] = message["kind"]
            if message.get("note"):
                item["note"] = message["note"]
            cleaned.append(item)
        return cleaned

    @staticmethod
    def _last_assistant(chat: Chat) -> str:
        return next(
            (m["content"] for m in reversed(chat.history) if m["role"] == "assistant"),
            "",
        )

    def _dedupe(self, chat_id: int, chat: Chat, reply: str) -> str:
        last = self._last_assistant(chat)
        # normalize() strips emoji, so an emoji-only reply would otherwise
        # compare equal to the empty last message and be "deduplicated".
        key = normalize(reply) or reply.strip().lower()
        last_key = normalize(last) or last.strip().lower()
        if key != last_key or not reply:
            return reply
        low = reply.lower()
        if "?" in reply:
            options = ["Huh", "Mic"]
        elif re.search(r"\b(nham|bay|prek|kteav|trey|food)\b", low):
            options = ["Nham hz", "Tv nham tv", "Ot nham teh"]
        elif re.search(r"\b(slnh|love|miss)\b", low):
            options = ["Slnh bong", "Kit bong"]
        elif re.search(r"\b(dek|sleep|night)\b", low):
            options = ["Dek hz", "Dek ban"]
        else:
            options = ["Hmm", "Okay", "Hxhx"]
        for alt in options:
            if normalize(alt) != normalize(last):
                log.info("duplicate reply avoided: %r -> %r", reply, alt)
                return alt
        return reply

    def _fallback_chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        session_id: str,
        code: bool = False,
    ) -> str:
        """Try each configured fallback in order, newest provider chain.

        Remote free models are often reasoning models: they need token room for
        their hidden reasoning and must not receive stop sequences, which would
        cut the reasoning before the answer starts.
        """
        for llm in self.fallbacks:
            remaining = self._remaining()
            if remaining <= 0.5:
                log.warning("reply deadline reached; skipping %s", llm.config.model)
                break
            local = (
                "localhost" in llm.config.base_url
                or "127.0.0.1" in llm.config.base_url
            )
            tokens = max_tokens if local else max(max_tokens, 900 if code else 700)
            self.last_fallbacks.append(llm.config.model)
            log.info("falling back to %s (%s)", llm.config.model, llm.config.base_url)
            send_messages = messages
            if not local:
                # Free remote models drift into English or Khmer script. Repeat
                # the house style at the very end, where it has the most weight.
                send_messages = [dict(message) for message in messages]
                if send_messages and send_messages[0].get("role") == "system":
                    examples = self._fallback_examples(send_messages)
                    extra = (
                        "\n\nSTRICT: reply only in Khmerlish written with Latin "
                        "letters, one short line of at most six words. Do not use "
                        "Khmer script and do not answer in English. Do not tell "
                        "stories or mention family members."
                    )
                    if examples:
                        extra += (
                            "\nReal replies to similar messages (borrow the voice, "
                            "answer Bong's message, never copy):\n" + examples
                        )
                    send_messages[0]["content"] += extra
            try:
                raw = llm.chat(
                    send_messages,
                    temperature=temperature,
                    max_tokens=tokens,
                    session_id=session_id,
                    stop=None if local else [],
                    timeout=min(remaining, 4.0 if local else 6.0),
                )
            except Exception as exc:  # noqa: BLE001 - try the next provider
                log.warning("fallback %s failed: %s", llm.config.model, exc)
                continue
            if raw.strip():
                self.last_provider = llm.config.model
                return raw
            log.warning("fallback %s returned nothing", llm.config.model)
        return ""

    def rewrite_with_note(
        self,
        chat_id: int,
        text: str,
        note: str,
        history: list[dict],
        rejected: str = "",
        reset_metrics: bool = True,
        timeout: float | None = None,
    ) -> list[str]:
        """Rewrite a rejected answer so it follows Bong's note.

        The note is an instruction ("too dry", "she is at school then", "say it
        softer"), not the literal reply. It is sent to the model together with
        the real few-shot examples, and the old line is explicitly banned.
        """
        if reset_metrics:
            self.last_provider = self.config.model
            self.last_fallbacks = []
            self.last_style_issues = []
        if not self._deadline:
            self.begin_turn()
        mood = self.mood.snapshot(chat_id)
        life = self.life.observe(chat_id)
        parts = self._note_reply(
            chat_id, text, note, rejected, mood, life, history, timeout=timeout
        )
        if note_emotion(note) and is_neutral_reply(parts):
            parts = self._note_reply(
                chat_id,
                text,
                note + " Show the mood clearly, not a neutral filler.",
                rejected,
                mood,
                life,
                history,
                timeout=timeout,
            )
        return parts

    def _note_reply(
        self,
        chat_id: int,
        text: str,
        note: str,
        rejected: str,
        mood: Mood,
        life: Activity,
        history: list[dict],
        timeout: float | None = None,
    ) -> list[str]:
        system = self._system_prompt(
            text, has_history=bool(history), mood=mood, life=life, chat_id=chat_id
        )
        system += (
            "\n\nBong rejected your last answer."
            f'\nHis note: "{note}"'
            "\nAnswer his message again so the new line follows that note."
            " Keep the same voice and keep it short."
        )
        if rejected:
            system += f'\nDo not repeat the rejected line: "{rejected}"'
        emotion = note_emotion(note)
        if emotion:
            system += (
                f"\nBong wants you to be clearly {emotion}. Show that mood in the "
                "line. Do not answer with Hmm, Ort, Okay or Mic."
            )

        messages = [{"role": "system", "content": system}]
        messages += history
        messages.append({"role": "user", "content": text})

        temperature, max_tokens, max_chars = MOOD_STYLE.get(
            mood.mood, MOOD_STYLE["neutral"]
        )
        temperature = max(temperature, 0.9)
        if self.config.temperature_override is not None:
            temperature = self.config.temperature_override
        if (
            "localhost" not in self.config.base_url
            and "127.0.0.1" not in self.config.base_url
        ):
            max_tokens = max(max_tokens, self.config.remote_min_tokens)

        raw = ""
        self.last_provider = self.config.model
        try:
            raw = self.llm.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=f"muffin-{chat_id}",
                timeout=min(timeout or self.config.timeout, self._llm_timeout()),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("note rewrite primary LLM failed: %s", exc)

        if not raw:
            raw = self._fallback_chat(
                messages,
                temperature=temperature,
                max_tokens=min(max_tokens, 80),
                session_id=f"muffin-{chat_id}",
            )

        reply = clean_reply(raw)
        if not reply or re.fullmatch(r"[\d\W_]+", reply) or BANNED.search(reply):
            reply = self._safe_fallback(chat_id, text, rejected, life)
        elif is_stock_echo(reply) or only_echoes(reply):
            log.info("note rewrite stock echo rejected: %r", reply)
            reply = self._safe_fallback(chat_id, text, rejected, life)
        elif self.unknown_words(reply):
            log.info("note rewrite unknown words: %s", self.unknown_words(reply))
            reply = self._safe_fallback(chat_id, text, rejected, life)
        elif self.life.conflicts(life, reply):
            log.info("note rewrite activity conflict: %r", reply)
            reply = self._safe_fallback(chat_id, text, rejected, life)
        elif len(reply) > min(max_chars, self.config.max_reply_chars) and not EMOTIONAL.search(text):
            limit = min(max_chars, self.config.max_reply_chars)
            reply = reply[:limit].rsplit(" ", 1)[0].rstrip(" ,;.")

        return split_messages(reply, text, mood.mood, generated=True)

    def begin_turn(self, seconds: float | None = None) -> None:
        self._deadline = time.monotonic() + (
            seconds if seconds is not None else self.config.max_reply_seconds
        )

    def _remaining(self) -> float:
        if not self._deadline:
            return self.config.timeout
        return max(0.0, self._deadline - time.monotonic())

    def _llm_timeout(self) -> float:
        return max(0.5, min(self.config.timeout, self._remaining()))

    def style_problems(self, text: str, user_text: str = "") -> list[str]:
        """What breaks her texting style for this message."""
        if KHMER_SCRIPT.search(user_text or ""):
            return []
        problems: list[str] = []
        plain = strip_code(text)
        if KHMER_SCRIPT.search(plain):
            converted = normalize_script(text, user_text)
            if KHMER_SCRIPT.search(strip_code(converted)):
                problems.append("khmer")
        if "," in plain:
            problems.append("comma")
        if FOREIGN_SCRIPT.search(plain):
            problems.append("foreign")
        if not looks_english(user_text) and looks_english(plain):
            problems.append("english")
        return problems

    def relevance_problems(self, user_text: str, reply: str) -> list[str]:
        """Catch a reply that volunteers food when he asked something else."""
        if FOOD_WORDS.search(reply) and not (
            FOOD_ASK.search(user_text) or DOING_ASK.search(user_text)
        ):
            return ["food"]
        return []

    def mood_problems(self, mood: str, reply: str) -> list[str]:
        """Catch a reply that contradicts the current mood."""
        low = reply.lower()
        problems: list[str] = []
        # Angry/upset mood must not be affectionate or romantic
        if mood in {"angry", "annoyed"}:
            if AFFECTIONATE.search(low):
                problems.append("mood:affectionate_while_angry")
            if ROMANTIC.search(low):
                problems.append("mood:romantic_while_angry")
        # Soft mood must not be cold/hostile
        if mood == "soft":
            if COLD_WORDS.search(low):
                problems.append("mood:cold_while_soft")
        return problems

    def style_fix(self, parts: list[str], user_text: str = "") -> list[str]:
        fixed: list[str] = []
        for part in parts:
            part = FOREIGN_SCRIPT.sub("", part)
            part = normalize_script(part, user_text)
            if not looks_english(user_text):
                part = drop_commas(part)
            if part.strip():
                fixed.append(part.strip())
        return fixed or parts

    def _fallback_examples(self, messages: list[dict], limit: int = 3) -> str:
        """Two or three short real lines to keep a fallback model on topic."""
        question = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                question = message.get("content") or ""
                break
        if not question:
            return ""
        try:
            hits = self.retriever.search(question, k=12)
        except Exception as exc:  # noqa: BLE001 - retrieval may be offline
            log.warning("fallback examples failed: %s", exc)
            return ""
        lines: list[str] = []
        for hit in hits:
            if hit["score"] < 0.60:
                continue
            reply = (hit.get("reply") or "").strip()
            if not reply or too_long_for_chat(reply, 10) or junk_reply(reply):
                continue
            lines.append(f"Bong: {clip(hit['user'])}\nMuffin: {clip(reply)}")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def _short_reply(
        self, chat_id: int, text: str, last_assistant: str, life: Activity
    ) -> str:
        """Deterministic short answer used when a provider drifts."""
        contextual = self._context_reply(text, last_assistant)
        if contextual:
            return contextual
        return self._safe_fallback(chat_id, text, last_assistant, life)

    def _safe_fallback(
        self, chat_id: int, text: str, last_assistant: str, life: Activity
    ) -> str:
        """Never invent a reply. Prefer the closest real chat response."""
        chat = self.chats.setdefault(chat_id, Chat())
        remembered = self.memory.lookup(text, last_assistant, max_len=160)
        if (
            remembered
            and not BANNED.search(remembered)
            and not junk_reply(remembered)
            and not re.fullmatch(r"[\d\W_]+", remembered)
            and not self.life.conflicts(life, remembered)
        ):
            log.info("safe fallback from real chat: %r", remembered)
            return self._dedupe(chat_id, chat, remembered)
        # No exact line, no model: answer with the closest real exchange instead
        # of a generic activity line.
        try:
            hits = [
                hit
                for hit in self.retriever.search(text, k=8)
                if hit["score"] >= self.config.fallback_min_score
            ]
            # Prefer a short, chat-sized line over a long archive sentence.
            hits.sort(key=lambda hit: (too_long_for_chat(hit["reply"], 14), -hit["score"]))
            for hit in hits:
                reply = hit["reply"]
                if (
                    BANNED.search(reply)
                    or junk_reply(reply)
                    or self.relevance_problems(text, reply)
                    or re.fullmatch(r"[\d\W_]+", reply)
                    or len(reply) > 160
                    or self.life.conflicts(life, reply)
                ):
                    continue
                log.info("safe fallback from retrieval: %r", reply)
                return self._dedupe(chat_id, chat, reply)
        except Exception as exc:  # noqa: BLE001 - retrieval must not break a reply
            log.warning("retrieval fallback failed: %s", exc)
        log.info("safe fallback to activity line for %r", life.key)
        return self._dedupe(chat_id, chat, self.life.fallback(life))

    @staticmethod
    def _context_reply(text: str, last_assistant: str) -> str | None:
        last = last_assistant.lower()
        if YES.match(text):
            if "takmao" in last:
                return "Tv takmao hz"
            if "can i" in last:
                return "Rvol men"
            if "nham" in last:
                return "Nham hz"
            if "dek" in last or "sleep" in last:
                return "Dek hz"
            if "slnh" in last or "love" in last:
                return "Slnh bong"
            return random.choice(["Yep", "Okay", "Yes"])
        if NO.match(text):
            if "takmao" in last:
                return "Ort"
            return random.choice(["Ort", "Ot", "No"])
        return None

    def reset(self, chat_id: int) -> None:
        self.chats.pop(chat_id, None)
        self.history.reset(chat_id)
        self.facts.reset(chat_id)

    def _remember(
        self,
        chat_id: int,
        chat: Chat,
        text: str,
        reply: str,
        kind: str = "reply",
    ) -> None:
        # Store what was actually sent, not the raw model line: otherwise a
        # Khmer-script fragment or comma in history teaches the next prompt to
        # repeat it.
        reply = clean_reply(normalize_script(reply, text))
        if not reply:
            reply = "Huh"
        if not looks_english(text):
            reply = drop_commas(reply)
        now = time.time()
        chat.history += [
            {"role": "user", "content": text, "ts": now},
            {"role": "assistant", "content": reply, "ts": now, "kind": kind},
        ]
        self._trim(chat)
        self.history.set(chat_id, chat.history)

    @staticmethod
    def _last_bot_line(chat: Chat) -> str:
        """The last thing she actually sent, without earlier burst lines."""
        for message in reversed(chat.history):
            if message.get("role") == "assistant":
                text = message.get("content") or ""
                return re.split(r"\s*\|\s*|\n", text)[-1].strip()
        return ""

    @staticmethod
    def _cold_reply(text: str, last: str, fallback: list[str]) -> str:
        """Stay cold, but react to what he actually said."""
        low = text.lower()
        if re.match(r"^\s*(morning|hi|hello|hey|bart)\b", low):
            # Even a cold morning gets a morning back, not a flat "Ohh".
            options = ["Morning", "Morning bong", "Ngob"]
        elif re.search(r"\b(rvol|mok|back|home)\b", low):
            options = ["Rvol hz?", "Nham bay nv?", "Ort mean ey"]
        elif re.search(r"\bjorb\b", low):
            options = ["Jorb hz?", "Rean tv jg", "Ort mean ey"]
        elif re.search(r"\b(rean|school|class|study|exam)\b", low):
            options = ["Rean tv jg", "Jorb hz?", "Ort mean ey"]
        elif re.search(r"\b(nham|bay|eat|hungry)\b", low):
            # A food question still needs an answer. "Nham hz?" turns her
            # answer into a confused question of her own.
            # Compound question: "ban ey nham bay nv" = what + whether
            if re.search(r"\b(ban|mean)\s+ey\b.*\b(nham|bay)\b|\b(nham|bay)\b.*\b(ban|mean)\s+ey\b|\bwhat\b.*\b(eat|ate)\b", low):
                options = ["Nham bay muy chicken", "Bay cha", "Nham eggs"]
            elif re.search(r"\b(ban|mean)\s+ey\b|\bwhat\b", low):
                options = ["Nham bay muy chicken", "Bay cha", "Nham eggs"]
            else:
                options = ["Nham tv", "Ot nham teh", "Nham"]
        elif text.strip().endswith("?") or re.search(
            r"\b(ey|mic|mix|tv na)\b", low
        ):
            options = ["Ort mean ey", "Hmm", "Ort"]
        else:
            # A statement is not a yes/no question. Even while cold, react to
            # what he said instead of answering "Ort" to a sentence.
            options = ["Ohh", "Hxhx", "Hmm", "Jah"]
        if not options:
            options = fallback
        return next(
            (line for line in options if normalize(line) != normalize(last)),
            options[0],
        )

    @staticmethod
    def _cold_streak(chat: Chat) -> int:
        """How many cold one-word replies she just sent in a row."""
        streak = 0
        for message in reversed(chat.history):
            if message.get("role") != "assistant":
                continue
            if COLD_LINE.match((message.get("content") or "").strip()):
                streak += 1
            else:
                break
        return streak

    def _trim(self, chat: Chat) -> None:
        limit = max(2 * self.config.history_messages, self.config.history_keep)
        if len(chat.history) > limit:
            chat.history = chat.history[-limit:]

    def remember_assistant(
        self, chat_id: int, text: str, kind: str = "proactive"
    ) -> str:
        chat = self.chats.setdefault(chat_id, Chat())
        chat.history.append(
            {"role": "assistant", "content": text, "ts": time.time(), "kind": kind}
        )
        self._trim(chat)
        self.history.set(chat_id, chat.history)
        self.mood.after_reply(chat_id)
        return text

    def context_summary(self, chat_id: int) -> str:
        """What the bot actually remembers right now, for /context."""
        lines: list[str] = []
        mood = self.mood.snapshot(chat_id)
        if mood.upset:
            lines.append(
                f"mood: {mood.mood} because {mood.upset}; "
                f"needs {mood.comfort_needed}/100 comfort"
            )
        else:
            lines.append(f"mood: {mood.mood}")
        topic = self.threads.describe(chat_id)
        if topic:
            lines.append(f"topic: {topic}")
        facts = self.facts.describe(chat_id)
        if facts:
            lines.append("remember:")
            lines += [fact[2:] if fact.startswith("- ") else fact for fact in facts]
        history = self._history(chat_id)
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        last_bot = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )
        if last_user:
            lines.append(f"last you said: {last_user}")
        if last_bot:
            lines.append(f"last she said: {last_bot}")
        gap = self._last_gap(chat_id)
        if gap:
            lines.append(f"quiet for {gap}")
        return "\n".join(lines) or "No context yet"

    def human_delay(self, chat_id: int) -> float:
        state = self.life.observe(chat_id)
        if state.key in {"school", "teach", "meeting", "commute"}:
            return random.uniform(1.5, 4.0)
        if state.key in {"sleep", "night"}:
            return random.uniform(0.8, 2.0)
        return random.uniform(0.2, 0.9)

    def sticker(self, chat_id: int) -> str | None:
        if not self.stickers or random.random() >= self.config.sticker_chance:
            return None
        mood = self.mood.snapshot(chat_id).mood
        if mood in {"annoyed", "tired", "sleepy", "soft"}:
            return None
        state = self.life.observe(chat_id)
        if state.key in {"sleep", "meeting", "teach", "school"}:
            return None
        return random.choice(self.stickers)

    def unknown_words(self, text: str) -> list[str]:
        if not self.vocab:
            return []
        tokens = re.findall(r"[A-Za-z][A-Za-z']{2,}", text)
        return sorted({t.lower() for t in tokens if t.lower() not in self.vocab})

    def ritual(self, chat_id: int) -> str | None:
        message = self.reactions.proactive(chat_id)
        if message:
            self.remember_assistant(chat_id, message, kind="ritual")
        return message

    def proactive(self, chat_id: int) -> list[str]:
        self.begin_turn()
        self.last_provider = self.config.model
        self.last_fallbacks = []
        self.last_style_issues = []
        chat = self.chats.setdefault(chat_id, Chat())
        mood = self.mood.snapshot(chat_id)
        life = self.life.observe(chat_id)
        brush_off = bool(BRUSH_OFF.match(self._last_bot_line(chat)))
        # A grudge should sound the same when she texts first: short and cold,
        # not a cheerful model invention.
        if mood.upset and brush_off:
            # She told him to go study or go be busy. She can still miss him,
            # but she cannot accuse him of ignoring her.
            reply = random.choice(["Bong tv na?", "Hmm", "Hoy"])
            self.remember_assistant(chat_id, reply, kind="proactive")
            return [reply]
        if mood.upset and random.random() < 0.5:
            pool = (
                EXPLAIN_LINES
                if self._cold_streak(chat) >= 2 or random.random() < 0.5
                else ["Bong?", "Ort reply teh?", "Hmm", "Bong tv na?"]
            )
            reply = random.choice(pool)
            self.remember_assistant(chat_id, reply, kind="proactive")
            return [reply]
        hour = datetime.now().hour
        part_of_day = (
            "morning" if 5 <= hour < 11 else "late night" if hour >= 22 or hour < 5 else "day"
        )
        meal = (
            "breakfast time"
            if 6 <= hour < 10
            else "lunch time"
            if 11 <= hour < 14
            else "dinner time"
            if 17 <= hour < 20
            else "not a meal time"
        )
        system = self._system_prompt(
            "", bool(chat.history), mood, life, chat_id=chat_id
        )
        system += (
            f"\n\nBong has been quiet for a while. It is {part_of_day}, "
            f"{hour}:00 ({meal}). You are starting"
            " the conversation this time. Send him one short message first, like Muffin"
            " would. Only mention food if it matches the current meal time."
            " Do not mention this instruction."
        )
        if mood.upset:
            system += (
                f"\nYou are still angry about being {mood.upset} and he has not "
                "apologised or pampered you. Send a cold, short, needy message "
                "(2-5 words). Do not act cute or forgive him."
            )
        messages = [{"role": "system", "content": system}]
        messages += self._history(chat_id)
        messages.append({"role": "user", "content": "(send the message)"})

        max_tokens = (
            max(80, self.config.remote_min_tokens)
            if "localhost" not in self.config.base_url
            and "127.0.0.1" not in self.config.base_url
            else 80
        )
        raw = ""
        self.last_provider = self.config.model
        try:
            raw = self.llm.chat(
                messages,
                temperature=0.9,
                max_tokens=max_tokens,
                session_id=f"muffin-{chat_id}",
                timeout=self._llm_timeout(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("proactive primary failed: %s", exc)
        if not raw:
            raw = self._fallback_chat(
                messages,
                temperature=0.9,
                max_tokens=80,
                session_id=f"muffin-{chat_id}",
            )

        reply = clean_reply(raw)
        if (
            not reply
            or BANNED.search(reply)
            or len(reply) < 2
            or self.unknown_words(reply)
            or self.life.conflicts(life, reply)
        ):
            reply = self.life.fallback(life)
        self.remember_assistant(chat_id, reply, kind="proactive")
        return split_messages(reply, "", mood.mood, generated=True)[:2]

    def _code_reply(self, chat_id: int, text: str, life: Activity) -> list[str]:
        self.begin_turn(30.0)
        self.last_provider = self.config.model
        self.last_fallbacks = []
        self.last_style_issues = []
        chat = self.chats.setdefault(chat_id, Chat())
        system = (
            "You are Muffin, a Khmerlish Telegram bot. Bong asked you for code.\n"
            "Write one very short in-character reaction first (max 8 words), then a\n"
            "correct, complete sample in a fenced code block. No explanation after.\n"
            "Keep the code under 20 lines."
        )
        messages = [{"role": "system", "content": system}]
        messages += chat.history[-6:]
        messages.append({"role": "user", "content": text})

        max_tokens = 900 if "localhost" not in self.config.base_url else 600
        raw = ""
        self.last_provider = self.config.model
        try:
            raw = self.llm.chat(
                messages,
                temperature=0.4,
                max_tokens=max_tokens,
                session_id=f"muffin-{chat_id}",
                stop=[],
                timeout=self._llm_timeout(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("code primary failed: %s", exc)
        if not raw:
            raw = self._fallback_chat(
                messages,
                temperature=0.4,
                max_tokens=600,
                session_id=f"muffin-{chat_id}",
                code=True,
            )

        raw = PREFIXES.sub("", raw.strip())
        parts: list[str] = []
        if "```" in raw:
            before, rest = raw.split("```", 1)
            intro = before.strip() or "Here bong:"
            if ASSISTANT_INTRO.search(intro):
                intro = "Here bong:"
            code = "```" + rest.strip()
            if not code.rstrip().endswith("```"):
                code += "\n```"
            parts = [intro, code]
        else:
            parts = self._fallback_code(text)

        full = "\n".join(parts)
        self._remember(chat_id, chat, text, full)
        self.mood.after_reply(chat_id)
        return parts

    @staticmethod
    def _fallback_code(text: str) -> list[str]:
        low = text.lower()
        if "python" in low:
            lang, code = "python", 'print("Hello, Bong!")'
        elif "javascript" in low or re.search(r"\bjs\b", low):
            lang, code = "javascript", 'console.log("Hello, Bong!");'
        elif "java" in low:
            lang, code = "java", (
                'public class Main {\n'
                '    public static void main(String[] args) {\n'
                '        System.out.println("Hello, Bong!");\n'
                "    }\n"
                "}"
            )
        elif "html" in low:
            lang, code = "html", (
                "<!DOCTYPE html>\n<html>\n<body>\n"
                "  <h1>Hello, Bong!</h1>\n</body>\n</html>"
            )
        elif "sql" in low:
            lang, code = "sql", "SELECT * FROM users WHERE name = 'Bong';"
        else:
            lang, code = "cpp", (
                "#include <iostream>\n\n"
                "int main() {\n"
                '    std::cout << "Hello, Bong!" << std::endl;\n'
                "    return 0;\n"
                "}"
            )
        return ["Here bong:", f"``` {lang}\n{code}\n```"]

    def reply(
        self,
        chat_id: int,
        text: str,
        prior: tuple[str, ...] | list[str] = (),
        avoid: tuple[str, ...] | list[str] = (),
    ) -> list[str]:
        """Answer `text`; earlier messages from the same burst stay in history.

        Concatenating a burst into one string broke every exact-match rule, so
        each message is kept as its own turn and only the last one is answered.

        `avoid` holds lines that must not be repeated. It is used by the
        Regenerate button: the same message is answered again with different
        randomness, never with the reply Bong just rejected.
        """
        chat = self.chats.setdefault(chat_id, Chat())
        self.last_provider = "handler"
        self.last_fallbacks = []
        self.last_style_issues = []
        self.begin_turn()
        for message in prior:
            if message.strip():
                self.threads.observe_user(chat_id, message)
                self.facts.observe(chat_id, message)
                chat.history.append({"role": "user", "content": message})
        self.threads.observe_user(chat_id, text)
        self.facts.observe(chat_id, text)
        start = list(chat.history)

        def key(value: str) -> str:
            return normalize(value) or value.strip().lower()

        blocked = {key(part) for part in avoid if part.strip()}
        avoid_text = "\n".join(avoid)
        parts = self._reply(chat_id, text)
        for _ in range(2):
            if not blocked or key(" ".join(parts)) not in blocked:
                break
            chat.history = list(start)
            parts = self._reply(chat_id, text, last_override=avoid_text)
        # A last deterministic style pass: Khmerlish conversations must not be
        # answered with Khmer script, commas, or a plain English paragraph.
        joined = "\n".join(parts)
        problems = self.style_problems(joined, text)
        if problems:
            self.last_style_issues.extend(problems)
            fixed = self.style_fix(parts, text)
            remaining = self.style_problems("\n".join(fixed), text)
            if not remaining:
                # Commas and known Khmer particles are fixed deterministically;
                # no need to spend another model call.
                parts = fixed
            else:
                log.info("style issues %s: %r", remaining, joined[:120])
                try:
                    rewritten = self.rewrite_with_note(
                        chat_id,
                        text,
                        "Rewrite in Khmerlish using Latin letters only. No Khmer "
                        "script, no commas, and no English sentences.",
                        start,
                        joined,
                        reset_metrics=False,
                        timeout=5.0,
                    )
                except Exception as exc:  # noqa: BLE001 - fix what we can
                    log.warning("style rewrite failed: %s", exc)
                    rewritten = []
                if rewritten and not self.style_problems("\n".join(rewritten), text):
                    parts = rewritten
                    self.apply_correction(chat_id, "\n".join(parts))
                else:
                    parts = fixed

        joined = "\n".join(parts)
        relevance = self.relevance_problems(text, joined)
        if relevance:
            self.last_style_issues.extend(relevance)
            log.info("relevance issue %s: %r", relevance, joined[:120])
            try:
                rewritten = self.rewrite_with_note(
                    chat_id,
                    text,
                    "Answer only what he asked. Do not mention food, eating, or "
                    "what you are doing.",
                    start,
                    joined,
                    reset_metrics=False,
                    timeout=5.0,
                )
            except Exception as exc:  # noqa: BLE001 - keep the reply
                log.warning("relevance rewrite failed: %s", exc)
                rewritten = []
            if rewritten and not self.relevance_problems(
                text, "\n".join(rewritten)
            ):
                parts = rewritten
                self.apply_correction(chat_id, "\n".join(parts))
            else:
                parts = self.style_fix(parts, text)

        # Mood consistency: angry must not turn affectionate/romantic
        mood = self.mood.snapshot(chat_id)
        mood_issues = self.mood_problems(mood.mood, "\n".join(parts))
        if mood_issues:
            self.last_style_issues.extend(mood_issues)
            log.info("mood issue %s: %r", mood_issues, joined[:120])
            try:
                rewritten = self.rewrite_with_note(
                    chat_id,
                    text,
                    f"You are {mood.mood}. Stay cold and short. Do not be affectionate, romantic, or warm.",
                    start,
                    joined,
                    reset_metrics=False,
                    timeout=5.0,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("mood rewrite failed: %s", exc)
                rewritten = []
            if rewritten and not self.mood_problems(mood.mood, "\n".join(rewritten)):
                parts = rewritten
                self.apply_correction(chat_id, "\n".join(parts))
            else:
                parts = self.style_fix(parts, text)
        self.threads.observe_reply(chat_id, text, "\n".join(parts))
        return parts

    def restore_history(self, chat_id: int, history: list[dict]) -> None:
        """Rewind the chat so a regenerated answer replaces the rejected one."""
        chat = self.chats.setdefault(chat_id, Chat())
        chat.history = [dict(message) for message in history]
        self.history.set(chat_id, chat.history)

    def apply_correction(
        self, chat_id: int, corrected: str, note: str = ""
    ) -> None:
        """Put Bong's recommended reply where the wrong answer was in history."""
        chat = self.chats.setdefault(chat_id, Chat())
        for index in range(len(chat.history) - 1, -1, -1):
            if chat.history[index]["role"] == "assistant":
                chat.history[index]["content"] = corrected
                chat.history[index]["kind"] = "note"
                if note:
                    chat.history[index]["note"] = note
                break
        self.history.set(chat_id, chat.history)

    def _reply(self, chat_id: int, text: str, last_override: str = "") -> list[str]:
        chat = self.chats.setdefault(chat_id, Chat())
        mood = self.mood.observe(chat_id, text)
        life = self.life.observe(chat_id)
        reaction = self.reactions.observe(chat_id, text)
        turn = classify(text)

        # A grudge is sticky. Being ignored for hours or hearing about another
        # girl starts one; only an apology or pampering clears it (mood.py).
        if reaction.ignored:
            self.mood.hurt(chat_id, "ignored", 35)
        if reaction.jealous and not reaction.apology:
            self.mood.hurt(chat_id, "jealous", 30)
        if reaction.apology:
            self.mood.adjust(chat_id, patience=4, affection=2)
        mood = self.mood.snapshot(chat_id)

        # Bong's explicit recommendation wins over every rule, real chat line
        # and model guess. It is exactly the text he typed.
        corrected = self.corrections.get(text)
        if corrected:
            self._remember(chat_id, chat, text, corrected)
            self.mood.after_reply(chat_id)
            return split_messages(corrected, text)

        # He asks why she is cold. That is a repair attempt, not a normal
        # question: say what is wrong and ask for what she wants.
        if mood.upset and WHY_COLD.search(text):
            last = next(
                (m["content"] for m in reversed(chat.history) if m["role"] == "assistant"),
                "",
            )
            reply = next(
                (line for line in EXPLAIN_LINES if normalize(line) != normalize(last)),
                EXPLAIN_LINES[0],
            )
            self.mood.comforted(chat_id, 5)
            log.info("grudge explained after why-question: %r", reply)
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # A "Dek hz" / "Good night" is a commitment. Do not answer "are you
        # awake?" with "Ngerb hx" ten minutes later; hold it until morning
        # gives her a real reason to be up. This applies even when upset.
        claim, claim_ts = self.threads.bot_claim(chat_id)
        asks_her_state = (
            turn.refers_to("sleep")
            or turn.refers_to("sleep_probe")
            or AWAKE_QUESTION.match(text)
            or SLEEP_QUESTION.search(text)
            or DOING_QUESTION.search(text)
        )
        if (
            claim == "sleep"
            and asks_her_state
            and not claim_allows_wake(claim, claim_ts)
        ):
            last = next(
                (m["content"] for m in reversed(chat.history) if m["role"] == "assistant"),
                "",
            )
            reply = next(
                (
                    line
                    for line in ("Dek nv", "Ngerb ot rouch teh", "Ngerb ot rouch")
                    if normalize(line) != normalize(last)
                ),
                "Dek nv",
            )
            log.info("holding sleep claim against %r: %r", text, reply)
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # While the grudge is active she is cold sometimes, not every turn.
        if mood.upset:
            lines = UPSET_LINES.get(mood.upset, UPSET_LINES["ignored"])
            last = next(
                (m["content"] for m in reversed(chat.history) if m["role"] == "assistant"),
                "",
            )
            if random.random() < 0.30:
                streak = self._cold_streak(chat)
                if streak >= 2:
                    reply = next(
                        (
                            line
                            for line in EXPLAIN_LINES
                            if normalize(line) != normalize(last)
                        ),
                        EXPLAIN_LINES[0],
                    )
                else:
                    reply = self._cold_reply(text, last, lines)
                log.info(
                    "angry mood persists after %s (comfort %d): %r",
                    mood.upset,
                    mood.comfort_needed,
                    reply,
                )
                self._remember(chat_id, chat, text, reply)
                self.mood.after_reply(chat_id)
                return split_messages(reply, text)

        # Real data: 22% of skipped mornings get a cold reply, and a long
        # message is a conversation, not a greeting to be punished.
        substantive = len(text.split()) >= 7
        cold = None
        if (
            reaction.missed_morning
            and not substantive
            and not AWAKE_QUESTION.match(text)
            and not MORNING_GREETING.search(text)
            and COLD_CONTEXT.search(text)
            and random.random() < 0.15
        ):
            cold = self.reactions.cold_for(reaction)
        elif reaction.jealous and random.random() < 0.25:
            cold = self.reactions.cold_for(reaction)
        if cold:
            # Do not drain affection on every cold line: that is what kept
            # the grudge alive for a whole day.
            self.mood.adjust(chat_id, patience=-2)
            self._remember(chat_id, chat, text, cold)
            self.mood.after_reply(chat_id)
            return split_messages(cold, text)

        last_assistant = last_override or next(
            (m["content"] for m in reversed(chat.history) if m["role"] == "assistant"),
            "",
        )
        # Once the conversation is running, context must beat archive replay:
        # a line copied from an unrelated chat is what makes replies feel random.
        ongoing = bool(chat.history)

        # He points at her own messages ("look at the chat"). That is a
        # reference to the conversation, not a normal question.
        if turn.act == ACT_META:
            reply = random.choice(["Hxhx oun merl hz", "Oun merl hz", "Oun kit teh"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # He corrects the time of day: do not answer the stale meal greeting.
        # Other corrections ("you said...") get acknowledged instead of being
        # treated as a fresh unrelated question.
        if turn.act == ACT_CALLOUT:
            if turn.refers_to("time") and (
                turn.refers_to("food") or MEAL_WORD.search(text)
            ):
                reply = random.choice(
                    [
                        "Hxhx oun kit nham bay",
                        "Oun kit nham bay hz",
                        "Hxhx kit tv nham bay",
                    ]
                )
            elif turn.refers_to("time") and turn.refers_to("sleep"):
                reply = random.choice(["Hxhx dek hz", "Oun kit dek hz", "Dek hz"])
            elif turn.refers_to("bot_claim") and not THOUGHT_AT.search(text):
                reply = random.choice(["Hxhx kit bong", "Oun kit teh", "Hxhx oun kit"])
            elif turn.refers_to("time"):
                # A time correction without food context still deserves an
                # acknowledgement, not a random activity line.
                reply = random.choice(["Hxhx oun kit", "Oun kit teh", "Hxhx"])
            else:
                reply = ""
            if reply:
                self._remember(chat_id, chat, text, reply)
                self.mood.after_reply(chat_id)
                return split_messages(reply, text)

        # Typo-like pairs ("der kg" for "der lg") are not safe to guess.
        if TYPO_QUESTION.search(text):
            reply = "Mean ey bong?"
            log.info("typo-like message %r -> ask", text)
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)


        # "Tv sala kal?" = what time do you go to school? Her real schedule
        # starts at 07:00, so do not let the model guess "Okay".
        if SCHOOL_QUESTION.match(text):
            if re.search(r"\b(kal|mong\s*marn)\b", text, re.I):
                reply = random.choice(["Mg 7", "7 jeang", "Tv 7h"])
            elif life.key in {"school"}:
                reply = random.choice(["Nv sala", "Rean nv", "Nv school"])
            elif life.key in {"sleep", "breakfast"}:
                reply = random.choice(["Mg 7", "7 jeang", "Ot ton"])
            else:
                reply = random.choice(["Rean hx", "Dl hx", "Jenh hx"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Ngerb nv?" = are you awake yet? Answer from her actual schedule,
        # not from a loose archive line.
        if AWAKE_QUESTION.match(text):
            if life.key == "sleep":
                reply = random.choice(
                    ["Ngerb ot rouch teh", "Ot", "Dek nv", "Ngerb ot rouch"]
                )
            else:
                reply = random.choice(["Ngerb hx", "Ngerb yu hx", "Hxhx"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Pong tvr ey" / "what are you doing" must answer from the actual
        # schedule, not a random archive line. Repeat the question and get a
        # different real variant instead of the same line twice.
        if DOING_QUESTION.search(text):
            english = bool(ENGLISH_DOING.search(text))
            options = (DOING_EN if english else DOING_KH).get(
                life.key, ["Just chilling"] if english else ["Dek merl tiktok"]
            )
            reply = next(
                (o for o in options if normalize(o) != normalize(last_assistant)),
                options[0],
            )
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Nham bay nv" is usually a statement, not a question: ask what.
        if EATING_STATEMENT.match(text) and "?" not in text:
            # Bong says he is eating; she says what she is eating.
            reply = random.choice(EATING_KH)
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Ort rean heh" = not studying; say why she is not studying instead
        # of inventing an unrelated activity line.
        if re.search(r"\b(ort|ot)\s+rean\b", text, re.I):
            reply = random.choice(NOT_STUDY_KH)
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Hak min men nv rean teh?" = still not studying? Answer from the
        # schedule; the fallback model otherwise invents a location line.
        if STUDY_STATUS.search(text):
            if life.key == "school":
                reply = random.choice(["Rean nv", "Tvr lomhat nv", "Pg nv pteas"])
            elif life.key in {"chill", "work", "dinner", "night"}:
                reply = random.choice(NOT_STUDY_KH)
            else:
                reply = random.choice(["Rean hx", "Tvr lomhat nv"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Jos minh tha nv ptes" = I thought you were at school. Confirm where
        # she really is instead of repeating the guess as nonsense.
        if THOUGHT_AT.search(text):
            if life.key in {"school", "teach"}:
                reply = "Nv pteas"
            elif life.key in {"chill", "work", "sleep", "breakfast", "lunch", "dinner", "night"}:
                reply = "Nv home"
            else:
                reply = "Nv psar"
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # "Jenh nv?" = leaving already? One fixed reply, no model guessing.
        if re.fullmatch(r"\s*jenh\s+nv\s*[?.!]*", text, re.I):
            reply = random.choice(["Jenh mk nv", "Jenh mk", "Jenh mk hx"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        if HUNGER_YES.search(text) and re.search(
            r"\b(nham|hungry|eat)\b", last_assistant, re.I
        ):
            reply = random.choice(["Tv nham tv", "Nham tt bong", "Tv nham bay"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        if (
            "takmao" in text.lower()
            and "takmao" in last_assistant.lower()
            and not re.search(r"\b(mk|mok)\s+pi\b|\bmix\b|\bsmarn\b", text, re.I)
        ):
            reply = random.choice(["Tv takmao hz", "Okay"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        question = self.threads.pending_for_answer(chat_id, text) or last_assistant
        if question.rstrip().endswith("?"):
            contextual = self._context_reply(text, question)
            if contextual:
                self._remember(chat_id, chat, text, contextual)
                self.mood.after_reply(chat_id)
                return split_messages(contextual, text)

        ruled = override_reply(text) if self.config.use_rules else None
        if ruled:
            if self.life.conflicts(life, ruled):
                ruled = self.life.fallback(life)
            ruled = self._dedupe(chat_id, chat, ruled)
            self._remember(chat_id, chat, text, ruled)
            self.mood.after_reply(chat_id)
            return split_messages(ruled, text)

        if CODE_REQUEST.search(text):
            return self._code_reply(chat_id, text, life)

        # "Yes nham ruc hz tas pov" is an answer plus "and you?"; do not
        # collapse it to a bare "Yes" and make him repeat himself.
        if RETURN_QUESTION.search(text):
            if FOOD_ASK.search(text) or FOOD_WORDS.search(last_assistant):
                reply = "Oun nham ruc hz"
            else:
                reply = random.choice(["Oun hz", "Oun tvr ey?", "Oun nham ruc hz"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        # An exact real replay beats generic heuristics such as the sleep
        # rule: "Are you sleeping yet?" really was answered "Nv teh" once.
        if AFFIRM.match(text):
            remembered = self.memory.lookup_context(text, last_assistant)
        else:
            remembered = self.memory.lookup(
                text,
                last_assistant,
                max_len=160,
                allow_generic=not ongoing,
            )
        if remembered:
            remembered = self._dedupe(chat_id, chat, remembered)
            log.info("memory replay: %r", remembered)
            self._remember(chat_id, chat, text, remembered)
            self.mood.after_reply(chat_id)
            return split_messages(remembered, text)

        if not remembered and not AFFIRM.match(text) and not ongoing:
            remembered = self.memory.lookup_time(text, life.key)
            if remembered:
                log.info("time-of-day replay: %r", remembered)
                remembered = self._dedupe(chat_id, chat, remembered)
                self._remember(chat_id, chat, text, remembered)
                self.mood.after_reply(chat_id)
                return split_messages(remembered, text)

        if SLEEP_QUESTION.search(text):
            if life.key == "sleep":
                reply = random.choice(["Dek ban", "Dek hz", "Hmm"])
            else:
                reply = random.choice(["Dek ban", "Dek ot ban", "Dek ban teh", "Ot"])
            self._remember(chat_id, chat, text, reply)
            self.mood.after_reply(chat_id)
            return split_messages(reply, text)

        if self.config.use_rules and self.config.use_intents:
            ruled = intent_reply(text)
            if ruled:
                if self.life.conflicts(life, ruled):
                    ruled = self.life.fallback(life)
                ruled = self._dedupe(chat_id, chat, ruled)
                self._remember(chat_id, chat, text, ruled)
                self.mood.after_reply(chat_id)
                return split_messages(ruled, text)

        if (
            not ongoing
            and self.config.direct_threshold > 0
            and len(text.split()) >= 4
            and not AFFIRM.match(text)
        ):
            matches = self.retriever.search(text, k=8)
            if matches and matches[0]["score"] >= self.config.direct_threshold:
                top_score = matches[0]["score"]
                tied = [m for m in matches if m["score"] >= top_score - 0.01]
                reply = Counter(m["reply"] for m in tied).most_common(1)[0][0]
                if (
                    len(reply) <= 100
                    and "http" not in reply.lower()
                    and not self.life.conflicts(life, reply)
                ):
                    reply = self._dedupe(chat_id, chat, reply)
                    log.info("direct match %.3f: %r", top_score, reply)
                    self._remember(chat_id, chat, text, reply)
                    self.mood.after_reply(chat_id)
                    return split_messages(reply, text)

        if (
            not ongoing
            and text.strip().endswith("?")
            and mood.mood != "soft"
            and random.random() < 0.15
        ):
            nearby = self.retriever.search(text, k=1)
            confidence = nearby[0]["score"] if nearby else 0.0
            if confidence < 0.35:
                reply = random.choice(["Huh", "Mic", "Nham bay nv", "Ort"])
                self._remember(chat_id, chat, text, reply)
                self.mood.after_reply(chat_id)
                return split_messages(reply, text)

        # Unknown short Khmerlish: ask what he means instead of guessing.
        unknown = self.unknown_words(text)
        if unknown and len(text.split()) <= 5 and len(unknown) >= 2:
            try:
                hits = self.retriever.search(text, k=1)
                confidence = hits[0]["score"] if hits else 0.0
            except Exception:  # noqa: BLE001 - retrieval must not block a reply
                confidence = 0.0
            if confidence < 0.5:
                reply = random.choice(["Mean ey bong?", "Mic ot?", "Huh?"])
                log.info(
                    "unclear message %r unknown=%s score=%.2f -> ask",
                    text,
                    unknown,
                    confidence,
                )
                self._remember(chat_id, chat, text, reply)
                self.mood.after_reply(chat_id)
                return split_messages(reply, text)

        history = self._history(chat_id)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    text, has_history=bool(history), mood=mood, life=life, chat_id=chat_id
                ),
            }
        ]
        messages += history
        messages.append({"role": "user", "content": text})

        temperature, max_tokens, max_chars = MOOD_STYLE.get(
            mood.mood, MOOD_STYLE["neutral"]
        )
        if self.config.temperature_override is not None:
            temperature = self.config.temperature_override
        if "localhost" not in self.config.base_url and "127.0.0.1" not in self.config.base_url:
            max_tokens = max(max_tokens, self.config.remote_min_tokens)
        log.info(
            "mood=%s energy=%d affection=%d patience=%d",
            mood.mood,
            mood.energy,
            mood.affection,
            mood.patience,
        )
        raw = ""
        self.last_provider = self.config.model
        try:
            raw = self.llm.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=f"muffin-{chat_id}",
                timeout=self._llm_timeout(),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("primary LLM failed: %s", exc)

        if not raw:
            raw = self._fallback_chat(
                messages,
                temperature=temperature,
                max_tokens=min(max_tokens, 80),
                session_id=f"muffin-{chat_id}",
            )

        reply = clean_reply(raw)
        drifted = (
            self.last_provider != self.config.model
            and too_long_for_chat(reply)
        )
        if not reply or re.fullmatch(r"[\d\W_]+", reply) or BANNED.search(reply):
            reply = self._safe_fallback(chat_id, text, last_assistant, life)
        elif drifted:
            log.info("fallback drift rejected: %r", reply[:100])
            reply = self._short_reply(chat_id, text, last_assistant, life)
        elif is_stock_echo(reply) or only_echoes(reply):
            log.info("stock glossary echo rejected: %r", reply)
            reply = self._safe_fallback(chat_id, text, last_assistant, life)
        elif self.unknown_words(reply):
            log.info("unknown words: %s", self.unknown_words(reply))
            reply = self._safe_fallback(chat_id, text, last_assistant, life)
        elif self.life.conflicts(life, reply):
            log.info("activity conflict: %r", reply)
            reply = self._safe_fallback(chat_id, text, last_assistant, life)
        elif len(reply) > min(max_chars, self.config.max_reply_chars) and not EMOTIONAL.search(text):
            limit = min(max_chars, self.config.max_reply_chars)
            reply = reply[:limit].rsplit(" ", 1)[0].rstrip(" ,;.")

        reply = self._dedupe(chat_id, chat, reply)
        self._remember(chat_id, chat, text, reply)
        self.mood.after_reply(chat_id)
        return split_messages(reply, text, mood.mood, generated=True)


def load_config(env: dict[str, str], persona_path: Path) -> Config:
    fallbacks: list[tuple[str, str, str]] = []
    for prefix in ("FALLBACK", "FALLBACK2", "FALLBACK3", "FALLBACK4"):
        base = env.get(f"{prefix}_BASE_URL", "").strip()
        model = env.get(f"{prefix}_MODEL", "").strip()
        key = env.get(f"{prefix}_API_KEY", "").strip()
        if not base or not model:
            continue
        if not key and "opencode.ai" in base:
            key = env.get("LLM_API_KEY", "").strip()
        if not key and "openrouter.ai" in base:
            key = env.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            continue
        if base == env.get("LLM_BASE_URL", "").strip() and model == env.get(
            "LLM_MODEL", ""
        ).strip():
            continue
        fallbacks.append((base, key, model))

    return Config(
        base_url=env.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=env.get("LLM_API_KEY", "ollama"),
        model=env.get("LLM_MODEL", "qwen2.5:7b-instruct"),
        temperature=float(env.get("LLM_TEMPERATURE", 0.8)),
        max_tokens=int(env.get("LLM_MAX_TOKENS", 120)),
        top_k=int(env.get("RETRIEVE_K", 8)),
        example_min_score=float(env.get("EXAMPLE_MIN_SCORE", 0.35)),
        fallback_min_score=float(env.get("FALLBACK_MIN_SCORE", 0.5)),
        direct_threshold=float(env.get("DIRECT_REPLY_THRESHOLD", 0)),
        timeout=float(env.get("LLM_TIMEOUT", 90)),
        max_reply_seconds=float(env.get("MAX_REPLY_SECONDS", 15)),
        history_messages=int(env.get("HISTORY_MESSAGES", 12)),
        history_chars=int(env.get("HISTORY_CHARS", 1800)),
        history_keep=int(env.get("HISTORY_KEEP", 200)),
        max_reply_chars=int(env.get("MAX_REPLY_CHARS", 160)),
        remote_min_tokens=int(env.get("REMOTE_MIN_TOKENS", 300)),
        fallbacks=fallbacks,
        double_text_chance=float(env.get("DOUBLE_TEXT_CHANCE", 0.25)),
        sticker_chance=float(env.get("STICKER_CHANCE", 0.10)),
        use_rules=env.get("USE_RULES", "true").lower() not in {"0", "false", "no"},
        use_intents=env.get("USE_INTENTS", "false").lower() in {"1", "true", "yes"},
        examples_mode=env.get("EXAMPLES_MODE", "always"),
        adapter_mode=env.get("ADAPTER_MODE", "false").lower() in {"1", "true", "yes"},
        compact_persona_file=env.get("COMPACT_PERSONA_FILE", "persona_compact.md"),
        temperature_override=(
            float(env["LLM_TEMPERATURE_OVERRIDE"])
            if env.get("LLM_TEMPERATURE_OVERRIDE")
            else None
        ),
        persona_path=persona_path,
    )
