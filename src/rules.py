"""Deterministic intent layer.

OVERRIDES are explicit corrections from Bong: they win even over a real chat
replay. INTENTS are generic courtesies/short questions; exact memory is allowed
to beat those so the bot can quote the real Muffin when the line is in the chat.
"""

from __future__ import annotations

import random
import re

# Patterns that stop the override layer entirely without answering, so a
# longer request that merely starts with a greeting reaches memory or the LLM.
OVERRIDE_BLOCK: list[re.Pattern[str]] = [
    re.compile(r"\b(good\s*night|gn)\b.*\b(call|phone|video)\b", re.I),
]

OVERRIDES: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"^\s*(good\s*)?morning\b", re.I),
        ["Morning bong", "Morning oun hz", "Morning"],
    ),
    (
        re.compile(r"\bsg\b.*\b(ey|pov|na|mean|ke)\b|\bmean\b.*\bsg\b", re.I),
        ["Sg = sangsar bong", "Sg sangsar ng", "Sg = srey ng bong"],
    ),
    (
        re.compile(r"^\s*kiss(\s*kiss)?\s*[?.!]*$", re.I),
        ["Ort Ot jg", "Tic tt", "Ort"],
    ),
    (
        re.compile(r"\b(bong|oun)\s+slnh\s+(oun|bong)\b", re.I),
        ["Oun sur bong jrern", "Slnh bong", "Kit bong"],
    ),
    (
        re.compile(r"^\s*(good\s*night|gn)\b", re.I),
        ["Good night bong", "Dek hz", "Good night"],
    ),
    (
        re.compile(r"^\s*nv teh\s*[.!]*$", re.I),
        ["Mix yu mles"],
    ),
    (
        re.compile(r"\bmix yu mles\b|\blern\b.*\bmles\b|\bmles\b.*\blern\b", re.I),
        ["Jam tt", "Vinh tvr", "Ort hz"],
    ),
    (
        re.compile(r"\b(tnaii|tngaii|today)\b.*\b(nham bay|nham ey|eat)\b", re.I),
        ["Nham bay prek nv", "Nham bay kteav", "Nham bay trey", "Maybe fried fish"],
    ),
    (
        re.compile(r"\b(i'?m|im|feeling|so|very|nas)\s+sad\b|\bsad\s+(today|now|nas)\b", re.I),
        ["Kom sad teh bong, oun nv nh"],
    ),
]

INTENTS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"^🙏+$"), ["Hxhx"]),
    (
        re.compile(r"\b(mk|mok)\s+pi\b.*\btakmao\b|\btakmao\b.*\b(mix|smarn|how)\b", re.I),
        ["Mix dea?", "Takmao mix dea?"],
    ),
    (
        re.compile(r"\btakmao\b", re.I),
        ["Nv teh\nTv takmao?"],
    ),
    (
        # lern = hurry, mk = come, tos = go: "hurry up, come on, let's go"
        re.compile(r"\blern\b.*\b(mk|mok)\b|\b(mk|mok)\b.*\blern\b", re.I),
        ["Tv tv", "Nv teh", "Where"],
    ),
    (
        re.compile(r"https?://|youtu\.be|youtube\.com", re.I),
        ["Can I??", "Why?"],
    ),
    (
        re.compile(r"^(have no idea|idk|i don'?t know|dunno|no idea)\b", re.I),
        ["Nham bay nv", "Huh", "Why?"],
    ),
    (
        re.compile(r"\blazy\b|\bjg mic\b", re.I),
        ["Jg mic haa kon pov", "Mic JG mic"],
    ),
    (
        re.compile(r"^(yes|yepp?|yep|ok(ay)?|okiee?|hxhx|haha+|lol)\s*[.!]*$", re.I),
        ["Yep", "Yes", "Okay"],
    ),
    (re.compile(r"^(hi|hii+|hello|helo|hey|heyy+|hallo|bart)\b", re.I), ["Bart"]),
    (
        re.compile(r"^\s*(good\s*morning|morning)\b", re.I),
        ["Morning bong", "Morning oun hz"],
    ),
    (re.compile(r"\bthank(s| you| u)\b", re.I), ["Okay"]),
    (re.compile(r"^(bye|byebye|goodbye|good bye)\b", re.I), ["Bye"]),
    (
        re.compile(r"\b(nham bay|nham ey|nham mi|nham ruc|nham hz|nham hx)\b", re.I),
        ["Pg nham eggs", "Nham bay muy chicken", "Nham saj phat", "Pg nham bay"],
    ),
    (
        re.compile(r"\b(pong|tvr|tver)\s+(tvr\s+)?ey\b|\bdoing what\b", re.I),
        ["Pg dek merl tiktok", "Dek merl tiktok", "Pg scroll"],
    ),
    (
        re.compile(r"^(tv|jg tv)\s+na\b", re.I),
        ["Nv ler", "Tv ler"],
    ),
]


def _match(rules: list[tuple[re.Pattern[str], list[str]]], text: str) -> str | None:
    if len(text) > 80:
        return None
    for pattern, replies in rules:
        if pattern.search(text):
            # An empty reply list means "this rule blocks every rule after it"
            # without answering itself, so the real chat or the model can.
            if not replies:
                return None
            return random.choice(replies)
    return None


def override_reply(text: str) -> str | None:
    if any(pattern.search(text) for pattern in OVERRIDE_BLOCK):
        return None
    return _match(OVERRIDES, text)


def intent_reply(text: str) -> str | None:
    return _match(INTENTS, text)


def rule_reply(text: str) -> str | None:
    return override_reply(text) or intent_reply(text)
