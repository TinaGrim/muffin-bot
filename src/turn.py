"""What Bong just did in the conversation, and what he is pointing at.

The reply pipeline used to treat almost everything as either a question or a
greeting. That is why a correction ("it is night already") could be answered
with a stale lunch line, and why a plain statement could get a cold "Ort".
This module classifies the turn once, so later rules can act on the act and on
the things the message refers to, instead of re-guessing with one-off regexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from threads import infer_topic

ACT_QUESTION = "question"
ACT_STATEMENT = "statement"
ACT_CALLOUT = "callout"
ACT_META = "meta"
ACT_GREETING = "greeting"
ACT_COMMAND = "command"
ACT_TEASE = "tease"
ACT_ANSWER = "answer"

# A whole-message yes/no. Longer "Ort rean" is a statement, not an answer.
ANSWER_ONLY = re.compile(
    r"^\s*(yes|yeah|yep|yup|no|nope|nah|ok(ay)?|okiee?|sure|hz|ort|ot|men)"
    r"\s*[.!]*$",
    re.I,
)
META = re.compile(
    r"\b(?:merl|look\s+at|check|read)\b[^.!?\n]{0,24}"
    r"\b(?:chat|history|messages?|tv\s*dg|texts?)\b"
    r"|\b(?:chat|history|messages?)\b[^.!?\n]{0,12}\b(?:tv\s*dg|above|earlier)\b",
    re.I,
)
TIME_WORDS = re.compile(
    r"\b(lngeac|hark\s+lerng|dark\s+already|evening\s+already|"
    r"it'?s\s+night|night\s+already|late\s+already|morning\s+already)\b",
    re.I,
)
CORRECTION = re.compile(
    r"\b(jos\s+minh?|jos\s+min|kit\s+tha|i\s+thought|you\s+said|bong\s+said|"
    r"you\s+told\s+me|actually)\b",
    re.I,
)
ATTENTION = re.compile(r"^\s*(hoy|ey|oi)\b", re.I)
GREETING = re.compile(
    r"^\s*(hi|hello|hey|bart|muffin|oun|bong|good\s+(morning|night)|"
    r"morning|gn)\s*[!.,]*$",
    re.I,
)
COMMAND = re.compile(
    r"^\s*(?:please\s+)?(call|text|send|tell|come|wait|listen|stop|answer|"
    r"reply|show|give|oy|plov)\b",
    re.I,
)
WH_QUESTION = re.compile(
    r"^\s*(what|why|how|where|when|who|which|can|could|do|did|does|are|is|"
    r"will|would|have|has|should|pong|bong|oun)\b",
    re.I,
)
KH_QUESTION = re.compile(
    r"\?|\b(?:tvr|tver|tv)\s+ey\b|\bey\b(?!\s*(?:hz|hx|teh)\b)|"
    r"\b(?:mic|mix)\b|\btv\s+na\b|\bkal\s+na\b|\bmen\s+ey\b",
    re.I,
)
# "Ngerb nv" asks if she is awake; "Ngerb hx" answers that she is.
AWAKE_ASK = re.compile(
    r"\bngerb\s+(?:nv|has|ha|haii|men|ban|yu|ruc|ort|ot|teh|jos|ng|pov|ah|la|der|lg)\b",
    re.I,
)
# "Lov pov nguy nv" / "Pong tvr ey" asking state while she claimed sleep.
STATE_PROBE = re.compile(
    r"\b(lov\s+pov\s+nguy|pong\s+tvr\s+ey|merl\s+tv\s+dg)\b",
    re.I,
)
TEASE = re.compile(
    r"\b(stupid|idiot|boring|annoying|dumb|ugly|hate\s+you|bitch)\b", re.I
)
# "Ort mean ey" means "nothing", not "what?".
NOTHING = re.compile(r"^\s*(ot|ort)?\s*(mean|men)\s+ey\s*[.!]*$", re.I)

SLEEP_REF = re.compile(
    r"\b(dek|sleep|sleepy|nguy|ngerb|awake|wake|woke|bed|night)\b", re.I
)
FOOD_REF = re.compile(
    r"\b(nham|bay|prek|kteav|trey|eat|eating|food|hungry|breakfast|"
    r"lunch|dinner)\b",
    re.I,
)
SCHOOL_REF = re.compile(
    r"\b(rean|school|sala|ptes|pteas|study|studied|class|exam|teach)\b", re.I
)
CHAT_REF = re.compile(
    r"\b(chat|history|messages?|tv\s*dg|merl)\b", re.I
)
SELF_REF = re.compile(r"\b(i|me|my|oun|bong)\b", re.I)
BOT_CLAIM_REF = re.compile(
    r"\b(you\s+said|bong\s+said|jos\s+minh?|kit\s+tha|you\s+told)\b", re.I
)

# Going to bed is a commitment, not a throwaway line. She holds it until
# morning; only the clock (or a new morning) can legitimately wake her.
SLEEP_HOLD = 8 * 3600
MORNING_WAKE = 2 * 3600


@dataclass(frozen=True)
class Turn:
    act: str
    topic: str = ""
    refs: tuple[str, ...] = ()
    has_vocative: bool = False

    def refers_to(self, name: str) -> bool:
        return name in self.refs

    @property
    def is_question(self) -> bool:
        return self.act == ACT_QUESTION


def _references(text: str) -> tuple[str, ...]:
    refs: list[str] = []
    for name, pattern in (
        ("sleep", SLEEP_REF),
        ("food", FOOD_REF),
        ("school", SCHOOL_REF),
        ("chat", CHAT_REF),
        ("self", SELF_REF),
        ("bot_claim", BOT_CLAIM_REF),
    ):
        if pattern.search(text):
            refs.append(name)
    if TIME_WORDS.search(text):
        refs.append("time")
    if STATE_PROBE.search(text):
        refs.append("sleep_probe")
    return tuple(refs)


def classify(text: str) -> Turn:
    """One deterministic read of the message. No model, no randomness."""
    body = (text or "").strip()
    low = body.lower()
    has_vocative = bool(
        re.match(r"^\s*(bart|bong|oun|muffin|muffin\s+bong)\b[,\s!.]*", low)
    )
    refs = _references(low)
    topic = infer_topic(low)

    if META.search(low):
        act = ACT_META
    elif ANSWER_ONLY.match(low):
        act = ACT_ANSWER
    elif NOTHING.match(low):
        act = ACT_STATEMENT
    elif TIME_WORDS.search(low) or ATTENTION.match(low):
        act = ACT_CALLOUT
    elif STATE_PROBE.search(low):
        act = ACT_CALLOUT
    elif CORRECTION.search(low):
        act = ACT_CALLOUT
    elif GREETING.match(low):
        act = ACT_GREETING
    elif COMMAND.match(low):
        act = ACT_COMMAND
    elif AWAKE_ASK.search(low) or KH_QUESTION.search(low) or WH_QUESTION.match(low):
        act = ACT_QUESTION
    elif TEASE.search(low):
        act = ACT_TEASE
    else:
        act = ACT_STATEMENT

    return Turn(act=act, topic=topic, refs=refs, has_vocative=has_vocative)


def claim_allows_wake(
    claim: str, claim_ts: float, now: datetime | None = None
) -> bool:
    """May she contradict a recent claim without looking like she forgot it?

    A "Dek hz" or "Good night" at night holds through the night. In the
    morning a few hours are enough, because waking up is the natural reason.
    """
    if claim != "sleep":
        return True
    now = now or datetime.now()
    age = max(0.0, now.timestamp() - float(claim_ts or 0.0))
    if now.hour >= 20 or now.hour < 5:
        return False
    if 5 <= now.hour < 12 and age >= MORNING_WAKE:
        return True
    return age >= SLEEP_HOLD
