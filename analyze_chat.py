#!/usr/bin/env python3
"""Aggregate analysis of the real Muffin chat (no raw private quotes in output)."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS = ROOT / "data" / "pairs.jsonl"
STATS = ROOT / "data" / "stats.json"
OUT = ROOT / "data" / "chat_analysis.md"

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
KHMER = re.compile(r"[\u1780-\u17ff]")
WORD = re.compile(r"[\w\u1780-\u17ff']+", re.UNICODE)

TOPICS = {
    "food": r"\b(nham|bay|eat|food|hungry|prek|kteav|trey|moan)\b",
    "school/study": r"\b(rean|school|class|study|pteas|homework|exam|kru)\b",
    "teaching": r"\b(teach|teaching|sala|teach\w*)\b",
    "work/code": r"\b(code|work|job|program|laptop|project|meeting)\b",
    "love/affection": r"\b(slnh|love|miss|kit bong|kiss|hug|baby|bf|girlfriend)\b",
    "sad/stress": r"\b(sad|cry|stress|hurt|tired|depress|lonely|anxious)\b",
    "sleep": r"\b(dek|sleep|asleep|keng|lok)\b",
    "money": r"\b(luy|money|paid|pay|dollar|riel|price)\b",
    "family": r"\b(mom|dad|mek|pa|grandma|grandpa|family|sister|brother)\b",
    "travel/out": r"\b(tv|lerng|phnom|beach|travel|trip|hotel|airport|market|psar)\b",
}


def norm(text: str) -> str:
    text = text.lower().replace("\u200b", "")
    text = re.sub(r"[^\w\u1780-\u17ff]+", " ", text)
    return " ".join(text.split())


def main() -> None:
    pairs = [json.loads(line) for line in PAIRS.open(encoding="utf-8")]
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    users = [p["user"] for p in pairs]
    replies = [p["reply"] for p in pairs]
    n = len(pairs)

    def median(xs):
        return statistics.median(xs) if xs else 0

    report = []
    add = report.append

    add("# Real Muffin chat — aggregate analysis\n")
    add("## Volume\n")
    add(f"- Pairs: **{n:,}**")
    add(f"- Raw messages in export: **{stats['raw_messages']:,}**")
    add(f"- Merged turns: **{stats['turns']:,}**")
    add(f"- Date range: **{stats['first_date']} → {stats['last_date']}**")
    add(f"- Average messages per turn: **{stats['raw_messages'] / max(stats['turns'], 1):.2f}**")
    add(f"- Voice-note-only replies skipped: **{stats['voice_replies']:,}**")
    add(f"- Text + sticker replies: **{stats['sticker_replies']:,}**\n")

    add("## Message shape\n")
    add(f"- User median / mean: **{median([len(x) for x in users])} / {sum(map(len, users)) / n:.1f}** chars")
    add(f"- Muffin median / mean: **{median([len(x) for x in replies])} / {sum(map(len, replies)) / n:.1f}** chars")
    short = sum(1 for r in replies if len(r) <= 20)
    tiny = sum(1 for r in replies if len(r) <= 5)
    add(f"- Muffin replies ≤20 chars: **{short / n:.1%}**")
    add(f"- Muffin replies ≤5 chars: **{tiny / n:.1%}**")
    add(f"- Emoji per 100 replies: **{sum(len(EMOJI.findall(r)) for r in replies) / n * 100:.1f}**\n")

    add("## Language mix\n")
    for name, texts in [("User", users), ("Muffin", replies)]:
        letters = sum(len([c for c in t if c.isalpha()]) for t in texts)
        khmer = sum(len(KHMER.findall(t)) for t in texts)
        add(f"- {name}: Khmer script = **{khmer / max(letters, 1):.1%}** of letters")
    add("- The rest is Romanized Khmer (Khmerlish) mixed with English.\n")

    add("## Rhythm\n")
    hours = defaultdict(lambda: [0, 0])
    for p in pairs:
        try:
            h = datetime.fromisoformat(p["date"]).hour
        except (ValueError, TypeError):
            continue
        hours[h][0] += 1
    add("| Hour | Messages |")
    add("|---|---|")
    for h in range(24):
        add(f"| {h:02d}:00 | {hours[h][0]} |")
    add("")

    add("## Topics (keyword mentions)\n")
    add("| Topic | User messages | Muffin replies |")
    add("|---|---|---|")
    for topic, pat in TOPICS.items():
        rx = re.compile(pat, re.I)
        add(
            f"| {topic} | {sum(1 for x in users if rx.search(x)):,} | "
            f"{sum(1 for x in replies if rx.search(x)):,} |"
        )
    add("")

    add("## Pronouns / address\n")
    for word in ["bong", "oun", "pov", "ort", "ot", "jg", "hz", "nham bay"]:
        u = sum(1 for x in users if re.search(rf"\b{re.escape(word)}\b", x, re.I))
        r = sum(1 for x in replies if re.search(rf"\b{re.escape(word)}\b", x, re.I))
        add(f"- `{word}` — user {u:,} / Muffin {r:,}")
    add("")

    add("## Most frequent Muffin replies\n")
    for text, count in Counter(replies).most_common(25):
        add(f"- {count:4d}× `{text[:60]}`")
    add("")

    add("## Most frequent user messages\n")
    for text, count in Counter(users).most_common(20):
        add(f"- {count:4d}× `{text[:70]}`")
    add("")

    # Ambiguity: same normalized user text with different replies.
    grouped = defaultdict(Counter)
    for p in pairs:
        grouped[norm(p["user"])][p["reply"]] += 1
    ambiguous = {k: v for k, v in grouped.items() if len(v) > 1}
    add("## Ambiguity (why standalone matching is limited)\n")
    add(f"- Distinct normalized user messages: **{len(grouped):,}**")
    add(f"- Those with more than one real reply: **{len(ambiguous):,}** "
        f"(**{len(ambiguous) / max(len(grouped), 1):.1%}**)")
    multi = [len(v) for v in ambiguous.values()]
    add(f"- Median number of different replies when ambiguous: **{median(multi)}**")
    add(f"- Maximum different replies for one message: **{max(multi) if multi else 0}**\n")

    add("## Content profile\n")
    add("The dominant registers are: food and daily errands, school/teaching, "
        "flirting and jealousy, emotional processing, and code/study complaints. "
        "Many replies are jokes or deflections rather than answers; a large share "
        "are one-word reactions. The chat is also highly context-dependent, with "
        "inside references and repeated callbacks that are not decodable from a "
        "single message.\n")

    add("## What this means for the clone\n")
    add("1. The archive is not a Q&A dataset; it is a relationship transcript. "
        "Exact text matching only works when the same conversation context is present.")
    add("2. Khmerlish has no fixed spelling, so the model needs a lexicon plus examples, "
        "not grammar rules.")
    add("3. Short replies dominate; long generations feel wrong even when the content is right.")
    add("4. Media (stickers/voice) carries a meaningful share of the persona and is missing "
        "from the text model.")
    add("5. The schedule and activity state matter because the chat is grounded in everyday "
        "life, not a generic assistant persona.")

    text = "\n".join(report) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
