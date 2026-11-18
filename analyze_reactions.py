#!/usr/bin/env python3
"""Analyze reaction patterns: mornings, gaps, jealousy, repair, initiation."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from build_dataset import merge_turns, parse_page, page_number

EXPORT = Path("/Users/grim/Downloads/ChatExport_Muffin ( pitite bot )_3")
BOT = "Muffin ( pitite bot )"
OWNER = "NHEB Tinarith"

GREETING = re.compile(r"^(good\s*morning|morning|bart)\b", re.I)
COLD = re.compile(r"\b(ort|ot|huh|ng|mic|ers|nah|bitch)\b|\?\?", re.I)
WARM = re.compile(r"\b(love|slnh|miss|kit|hun|haha|lol|okay|yes|yep)\b|😘|❤️|🥰", re.I)
JEALOUS = re.compile(r"\b(srey|girl|ex|tina|vivian|neary|khat|jealous|angry|mad)\b", re.I)
APOLOGY = re.compile(r"\b(sorry|khos|my bad)\b", re.I)
BUSY = re.compile(r"\b(busy|late|work|meeting|class|rean)\b", re.I)


def pages():
    files = sorted(EXPORT.glob("messages*.html"), key=page_number)
    messages = []
    prev = None
    for page in files:
        parsed, prev = parse_page(page, prev)
        messages.extend(parsed)
    return merge_turns(messages)


def label(text: str) -> str:
    cold = bool(COLD.search(text))
    warm = bool(WARM.search(text))
    if cold and not warm:
        return "cold"
    if warm and not cold:
        return "warm"
    if cold and warm:
        return "mixed"
    return "neutral"


def main() -> None:
    turns = [t for t in pages() if t.sender in {BOT, OWNER}]
    print(f"turns: {len(turns)}")

    # First message of each day.
    days = defaultdict(list)
    for i, t in enumerate(turns):
        days[t.ts.date()].append(i)

    morning_stats = Counter()
    examples = []
    for date, idxs in sorted(days.items()):
        first = turns[idxs[0]]
        if first.sender == OWNER:
            if GREETING.search(first.text):
                morning_stats["user_greeted"] += 1
            else:
                morning_stats["user_no_greeting"] += 1
                # first Muffin reply after this
                for j in idxs[1:]:
                    if turns[j].sender == BOT:
                        morning_stats[f"reply_{label(turns[j].text)}"] += 1
                        if len(examples) < 8 and label(turns[j].text) == "cold":
                            examples.append((first.text, turns[j].text))
                        break
        else:
            morning_stats["bot_first"] += 1
            if GREETING.search(first.text):
                morning_stats["bot_greeted"] += 1

    print("\nMORNING")
    print(dict(morning_stats))
    print("cold examples when user skipped morning:")
    for a, b in examples[:5]:
        print("  B:", a[:50], "->", b[:50])

    # Gaps: user returns after 6h
    gap_stats = Counter()
    gap_examples = []
    for i in range(1, len(turns)):
        if turns[i].sender != OWNER or turns[i - 1].sender != BOT:
            continue
        gap = (turns[i].ts - turns[i - 1].ts).total_seconds() / 3600
        if gap < 6:
            continue
        gap_stats["returns"] += 1
        # next bot turn
        for j in range(i + 1, min(i + 4, len(turns))):
            if turns[j].sender == BOT:
                gap_stats[label(turns[j].text)] += 1
                if len(gap_examples) < 8 and label(turns[j].text) == "cold":
                    gap_examples.append((turns[i].text, turns[j].text, gap))
                break
    print("\nAFTER 6H+ GAP")
    print(dict(gap_stats))
    for a, b, g in gap_examples[:5]:
        print(f"  {g:.1f}h B: {a[:45]} -> {b[:50]}")

    # Jealousy / apology
    for name, rx in [("JEALOUSY", JEALOUS), ("APOLOGY", APOLOGY)]:
        rows = []
        for i, t in enumerate(turns):
            if t.sender != OWNER or not rx.search(t.text):
                continue
            for j in range(i + 1, min(i + 4, len(turns))):
                if turns[j].sender == BOT:
                    rows.append(turns[j].text)
                    break
        print(f"\n{name}: {len(rows)}")
        print("  tone:", dict(Counter(label(r) for r in rows)))
        print("  top:", Counter(rows).most_common(5))

    # Bot-initiated messages: bot turn after >=2h since previous turn
    init = []
    for i in range(1, len(turns)):
        if turns[i].sender != BOT:
            continue
        gap = (turns[i].ts - turns[i - 1].ts).total_seconds() / 3600
        if gap >= 2:
            init.append((gap, turns[i].text))
    print(f"\nBOT-INITIATED (2h+): {len(init)}")
    print("  top:", Counter(t for _, t in init).most_common(10))

    # Reply length distribution by tone
    by_tone = defaultdict(list)
    for t in turns:
        if t.sender == BOT:
            by_tone[label(t.text)].append(len(t.text))
    print("\nLENGTH BY TONE")
    for k, v in by_tone.items():
        print(f"  {k}: n={len(v)} median={statistics.median(v):.1f}")


if __name__ == "__main__":
    main()
